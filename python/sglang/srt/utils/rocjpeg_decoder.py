"""rocJPEG (torchvision on ROCm) JPEG decoding for multimodal image processors."""

from __future__ import annotations

import logging
from functools import lru_cache
from io import BytesIO
from typing import Optional, Union
from urllib.parse import unquote, urlparse

import numpy as np
import torch
from PIL import Image

from sglang.srt.environ import envs
from sglang.srt.utils.common import (
    CLIENT_MEDIA_EXCEPTIONS,
    ImageData,
    get_image_bytes,
    is_hip,
    load_image,
)

logger = logging.getLogger(__name__)


# Full-range RGB to YCbCr for luma weights kr and kb:
# https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion
def _ycbcr_from_rgb(kr: float, kb: float) -> np.ndarray:
    kg = 1 - kr - kb
    return np.array(
        [
            [kr, kg, kb],
            [-kr / (2 - 2 * kb), -kg / (2 - 2 * kb), 0.5],
            [0.5, -kg / (2 - 2 * kr), -kb / (2 - 2 * kr)],
        ]
    )


# JPEG (JFIF, ITU-T T.871 section 7) uses the ITU-R BT.601 luma weights
# kr = 0.299, kb = 0.114. rocJPEG before 1.10.0 converts YCbCr to RGB with the
# ITU-R BT.709 weights kr = 0.2126, kb = 0.0722 (R = Y + 1.5748 Cr, ...):
# https://github.com/ROCm/rocJPEG/blob/rocm-7.2.4/src/rocjpeg_hip_kernels.cpp#L76-L78
# 1.10.0 switched to BT.601 (https://github.com/ROCm/rocm-systems/pull/11520),
# and BT.709 remains a build option, so the self-check decides whether to correct.
# Both are full range, so inv(M601) @ M709 maps the BT.709 output to the JFIF result:
#   [[ 0.9136,  0.0785,  0.0079],
#    [-0.1050,  1.1722, -0.0671],
#    [ 0.0096,  0.0322,  0.9582]]
BT709_TO_BT601 = np.linalg.inv(_ycbcr_from_rgb(0.299, 0.114)) @ _ycbcr_from_rgb(
    0.2126, 0.0722
)


def bt709_to_bt601(image: torch.Tensor) -> torch.Tensor:
    """Map a (3, H, W) uint8 image decoded with BT.709 colors to JPEG's BT.601."""
    matrix = torch.tensor(BT709_TO_BT601, dtype=torch.float32, device=image.device)
    rgb = matrix @ image.reshape(3, -1).float()
    return rgb.round_().clamp_(0, 255).to(torch.uint8).reshape(image.shape)


@lru_cache(maxsize=1)
def _rocjpeg_needs_correction() -> Optional[bool]:
    """Decode a 128x128 color test image with rocJPEG once per process and compare
    it with PIL, as decoded and after bt709_to_bt601. Returns whether the correction
    is needed, or None if rocJPEG is unusable. Only the test image is copied to the
    host; request images stay on the GPU."""
    try:
        from torchvision.io import decode_jpeg  # lazy: ~1 s of torch._dynamo

        ramp = np.linspace(0, 255, 128, dtype=np.uint8)
        r, g = np.meshgrid(ramp, ramp[::-1])
        buf = BytesIO()
        Image.fromarray(np.stack([r, g, 255 - r], axis=-1)).save(
            buf, format="JPEG", quality=95, subsampling=0
        )
        data = buf.getvalue()
        expected = torch.from_numpy(np.array(Image.open(BytesIO(data)))).permute(
            2, 0, 1
        )
        decoded = decode_jpeg(
            torch.frombuffer(bytearray(data), dtype=torch.uint8), device="cuda"
        )
        errors = [
            (output.cpu().float() - expected.float()).abs().mean().item()
            for output in (decoded, bt709_to_bt601(decoded))
        ]
        if min(errors) > 1.0:
            raise ValueError(
                f"output differs from PIL (mean abs error {errors[0]:.2f}, "
                f"{errors[1]:.2f} with color correction)"
            )
    except Exception as e:
        logger.warning(
            "rocJPEG GPU JPEG decoding: off, using PIL: %s. It needs a torchvision "
            "built with rocJPEG and, for ROCm's bundled libva, LIBVA_DRIVERS_PATH="
            "<ROCm>/lib/rocm_sysdeps/lib and LIBVA_DRIVER_NAME=radeonsi.",
            e,
        )
        return None
    correct = errors[1] < errors[0]
    logger.info(
        "rocJPEG GPU JPEG decoding: on, color correction %s (mean abs error vs PIL %.2f)",
        "on" if correct else "off",
        min(errors),
    )
    return correct


def rocjpeg_decode_enabled(gpu_image_decode) -> bool:
    return (
        gpu_image_decode is True
        and is_hip()
        and envs.SGLANG_ROCM_GPU_JPEG_DECODE.get()
        and _rocjpeg_needs_correction() is not None
    )


def image_input_bytes(image_file: Union[str, bytes, ImageData]) -> bytes:
    """Raw bytes of any image input load_image accepts, except a PIL image."""
    if isinstance(image_file, ImageData):
        image_file = image_file.url
    if isinstance(image_file, str) and image_file.startswith("file://"):
        image_file = unquote(urlparse(image_file).path)
    return get_image_bytes(image_file)


def rocjpeg_can_decode(data: bytes) -> bool:
    """Whether rocJPEG's hardware decoder accepts this JPEG: complete, 8-bit, not
    progressive, gray or YCbCr, at least 64x64, not 4:1:1. Reads only the header."""
    try:
        image = Image.open(BytesIO(data), formats=["JPEG"])
    except Exception:
        return False
    return (
        data.endswith(b"\xff\xd9")
        and image.mode in ("L", "RGB")
        and min(image.size) >= 64
        and "progressive" not in image.info
        and image.layer[0][1] != 4  # horizontal luma sampling factor, 4 in 4:1:1
    )


class DeferredJpeg:
    """JPEG bytes held back so that a request's images are decoded in one rocJPEG call."""

    def __init__(self, data: bytes):
        self.data = data


def decode_deferred_jpegs(
    images: list, color_correction: Optional[bool] = None
) -> None:
    """Replace the DeferredJpeg entries of `images` in place with RGB images,
    decoded in one rocJPEG call; if that call fails they are decoded with PIL.
    color_correction forces the BT.709-to-BT.601 correction on or off; by
    default SGLANG_ROCM_JPEG_COLOR_CORRECTION, then the self-check, decide."""
    indices = [i for i, image in enumerate(images) if isinstance(image, DeferredJpeg)]
    if not indices:
        return
    from torchvision.io import ImageReadMode, decode_jpeg

    encoded = [
        torch.frombuffer(bytearray(images[i].data), dtype=torch.uint8) for i in indices
    ]
    try:
        decoded = decode_jpeg(encoded, mode=ImageReadMode.RGB, device="cuda")
    except Exception as e:
        logger.warning("rocJPEG batch decode failed, decoding with PIL: %s", e)
        for i in indices:
            try:
                image, _ = load_image(images[i].data, gpu_image_decode=False)
                images[i] = image.convert("RGB")
            except CLIENT_MEDIA_EXCEPTIONS as e:
                # Same client error that _load_single_item raises for these.
                raise ValueError(f"Could not decode image: {e}") from e
        return
    if color_correction is None:
        color_correction = envs.SGLANG_ROCM_JPEG_COLOR_CORRECTION.get()
    if color_correction is None:
        color_correction = _rocjpeg_needs_correction()
    for i, image in zip(indices, decoded):
        images[i] = bt709_to_bt601(image) if color_correction else image
    logger.info("rocJPEG request batch decoded: %d image(s)", len(indices))
