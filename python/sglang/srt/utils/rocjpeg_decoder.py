"""rocJPEG (torchvision on ROCm) JPEG decoding for multimodal image processors."""

from __future__ import annotations

import logging
from functools import lru_cache
from io import BytesIO
from typing import Union
from urllib.parse import unquote, urlparse

import numpy as np
import torch
import triton
import triton.language as tl
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


def _ycbcr_from_rgb(kr: float, kb: float) -> np.ndarray:
    kg = 1 - kr - kb
    return np.array(
        [
            [kr, kg, kb],
            [-kr / (2 - 2 * kb), -kg / (2 - 2 * kb), 0.5],
            [0.5, -kg / (2 - 2 * kr), -kb / (2 - 2 * kr)],
        ]
    )


# rocJPEG converts YCbCr to RGB with BT.709 coefficients, while JPEG (JFIF) uses
# BT.601. Both are full range, so one matrix maps its output to the JFIF result.
BT709_TO_BT601 = np.linalg.inv(_ycbcr_from_rgb(0.299, 0.114)) @ _ycbcr_from_rgb(
    0.2126, 0.0722
)
_BT709_TO_BT601_ARGS = tuple(BT709_TO_BT601.ravel().tolist())


@triton.jit
def _bt709_to_bt601_kernel(
    src,
    dst,
    n,
    width,
    stride_c,
    stride_h,
    m00,
    m01,
    m02,
    m10,
    m11,
    m12,
    m20,
    m21,
    m22,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    src_offs = (offs // width) * stride_h + offs % width
    r = tl.load(src + src_offs, mask=mask).to(tl.float32)
    g = tl.load(src + stride_c + src_offs, mask=mask).to(tl.float32)
    b = tl.load(src + 2 * stride_c + src_offs, mask=mask).to(tl.float32)
    out_r = tl.floor(m00 * r + m01 * g + m02 * b + 0.5)
    out_g = tl.floor(m10 * r + m11 * g + m12 * b + 0.5)
    out_b = tl.floor(m20 * r + m21 * g + m22 * b + 0.5)
    tl.store(
        dst + offs, tl.minimum(tl.maximum(out_r, 0.0), 255.0).to(tl.uint8), mask=mask
    )
    tl.store(
        dst + n + offs,
        tl.minimum(tl.maximum(out_g, 0.0), 255.0).to(tl.uint8),
        mask=mask,
    )
    tl.store(
        dst + 2 * n + offs,
        tl.minimum(tl.maximum(out_b, 0.0), 255.0).to(tl.uint8),
        mask=mask,
    )


def bt709_to_bt601(image: torch.Tensor) -> torch.Tensor:
    """Correct a (3, H, W) uint8 rocJPEG output in one pass; returns a contiguous tensor."""
    if image.stride(2) != 1:
        image = image.contiguous()
    _, height, width = image.shape
    n = height * width
    out = torch.empty((3, height, width), dtype=torch.uint8, device=image.device)
    _bt709_to_bt601_kernel[(triton.cdiv(n, 1024),)](
        image,
        out,
        n,
        width,
        image.stride(0),
        image.stride(1),
        *_BT709_TO_BT601_ARGS,
        BLOCK=1024,
    )
    return out


@lru_cache(maxsize=1)
def _rocjpeg_matches_pil() -> bool:
    """Decode a color test image with rocJPEG and compare the corrected output with PIL."""
    try:
        from torchvision.io import decode_jpeg  # lazy: ~1 s of torch._dynamo

        ramp = np.linspace(0, 255, 128, dtype=np.uint8)
        r, g = np.meshgrid(ramp, ramp[::-1])
        buf = BytesIO()
        Image.fromarray(np.stack([r, g, 255 - r], axis=-1)).save(
            buf, format="JPEG", quality=95, subsampling=0
        )
        data = buf.getvalue()
        expected = torch.from_numpy(np.array(Image.open(BytesIO(data)).convert("RGB")))
        decoded = decode_jpeg(
            torch.frombuffer(bytearray(data), dtype=torch.uint8), device="cuda"
        )
        corrected = bt709_to_bt601(decoded).cpu().float()
        error = (corrected - expected.permute(2, 0, 1).float()).abs().mean().item()
    except Exception as e:
        logger.warning("rocJPEG decode is unavailable, decoding JPEG with PIL: %s", e)
        return False
    if error > 1.0:
        logger.warning(
            "rocJPEG output differs from PIL (mean abs error %.2f), decoding JPEG with PIL",
            error,
        )
        return False
    logger.info("Decoding JPEG with rocJPEG (mean abs error vs PIL %.2f)", error)
    return True


def rocjpeg_decode_enabled(gpu_image_decode) -> bool:
    return (
        gpu_image_decode is True
        and is_hip()
        and envs.SGLANG_ROCM_GPU_JPEG_DECODE.get()
        and _rocjpeg_matches_pil()
    )


def image_input_bytes(image_file: Union[str, bytes, ImageData]) -> bytes:
    """Raw bytes of any image input load_image accepts, except a PIL image."""
    if isinstance(image_file, ImageData):
        image_file = image_file.url
    if isinstance(image_file, str) and image_file.startswith("file://"):
        image_file = unquote(urlparse(image_file).path)
    return get_image_bytes(image_file)


def rocjpeg_can_decode(data: bytes) -> bool:
    """Whether rocJPEG's hardware decoder accepts this JPEG: complete, 8-bit
    baseline or extended sequential, 1 or 3 components, at least 64x64, not 4:1:1."""
    if not (data.startswith(b"\xff\xd8") and data.endswith(b"\xff\xd9")):
        return False
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            return False
        marker = data[i + 1]
        if marker == 0xFF:
            i += 1
            continue
        if marker == 0x01 or 0xD0 <= marker <= 0xD8:
            i += 2
            continue
        if marker in (0xC0, 0xC1):
            if i + 12 > len(data):
                return False
            height = int.from_bytes(data[i + 5 : i + 7], "big")
            width = int.from_bytes(data[i + 7 : i + 9], "big")
            components = data[i + 9]
            if data[i + 4] != 8 or components not in (1, 3) or min(height, width) < 64:
                return False
            return components == 1 or data[i + 11] >> 4 != 4
        if marker == 0xDA or (
            0xC2 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC)
        ):
            return False
        i += 2 + int.from_bytes(data[i + 2 : i + 4], "big")
    return False


class DeferredJpeg:
    """JPEG bytes held back so that a request's images are decoded in one rocJPEG call."""

    __slots__ = ("data",)

    def __init__(self, data: bytes):
        self.data = data


def decode_deferred_jpegs(images: list) -> None:
    """Replace the DeferredJpeg entries of `images` in place with RGB images,
    decoded in one rocJPEG call; if that call fails they are decoded with PIL."""
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
        logger.debug("rocJPEG batch decode failed, decoding with PIL: %s", e)
        for i in indices:
            try:
                image, _ = load_image(images[i].data, gpu_image_decode=False)
                images[i] = image.convert("RGB")
            except CLIENT_MEDIA_EXCEPTIONS as e:
                # Same client error that _load_single_item raises for these.
                raise ValueError(f"Could not decode image: {e}") from e
        return
    for i, image in zip(indices, decoded):
        images[i] = bt709_to_bt601(image)
