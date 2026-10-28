"""Unit tests for JPEG decoding with rocJPEG on ROCm.

No GPU: torchvision's decode_jpeg and the rocJPEG self-check are mocked, and the
color correction matrix is checked against the JFIF (BT.601) and BT.709 formulas.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import asyncio
import base64
import concurrent.futures
import io
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from PIL import Image

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.utils import common, rocjpeg_decoder
from sglang.srt.utils.rocjpeg_decoder import (
    BT709_TO_BT601,
    DeferredJpeg,
    decode_deferred_jpegs,
    rocjpeg_can_decode,
)
from sglang.test.test_utils import CustomTestCase

BASE = "sglang.srt.multimodal.processors.base_processor"


class _StubProcessor(BaseMultimodalProcessor):
    async def process_mm_data_async(self, *args, **kwargs):
        raise NotImplementedError


def _image(size=(96, 80), mode="RGB") -> Image.Image:
    arr = (np.random.RandomState(0).rand(size[1], size[0], 3) * 255).astype("uint8")
    return Image.fromarray(arr, "RGB").convert(mode)


def _jpeg(size=(96, 80), mode="RGB", **kwargs) -> bytes:
    buf = io.BytesIO()
    _image(size, mode).save(buf, format="JPEG", quality=90, **kwargs)
    return buf.getvalue()


def _png(size=(96, 80)) -> bytes:
    buf = io.BytesIO()
    _image(size).save(buf, format="PNG")
    return buf.getvalue()


class TestRocJpegCanDecode(CustomTestCase):
    def test_baseline_jpegs_are_accepted(self):
        self.assertTrue(rocjpeg_can_decode(_jpeg()))
        self.assertTrue(rocjpeg_can_decode(_jpeg(subsampling=0)))
        self.assertTrue(rocjpeg_can_decode(_jpeg(mode="L")))

    def test_jpegs_the_hardware_rejects_are_not_accepted(self):
        self.assertFalse(rocjpeg_can_decode(_jpeg(progressive=True)))
        self.assertFalse(rocjpeg_can_decode(_jpeg(mode="CMYK")))
        self.assertFalse(rocjpeg_can_decode(_jpeg(size=(32, 96))))

    def test_incomplete_or_non_jpeg_data_is_not_accepted(self):
        self.assertFalse(rocjpeg_can_decode(_jpeg()[:-100]))
        self.assertFalse(rocjpeg_can_decode(_png()))
        self.assertFalse(rocjpeg_can_decode(b""))


class TestColorCorrection(CustomTestCase):
    def test_matrix_maps_bt709_rgb_to_jfif_rgb(self):
        rng = np.random.RandomState(0)
        y = rng.uniform(0, 255, 256)
        cb, cr = rng.uniform(-128, 127, (2, 256))
        jfif = np.stack(
            [y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb]
        )
        bt709 = np.stack(
            [y + 1.5748 * cr, y - 0.187324 * cb - 0.468124 * cr, y + 1.8556 * cb]
        )
        np.testing.assert_allclose(BT709_TO_BT601 @ bt709, jfif, atol=1e-3)


class TestRocJpegGate(CustomTestCase):
    def _enabled(self, *, env=True, hip=True, usable=True, gpu_image_decode=True):
        with (
            envs.SGLANG_ROCM_GPU_JPEG_DECODE.override(env),
            patch.object(rocjpeg_decoder, "is_hip", return_value=hip),
            patch.object(rocjpeg_decoder, "_rocjpeg_matches_pil", return_value=usable),
        ):
            return rocjpeg_decoder.rocjpeg_decode_enabled(gpu_image_decode)

    def test_enabled_only_with_env_hip_and_a_usable_decoder(self):
        self.assertTrue(self._enabled())
        self.assertFalse(self._enabled(env=False))
        self.assertFalse(self._enabled(hip=False))
        self.assertFalse(self._enabled(usable=False))
        self.assertFalse(self._enabled(gpu_image_decode=False))
        self.assertFalse(self._enabled(gpu_image_decode="nvjpeg_fancy"))


class TestDeferredJpegLoading(CustomTestCase):
    def _load(self, data, **kwargs):
        with patch(f"{BASE}.rocjpeg_decode_enabled", return_value=True):
            return _StubProcessor._load_single_item(data, Modality.IMAGE, **kwargs)

    def test_decodable_jpeg_is_deferred(self):
        data = _jpeg()
        for item in (data, "data:image/jpeg;base64," + base64.b64encode(data).decode()):
            deferred = self._load(item)
            self.assertIsInstance(deferred, DeferredJpeg)
            self.assertEqual(deferred.data, data)

    def test_other_images_are_decoded_with_pil_only(self):
        with patch(f"{BASE}.load_image", wraps=common.load_image) as load:
            for data in (_png(), _jpeg(progressive=True)):
                image = self._load(data)
                self.assertIsInstance(image, Image.Image)
                self.assertEqual(image.mode, "RGB")
        self.assertEqual([call.args[1] for call in load.call_args_list], [False, False])

    def test_images_keeping_alpha_are_not_deferred(self):
        with patch.object(common, "is_cuda", return_value=False):
            image = self._load(_jpeg(), discard_alpha_channel=False)
        self.assertIsInstance(image, Image.Image)


class TestDecodeDeferredJpegs(CustomTestCase):
    def test_one_batch_call_decodes_and_corrects_in_order(self):
        first, second, other = _jpeg(), _jpeg(size=(128, 64)), _image()
        decoded = [
            torch.zeros((3, 80, 96), dtype=torch.uint8),
            torch.zeros((3, 64, 128), dtype=torch.uint8),
        ]
        corrected = [torch.ones_like(image) for image in decoded]
        images = [DeferredJpeg(first), other, DeferredJpeg(second)]
        with (
            patch("torchvision.io.decode_jpeg", return_value=decoded) as decode,
            patch.object(
                rocjpeg_decoder, "bt709_to_bt601", side_effect=corrected
            ) as correct,
        ):
            decode_deferred_jpegs(images)

        decode.assert_called_once()
        encoded = decode.call_args.args[0]
        self.assertEqual([t.numpy().tobytes() for t in encoded], [first, second])
        self.assertEqual(decode.call_args.kwargs["device"], "cuda")
        for call, image in zip(correct.call_args_list, decoded):
            self.assertIs(call.args[0], image)
        self.assertIs(images[0], corrected[0])
        self.assertIs(images[1], other)
        self.assertIs(images[2], corrected[1])

    def test_batch_failure_falls_back_to_pil(self):
        data = _jpeg()
        images = [DeferredJpeg(data)]
        with patch("torchvision.io.decode_jpeg", side_effect=RuntimeError("failed")):
            decode_deferred_jpegs(images)
        self.assertIsInstance(images[0], Image.Image)
        reference = Image.open(io.BytesIO(data)).convert("RGB")
        np.testing.assert_array_equal(np.asarray(images[0]), np.asarray(reference))

    def test_undecodable_data_is_a_client_error(self):
        images = [DeferredJpeg(b"\xff\xd8" + b"\x00" * 64 + b"\xff\xd9")]
        with patch("torchvision.io.decode_jpeg", side_effect=RuntimeError("failed")):
            with self.assertRaises(ValueError):
                decode_deferred_jpegs(images)

    def test_fast_loader_decodes_deferred_jpegs(self):
        processor = object.__new__(_StubProcessor)
        processor.io_executor = concurrent.futures.ThreadPoolExecutor(1)
        self.addCleanup(processor.io_executor.shutdown)
        other = _image()
        futures = []
        for idx, item in enumerate((DeferredJpeg(b"jpeg"), other)):
            future = concurrent.futures.Future()
            future.set_result(item)
            futures.append((Modality.IMAGE, idx, future))
        processor._submit_mm_data_loading_tasks_simple = Mock(
            side_effect=[futures, [], []]
        )
        decoded = torch.zeros((3, 80, 96), dtype=torch.uint8)

        def decode(images):
            images[0] = decoded

        with patch(f"{BASE}.decode_deferred_jpegs", side_effect=decode) as batch:
            output = asyncio.run(
                processor.fast_load_mm_data(
                    prompt="<image><image>",
                    multimodal_tokens=Mock(),
                    image_data=["a", "b"],
                )
            )
        batch.assert_called_once()
        self.assertIs(output.images[0], decoded)
        self.assertIs(output.images[1], other)


if __name__ == "__main__":
    unittest.main()
