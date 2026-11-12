"""CPU regressions for the swizzled GEMV caller-owned output contract."""

import importlib.util
import unittest
from pathlib import Path

import torch

_KERNEL_PATH = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/kernels/ops/gemm/swizzled_bf16_gemv.py"
)
_SPEC = importlib.util.spec_from_file_location("output_guard_kernel", _KERNEL_PATH)
_KERNEL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_KERNEL)


class TestSwizzledGemvOutputGuard(unittest.TestCase):
    def setUp(self):
        self.x = torch.empty(3, 4096, dtype=torch.bfloat16)
        self.weight_storage = torch.empty(16 * 4096, dtype=torch.bfloat16)
        self.weight = self.weight_storage.view(16, 4096).t()

    def assert_rejected_before_launch(self, output):
        self.assertFalse(
            _KERNEL.can_write_swizzled_gemv_output(self.x, self.weight, output)
        )
        # These are CPU tensors: a valid guard must reject without trying
        # to launch the Triton kernel on an unavailable device.
        with self.assertRaisesRegex(ValueError, "overlapping"):
            _KERNEL.swizzled_gemv(self.x, self.weight, output=output)

    def test_dlpack_input_alias_is_rejected(self):
        output = torch.utils.dlpack.from_dlpack(self.x[:, 100:116])
        self.assertNotEqual(
            output.untyped_storage().data_ptr(), self.x.untyped_storage().data_ptr()
        )
        self.assert_rejected_before_launch(output)

    def test_dlpack_weight_alias_is_rejected(self):
        output = torch.utils.dlpack.from_dlpack(
            self.weight_storage[100:148].view(3, 16)
        )
        self.assertNotEqual(
            output.untyped_storage().data_ptr(),
            self.weight.untyped_storage().data_ptr(),
        )
        self.assert_rejected_before_launch(output)

    def test_same_storage_disjoint_views_remain_rejected(self):
        storage = torch.empty(3 * 4096 + 3 * 16, dtype=torch.bfloat16)
        x = storage[: 3 * 4096].view(3, 4096)
        output = storage[3 * 4096 :].view(3, 16)
        self.assertFalse(
            _KERNEL.can_write_swizzled_gemv_output(x, self.weight, output)
        )

    def test_disjoint_output_layouts_remain_eligible(self):
        for rows in (1, 3, 8, 16, 32):
            x = torch.empty(rows, 4096, dtype=torch.bfloat16)
            base = torch.empty(rows, 39, dtype=torch.bfloat16)
            outputs = {
                "contiguous": torch.empty(rows, 16, dtype=torch.bfloat16),
                "row_gap": base[:, 3:19],
                "column_stride": base[:, 3:35:2],
            }
            for name, output in outputs.items():
                for dlpack in (False, True):
                    with self.subTest(rows=rows, layout=name, dlpack=dlpack):
                        view = (
                            torch.utils.dlpack.from_dlpack(output) if dlpack else output
                        )
                        # DLPack may normalize the irrelevant row stride for
                        # a single-row tensor. Keep the declared output layout.
                        view = view.as_strided(output.shape, output.stride())
                        self.assertTrue(
                            _KERNEL.can_write_swizzled_gemv_output(x, self.weight, view)
                        )

    def test_invalid_output_metadata_remains_rejected(self):
        invalid_outputs = (
            torch.empty(16, dtype=torch.bfloat16),
            torch.empty(3, 17, dtype=torch.bfloat16),
            torch.empty(3, 16, dtype=torch.float32),
            torch.empty(16, dtype=torch.bfloat16).as_strided((3, 16), (0, 1)),
            torch.empty(80, dtype=torch.bfloat16).as_strided((3, 16), (15, 1)),
        )
        for output in invalid_outputs:
            with self.subTest(shape=tuple(output.shape), stride=output.stride()):
                self.assertFalse(
                    _KERNEL.can_write_swizzled_gemv_output(self.x, self.weight, output)
                )


if __name__ == "__main__":
    unittest.main()
