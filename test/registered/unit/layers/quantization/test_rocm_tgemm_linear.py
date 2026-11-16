"""CPU regression checks for AITER linear dispatch and packed-weight ownership.

Load the production methods without importing ROCm extensions. Numerical backend
stand-ins deliberately scramble packed weights, so a dense fallback cannot
silently consume packed storage.
"""

import ast
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[5]
QUANT = ROOT / "python/sglang/srt/layers/quantization"


def load_definition(path, name, namespace, methods=None):
    tree = ast.parse(path.read_text())
    node = next(item for item in tree.body if getattr(item, "name", None) == name)
    if methods is not None:
        node.bases = []
        node.body = [
            item for item in node.body if getattr(item, "name", None) in methods
        ]
    module = ast.Module(
        body=[ast.ImportFrom("__future__", [ast.alias("annotations")], 0), node],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


class Flag:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


def pack(weight, layout=None):
    return weight.float().flip(1).to(weight.dtype)


class Backends:
    def __init__(self):
        self.calls = []

    def mm(self, x, weight, bias=None, *, otype, scale_a=None, scale_b=None):
        packed = getattr(weight, "is_shuffled", False)
        self.calls.append(("tgemm", weight, x.shape, packed))
        dense = weight.float().flip(1) if packed else weight.float()
        if scale_a is not None:
            return ((x.float() * scale_a) @ (dense * scale_b.t()).t()).to(otype)
        if packed:
            # The native non-viewable-input fallback requires dense weights.
            assert x.ndim == 2 and x.is_contiguous()
        result = F.linear(x.float(), dense, None if bias is None else bias.float())
        return result.to(otype)

    def ck(self, x, weight, x_scale, weight_scale, unused_bias, out_dtype):
        self.calls.append(("ck", weight, x.shape, True))
        assert x.is_contiguous()
        assert x_scale.shape == (x.shape[0], 1) and x_scale.is_contiguous()
        dense = weight.float().flip(1)
        return ((x.float() * x_scale) @ (dense * weight_scale.view(-1, 1)).t()).to(
            out_dtype
        )

    def quant(self, *args, **kwargs):
        raise AssertionError("pre-quantized tuples must not be quantized again")


class TestBF16TgemmWeightOwnership(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.backends = Backends()
        self.shuffle = Flag(True)
        self.namespace = {
            "torch": torch,
            "F": F,
            "Parameter": Parameter,
            "_is_cpu": False,
            "_is_cpu_amx_available": False,
            "_use_aiter": True,
            "is_gfx942_supported": lambda: True,
            "get_bool_env_var": lambda name: self.shuffle.get(),
            "shuffle_weight": pack,
            "use_intel_amx_backend": lambda layer: False,
            "tgemm": self.backends,
            "get_bf16_gemm_backend": lambda: SimpleNamespace(is_cutedsl=lambda: False),
        }
        self.method = load_definition(
            QUANT / "unquant.py",
            "UnquantizedLinearMethod",
            self.namespace,
            {"process_weights_after_loading", "apply"},
        )()

    def layer(self, n=16, k=64, dtype=torch.bfloat16):
        layer = torch.nn.Module()
        layer.weight = Parameter(torch.randn(n, k, dtype=dtype) / 8, False)
        return layer

    def test_dense_parameter_is_preserved_for_decode_and_kv_slicing(self):
        layer = self.layer()
        original = layer.weight
        dense = original.detach().clone()
        self.method.process_weights_after_loading(layer)
        self.assertIs(layer.weight, original)
        torch.testing.assert_close(layer.weight, dense, rtol=0, atol=0)
        self.assertIn("aiter_prefill_weight", dict(layer.named_parameters()))
        self.assertTrue(layer.aiter_prefill_weight.is_shuffled)
        hidden = torch.randn(3, 64)
        torch.testing.assert_close(
            F.linear(hidden, layer.weight[8:].float()),
            F.linear(hidden, dense.float())[..., 8:],
        )

    def test_small_m_dense_and_large_m_packed(self):
        layer = self.layer()
        dense = layer.weight.detach().clone()
        self.method.process_weights_after_loading(layer)
        for m in (1, 16, 17, 256):
            with self.subTest(m=m):
                x = torch.randn(1, m, 64, dtype=torch.bfloat16)
                bias = torch.randn(16, dtype=torch.bfloat16)
                output = self.method.apply(layer, x, bias)
                expected = F.linear(x.float(), dense.float(), bias.float()).to(x.dtype)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                name, used_weight, _, packed = self.backends.calls[-1]
                self.assertEqual(name, "tgemm")
                self.assertEqual(packed, m > 16)
                self.assertIs(
                    used_weight, layer.aiter_prefill_weight if m > 16 else layer.weight
                )

    def test_shared_method_does_not_mix_layer_layouts(self):
        aligned, unaligned = self.layer(), self.layer(n=15, k=63)
        for layer in (aligned, unaligned):
            self.method.process_weights_after_loading(layer)
        self.assertFalse(hasattr(unaligned, "aiter_prefill_weight"))
        for layer in (aligned, unaligned, aligned):
            x = torch.randn(17, layer.weight.shape[1], dtype=torch.bfloat16)
            output = self.method.apply(layer, x)
            torch.testing.assert_close(
                output, F.linear(x.float(), layer.weight.float()).to(x.dtype)
            )
            self.assertEqual(self.backends.calls[-1][3], layer is aligned)

    def test_shuffle_off_keeps_existing_dense_tgemm_path(self):
        self.shuffle.value = False
        layer = self.layer()
        self.method.process_weights_after_loading(layer)
        self.assertFalse(hasattr(layer, "aiter_prefill_weight"))
        x = torch.randn(17, 64, dtype=torch.bfloat16)
        output = self.method.apply(layer, x)
        self.assertIs(self.backends.calls[-1][1], layer.weight)
        self.assertFalse(self.backends.calls[-1][3])
        torch.testing.assert_close(
            output, F.linear(x.float(), layer.weight.float()).to(x.dtype)
        )

    def test_non_viewable_prefill_is_flattened_before_packed_dispatch(self):
        layer = self.layer()
        self.method.process_weights_after_loading(layer)
        x = torch.randn(2, 12, 64, dtype=torch.bfloat16).transpose(0, 1)
        self.assertFalse(x.is_contiguous())
        output = self.method.apply(layer, x)
        self.assertEqual(self.backends.calls[-1][2], (24, 64))
        self.assertEqual(output.shape, (12, 2, 16))
        torch.testing.assert_close(
            output, F.linear(x.float(), layer.weight.float()).to(x.dtype)
        )


class TestPackedFP8Tuples(unittest.TestCase):
    def setUp(self):
        self.backends = Backends()
        self.enabled = Flag(True)
        self.namespace = {
            "torch": torch,
            "_use_aiter": True,
            "envs": SimpleNamespace(SGLANG_ROCM_USE_AITER_LINEAR_FP8HIPB=self.enabled),
            "is_gfx942_supported": lambda: True,
            "cutlass_fp8_supported": lambda: False,
            "aiter": SimpleNamespace(
                gemm_a8w8_bpreshuffle=self.backends.ck,
                per_token_quant_hip=self.backends.quant,
            ),
        }
        ptpc = load_definition(
            QUANT / "fp8_utils.py", "apply_fp8_ptpc_linear", self.namespace
        )
        self.namespace["apply_fp8_ptpc_linear"] = ptpc
        self.namespace["apply_fp8_linear"] = lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("packed CHANNEL weights reached the dense [K,N] path")
        )
        self.channel = object()
        self.namespace["QuantizationStrategy"] = SimpleNamespace(CHANNEL=self.channel)
        cls = load_definition(
            QUANT / "compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py",
            "CompressedTensorsW8A8Fp8",
            self.namespace,
            {"apply_weights"},
        )
        self.method = cls()
        self.method.strategy = self.channel
        self.method.weight_block_size = None
        self.dense = (
            torch.arange(32 * 64, dtype=torch.float32).remainder(13).view(32, 64) / 8
        )
        self.dense = self.dense.to(torch.float8_e4m3fnuz).float()
        self.layer = torch.nn.Module()
        self.layer.weight = Parameter(pack(self.dense).to(torch.float8_e4m3fnuz), False)
        self.layer.weight.is_shuffled = True
        self.layer.weight_scale = Parameter(
            torch.linspace(0.01, 0.02, 32).view(32, 1), False
        )
        self.layer.input_scale = None
        module = ModuleType("aiter.tuned_gemm")
        module.tgemm = self.backends
        parent = ModuleType("aiter")
        parent.__path__ = []
        parent.tuned_gemm = module
        self.modules = patch.dict(
            sys.modules, {"aiter": parent, "aiter.tuned_gemm": module}
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def activation(self, m):
        return (
            torch.arange(m * 64, dtype=torch.float32).remainder(11).view(1, m, 64) / 8
        ).to(torch.float8_e4m3fnuz)

    def reference(self, qx, scale, dtype, bias):
        x = qx.float().reshape(-1, 64)
        scale = scale.reshape(-1, 1) if scale.numel() > 1 else scale
        output = ((x * scale) @ (self.dense * self.layer.weight_scale).t()).to(dtype)
        return (output + bias).view(*qx.shape[:-1], 32)

    def test_tuple_dispatch_preserves_layout_scale_and_dtype(self):
        for m in (1, 16, 17):
            qx = self.activation(m)
            for enabled in (False, True):
                self.enabled.value = enabled
                for per_token in (False, True):
                    scale = (
                        torch.linspace(0.1, 0.2, m).view(m, 1)
                        if per_token
                        else torch.tensor(0.125)
                    )
                    for dtype in (None, torch.bfloat16, torch.float16):
                        with self.subTest(
                            m=m, enabled=enabled, per_token=per_token, dtype=dtype
                        ):
                            output_dtype = dtype or torch.bfloat16
                            bias = torch.linspace(-0.1, 0.1, 32).to(output_dtype)
                            value = (qx, scale) if dtype is None else (qx, scale, dtype)
                            output = self.method.apply_weights(self.layer, value, bias)
                            expected = self.reference(qx, scale, output_dtype, bias)
                            self.assertEqual(output.dtype, output_dtype)
                            torch.testing.assert_close(output, expected, rtol=0, atol=0)
                            route = (
                                "tgemm"
                                if enabled
                                and m > 16
                                and per_token
                                and output_dtype == torch.bfloat16
                                else "ck"
                            )
                            self.assertEqual(self.backends.calls[-1][0], route)

    def test_invalid_scale_counts_are_rejected_before_kernel(self):
        qx = self.activation(17)
        for scale in (torch.ones(2), torch.ones(17, 2)):
            with self.subTest(shape=scale.shape):
                with self.assertRaisesRegex(ValueError, "scale"):
                    self.method.apply_weights(self.layer, (qx, scale))
        self.layer.weight_scale = Parameter(torch.ones(31, 1), False)
        with self.assertRaisesRegex(ValueError, "scale"):
            self.method.apply_weights(self.layer, (qx, torch.ones(17, 1)))

    def test_non_viewable_tuple_preserves_leading_dimensions(self):
        qx = self.activation(34).view(2, 17, 64).transpose(0, 1)
        scale = torch.ones(34, 1) / 8
        bias = torch.zeros(32, dtype=torch.float16)
        output = self.method.apply_weights(self.layer, (qx, scale, torch.float16), bias)
        self.assertEqual(output.shape, (17, 2, 32))
        torch.testing.assert_close(
            output, self.reference(qx, scale, torch.float16, bias), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
