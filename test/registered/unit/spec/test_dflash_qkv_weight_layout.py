"""DFlash KV materialization must respect post-load linear weight layouts."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models.dflash import DFlashAttention
from sglang.srt.speculative.dflash_utils import (
    can_dflash_slice_qkv_weight,
    can_dflash_use_fused_qkv_proj,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class LayoutAwareQKV(nn.Module):
    """CPU stand-in: expose packed storage and consume canonical values in forward."""

    def __init__(self, canonical, *, packed, bias=None):
        super().__init__()
        self.canonical = canonical
        physical = canonical.t().flip(1) if packed else canonical
        self.weight = nn.Parameter(physical, requires_grad=False)
        self.weight.aiter_trans_weight = packed
        self.bias = bias
        self.quant_method = UnquantizedLinearMethod()
        self.calls = 0

    def forward(self, hidden):
        self.calls += 1
        return torch.nn.functional.linear(hidden, self.canonical, self.bias), None


def attention(proj, q_size, kv_size, v_scale=None):
    result = DFlashAttention.__new__(DFlashAttention)
    nn.Module.__init__(result)
    result.qkv_proj = proj
    result.q_size = q_size
    result.kv_size = kv_size
    result.v_scale = v_scale
    return result


class TestDFlashQKVWeightLayout(unittest.TestCase):
    def test_dense_weights_keep_kv_only_path(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                proj = LayoutAwareQKV(torch.arange(48).reshape(12, 4).to(dtype), packed=False)
                attn = attention(proj, 8, 2)
                hidden = torch.ones(3, 4, dtype=dtype)
                expected = torch.nn.functional.linear(hidden, proj.canonical)[..., 8:]
                k, v = attn.kv_proj_only(hidden)
                torch.testing.assert_close(torch.cat((k, v), dim=-1), expected)
                self.assertEqual(proj.calls, 0)
                self.assertTrue(can_dflash_slice_qkv_weight(proj)[0])
                self.assertTrue(can_dflash_use_fused_qkv_proj(proj)[0])

    def test_packed_weights_use_layout_aware_fallback(self):
        for hidden_size in (4, 12):
            with self.subTest(hidden_size=hidden_size):
                canonical = torch.arange(12 * hidden_size, dtype=torch.float32).reshape(12, hidden_size) / 16
                proj = LayoutAwareQKV(canonical, packed=True)
                attn = attention(proj, 8, 2, v_scale=torch.tensor(0.5))
                hidden = torch.arange(3 * hidden_size, dtype=torch.float32).reshape(3, hidden_size) / 8
                qkv = torch.nn.functional.linear(hidden, canonical)
                k, v = attn.kv_proj_only(hidden)
                torch.testing.assert_close(k, qkv[..., 8:10])
                torch.testing.assert_close(v, qkv[..., 10:12] * 0.5)
                self.assertEqual(proj.calls, 1)
                eligible, reason = can_dflash_use_fused_qkv_proj(proj)
                self.assertFalse(eligible)
                self.assertIn("AITER-swizzled", reason)

    def test_ci_qkv_shape_is_rejected_before_empty_slice(self):
        weight = nn.Parameter(torch.empty(2048, 6144, dtype=torch.bfloat16), requires_grad=False)
        weight.aiter_trans_weight = True
        proj = SimpleNamespace(quant_method=UnquantizedLinearMethod(), weight=weight, bias=None)
        self.assertEqual(tuple(weight[4096:6144].shape), (0, 6144))
        self.assertFalse(can_dflash_slice_qkv_weight(proj)[0])
        self.assertFalse(can_dflash_use_fused_qkv_proj(proj)[0])

    def test_invalid_dense_storage_is_rejected(self):
        for weight in (None, torch.empty(2, 3, 4), torch.empty(3, 4, dtype=torch.uint8)):
            with self.subTest(shape=None if weight is None else tuple(weight.shape)):
                proj = SimpleNamespace(quant_method=UnquantizedLinearMethod(), weight=weight, bias=None)
                self.assertFalse(can_dflash_slice_qkv_weight(proj)[0])

    def test_quantized_and_biased_fused_guards_remain(self):
        proj = SimpleNamespace(quant_method=object(), weight=torch.empty(12, 4), bias=None)
        self.assertFalse(can_dflash_slice_qkv_weight(proj)[0])
        proj.quant_method = UnquantizedLinearMethod()
        proj.bias = torch.ones(12)
        self.assertTrue(can_dflash_slice_qkv_weight(proj)[0])
        self.assertFalse(can_dflash_use_fused_qkv_proj(proj)[0])


if __name__ == "__main__":
    unittest.main()
