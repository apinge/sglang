import unittest

import torch

from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd
from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    fused_qkv_split_gdn_prefill,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.utils.common import is_gfx942_supported
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")

D = 128

from sglang.srt.layers.attention.linear.kernels.gdn_flydsl import (
    FlyDSLGDNKernel,
    flydsl_gdn_decode_ops,
    flydsl_gdn_prefill_op,
)
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel

_HAS_FLYDSL_GDN = (
    torch.cuda.is_available()
    and flydsl_gdn_decode_ops() is not None
    and flydsl_gdn_prefill_op() is not None
)


def _split(qkvz, H, HV):
    kd, vd = H * D, HV * D
    return qkvz.split([kd, kd, vd, vd], dim=-1)


@unittest.skipIf(
    not (_HAS_FLYDSL_GDN and is_gfx942_supported()),
    "requires gfx942 and aiter.ops.flydsl GDN kernels",
)
class TestGdnFlyDSLGfx942(unittest.TestCase):
    """The FlyDSL GDN kernels must reproduce the Triton chain they replace in
    qwen3_5 (ratio 3: 8 k-heads / 24 v-heads), including both caches."""

    H, HV = 8, 24

    def test_fused_decode_matches_triton_chain(self):
        """q/k conv channels are shared by 3 v-head workgroups; the conv cache must
        be shifted exactly once, and a -1 (CUDA-graph pad) slot left untouched.
        Both FlyDSLGDNKernel entry points must match TritonGDNKernel."""
        torch.manual_seed(0)
        H, HV, B, slots = self.H, self.HV, 9, 23
        qkv_dim = 2 * H * D + HV * D
        qkvz = torch.randn(B, qkv_dim + HV * D, device="cuda", dtype=torch.bfloat16)
        ba = torch.randn(B, 2 * HV, device="cuda", dtype=torch.bfloat16)
        conv_w = (torch.randn(qkv_dim, 4, device="cuda") * 0.5).to(torch.bfloat16)
        conv = torch.randn(slots, qkv_dim, 3, device="cuda").to(torch.bfloat16)
        ssm = torch.randn(slots, HV, D, D, device="cuda") * 0.1
        A_log = torch.rand(HV, device="cuda") * 2 - 1
        dt_bias = torch.rand(HV, device="cuda").to(torch.bfloat16)
        idx = (torch.randperm(slots - 1, device="cuda")[:B] + 1).to(torch.int32)
        idx[-1] = -1
        live = idx >= 0
        common = dict(A_log=A_log, dt_bias=dt_bias, scale=D**-0.5, cache_indices=idx)
        triton, flydsl = TritonGDNKernel(), FlyDSLGDNKernel(TritonGDNKernel())

        def triton_chain(conv_pool, ssm_pool):
            q, k, v, z = _split(qkvz, H, HV)
            mixed = causal_conv1d_update(
                torch.cat((q, k, v), dim=-1),
                conv_pool,
                conv_w,
                None,
                "silu",
                conv_state_indices=idx,
            )
            o = triton.packed_decode(
                mixed,
                ba[:, HV:].contiguous(),
                ba[:, :HV].contiguous(),
                ssm_states=ssm_pool,
                num_v_heads=HV,
                head_v_dim=D,
                **common,
            )
            return mixed, o, z.reshape(B, HV, D)

        conv_ref, ssm_ref = conv.clone(), ssm.clone()
        mixed, o_ref, z_ref = triton_chain(conv_ref, ssm_ref)

        conv_f, ssm_f = conv.clone(), ssm.clone()
        o, z = flydsl.fused_qkvzba_conv1d_decode(
            qkvz,
            ba,
            conv_states=conv_f,
            conv_weights=conv_w,
            ssm_states=ssm_f,
            num_k_heads=H,
            num_v_heads=HV,
            head_v_dim=D,
            **common,
        )
        torch.testing.assert_close(
            o[0, live].float(), o_ref[0, live].float(), rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(z, z_ref, rtol=0, atol=0)
        torch.testing.assert_close(ssm_f, ssm_ref, rtol=0, atol=1e-5)
        torch.testing.assert_close(conv_f, conv_ref, rtol=0, atol=0)

        ssm_p = ssm.clone()
        o_p = flydsl.packed_decode(
            mixed,
            ba[:, HV:].contiguous(),
            ba[:, :HV].contiguous(),
            ssm_states=ssm_p,
            num_v_heads=HV,
            head_v_dim=D,
            **common,
        )
        torch.testing.assert_close(
            o_p[0, live].float(), o_ref[0, live].float(), rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(ssm_p, ssm_ref, rtol=0, atol=1e-5)

        # The FlyDSL kernel reads bf16 activations only; fp16 must take the Triton path.
        mixed16, a16, b16 = mixed.half(), ba[:, HV:].half(), ba[:, :HV].half()
        kw16 = dict(common, dt_bias=dt_bias.half(), num_v_heads=HV, head_v_dim=D)
        ssm_t, ssm_h = ssm.clone(), ssm.clone()
        o_t = triton.packed_decode(mixed16, a16, b16, ssm_states=ssm_t, **kw16)
        o_h = flydsl.packed_decode(mixed16, a16, b16, ssm_states=ssm_h, **kw16)
        torch.testing.assert_close(o_h, o_t, rtol=0, atol=0)
        torch.testing.assert_close(ssm_h, ssm_t, rtol=0, atol=0)

    def test_prefill_front_matches_triton_chain(self):
        """Varlen boundaries: sequences shorter than the conv width, with and without
        an initial conv state, must give the same q/k/v and final conv cache."""
        torch.manual_seed(1)
        H, HV = self.H, self.HV
        lens = [1, 2, 3, 130, 300]
        has_init = torch.tensor([True, False, True, True, False], device="cuda")
        T, N = sum(lens), len(lens)
        qkv_dim = 2 * H * D + HV * D
        qkvz = torch.randn(T, qkv_dim + HV * D, device="cuda", dtype=torch.bfloat16)
        conv_w = (torch.randn(qkv_dim, 4, device="cuda") * 0.5).to(torch.bfloat16)
        conv = torch.randn(N + 3, qkv_dim, 3, device="cuda").to(torch.bfloat16)
        idx = torch.arange(1, N + 1, device="cuda", dtype=torch.int32)
        cu = torch.tensor(
            [0] + torch.tensor(lens).cumsum(0).tolist(),
            device="cuda",
            dtype=torch.int32,
        )

        conv_ref = conv.clone()
        q, k, v, _ = _split(qkvz, H, HV)
        out = causal_conv1d_fn(
            torch.cat((q, k, v), dim=-1).transpose(0, 1),
            conv_w,
            None,
            activation="silu",
            conv_states=conv_ref,
            has_initial_state=has_init,
            cache_indices=idx,
            query_start_loc=cu,
            seq_lens_cpu=lens,
        ).transpose(0, 1)[:T]
        rq, rk, rv = fused_qkv_split_gdn_prefill(out, H, H, HV, D, D, D)
        rq, rk = l2norm_fwd(rq), l2norm_fwd(rk)

        fq, fk, fv = flydsl_gdn_prefill_op()(
            qkvz, conv_w, conv, idx, has_init, cu, H, HV, max(lens)
        )
        for ref, got in ((rq, fq), (rk, fk), (rv, fv)):
            torch.testing.assert_close(
                got.float().reshape(-1), ref.float().reshape(-1), rtol=1e-2, atol=1e-2
            )
        torch.testing.assert_close(conv, conv_ref, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
