"""Qwen4 gfx942 fused MoE paths, which are on by default: the routing-metadata round trip
through the fused prefill MoE, and the gates that keep other models off these paths."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import qwen4_fused_moe as mm
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.utils import is_gfx942_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd")

# Qwen3.8 flash-next MoE: 512 routed + 1 fused shared expert (id 512), top-10 + shared.
HIDDEN, EXPERTS, TOPK, SHARED_ID = 2560, 513, 10, 512
FP8 = torch.float8_e4m3fnuz


def _ptpc_experts(inter: int, shuffled: bool = True) -> torch.nn.Module:
    """Experts laid out like CompressedTensorsW8A8Fp8MoE's aiter PTPC weights."""
    from aiter.ops.shuffle import shuffle_weight

    gen = torch.Generator(device="cuda").manual_seed(0)

    def quant(n, k):
        w = torch.randn(EXPERTS, n, k, device="cuda", generator=gen) * 0.05
        scale = w.abs().amax(dim=-1, keepdim=True) / 240.0
        return (w / scale).clamp(-240, 240).to(FP8), scale.float()

    w13, s13 = quant(2 * inter, HIDDEN)
    w2, s2 = quant(HIDDEN, inter)
    experts = torch.nn.Module()
    experts.hidden_size = HIDDEN
    for name, value in (("w13_weight_scale", s13), ("w2_weight_scale", s2)):
        setattr(experts, name, torch.nn.Parameter(value, requires_grad=False))
    for name, w in (("w13_weight", w13), ("w2_weight", w2)):
        if shuffled:
            w = shuffle_weight(w, (16, 16))
        param = torch.nn.Parameter(w, requires_grad=False)
        param.is_shuffled = shuffled
        setattr(experts, name, param)
    return experts


class _TopK:
    topk_config = SimpleNamespace(top_k=TOPK)

    def __call__(self, rows, router_logits):
        w, ids = torch.topk(torch.softmax(router_logits.float(), -1), TOPK, dim=-1)
        return StandardTopKOutput(
            topk_weights=w, topk_ids=ids.to(torch.int32), router_logits=router_logits
        )


class _MoeBlock:
    """The attributes of a Qwen4 MoE block that qwen4_fused_moe reads."""

    num_experts = EXPERTS - 1
    num_fused_shared_experts = 1
    shared_expert = None

    def __init__(self, experts):
        self.experts = experts
        self.topk = _TopK()
        gen = torch.Generator(device="cuda").manual_seed(1)
        self._gate = torch.randn(HIDDEN, EXPERTS - 1, device="cuda", generator=gen)

    def gate(self, rows):
        return rows.float() @ self._gate, None

    def _append_shared_to_topk_output(self, topk_output, rows):
        n = rows.shape[0]
        shared_w = torch.full((n, 1), 0.5, device="cuda")
        shared_id = torch.full((n, 1), SHARED_ID, dtype=torch.int32, device="cuda")
        return StandardTopKOutput(
            topk_weights=torch.cat([topk_output.topk_weights, shared_w], 1),
            topk_ids=torch.cat([topk_output.topk_ids, shared_id], 1),
            router_logits=topk_output.router_logits,
        )


def _aiter_fused_moe(x, topk_output, experts):
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    return fused_moe(
        hidden_states=x,
        w1=experts.w13_weight,
        w2=experts.w2_weight,
        topk_weight=topk_output.topk_weights,
        topk_ids=topk_output.topk_ids,
        quant_type=QuantType.per_Token,
        activation=ActivationType.Silu,
        w1_scale=experts.w13_weight_scale,
        w2_scale=experts.w2_weight_scale,
    )


class TestQwen4FusedMoeGfx942(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or not is_gfx942_supported():
            raise unittest.SkipTest("gfx942 only")
        if not mm.fused_moe_available():
            raise unittest.SkipTest("aiter has no moe_fused_quant_gfx942")

    def test_route_partial_bit_exact_with_aiter_runner(self):
        """fused_route ships ids (bitcast to f32) and weights in one f32 row per token next to
        the bf16 rows; unpacking must give the aiter runner's exact output."""
        experts = _ptpc_experts(inter=192)
        block = _MoeBlock(experts)
        gen = torch.Generator(device="cuda").manual_seed(2)
        # SP prefill (the only caller) needs >= SGLANG_QWEN4_SEQ_PARALLEL_MIN_TOKENS rows;
        # aiter's fused_moe runs its FlyDSL prefill path there (not below ~1000 rows).
        m = envs.SGLANG_QWEN4_SEQ_PARALLEL_MIN_TOKENS.get() + 1
        x = torch.randn(m, HIDDEN, device="cuda", generator=gen, dtype=torch.bfloat16)

        meta = mm.route_meta(block, x)
        out = mm.route_partial(block, x, meta)
        topk_output = block._append_shared_to_topk_output(
            block.topk(x, block.gate(x)[0]), x
        )
        ref = _aiter_fused_moe(x, topk_output, experts)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)

    def test_prefill_gate_takes_only_what_the_fused_path_computes(self):
        """The default-on prefill path replaces self.mlp, so it must reject every block whose
        MoE it would compute differently: unshuffled (Triton-runner) weights, a separate
        shared-expert MLP, EP, non-per-channel scales, or a shape with no tuned config."""
        parallel = SimpleNamespace(moe_ep_size=1)
        a2a = SimpleNamespace(is_none=lambda: True)
        no_capture = dict(
            get_global_experts_capturer=lambda: None,
            get_global_expert_distribution_recorder=lambda: SimpleNamespace(
                recording=False
            ),
        )
        with (
            mock.patch.object(mm, "get_parallel", lambda: parallel),
            mock.patch.object(mm, "get_moe_a2a_backend", lambda: a2a),
            mock.patch.multiple(mm, **no_capture),
        ):
            block = _MoeBlock(_ptpc_experts(inter=192))
            self.assertTrue(mm.fused_moe_supported(block))
            self.assertTrue(mm.fused_route_supported(block, num_tokens=12000))

            self.assertFalse(
                mm.fused_moe_supported(_MoeBlock(_ptpc_experts(192, shuffled=False)))
            )
            separate_shared = _MoeBlock(block.experts)
            separate_shared.shared_expert = torch.nn.Module()
            self.assertFalse(mm.fused_moe_supported(separate_shared))
            per_tensor = _MoeBlock(_ptpc_experts(inter=192))
            per_tensor.experts.w13_weight_scale = torch.nn.Parameter(
                torch.ones(EXPERTS, 2, device="cuda"), requires_grad=False
            )
            self.assertFalse(mm.fused_moe_supported(per_tensor))
            parallel.moe_ep_size = 2
            self.assertFalse(mm.fused_moe_supported(block))
            parallel.moe_ep_size = 1

            # TP8 pads inter 80 to 128, which has no tuned FlyDSL prefill config.
            tp8 = _MoeBlock(_ptpc_experts(inter=128))
            self.assertTrue(mm.fused_moe_supported(tp8))
            self.assertFalse(mm.fused_route_supported(tp8, num_tokens=12000))

    def test_decode_small_scope_and_graph_replay(self):
        """Default-on decode path: only Qwen4 shapes at small M; CUDA-graph replay matches
        the aiter runner within bf16 atomic-accumulation noise."""
        experts = _ptpc_experts(inter=320)
        block = _MoeBlock(experts)
        gen = torch.Generator(device="cuda").manual_seed(3)

        def run(m, w13=experts.w13_weight):
            x = torch.randn(
                m, HIDDEN, device="cuda", generator=gen, dtype=torch.bfloat16
            )
            topk_output = block._append_shared_to_topk_output(
                block.topk(x, block.gate(x)[0]), x
            )
            args = dict(
                hidden_states=x,
                w13=w13,
                w13_scale=experts.w13_weight_scale,
                w2=experts.w2_weight,
                w2_scale=experts.w2_weight_scale,
                topk_weights=topk_output.topk_weights,
                topk_ids=topk_output.topk_ids,
            )
            return x, topk_output, args

        _, _, args = run(16)
        self.assertIsNone(mm.decode_small(m=16, **args))  # above the measured M range
        _, _, args = run(8, w13=experts.w13_weight[:256])
        self.assertIsNone(mm.decode_small(m=8, **args))  # not the Qwen4 expert count

        x, topk_output, args = run(8)
        graph = torch.cuda.CUDAGraph()
        mm.decode_small(m=8, **args)
        with torch.cuda.graph(graph):
            out = mm.decode_small(m=8, **args)
        graph.replay()
        ref = _aiter_fused_moe(x, topk_output, experts)
        torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
