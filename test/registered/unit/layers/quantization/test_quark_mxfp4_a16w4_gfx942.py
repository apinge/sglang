"""Focused integration coverage for Quark MXFP4 MoE on gfx942."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.moe.moe_op_gemm_a16w4 import moe_gemm_torch
from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp, upcast_from_mxfp
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4_moe import (
    QuarkW4A4MXFp4MoE,
)
from sglang.srt.utils.common import is_gfx942_supported


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_gfx942_supported(),
    reason="MI308X/gfx942-specific test",
)


class _Dispatcher:
    expert_mask_gpu = None

    def __init__(self):
        self.quant_config = None

    def set_quant_config(self, quant_config):
        self.quant_config = quant_config


def _scheme() -> QuarkW4A4MXFp4MoE:
    scheme = QuarkW4A4MXFp4MoE(
        weight_config={"qscheme": "per_group"},
        input_config={"qscheme": "per_group", "is_dynamic": True},
    )
    scheme.use_aiter_a16w4 = True
    scheme.moe_runner_config = SimpleNamespace(top_k=10)
    return scheme


def _assert_close(ref: torch.Tensor, actual: torch.Tensor) -> None:
    ref = ref.float()
    actual = actual.float()
    diff = (ref - actual).abs()
    scale = ref.abs().amax().clamp_min(1.0e-30)
    ref_normalized = ref / scale
    actual_normalized = actual / scale
    ref_rms = torch.sqrt(torch.square(ref_normalized).mean()).clamp_min(1.0e-30)
    rel = (ref_normalized - actual_normalized).abs() / torch.maximum(
        ref_normalized.abs(), ref_rms
    )

    assert torch.isfinite(actual).all()
    assert diff.max().item() <= 5.0e-2
    assert rel.max().item() <= 4.0e-1
    assert torch.sqrt(torch.square(rel).mean()).item() <= 4.0e-2


def test_create_weights_selects_qwen3_8_tp8_a16w4() -> None:
    scheme = QuarkW4A4MXFp4MoE(
        weight_config={"qscheme": "per_group"},
        input_config={"qscheme": "per_group", "is_dynamic": True},
    )
    layer = torch.nn.Module()

    with torch.device("meta"):
        scheme.create_weights(
            layer=layer,
            num_experts=512,
            hidden_size=8192,
            intermediate_size_per_partition=256,
            params_dtype=torch.bfloat16,
            weight_loader=lambda *args: None,
        )

    assert scheme.use_aiter_a16w4
    assert tuple(layer.w13_weight.shape) == (512, 512, 4096)
    assert tuple(layer.w2_weight.shape) == (512, 8192, 128)
    assert tuple(layer.w13_weight_scale.shape) == (512, 512, 256)
    assert tuple(layer.w2_weight_scale.shape) == (512, 8192, 8)


def test_process_weights_preserves_checkpoint_layout() -> None:
    scheme = _scheme()
    layer = SimpleNamespace(
        w13_weight=torch.nn.Parameter(
            torch.arange(64, dtype=torch.uint8).view(2, 4, 8), requires_grad=False
        ),
        w2_weight=torch.nn.Parameter(
            torch.arange(64, dtype=torch.uint8).view(2, 8, 4), requires_grad=False
        ),
        w13_weight_scale=torch.nn.Parameter(
            torch.arange(16, dtype=torch.uint8).view(2, 4, 2), requires_grad=False
        ),
        w2_weight_scale=torch.nn.Parameter(
            torch.arange(16, dtype=torch.uint8).view(2, 8, 1), requires_grad=False
        ),
        dispatcher=_Dispatcher(),
    )
    before = [
        tensor.detach().clone()
        for tensor in (
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
        )
    ]

    scheme.process_weights_after_loading(layer)

    after = (
        layer.w13_weight,
        layer.w2_weight,
        layer.w13_weight_scale,
        layer.w2_weight_scale,
    )
    assert all(torch.equal(expected, actual) for expected, actual in zip(before, after))
    assert layer.dispatcher.quant_config == {"weight_dtype": torch.float4_e2m1fn_x2}


def test_quark_a16w4_apply_matches_dequantized_reference() -> None:
    torch.manual_seed(3807)
    m = 4
    hidden_size = 8192
    intermediate_size_per_rank = 256
    num_experts = 16
    topk = 10

    # Build logical AITER weights, then transpose back to SGLang's checkpoint
    # parameter layout so the integration's zero-copy views are exercised.
    w13 = (
        torch.randn(
            (num_experts, hidden_size, 2 * intermediate_size_per_rank),
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.02
    )
    w13_quant, w13_scale = downcast_to_mxfp(w13, torch.uint8, axis=1)
    w2 = (
        torch.randn(
            (num_experts, intermediate_size_per_rank, hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.02
    )
    w2_quant, w2_scale = downcast_to_mxfp(w2, torch.uint8, axis=1)

    layer = SimpleNamespace(
        w13_weight=torch.nn.Parameter(w13_quant.transpose(1, 2), requires_grad=False),
        w13_weight_scale=torch.nn.Parameter(
            w13_scale.transpose(1, 2), requires_grad=False
        ),
        w2_weight=torch.nn.Parameter(w2_quant.transpose(1, 2), requires_grad=False),
        w2_weight_scale=torch.nn.Parameter(
            w2_scale.transpose(1, 2), requires_grad=False
        ),
        moe_ep_size=1,
        moe_tp_size=8,
    )
    hidden_states = (
        torch.randn((m, hidden_size), device="cuda", dtype=torch.bfloat16) * 0.02
    )
    router_logits = torch.randn((m, num_experts), device="cuda", dtype=torch.float16)
    dispatch_output = StandardDispatchOutput(
        hidden_states=hidden_states,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=torch.empty((m, topk), device="cuda"),
            topk_ids=torch.empty((m, topk), dtype=torch.int32, device="cuda"),
            router_logits=router_logits,
        ),
    )

    actual = _scheme()._apply_aiter_a16w4(layer, dispatch_output).hidden_states

    routing_data, gather_idx, scatter_idx = routing(router_logits, topk)
    w13_dequant = upcast_from_mxfp(w13_quant, w13_scale, torch.bfloat16, axis=1)
    stage1_ref = moe_gemm_torch(
        hidden_states,
        w13_dequant,
        None,
        routing_data,
        gather_idx,
        None,
        None,
        False,
    )
    intermediate_ref = (
        F.silu(stage1_ref[:, :intermediate_size_per_rank].float())
        * stage1_ref[:, intermediate_size_per_rank:].float()
    ).to(torch.bfloat16)
    w2_dequant = upcast_from_mxfp(w2_quant, w2_scale, torch.bfloat16, axis=1)
    expected = moe_gemm_torch(
        intermediate_ref,
        w2_dequant,
        None,
        routing_data,
        None,
        scatter_idx,
        routing_data.gate_scal,
        False,
    )

    _assert_close(expected, actual)
