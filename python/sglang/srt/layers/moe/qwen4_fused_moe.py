"""Qwen4 MoE on aiter's gfx942 fused-quant MoE kernels
(aiter.ops.flydsl.kernels.moe_fused_quant_gfx942).

Prefill (SP + SDMA reduce-scatter): this rank's partial sum of the routed + fused shared
experts, bit-exact with the aiter runner. Decode: small-M MoE with the output clear fused
into gate-up, used only where it beats the tuned path.
"""

import functools
import logging

import torch

from sglang.srt.layers.moe.topk import TopKOutputChecker
from sglang.srt.utils import is_gfx942_supported

logger = logging.getLogger(__name__)


def _prefill_moe():
    from aiter.ops.flydsl.kernels.moe_fused_quant_gfx942.moe import prefill_moe

    return prefill_moe


def fused_moe_supported(experts: torch.nn.Module) -> bool:
    """gfx942 PTPC fp8 experts: fp8 weights with one scale per output channel."""
    if not is_gfx942_supported():
        return False
    params = dict(experts.named_parameters(recurse=False))
    w13, s13 = params.get("w13_weight"), params.get("w13_weight_scale")
    return (
        w13 is not None
        and s13 is not None
        and w13.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
        and tuple(s13.shape) == (*w13.shape[:2], 1)
    )


@functools.cache
def fused_moe_available() -> bool:
    try:
        _prefill_moe()
    except ImportError as e:
        logger.warning(
            "Qwen4 fused MoE unavailable (%s); using the aiter MoE runner", e
        )
        return False
    return True


def _standard_topk(moe_block, rows: torch.Tensor):
    router_logits, _ = moe_block.gate(rows)
    topk_output = moe_block.topk(rows, router_logits)
    if not TopKOutputChecker.format_is_standard(topk_output):
        return None
    return moe_block._append_shared_to_topk_output(topk_output, rows)


def _run(moe_block, hidden_states, topk_weights, topk_ids) -> torch.Tensor:
    experts = moe_block.experts
    return _prefill_moe()(
        hidden_states=hidden_states,
        w13=experts.w13_weight,
        w13_scale=experts.w13_weight_scale,
        w2=experts.w2_weight,
        w2_scale=experts.w2_weight_scale,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )


# Routing metadata all-gathered next to the bf16 MoE input rows, one f32 row per token:
# [unused, topk_ids (bitcast i32) x topk, topk_weights x topk, pad to a multiple of 8].
def meta_words(topk: int) -> int:
    return -(-(1 + 2 * topk) // 8) * 8


def route_meta(moe_block, rows: torch.Tensor) -> torch.Tensor:
    """[n, meta_words] f32 routing metadata of this rank's bf16 rows."""
    topk_output = _standard_topk(moe_block, rows)
    if topk_output is None:
        raise RuntimeError("Qwen4 fused MoE needs the standard top-k output")
    ids = topk_output.topk_ids.to(torch.int32)
    topk = ids.shape[1]
    meta = rows.new_zeros(rows.shape[0], meta_words(topk), dtype=torch.float32)
    meta[:, 1 : 1 + topk] = ids.view(torch.float32)
    meta[:, 1 + topk : 1 + 2 * topk] = topk_output.topk_weights.float()
    return meta


def route_partial(moe_block, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
    """This rank's MoE partial sum from gathered bf16 rows and their routing metadata."""
    topk = moe_block.topk.topk_config.top_k + moe_block.num_fused_shared_experts
    topk_weights = meta[:, 1 + topk : 1 + 2 * topk].contiguous()
    topk_ids = meta[:, 1 : 1 + topk].contiguous().view(torch.int32)
    return _run(moe_block, x, topk_weights, topk_ids)


_DECODE_SMALL_LOGGED: list = []
# decode_small_supported's M thresholds were measured on Qwen3.8 flash-next only
# (512 routed + 1 fused shared expert, hidden 2560).
_QWEN4_DECODE_EXPERTS_HIDDEN = (513, 2560)


def decode_small(
    *, m, hidden_states, w13, w13_scale, w2, w2_scale, topk_weights, topk_ids
):
    """Small-M decode MoE when it beats the tuned path, else None."""
    try:
        from aiter.ops.flydsl.kernels.moe_fused_quant_gfx942 import decode_small as ds
    except ImportError:
        return None
    if (w13.shape[0], hidden_states.shape[1]) != _QWEN4_DECODE_EXPERTS_HIDDEN:
        return None
    if not ds.decode_small_supported(m, w13.shape[1] // 2):
        return None
    if not _DECODE_SMALL_LOGGED:
        _DECODE_SMALL_LOGGED.append(m)
        logger.info("Qwen4 fused MoE decode small-M path active (first M=%d)", m)
    return ds.decode_moe_small(
        hidden_states=hidden_states,
        w13=w13,
        w13_scale=w13_scale,
        w2=w2,
        w2_scale=w2_scale,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )
