"""Qwen4 MoE on aiter's gfx942 fused-quant MoE kernels
(aiter.ops.flydsl.kernels.moe_fused_quant_gfx942).

Prefill (SP + SDMA reduce-scatter): this rank's partial sum of the routed + fused shared
experts, bit-exact with the aiter runner. Decode: small-M MoE with the output clear fused
into gate-up, used only where it beats the tuned path.
"""

import functools
import logging
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.moe.topk import TopKOutputChecker
from sglang.srt.runtime_context import get_parallel
from sglang.srt.state_capturer.routed_experts import get_global_experts_capturer
from sglang.srt.utils import is_gfx942_supported

logger = logging.getLogger(__name__)


def _kernels():
    from aiter.ops.flydsl.kernels.moe_fused_quant_gfx942 import moe

    return moe


@functools.cache
def fused_moe_available() -> bool:
    try:
        _kernels().prefill_moe_supported
    except (ImportError, AttributeError) as e:
        logger.warning(
            "Qwen4 fused MoE unavailable (%s); using the aiter MoE runner", e
        )
        return False
    return True


def _is_ptpc_aiter_weight(w: torch.Tensor, scale: Optional[torch.Tensor]) -> bool:
    # is_shuffled is only set by the quant methods that preshuffle for aiter; an
    # unshuffled (e.g. Triton-runner) weight must never reach the FlyDSL kernels.
    return (
        w.dtype == torch.float8_e4m3fnuz
        and getattr(w, "is_shuffled", False)
        and scale is not None
        and scale.dtype == torch.float32
        and tuple(scale.shape) == (*w.shape[:2], 1)
    )


def fused_moe_supported(moe_block) -> bool:
    """Whether the block's MoE is exactly what the fused path computes: gfx942, aiter
    preshuffled PTPC fp8 experts, no EP, and the shared expert fused into the experts
    (a separate shared_expert MLP would be dropped). Call after weight loading."""
    if not (
        is_gfx942_supported()
        and moe_block.shared_expert is None
        and get_parallel().moe_ep_size == 1
        and get_moe_a2a_backend().is_none()
    ):
        return False
    params = dict(moe_block.experts.named_parameters(recurse=False))
    w13, w2 = params.get("w13_weight"), params.get("w2_weight")
    if w13 is None or w2 is None:
        return False
    num_experts = moe_block.num_experts + moe_block.num_fused_shared_experts
    return (
        _is_ptpc_aiter_weight(w13, params.get("w13_weight_scale"))
        and _is_ptpc_aiter_weight(w2, params.get("w2_weight_scale"))
        and w13.shape[0] == num_experts
        # No hidden padding, and gate-up / down agree on the intermediate size.
        and w13.shape[2] == w2.shape[1] == moe_block.experts.hidden_size
        and w2.shape[2] == w13.shape[1] // 2
        and fused_moe_available()
    )


def fused_route_supported(moe_block, num_tokens: int) -> bool:
    """Per-forward conditions under which routing on local rows is invisible: no routing
    side effects that index rows or count them per rank, and a tuned kernel config."""
    if (
        get_global_experts_capturer() is not None
        or get_global_expert_distribution_recorder().recording
        or envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.get()
        or envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.get()
    ):
        return False
    w13 = moe_block.experts.w13_weight
    return _kernels().prefill_moe_supported(
        m=num_tokens,
        model_dim=w13.shape[2],
        inter=w13.shape[1] // 2,
        experts=w13.shape[0],
        topk=moe_block.topk.topk_config.top_k + moe_block.num_fused_shared_experts,
    )


def _standard_topk(moe_block, rows: torch.Tensor):
    router_logits, _ = moe_block.gate(rows)
    topk_output = moe_block.topk(rows, router_logits)
    if not TopKOutputChecker.format_is_standard(topk_output):
        return None
    return moe_block._append_shared_to_topk_output(topk_output, rows)


def _run(moe_block, hidden_states, topk_weights, topk_ids) -> torch.Tensor:
    experts = moe_block.experts
    return _kernels().prefill_moe(
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


_PREFILL_LOGGED: list = []


def route_partial(moe_block, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
    """This rank's MoE partial sum from gathered bf16 rows and their routing metadata."""
    if not _PREFILL_LOGGED:
        _PREFILL_LOGGED.append(x.shape[0])
        logger.info("Qwen4 fused MoE prefill path active (first T=%d)", x.shape[0])
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
