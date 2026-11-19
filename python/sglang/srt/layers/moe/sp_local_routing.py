"""MoE routing on this rank's sequence-parallel rows, before their all-gather.

Each rank runs the router (gate GEMM, top-k, fused shared expert) on its own rows only. The
ids and weights travel next to the gathered bf16 rows as one f32 metadata row per token, and
the experts then run on the gathered rows exactly as the block's own forward runs them.
"""

import torch

from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.moe import get_moe_a2a_backend, get_moe_runner_backend
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKOutputFormat
from sglang.srt.state_capturer.routed_experts import get_global_experts_capturer


def local_routing_supported(moe_block) -> bool:
    """Whether router + experts are the block's whole MoE: no separate shared-expert MLP,
    no a2a dispatch, standard top-k output, and no simulated routing."""
    backend = get_moe_runner_backend()
    return (
        moe_block.shared_expert is None
        and get_moe_a2a_backend().is_none()
        and (backend.is_auto() or backend.is_aiter() or backend.is_triton())
        and moe_block.topk.topk_config.output_format
        in (None, TopKOutputFormat.STANDARD)
        and not envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.get()
        and not envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.get()
    )


def local_routing_allowed() -> bool:
    # Routed-experts capture and expert-distribution recording index or count the rows the
    # router sees, so they need it on the gathered rows.
    return (
        get_global_experts_capturer() is None
        and not get_global_expert_distribution_recorder().recording
    )


def _topk(moe_block) -> int:
    return moe_block.topk.topk_config.top_k + moe_block.num_fused_shared_experts


def meta_width(moe_block) -> int:
    """f32 words per token: topk ids (bitcast int32), then topk weights, padded to 32 B."""
    return -(-2 * _topk(moe_block) // 8) * 8


def route_rows(moe_block, rows: torch.Tensor) -> torch.Tensor:
    """[n, meta_width] f32 routing metadata of this rank's rows."""
    router_logits, _ = moe_block.gate(rows)
    topk_output = moe_block._append_shared_to_topk_output(
        moe_block.topk(rows, router_logits), rows
    )
    k = _topk(moe_block)
    meta = rows.new_zeros(rows.shape[0], meta_width(moe_block), dtype=torch.float32)
    meta[:, :k] = topk_output.topk_ids.to(torch.int32).view(torch.float32)
    meta[:, k : 2 * k] = topk_output.topk_weights
    return meta


def run_routed_experts(moe_block, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
    """The block's experts on gathered rows routed by `route_rows`."""
    k = _topk(moe_block)
    topk_output = StandardTopKOutput(
        topk_weights=meta[:, k : 2 * k].contiguous(),
        topk_ids=meta[:, :k].contiguous().view(torch.int32),
        router_logits=None,
    )
    return moe_block.experts(x, topk_output)
