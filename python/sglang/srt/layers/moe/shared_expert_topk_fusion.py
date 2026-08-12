"""Fused helpers for appending shared experts to MoE top-k output."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

_fused_linear_sigmoid = None
_fused_append_shared_experts_gated = None

try:
    from sglang.srt.layers.fused_linear_sigmoid_mul_triton import fused_linear_sigmoid

    _fused_linear_sigmoid = fused_linear_sigmoid
except ImportError:
    pass

try:
    # PR#286 upstream path was layers/moe/fused_moe_triton/; in this tree the
    # Triton MoE kernels live under layers/moe/moe_runner/triton_utils/. Importing
    # the old path would silently ImportError and fall back to the eager
    # append_shared_to_topk_ref, so the fused kernel would never run.
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels import (
        fused_append_shared_experts_gated,
    )

    _fused_append_shared_experts_gated = fused_append_shared_experts_gated
except ImportError:
    pass


def can_fuse_shared_expert_gate(gate, hidden_states: torch.Tensor) -> bool:
    if gate is None or not hidden_states.is_cuda or _fused_linear_sigmoid is None:
        return False
    weight = getattr(gate, "weight", None)
    if weight is None:
        return False
    return (
        getattr(gate, "bias", None) is None
        and weight.dim() == 2
        and weight.shape[0] == 1
        and weight.shape[1] == hidden_states.shape[1]
        and hidden_states.is_contiguous()
        and weight.is_contiguous()
    )


def get_shared_expert_gate_weights(
    gate,
    hidden_states: torch.Tensor,
) -> Optional[torch.Tensor]:
    if gate is None:
        return None
    if can_fuse_shared_expert_gate(gate, hidden_states):
        return _fused_linear_sigmoid(hidden_states, gate.weight)
    shared_out = gate(hidden_states)
    shared_logits = shared_out[0] if isinstance(shared_out, tuple) else shared_out
    return F.sigmoid(shared_logits)


def append_shared_to_topk_ref(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    shared_weights: torch.Tensor,
    num_experts: int,
    num_fused_shared_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    m = topk_ids.shape[0]
    shared_ids = torch.full(
        (m, num_fused_shared_experts),
        num_experts,
        dtype=topk_ids.dtype,
        device=topk_ids.device,
    )
    if shared_weights.dim() == 1:
        shared_weights = shared_weights.unsqueeze(-1)
    shared_w = shared_weights.expand(m, num_fused_shared_experts)
    if shared_w.dtype != topk_weights.dtype:
        shared_w = shared_w.to(topk_weights.dtype)
    return (
        torch.cat([topk_ids, shared_ids], dim=-1),
        torch.cat([topk_weights, shared_w], dim=-1),
    )


def append_shared_to_topk(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    shared_weights: torch.Tensor,
    num_experts: int,
    num_fused_shared_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if (
        topk_ids.is_cuda
        and _fused_append_shared_experts_gated is not None
        and num_fused_shared_experts > 0
    ):
        return _fused_append_shared_experts_gated(
            topk_ids,
            topk_weights,
            shared_weights,
            num_experts,
            num_fused_shared_experts,
        )
    return append_shared_to_topk_ref(
        topk_ids,
        topk_weights,
        shared_weights,
        num_experts,
        num_fused_shared_experts,
    )
