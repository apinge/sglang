from typing import Optional

import torch

_HIP_GDN_SUPPORTED_LOCAL_HEAD_SHAPES = frozenset(
    {
        (2, 4),
        (4, 8),
        (2, 8),
        (8, 16),
        (16, 32),
    }
)


def get_local_gdn_head_shape(
    total_num_k_heads: int, total_num_v_heads: int, tp_size: int
) -> tuple[int, int]:
    if tp_size <= 0:
        raise ValueError(f"tp_size must be positive, got {tp_size}")
    if total_num_k_heads % tp_size != 0 or total_num_v_heads % tp_size != 0:
        raise ValueError(
            "GDN head counts must be divisible by tp_size: "
            f"num_k_heads={total_num_k_heads}, "
            f"num_v_heads={total_num_v_heads}, tp_size={tp_size}"
        )
    return total_num_k_heads // tp_size, total_num_v_heads // tp_size


def supports_hip_gdn_decode_local_heads(
    local_num_k_heads: int, local_num_v_heads: int
) -> bool:
    return (local_num_k_heads, local_num_v_heads) in (
        _HIP_GDN_SUPPORTED_LOCAL_HEAD_SHAPES
    )


def supports_hip_gdn_decode_runtime(
    *,
    local_num_k_heads: int,
    local_num_v_heads: int,
    q_dtype: torch.dtype,
    k_dtype: torch.dtype,
    v_dtype: torch.dtype,
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
    dt_bias_dtype: torch.dtype,
    state_dtype: torch.dtype,
    head_k_dim: int,
    head_v_dim: int,
    state_shape: tuple[int, ...],
) -> bool:
    """Return whether the native HIP ASM GDN decode can safely read tensors.

    The kernel has fixed ABI assumptions: bf16 activations, fp32 state,
    128x128 per-head state matrices, and one of the compiled local head shapes.
    """
    if not supports_hip_gdn_decode_local_heads(local_num_k_heads, local_num_v_heads):
        return False
    if head_k_dim != 128 or head_v_dim != 128:
        return False
    if any(
        dtype != torch.bfloat16
        for dtype in (q_dtype, k_dtype, v_dtype, a_dtype, b_dtype, dt_bias_dtype)
    ):
        return False
    if state_dtype != torch.float32:
        return False
    if len(state_shape) != 4:
        return False
    return tuple(state_shape[-3:]) == (local_num_v_heads, 128, 128)


def sync_gdn_slot_layout_after_copy(
    slot_layout: Optional[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
) -> None:
    """Keep layout metadata consistent with SSM state row copies.

    Mirrors the ``ssm_states[dst] = ssm_states[src]`` scatter at the call site
    using the exact same index tensors. Those indices are already valid slot ids
    (the state scatter applies them unguarded), so we deliberately avoid any
    ``torch.any`` / boolean-mask / ``.to(long)`` here: each of those forces a
    device->host sync (or an int32->int64 copy) and stalls the GDN prefill stream.
    """
    if slot_layout is None or src_indices.numel() == 0 or dst_indices.numel() == 0:
        return
    slot_layout[dst_indices] = slot_layout[src_indices]


def target_gdn_state_layout(
    *,
    is_extend: bool,
    is_target_verify: bool,
    is_decode_or_idle: bool,
    layout_kv: int,
    layout_vk: int,
) -> Optional[int]:
    """Choose the state layout required by the next GDN forward pass."""
    if is_extend or is_target_verify:
        return layout_kv
    if is_decode_or_idle:
        return layout_vk
    return None


def select_vk_gdn_decode_backend(
    local_num_k_heads: int,
    local_num_v_heads: int,
    hip_decode_available: bool,
    flydsl_decode_available: bool,
) -> Optional[str]:
    if hip_decode_available and supports_hip_gdn_decode_local_heads(
        local_num_k_heads, local_num_v_heads
    ):
        return "hip"
    if flydsl_decode_available:
        return "flydsl"
    return None
