from __future__ import annotations

import torch
import triton
import triton.language as tl

# =============================================================================
# Fused kernel — reads INTERLEAVED input format
# Used by Qwen3-Next whose checkpoint stores fused in_proj_qkvz weights
# in per-head-group interleaved layout:
#   [g0_q, g0_k, g0_v, g0_z, g1_q, g1_k, g1_v, g1_z, ...]
# =============================================================================


@triton.jit
def fused_qkvzba_split_reshape_cat_kernel(
    mixed_qkv,
    z,
    b,
    a,
    mixed_qkvz,
    mixed_ba,
    NUM_HEADS_QK: tl.constexpr,
    NUM_HEADS_V: tl.constexpr,
    HEAD_QK: tl.constexpr,
    HEAD_V: tl.constexpr,
):
    i_bs, i_qk = tl.program_id(0), tl.program_id(1)
    QKVZ_DIM_T: tl.constexpr = HEAD_QK * 2 + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V * 2
    BA_DIM_T: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK * 2
    QKV_DIM_T: tl.constexpr = HEAD_QK * 2 + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V
    q_end: tl.constexpr = HEAD_QK
    blk_q_ptr = (
        mixed_qkvz
        + i_bs * NUM_HEADS_QK * QKVZ_DIM_T
        + i_qk * QKVZ_DIM_T
        + tl.arange(0, q_end)
    )
    k_end: tl.constexpr = q_end + HEAD_QK
    blk_k_ptr = (
        mixed_qkvz
        + i_bs * NUM_HEADS_QK * QKVZ_DIM_T
        + i_qk * QKVZ_DIM_T
        + tl.arange(q_end, k_end)
    )
    v_end: tl.constexpr = k_end + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V
    blk_v_ptr = (
        mixed_qkvz
        + i_bs * NUM_HEADS_QK * QKVZ_DIM_T
        + i_qk * QKVZ_DIM_T
        + tl.arange(k_end, v_end)
    )
    z_end: tl.constexpr = v_end + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V
    blk_z_ptr = (
        mixed_qkvz
        + i_bs * NUM_HEADS_QK * QKVZ_DIM_T
        + i_qk * QKVZ_DIM_T
        + tl.arange(v_end, z_end)
    )
    blk_q_st_ptr = (
        mixed_qkv
        + i_bs * NUM_HEADS_QK * QKV_DIM_T
        + i_qk * HEAD_QK
        + tl.arange(0, HEAD_QK)
    )
    blk_k_st_ptr = (
        mixed_qkv
        + i_bs * NUM_HEADS_QK * QKV_DIM_T
        + NUM_HEADS_QK * HEAD_QK
        + i_qk * HEAD_QK
        + tl.arange(0, HEAD_QK)
    )
    blk_v_st_ptr = (
        mixed_qkv
        + i_bs * NUM_HEADS_QK * QKV_DIM_T
        + NUM_HEADS_QK * HEAD_QK * 2
        + i_qk * HEAD_V * NUM_HEADS_V // NUM_HEADS_QK
        + tl.arange(0, HEAD_V * NUM_HEADS_V // NUM_HEADS_QK)
    )
    blk_z_st_ptr = (
        z
        + i_bs * NUM_HEADS_V * HEAD_V
        + i_qk * HEAD_V * NUM_HEADS_V // NUM_HEADS_QK
        + tl.arange(0, HEAD_V * NUM_HEADS_V // NUM_HEADS_QK)
    )
    tl.store(blk_q_st_ptr, tl.load(blk_q_ptr))
    tl.store(blk_k_st_ptr, tl.load(blk_k_ptr))
    tl.store(blk_v_st_ptr, tl.load(blk_v_ptr))
    tl.store(blk_z_st_ptr, tl.load(blk_z_ptr))
    b_end: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK
    a_end: tl.constexpr = b_end + NUM_HEADS_V // NUM_HEADS_QK
    for i in tl.static_range(b_end):
        blk_b_ptr = mixed_ba + i_bs * NUM_HEADS_QK * BA_DIM_T + i_qk * BA_DIM_T + i
        blk_b_st_ptr = b + i_bs * NUM_HEADS_V + i_qk * NUM_HEADS_V // NUM_HEADS_QK + i
        tl.store(blk_b_st_ptr, tl.load(blk_b_ptr))
    for i in tl.static_range(b_end, a_end):
        blk_a_ptr = mixed_ba + i_bs * NUM_HEADS_QK * BA_DIM_T + i_qk * BA_DIM_T + i
        blk_a_st_ptr = (
            a + i_bs * NUM_HEADS_V + i_qk * NUM_HEADS_V // NUM_HEADS_QK + (i - b_end)
        )
        tl.store(blk_a_st_ptr, tl.load(blk_a_ptr))


def fused_qkvzba_split_reshape_cat(
    mixed_qkvz,
    mixed_ba,
    num_heads_qk,
    num_heads_v,
    head_qk,
    head_v,
):
    batch, seq_len = mixed_qkvz.shape[0], 1
    qkv_dim_t = num_heads_qk * head_qk * 2 + num_heads_v * head_v
    mixed_qkv = torch.empty(
        [batch * seq_len, qkv_dim_t],
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    z = torch.empty(
        [batch * seq_len, num_heads_v, head_v],
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    b = torch.empty(
        [batch * seq_len, num_heads_v],
        dtype=mixed_ba.dtype,
        device=mixed_ba.device,
    )
    a = torch.empty_like(b)
    grid = (batch * seq_len, num_heads_qk)
    fused_qkvzba_split_reshape_cat_kernel[grid](
        mixed_qkv,
        z,
        b,
        a,
        mixed_qkvz,
        mixed_ba,
        num_heads_qk,
        num_heads_v,
        head_qk,
        head_v,
        num_warps=1,
        num_stages=3,
    )
    return mixed_qkv, z, b, a


@triton.jit
def _scatter_fused_proj_kernel(
    fused_ptr,
    qkv_ptr,
    z_ptr,
    b_ptr,
    a_ptr,
    stride_fused,
    qkv_size: tl.constexpr,
    z_size: tl.constexpr,
    ba_size: tl.constexpr,
    BLK_LARGE: tl.constexpr,
    BLK_SMALL: tl.constexpr,
):
    row = tl.program_id(0)
    src = row * stride_fused

    for off in range(0, qkv_size, BLK_LARGE):
        col = off + tl.arange(0, BLK_LARGE)
        mask = col < qkv_size
        tl.store(
            qkv_ptr + row * qkv_size + col,
            tl.load(fused_ptr + src + col, mask=mask),
            mask=mask,
        )

    z_off = qkv_size
    for off in range(0, z_size, BLK_LARGE):
        col = off + tl.arange(0, BLK_LARGE)
        mask = col < z_size
        tl.store(
            z_ptr + row * z_size + col,
            tl.load(fused_ptr + src + z_off + col, mask=mask),
            mask=mask,
        )

    ba_off = qkv_size + z_size
    col_s = tl.arange(0, BLK_SMALL)
    b_mask = col_s < ba_size
    tl.store(
        b_ptr + row * ba_size + col_s,
        tl.load(fused_ptr + src + ba_off + col_s, mask=b_mask),
        mask=b_mask,
    )
    a_mask = col_s < ba_size
    tl.store(
        a_ptr + row * ba_size + col_s,
        tl.load(fused_ptr + src + ba_off + ba_size + col_s, mask=a_mask),
        mask=a_mask,
    )


def scatter_fused_proj(fused_out, qkv_size, z_size, b_size, a_size):
    """Scatter fused output [qkv|z|b|a] into 4 contiguous buffers."""
    assert b_size == a_size
    rows = fused_out.shape[0]
    qkv = torch.empty(rows, qkv_size, device=fused_out.device, dtype=fused_out.dtype)
    z = torch.empty(rows, z_size, device=fused_out.device, dtype=fused_out.dtype)
    b = torch.empty(rows, b_size, device=fused_out.device, dtype=fused_out.dtype)
    a = torch.empty(rows, a_size, device=fused_out.device, dtype=fused_out.dtype)

    BLK_LARGE = triton.next_power_of_2(max(qkv_size, z_size))
    BLK_SMALL = triton.next_power_of_2(max(b_size, a_size))
    _scatter_fused_proj_kernel[(rows,)](
        fused_out,
        qkv,
        z,
        b,
        a,
        fused_out.stride(0),
        qkv_size,
        z_size,
        b_size,
        BLK_LARGE,
        BLK_SMALL,
        num_warps=4,
    )
    return qkv, z, b, a
