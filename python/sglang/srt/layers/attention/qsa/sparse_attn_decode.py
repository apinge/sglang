# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MI300 QSA decode over prepared packed K/V with split-K reduction.

Adapted from apinge/sglang PR #40, commit
4b5faf7343cb76afde8b92657997197140a46fbd, which ports ROCm/ATOM PR #2311,
commit b97551f1d03a47027d286a5846a63a888acad237,
atom/model_ops/qwen4_exp/ops/qsa.py (MIT). The ATOM implementation in turn
credits ROCm/aiter PR #4882 for the paged sparse-attention kernel.

The PR #40 attention algorithm operates here on the packed K/V and relative
indices emitted by ``prepare_qsa_packed_kv``. Gathering the original paged
cache remains a separate preparation step; this module has no cache-page or
request-table addressing. Reduction uses the installed AITER implementation.
"""

from functools import lru_cache

import torch
import triton
import triton.language as tl


@lru_cache(maxsize=None)
def _device_info(device: torch.device) -> tuple[str, int]:
    properties = torch.cuda.get_device_properties(device)
    return properties.gcnArchName.split(":")[0], properties.multi_processor_count


def supports_sparse_packed_gqa(q, k, v) -> bool:
    return (
        torch.version.hip is not None
        and q.is_cuda
        and q.ndim == 3
        and k.ndim == 3
        and v.shape == k.shape
        and q.device == k.device == v.device
        and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and q.shape[2] == k.shape[2]
        and q.shape[2] in (128, 256)
        and k.shape[1] > 0
        and q.shape[1] > 0
        and q.shape[1] % k.shape[1] == 0
        and q.shape[1] // k.shape[1] <= 32
        and _device_info(q.device)[0] == "gfx942"
    )


class SparsePackedGQAWorkspace:
    """Per-backend scratch retained across layers and graph replay.

    Calls on one stream are serialized by the attention backend. Different
    streams have separate scratch, and returned outputs never alias it.
    """

    def __init__(self):
        self.buffers = {}

    def get(self, q, splits):
        key = (
            q.device,
            tuple(q.shape),
            splits,
            torch.cuda.current_stream(q.device).cuda_stream,
        )
        if key not in self.buffers:
            shape = (*q.shape[:2], splits)
            maxima = torch.empty(shape, device=q.device, dtype=torch.float32)
            sums = torch.empty_like(maxima)
            partials = torch.empty(
                (*shape, q.shape[2]), device=q.device, dtype=torch.float32
            )
            widths = torch.empty(q.shape[0], device=q.device, dtype=torch.int32)
            self.buffers[key] = (partials, maxima, sums, widths)
        return self.buffers[key]


def _prev_pow2(n: int) -> int:
    if n < 1:
        return 1
    return 1 << (n.bit_length() - 1)


def _kv_splits_heuristic(
    T: int,
    kv_heads: int,
    topk: int,
    num_cu: int,
    target_wg_per_cu: float = 4.0,
    max_kv_splits: int = 64,
) -> int:
    """PR #40 occupancy heuristic, using the number of query token rows."""
    if topk < 512:
        return 1
    target_wg = max(1, int(target_wg_per_cu * num_cu))
    base_ctas = max(1, T * kv_heads)
    if base_ctas >= target_wg:
        return 1
    return _prev_pow2(min(target_wg // base_ctas, max_kv_splits))


def _kernel_config(
    T: int,
    kv_heads: int,
    kv_splits: int,
    group_size: int,
    num_cu: int,
) -> tuple[int, int, int, int, int]:
    """Pick (BLOCK_N, num_warps, num_stages, waves_per_eu, sub_group).

    Keep the full query-head group for small grids. Larger grids subdivide
    groups exceeding 16 heads to reduce accumulator size. The middle
    occupancy tier hints the register allocator to target three waves/EU.
    """
    sub_group = group_size
    if group_size > 16:
        sub_group = 8
    num_head_groups = (group_size + sub_group - 1) // sub_group
    grid_size = T * kv_heads * num_head_groups * kv_splits
    if grid_size <= num_cu:
        return 64, 4, 1, 0, group_size
    if grid_size >= num_cu * 8:
        return 32, 2, 1, 0, sub_group
    return 32, 4, 1, 3, sub_group


@triton.jit
def _qsa_sparse_packed_gqa_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    relative_indices_ptr,
    cu_q_ptr,
    cu_k_ptr,
    kv_lens_ptr,
    output_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    widths_ptr,
    stride_q_token,
    stride_q_head,
    stride_q_dim,
    stride_k_token,
    stride_k_head,
    stride_k_dim,
    stride_v_token,
    stride_v_head,
    stride_v_dim,
    stride_indices_token,
    stride_indices_column,
    stride_cu_q,
    stride_cu_k,
    stride_kv_lens,
    stride_output_token,
    stride_output_head,
    stride_output_dim,
    num_packed_tokens,
    softmax_scale,
    TOPK: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    KV_SPLITS: tl.constexpr = 1,
    SUB_GROUP: tl.constexpr = 0,
) -> None:
    """Apply PR #40 GQA to relative indices in prepared packed K/V rows."""
    token = tl.program_id(0)
    part = tl.program_id(2)
    ACTIVE_SUB: tl.constexpr = SUB_GROUP if SUB_GROUP > 0 else GROUP_SIZE
    NUM_HEAD_GROUPS: tl.constexpr = (GROUP_SIZE + ACTIVE_SUB - 1) // ACTIVE_SUB
    kv_head = tl.program_id(1) // NUM_HEAD_GROUPS
    head_group = tl.program_id(1) % NUM_HEAD_GROUPS
    if KV_SPLITS > 1:  # noqa: SIM102 -- compile-time guard for widths_ptr=None
        if tl.program_id(1) == 0 and part == 0:
            # AITER must use the same TOPK-based partition boundaries as this
            # kernel. Actual valid-prefix lengths can differ on every replay.
            tl.store(widths_ptr + token, TOPK)

    q_start = tl.load(cu_q_ptr + token * stride_cu_q)
    q_end = tl.load(cu_q_ptr + (token + 1) * stride_cu_q)
    packed_start = tl.load(cu_k_ptr + token * stride_cu_k).to(tl.int64)
    packed_end = tl.load(cu_k_ptr + (token + 1) * stride_cu_k).to(tl.int64)
    selected_count = tl.minimum(
        tl.maximum(tl.load(kv_lens_ptr + token * stride_kv_lens), 0), TOPK
    )
    row_valid = (
        (q_start == token)
        & (q_end == token + 1)
        & (packed_start >= 0)
        & (packed_end >= packed_start)
        & (packed_end <= num_packed_tokens)
    )
    selected_count = tl.where(row_valid, selected_count, 0)
    packed_width = packed_end - packed_start

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, BLOCK_D)
    valid_head = (head_offsets < ACTIVE_SUB) & (
        head_group * ACTIVE_SUB + head_offsets < GROUP_SIZE
    )
    first_q_head = kv_head * GROUP_SIZE + head_group * ACTIVE_SUB
    query = tl.load(
        q_ptr
        + token * stride_q_token
        + (first_q_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :] * stride_q_dim,
        mask=(valid_head[:, None]) & (dim_offsets[None, :] < HEAD_DIM),
        other=0.0,
    )
    query = (query * softmax_scale * 1.4426950408889634).to(query.dtype)

    # A finite maximum avoids -inf - -inf when an entire tile/split is empty.
    running_max = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    column_offsets = tl.arange(0, BLOCK_N)

    partition_size = tl.cdiv(TOPK, KV_SPLITS * BLOCK_N) * BLOCK_N
    # Every split writes its scratch, including splits with no selected tokens.
    # This makes workspace reuse and graph replay independent of stale values.
    for start in tl.range(
        part * partition_size,
        tl.minimum((part + 1) * partition_size, selected_count),
        BLOCK_N,
    ):
        columns = start + column_offsets
        relative_token = tl.load(
            relative_indices_ptr
            + token * stride_indices_token
            + columns * stride_indices_column,
            mask=columns < selected_count,
            other=-1,
        ).to(tl.int64)
        # Preparation compacts the index prefix, but leaves K/V in their
        # original selected slots. A relative slot may therefore exceed the
        # count of valid selections, including the empty-row zero sentinel.
        valid = (
            (columns < selected_count)
            & (relative_token >= 0)
            & (relative_token < packed_width)
        )
        packed_token = tl.where(valid, packed_start + relative_token, 0)
        keys = tl.load(
            k_ptr
            + packed_token[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None] * stride_k_dim,
            mask=(dim_offsets[:, None] < HEAD_DIM) & valid[None, :],
            other=0.0,
            cache_modifier=".cg",
        )
        values = tl.load(
            v_ptr
            + packed_token[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :] * stride_v_dim,
            mask=valid[:, None] & (dim_offsets[None, :] < HEAD_DIM),
            other=0.0,
            cache_modifier=".cg",
        )

        scores = tl.where(valid[None, :], tl.dot(query, keys), -1.0e20)
        next_max = tl.maximum(running_max, tl.max(scores, axis=1))
        alpha = tl.math.exp2(running_max - next_max)
        probabilities = tl.where(
            valid[None, :],
            tl.math.exp2(scores - next_max[:, None]),
            0.0,
        )
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            acc=accumulator * alpha[:, None],
        )
        running_sum = running_sum * alpha + tl.sum(probabilities, axis=1)
        running_max = next_max

    if KV_SPLITS > 1:
        # AITER consumes unnormalized FP32 [token, head, split, dim] partials
        # with each split's maximum and sum for base-2 softmax.
        partial_offset = (
            token * NUM_KV_HEADS * GROUP_SIZE + first_q_head + head_offsets
        ) * KV_SPLITS + part
        tl.store(partial_max_ptr + partial_offset, running_max, valid_head)
        tl.store(partial_sum_ptr + partial_offset, running_sum, valid_head)
        output_ptr += part * HEAD_DIM
        output = accumulator
    else:
        output = tl.where(
            running_sum[:, None] > 0,
            accumulator / tl.maximum(running_sum[:, None], 1.0e-20),
            0.0,
        )
    tl.store(
        output_ptr
        + token * stride_output_token
        + (first_q_head + head_offsets[:, None]) * stride_output_head
        + dim_offsets[None, :] * stride_output_dim,
        output,
        mask=(valid_head[:, None]) & (dim_offsets[None, :] < HEAD_DIM),
    )


def qsa_sparse_packed_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    relative_indices: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    kv_lens: torch.Tensor,
    scale: float | None = None,
    *,
    workspace: SparsePackedGQAWorkspace | None = None,
    kv_splits: int | None = None,
) -> torch.Tensor:
    """Attend to one query per prepared packed K/V segment on gfx942.

    ``cu_q`` must contain ``0, 1, ..., q.shape[0]``. ``cu_k`` delimits each
    row's packed K/V segment. The first ``kv_lens[row]`` columns of
    ``relative_indices[row]`` select slots within that segment; negative and
    out-of-segment slots are masked. Valid slots need not form a contiguous
    prefix in K/V. In particular, preparation's empty-row sentinel is valid
    even though its relative slot exceeds ``kv_lens[row]``.

    Invalid segment boundaries and rows without any valid selections return
    zero. All mutable metadata is read on device for graph replay, and output
    storage is independent of the workspace. Tensor views with strides are
    supported, although preparation emits contiguous packed K/V and metadata.
    """
    if not supports_sparse_packed_gqa(q, k, v):
        raise ValueError("packed QSA requires gfx942 BF16 GQA with head_dim 128/256")
    tensors = (relative_indices, cu_q, cu_k, kv_lens)
    if any(t.device != q.device for t in tensors):
        raise ValueError("all packed QSA inputs must be on the same device")
    integer_dtypes = (torch.int32, torch.int64)
    if (
        relative_indices.ndim != 2
        or relative_indices.shape[0] != q.shape[0]
        or relative_indices.dtype not in integer_dtypes
    ):
        raise ValueError(
            "relative_indices must be int32/int64 [tokens, selection_width]"
        )
    for name, tensor, length in (
        ("cu_q", cu_q, q.shape[0] + 1),
        ("cu_k", cu_k, q.shape[0] + 1),
        ("kv_lens", kv_lens, q.shape[0]),
    ):
        if tensor.shape != (length,) or tensor.dtype not in integer_dtypes:
            raise ValueError(f"{name} must be int32/int64 [{length}]")

    softmax_scale = q.shape[2] ** -0.5 if scale is None else scale
    # AITER reduction requires contiguous output head dimensions.
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    if q.shape[0] == 0:
        return out

    _, num_cu = _device_info(q.device)
    group_size = q.shape[1] // k.shape[1]
    block_d = max(16, triton.next_power_of_2(q.shape[2]))
    if kv_splits is None:
        kv_splits = _kv_splits_heuristic(
            q.shape[0], k.shape[1], relative_indices.shape[1], num_cu
        )
    if kv_splits < 1 or kv_splits > 64 or kv_splits & (kv_splits - 1):
        raise ValueError("kv_splits must be a power of two in [1, 64]")
    block_n, num_warps, num_stages, waves_per_eu, sub_group = _kernel_config(
        q.shape[0], k.shape[1], kv_splits, group_size, num_cu
    )
    num_head_groups = (group_size + sub_group - 1) // sub_group
    block_m = max(16, triton.next_power_of_2(sub_group))
    if relative_indices.shape[1] == 0 or k.shape[0] == 0:
        # Skip reduction because its segment-count calculation needs TOPK > 0.
        return out.zero_()

    target = out
    partial_max = partial_sum = out
    widths = None
    if kv_splits > 1:
        if workspace is None:
            workspace = SparsePackedGQAWorkspace()
        target, partial_max, partial_sum, widths = workspace.get(q, kv_splits)
    grid_y = k.shape[1] * num_head_groups
    _qsa_sparse_packed_gqa_kernel[(q.shape[0], grid_y, kv_splits)](
        q,
        k,
        v,
        relative_indices,
        cu_q,
        cu_k,
        kv_lens,
        target,
        partial_max,
        partial_sum,
        widths,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        relative_indices.stride(0),
        relative_indices.stride(1),
        cu_q.stride(0),
        cu_k.stride(0),
        kv_lens.stride(0),
        target.stride(0),
        target.stride(1),
        target.stride(-1),
        k.shape[0],
        float(softmax_scale),
        TOPK=relative_indices.shape[1],
        NUM_KV_HEADS=k.shape[1],
        GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[2],
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        KV_SPLITS=kv_splits,
        SUB_GROUP=sub_group if sub_group < group_size else 0,
        num_warps=num_warps,
        num_stages=num_stages,
        waves_per_eu=waves_per_eu,
    )
    if kv_splits > 1:
        from aiter.ops.triton._triton_kernels.attention.mla import (
            _mla_decode_fwd_reduce_kernel,
        )

        _mla_decode_fwd_reduce_kernel[(q.shape[0], q.shape[1])](
            out,
            target,
            partial_max,
            partial_sum,
            widths,
            None,
            q.shape[0],
            q.shape[1],
            out.stride(0),
            out.stride(1),
            1,
            1,
            q.shape[0],
            TILE_SIZE=block_n,
            KV_LORA_RANK=q.shape[2],
            query_start_len_ptr=None,
            BLOCK_Q=1,
            NUM_SEGMENTS_PER_SEQ=kv_splits,
            ALL_DECODE=True,
        )
    return out
