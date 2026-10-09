# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MI300 QSA decode with direct token-slot loads and split-K reduction.

Adapted from ROCm/ATOM PR #2311, commit
b97551f1d03a47027d286a5846a63a888acad237,
atom/model_ops/qwen4_exp/ops/qsa.py (MIT). The ATOM implementation in turn
credits ROCm/aiter PR #4882 for the paged sparse-attention kernel.

SGLang uses page_size=1 over its token-slot K/V buffers. Each query supplies
its request-pool row and visible sequence length, including MTP verify rows.
This module does not depend on ATOM; reduction uses the installed AITER.
"""

from functools import lru_cache

import torch
import triton
import triton.language as tl


@lru_cache(maxsize=None)
def _device_info(device: torch.device) -> tuple[str, int]:
    properties = torch.cuda.get_device_properties(device)
    return properties.gcnArchName.split(":")[0], properties.multi_processor_count


def supports_sparse_paged_gqa(q, k_cache, v_cache) -> bool:
    return (
        torch.version.hip is not None
        and q.is_cuda
        and q.ndim == 3
        and k_cache.ndim == 3
        and v_cache.shape == k_cache.shape
        and q.device == k_cache.device == v_cache.device
        and q.dtype == k_cache.dtype == v_cache.dtype == torch.bfloat16
        and q.shape[2] == k_cache.shape[2]
        and q.shape[2] in (128, 256)
        and k_cache.shape[1] > 0
        and q.shape[1] > 0
        and q.shape[1] % k_cache.shape[1] == 0
        and q.shape[1] // k_cache.shape[1] <= 32
        and _device_info(q.device)[0] == "gfx942"
    )


class SparsePagedGQAWorkspace:
    """Per-backend scratch, reused across layers and retained for graph replay.

    Separate streams get separate buffers. Calls on one stream are serialized
    by the attention backend; outputs are never aliased to this scratch.
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

    When ``group_size`` exceeds 16, large grids split Q heads into
    sub-groups of 8 to shrink the per-CTA accumulator.  Small grids keep
    the full group for maximum per-CTA throughput.

    ``waves_per_eu`` hints the register allocator to keep VGPR usage low
    enough for the target occupancy.
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
def _qsa_sparse_paged_gqa_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    logical_indices_ptr,
    block_table_ptr,
    token_to_request_ptr,
    sequence_lengths_ptr,
    output_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    widths_ptr,
    stride_q_token,
    stride_q_head,
    stride_q_dim,
    stride_k_page,
    stride_k_token,
    stride_k_head,
    stride_k_dim,
    stride_v_page,
    stride_v_token,
    stride_v_head,
    stride_v_dim,
    stride_indices_token,
    stride_indices_column,
    stride_table_request,
    stride_table_page,
    stride_output_token,
    stride_output_head,
    stride_output_dim,
    num_tokens,
    num_cache_pages,
    num_requests,
    softmax_scale,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    KV_SPLITS: tl.constexpr = 1,
    SUB_GROUP: tl.constexpr = 0,
) -> None:
    """Apply GQA over arbitrary logical tokens in separate paged BF16 K/V."""
    token = tl.program_id(0)
    part = tl.program_id(2)
    ACTIVE_SUB: tl.constexpr = SUB_GROUP if SUB_GROUP > 0 else GROUP_SIZE
    NUM_HEAD_GROUPS: tl.constexpr = (GROUP_SIZE + ACTIVE_SUB - 1) // ACTIVE_SUB
    kv_head = tl.program_id(1) // NUM_HEAD_GROUPS
    head_group = tl.program_id(1) % NUM_HEAD_GROUPS
    if KV_SPLITS > 1:  # noqa: SIM102 -- compile-time guard for widths_ptr=None
        if tl.program_id(1) == 0 and part == 0:
            tl.store(widths_ptr + token, TOPK)
    visible_length = tl.load(sequence_lengths_ptr + token)
    request = tl.load(token_to_request_ptr + token)
    request_valid = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

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

    running_max = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    column_offsets = tl.arange(0, BLOCK_N)

    partition_size = tl.cdiv(TOPK, KV_SPLITS * BLOCK_N) * BLOCK_N
    if part * partition_size >= TOPK:
        return
    for start in tl.range(
        part * partition_size, tl.minimum((part + 1) * partition_size, TOPK), BLOCK_N
    ):
        columns = start + column_offsets
        logical_token = tl.load(
            logical_indices_ptr
            + token * stride_indices_token
            + columns * stride_indices_column,
            mask=columns < TOPK,
            other=-1,
        )
        safe_logical_token = tl.maximum(logical_token, 0)
        logical_page = safe_logical_token // PAGE_SIZE
        page_offset = safe_logical_token % PAGE_SIZE
        valid = (
            (token < num_tokens)
            & request_valid
            & (logical_token >= 0)
            & (logical_token < visible_length)
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request * stride_table_request
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1) * stride_table_page,
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_pages)
        safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)

        keys = tl.load(
            k_cache_ptr
            + safe_physical_page[None, :] * stride_k_page
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None] * stride_k_dim,
            mask=(dim_offsets[:, None] < HEAD_DIM) & valid[None, :],
            other=0.0,
            cache_modifier=".cg",
        )
        values = tl.load(
            v_cache_ptr
            + safe_physical_page[:, None] * stride_v_page
            + page_offset[:, None] * stride_v_token
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
        # Reduction consumes unnormalized [token, head, split, dim] partials
        # with each split's running maximum and sum for base-2 softmax.
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
        mask=(token < num_tokens)
        & (valid_head[:, None])
        & (dim_offsets[None, :] < HEAD_DIM),
    )


def qsa_sparse_paged_gqa(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_request: torch.Tensor,
    softmax_scale: float | None = None,
    kv_splits: int | None = None,
    *,
    sequence_lengths: torch.Tensor,
    workspace: SparsePagedGQAWorkspace | None = None,
) -> torch.Tensor:
    """Attend to logical indices in 3D token-slot K/V (negative = padding).

    Request IDs index the real request pool, not a batch-local/dummy graph
    table. Invalid requests, logical positions, physical slots and positions
    past each query's visible length are masked. Empty rows return zero.
    """
    if not supports_sparse_paged_gqa(q, k_cache, v_cache):
        raise ValueError("paged QSA requires gfx942 BF16 GQA with head_dim 128/256")
    tensors = (logical_indices, block_table, token_to_request, sequence_lengths)
    if any(t.device != q.device for t in tensors):
        raise ValueError("all paged QSA inputs must be on the same device")
    integer_dtypes = (torch.int32, torch.int64)
    if (
        logical_indices.ndim != 2
        or logical_indices.shape[0] != q.shape[0]
        or logical_indices.dtype != torch.int32
    ):
        raise ValueError("logical_indices must be int32 [tokens, selection_width]")
    if block_table.ndim != 2 or block_table.dtype not in integer_dtypes:
        raise ValueError("block_table must be int32/int64 [requests, context]")
    for name, tensor in (
        ("token_to_request", token_to_request),
        ("sequence_lengths", sequence_lengths),
    ):
        if (
            tensor.shape != (q.shape[0],)
            or tensor.dtype not in integer_dtypes
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous int32/int64 [tokens]")
    # A SGLang physical token slot is a one-token page; unsqueeze is a view.
    k_cache = k_cache.unsqueeze(1)
    v_cache = v_cache.unsqueeze(1)

    scale = q.shape[2] ** -0.5 if softmax_scale is None else softmax_scale
    # Split-K reduction requires contiguous output head dimensions.
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    if q.shape[0] == 0:
        return out

    _, num_cu = _device_info(q.device)
    group_size = q.shape[1] // k_cache.shape[2]
    block_d = max(16, triton.next_power_of_2(q.shape[2]))
    if kv_splits is None:
        split_rows = q.shape[0]
        if split_rows <= 0:
            raise ValueError("num_decode_requests must be positive")
        kv_splits = _kv_splits_heuristic(
            split_rows, k_cache.shape[2], logical_indices.shape[1], num_cu
        )
    if kv_splits < 1 or kv_splits > 64 or kv_splits & (kv_splits - 1):
        raise ValueError("kv_splits must be a power of two in [1, 64]")
    block_n, num_warps, num_stages, waves_per_eu, sub_group = _kernel_config(
        q.shape[0], k_cache.shape[2], kv_splits, group_size, num_cu
    )
    num_head_groups = (group_size + sub_group - 1) // sub_group
    block_m = max(16, triton.next_power_of_2(sub_group))
    if logical_indices.shape[1] == 0 or 0 in block_table.shape or k_cache.shape[0] == 0:
        # Empty selections produce zero attention output; skip split reduction.
        return out.zero_()
    target = out
    partial_max = partial_sum = out
    widths = None
    if kv_splits > 1:
        if workspace is None:
            workspace = SparsePagedGQAWorkspace()
        target, partial_max, partial_sum, widths = workspace.get(q, kv_splits)
    grid_y = k_cache.shape[2] * num_head_groups
    _qsa_sparse_paged_gqa_kernel[(q.shape[0], grid_y, kv_splits)](
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_request,
        sequence_lengths,
        target,
        partial_max,
        partial_sum,
        widths,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        logical_indices.stride(0),
        logical_indices.stride(1),
        block_table.stride(0),
        block_table.stride(1),
        target.stride(0),
        target.stride(1),
        target.stride(-1),
        q.shape[0],
        k_cache.shape[0],
        block_table.shape[0],
        float(scale),
        TOPK=logical_indices.shape[1],
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        NUM_KV_HEADS=k_cache.shape[2],
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
