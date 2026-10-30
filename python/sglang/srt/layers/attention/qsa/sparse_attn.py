"""Validated sparse GQA operators migrated from the QSA reference branch."""

import functools
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.utils.common import is_hip

# (BLOCK_N, num_warps, num_stages), selected by total query rows.
_H20_CONFIGS = [
    (32, (32, 8, 2)),
    (64, (64, 8, 2)),
    (1024, (32, 4, 2)),
    (float("inf"), (16, 1, 2)),
]
_L20_CONFIGS = [
    (32, (32, 8, 2)),
    (64, (64, 8, 2)),
    (128, (64, 4, 2)),
    (512, (32, 4, 2)),
    (float("inf"), (16, 1, 2)),
]
# Measured on MI308X against _sparse_gqa_chunk_prefill at head_dim 256 / group
# size 12. Dropping num_stages to 1 is worth ~30% on its own: the software
# pipeline spills on gfx942 at this tile size.
_GFX942_CONFIGS = [
    (32, (64, 4, 1)),
    (128, (32, 2, 1)),
    (float("inf"), (16, 1, 1)),
]
# Split-KV decode: (BLOCK_N, SPLIT_SIZE, num_warps, num_stages) by query rows.
# Measured on MI308X at head_dim 256, group size 12, topk 2051; SPLIT_SIZE
# grows with the batch so the program count stays near the 80 compute units.
_GFX942_DECODE_CONFIGS = [
    (2, (64, 64, 4, 1)),
    (8, (64, 128, 4, 1)),
    (16, (32, 128, 2, 1)),
    (32, (32, 256, 2, 1)),
    (float("inf"), (32, 512, 2, 1)),
]
# Elementwise passes over the topk index row. A wavefront is 64 lanes on
# gfx942, so the NVIDIA num_warps=8 is 512 threads there; measured on MI308X.
_VALID_COUNTS_NUM_WARPS = 4 if is_hip() else 8
_COMPACT_KV_NUM_WARPS = 1 if is_hip() else 8
# Split count for decode on architectures without a measured table above.
# Arbitrary; picked to oversubscribe a large GPU rather than from measurement.
_DECODE_TARGET_PROGRAMS = 512
_DECODE_COMBINE_BLOCK_D = 128
_DECODE_COMBINE_NUM_WARPS = 1
# Folding the prefix sum into the count kernel rescans rows 0..b per program,
# which wins back a launch only while the batch is small. Measured crossover.
_FUSED_PREFIX_MAX_BATCH = 8


@functools.lru_cache(maxsize=1)
def _config_table():
    if is_hip():
        return _GFX942_CONFIGS
    return _H20_CONFIGS if "H20" in torch.cuda.get_device_name(0) else _L20_CONFIGS


def _get_best_config(total_q: int):
    return next(cfg for limit, cfg in _config_table() if total_q <= limit)


def _get_decode_config(rows: int, topk: int):
    """Pick (BLOCK_N, SPLIT_SIZE, num_warps, num_stages) for split-KV decode.

    ``rows`` is ``batch * num_kv_heads``, the program count of the unsplit
    kernel; splitting the topk budget brings the grid back up to the target.
    """

    if is_hip():
        block_n, split_size, warps, stages = next(
            cfg for limit, cfg in _GFX942_DECODE_CONFIGS if rows <= limit
        )
    else:
        block_n, warps, stages = _get_best_config(rows)
        target_splits = max(1, triton.cdiv(_DECODE_TARGET_PROGRAMS, max(rows, 1)))
        split_size = triton.next_power_of_2(triton.cdiv(topk, target_splits))
    # A split must end on a BLOCK_N boundary, or its last block would be all
    # padding and its running maximum would stay at -inf.
    return block_n, triton.cdiv(split_size, block_n) * block_n, warps, stages


@triton.jit
def _sparse_gqa_prefill(
    q,
    k,
    v,
    out,
    indices,
    cu_seqlens,
    scale,
    topk,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    sk_n: tl.constexpr,
    sk_h: tl.constexpr,
    sk_d: tl.constexpr,
    sv_n: tl.constexpr,
    sv_h: tl.constexpr,
    sv_d: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    si_m: tl.constexpr,
    si_g: tl.constexpr,
    si_n: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    batch_group = tl.program_id(1)
    group = batch_group % NUM_KV_HEADS
    batch = batch_group // NUM_KV_HEADS
    seq_start = tl.load(cu_seqlens + batch).to(tl.int64)
    seq_end = tl.load(cu_seqlens + batch + 1).to(tl.int64)
    query_relative = tl.program_id(0).to(tl.int64)
    query = seq_start + query_relative
    if query >= seq_end:
        return

    row_topk = tl.minimum(topk, query_relative + 1)
    row_limit = tl.minimum(topk, ((row_topk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    head_start = group * GROUP_SIZE
    q_values = tl.load(
        q
        + query * sq_m
        + (head_start + offs_h[:, None]) * sq_h
        + offs_d[None, :] * sq_d,
        mask=(offs_h < GROUP_SIZE)[:, None],
        other=0.0,
    )
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    k_base = k + seq_start * sk_n + group * sk_h
    v_base = v + seq_start * sv_n + group * sv_h
    idx_row = indices + query * si_m + group * si_g
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in range(0, row_limit, BLOCK_N):
        current = start + offs_n
        token = tl.load(idx_row + current * si_n, mask=current < topk, other=-1)
        valid = token >= 0
        keys = tl.load(
            k_base + token[None, :] * sk_n + offs_d[:, None] * sk_d,
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_base + token[:, None] * sv_n + offs_d[None, :] * sv_d,
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.where(valid[None, :], tl.dot(q_values, keys), -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, accumulator * alpha[:, None]
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = next_max
    output = accumulator / normalizer[:, None]
    tl.store(
        out
        + query * so_m
        + (head_start + offs_h[:, None]) * so_h
        + offs_d[None, :] * so_d,
        output,
        mask=(offs_h < GROUP_SIZE)[:, None],
    )


def sparse_gqa_fwd_interface_triton(q, k, v, max_seqlen_k, indices, cu_seqlens, scale):
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    block_m = max(16, triton.next_power_of_2(group_size))
    block_n, warps, stages = _get_best_config(total_q)
    out = torch.empty_like(q)
    _sparse_gqa_prefill[(max_seqlen_k, (cu_seqlens.shape[0] - 1) * num_kv_heads)](
        q,
        k,
        v,
        out,
        indices,
        cu_seqlens,
        scale,
        indices.shape[-1],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        indices.stride(0),
        indices.stride(1) if indices.ndim == 3 else 0,
        indices.stride(2) if indices.ndim == 3 else indices.stride(1),
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        num_warps=warps,
        num_stages=stages,
    )
    return out


@triton.jit
def _sparse_gqa_chunk_prefill(
    q,
    k,
    v,
    out,
    indices,
    cu_q,
    cu_k,
    kv_lens,
    scale,
    topk,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    sk_n: tl.constexpr,
    sk_h: tl.constexpr,
    sk_d: tl.constexpr,
    sv_n: tl.constexpr,
    sv_h: tl.constexpr,
    sv_d: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    si_m: tl.constexpr,
    si_g: tl.constexpr,
    si_n: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    query_relative = tl.program_id(0).to(tl.int64)
    batch_group = tl.program_id(1)
    group = batch_group % NUM_KV_HEADS
    batch = batch_group // NUM_KV_HEADS
    q_start = tl.load(cu_q + batch)
    q_end = tl.load(cu_q + batch + 1)
    query = (q_start + query_relative).to(tl.int64)
    if query >= q_end:
        return
    k_start = tl.load(cu_k + batch).to(tl.int64)
    kv_len = tl.load(kv_lens + batch).to(tl.int64)
    visible = query_relative + kv_len - (q_end - q_start) + 1
    row_topk = tl.minimum(topk, visible)
    row_limit = tl.minimum(topk, ((row_topk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_values = tl.load(
        q
        + query * sq_m
        + (group * GROUP_SIZE + offs_h[:, None]) * sq_h
        + offs_d[None, :] * sq_d,
        mask=(offs_h < GROUP_SIZE)[:, None],
        other=0.0,
    )
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    k_base = k + k_start * sk_n + group * sk_h
    v_base = v + k_start * sv_n + group * sv_h
    idx_row = indices + query * si_m + group * si_g
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in range(0, row_limit, BLOCK_N):
        current = start + offs_n
        token = tl.load(idx_row + current * si_n, mask=current < topk, other=-1)
        valid = token >= 0
        keys = tl.load(
            k_base + token[None, :] * sk_n + offs_d[:, None] * sk_d,
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_base + token[:, None] * sv_n + offs_d[None, :] * sv_d,
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.where(valid[None, :], tl.dot(q_values, keys), -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, accumulator * alpha[:, None]
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = next_max
    output = accumulator / normalizer[:, None]
    tl.store(
        out
        + query * so_m
        + (group * GROUP_SIZE + offs_h[:, None]) * so_h
        + offs_d[None, :] * so_d,
        output,
        mask=(offs_h < GROUP_SIZE)[:, None],
    )


def sparse_gqa_fwd_interface_triton_ck(q, k, v, indices, cu_q, cu_k, kv_lens, scale):
    k, v = k.contiguous(), v.contiguous()
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    max_q = int((cu_q[1:] - cu_q[:-1]).max().item())
    block_m = max(16, triton.next_power_of_2(group_size))
    block_n, warps, stages = _get_best_config(total_q)
    out = torch.empty_like(q)
    _sparse_gqa_chunk_prefill[(max_q, (cu_q.shape[0] - 1) * num_kv_heads)](
        q,
        k,
        v,
        out,
        indices,
        cu_q,
        cu_k,
        kv_lens,
        scale,
        indices.shape[-1],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        indices.stride(0),
        indices.stride(1) if indices.ndim == 3 else 0,
        indices.stride(2) if indices.ndim == 3 else indices.stride(1),
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        num_warps=warps,
        num_stages=stages,
    )
    return out


@triton.jit
def _sparse_gqa_split_decode(
    q,
    k,
    v,
    partial_out,
    partial_max,
    partial_sum,
    indices,
    cu_q,
    cu_k,
    kv_lens,
    scale,
    topk,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    sk_n: tl.constexpr,
    sk_h: tl.constexpr,
    sk_d: tl.constexpr,
    sv_n: tl.constexpr,
    sv_h: tl.constexpr,
    sv_d: tl.constexpr,
    si_m: tl.constexpr,
    si_g: tl.constexpr,
    si_n: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    SPLIT_SIZE: tl.constexpr,
    IDENTITY_INDICES: tl.constexpr,
):
    split = tl.program_id(0)
    batch_group = tl.program_id(1)
    group = batch_group % NUM_KV_HEADS
    batch = batch_group // NUM_KV_HEADS
    query = tl.load(cu_q + batch).to(tl.int64)
    kv_len = tl.load(kv_lens + batch).to(tl.int64)
    row_topk = tl.minimum(topk, kv_len)
    row_limit = tl.minimum(topk, ((row_topk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    offs_h = tl.arange(0, BLOCK_M)
    part = batch_group * NUM_SPLITS + split
    start = split * SPLIT_SIZE
    # An empty split still publishes its state; the combine pass reads every
    # split unconditionally and drops the ones with a zero normalizer.
    if start >= row_limit:
        tl.store(partial_max + part * BLOCK_M + offs_h, -float("inf"))
        tl.store(partial_sum + part * BLOCK_M + offs_h, 0.0)
        return

    k_start = tl.load(cu_k + batch).to(tl.int64)
    stop = tl.minimum(start + SPLIT_SIZE, row_limit)
    offs_d = tl.arange(0, HEAD_DIM)
    q_values = tl.load(
        q
        + query * sq_m
        + (group * GROUP_SIZE + offs_h[:, None]) * sq_h
        + offs_d[None, :] * sq_d,
        mask=(offs_h < GROUP_SIZE)[:, None],
        other=0.0,
    )
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    k_base = k + k_start * sk_n + group * sk_h
    v_base = v + k_start * sv_n + group * sv_h
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for begin in range(start, stop, BLOCK_N):
        current = begin + offs_n
        if IDENTITY_INDICES:
            token = current
            valid = current < kv_len
        else:
            token = tl.load(
                indices + query * si_m + group * si_g + current * si_n,
                mask=current < topk,
                other=-1,
            )
            valid = token >= 0
        keys = tl.load(
            k_base + token[None, :] * sk_n + offs_d[:, None] * sk_d,
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_base + token[:, None] * sv_n + offs_d[None, :] * sv_d,
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.where(valid[None, :], tl.dot(q_values, keys), -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, accumulator * alpha[:, None]
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = next_max
    tl.store(partial_max + part * BLOCK_M + offs_h, max_value)
    tl.store(partial_sum + part * BLOCK_M + offs_h, normalizer)
    tl.store(
        partial_out
        + part * BLOCK_M * HEAD_DIM
        + offs_h[:, None] * HEAD_DIM
        + offs_d[None, :],
        accumulator,
    )


@triton.jit
def _sparse_gqa_combine_splits(
    partial_out,
    partial_max,
    partial_sum,
    out,
    cu_q,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    batch_group = tl.program_id(0)
    head = tl.program_id(1)
    dim_block = tl.program_id(2)
    group = batch_group % NUM_KV_HEADS
    batch = batch_group // NUM_KV_HEADS
    query = tl.load(cu_q + batch).to(tl.int64)
    offs_s = tl.arange(0, BLOCK_S)
    offs_d = dim_block * BLOCK_D + tl.arange(0, BLOCK_D)
    in_range = offs_s < NUM_SPLITS
    parts = batch_group * NUM_SPLITS + offs_s
    split_max = tl.load(
        partial_max + parts * BLOCK_M + head, mask=in_range, other=-float("inf")
    )
    split_sum = tl.load(partial_sum + parts * BLOCK_M + head, mask=in_range, other=0.0)
    # A split that scored nothing has an -inf maximum and an unwritten
    # accumulator row; drop it rather than scaling garbage by exp2(-inf).
    live = in_range & (split_sum > 0)
    alpha = tl.where(live, tl.math.exp2(split_max - tl.max(split_max, 0)), 0.0)
    values = tl.load(
        partial_out
        + parts[:, None] * (BLOCK_M * HEAD_DIM)
        + head * HEAD_DIM
        + offs_d[None, :],
        mask=live[:, None],
        other=0.0,
    )
    tl.store(
        out + query * so_m + (group * GROUP_SIZE + head) * so_h + offs_d * so_d,
        tl.sum(alpha[:, None] * values, 0) / tl.sum(alpha * split_sum, 0),
    )


def sparse_gqa_packed_decode_triton(
    q, k, v, indices, cu_q, cu_k, kv_lens, scale, identity_topk: Optional[int] = None
):
    """Run one packed sparse-attention row per request without a host sync.

    Decode and speculative-verify already compact the selected K/V rows into
    contiguous request segments.  The topk budget is split across programs
    flash-decoding style: one row per request would otherwise leave all but
    ``batch * num_kv_heads`` compute units idle.

    ``indices`` is None when the packed rows are already the selected rows in
    order, i.e. every index row is ``[0, 1, ..., kv_lens[row] - 1, -1, ...]``;
    pass its width as ``identity_topk`` instead of materializing it.
    """

    k, v = k.contiguous(), v.contiguous()
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    block_m = max(16, triton.next_power_of_2(group_size))
    topk = identity_topk if indices is None else indices.shape[-1]
    rows = (cu_q.shape[0] - 1) * num_kv_heads
    block_n, split_size, warps, stages = _get_decode_config(rows=rows, topk=topk)
    num_splits = triton.cdiv(topk, split_size)
    out = torch.empty_like(q)
    partial_out = torch.empty(
        (rows * num_splits, block_m, head_dim), dtype=torch.float32, device=q.device
    )
    partial_state = torch.empty(
        (2, rows * num_splits, block_m), dtype=torch.float32, device=q.device
    )
    _sparse_gqa_split_decode[(num_splits, rows)](
        q,
        k,
        v,
        partial_out,
        partial_state[0],
        partial_state[1],
        indices,
        cu_q,
        cu_k,
        kv_lens,
        scale,
        topk,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        0 if indices is None else indices.stride(0),
        0 if indices is None or indices.ndim != 3 else indices.stride(1),
        0 if indices is None else indices.stride(-1),
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        NUM_SPLITS=num_splits,
        SPLIT_SIZE=split_size,
        IDENTITY_INDICES=indices is None,
        num_warps=warps,
        num_stages=stages,
    )
    block_d = min(head_dim, _DECODE_COMBINE_BLOCK_D)
    _sparse_gqa_combine_splits[(rows, group_size, head_dim // block_d)](
        partial_out,
        partial_state[0],
        partial_state[1],
        out,
        cu_q,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        NUM_SPLITS=num_splits,
        BLOCK_S=triton.next_power_of_2(num_splits),
        num_warps=_DECODE_COMBINE_NUM_WARPS,
        num_stages=1,
    )
    return out


@triton.jit
def _fa2_valid_counts(
    seq_lens,
    indices,
    counts,
    topk: tl.constexpr,
    stride_i: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_TOPK)
    length = tl.load(seq_lens + row)
    positions = tl.load(
        indices + row * stride_i + cols,
        mask=cols < topk,
        other=-1,
    )
    valid = (positions >= 0) & (positions < length)
    tl.store(counts + row, tl.sum(valid.to(tl.int32), axis=0))


@triton.jit
def _fa2_prefix_sum(counts, cu_k, batch, BLOCK_B: tl.constexpr):
    rows = tl.arange(0, BLOCK_B)
    valid_rows = rows < batch
    row_counts = tl.load(counts + rows, mask=valid_rows, other=0)
    tl.store(cu_k, 0)
    tl.store(cu_k + rows + 1, tl.cumsum(row_counts, 0), mask=valid_rows)


@triton.jit
def _fa2_valid_counts_and_prefix_sum(
    seq_lens,
    indices,
    counts,
    cu_k,
    topk: tl.constexpr,
    stride_i: tl.constexpr,
    BATCH: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_B)
    in_batch = rows < BATCH
    lengths = tl.load(seq_lens + rows, mask=in_batch, other=0)
    row_counts = tl.zeros([BLOCK_B], tl.int32)
    # The whole [BLOCK_B, BLOCK_TOPK] index tile would exceed Triton's
    # 1M-element limit at prefill batches, so walk topk in chunks.
    for start in range(0, topk, BLOCK_TOPK):
        cols = start + tl.arange(0, BLOCK_TOPK)
        positions = tl.load(
            indices + rows[:, None] * stride_i + cols[None, :],
            mask=in_batch[:, None] & (cols < topk)[None, :],
            other=-1,
        )
        valid = (positions >= 0) & (positions < lengths[:, None])
        row_counts += tl.sum(valid.to(tl.int32), axis=1)
    tl.store(counts + rows, row_counts, mask=in_batch)
    tl.store(cu_k, 0)
    tl.store(cu_k + rows + 1, tl.cumsum(row_counts, 0), mask=in_batch)


def qwen_sparse_fa2_cu_seqlens_triton(
    seq_lens, indices, counts, cu_k, batch, topk, block_b: Optional[int] = None
):
    block_b = block_b or triton.next_power_of_2(batch)
    if batch <= _FUSED_PREFIX_MAX_BATCH:
        _fa2_valid_counts_and_prefix_sum[(1,)](
            seq_lens,
            indices,
            counts,
            cu_k,
            topk,
            indices.stride(0),
            BATCH=batch,
            BLOCK_B=block_b,
            BLOCK_TOPK=triton.next_power_of_2(topk),
            num_warps=_VALID_COUNTS_NUM_WARPS,
        )
        return
    # Count one request per program. The previous implementation formed a
    # [next_power_of_2(topk), next_power_of_2(batch)] tensor in one program;
    # topk=2051 and batch=512 therefore exceeded Triton's 1M-element limit.
    _fa2_valid_counts[(batch,)](
        seq_lens,
        indices,
        counts,
        topk,
        indices.stride(0),
        BLOCK_TOPK=triton.next_power_of_2(topk),
        num_warps=_VALID_COUNTS_NUM_WARPS,
    )
    # Prefix sum is only over the batch dimension and remains a small 1-D
    # tensor, including during CUDA graph capture.
    _fa2_prefix_sum[(1,)](
        counts,
        cu_k,
        batch,
        BLOCK_B=block_b,
        num_warps=_VALID_COUNTS_NUM_WARPS,
    )


@triton.jit
def _compact_kv(
    k,
    v,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    topk: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    req_stride: tl.constexpr,
    idx_stride: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cols = block * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
    dims = tl.arange(0, BLOCK_D)
    length = tl.load(seq_lens + batch)
    req = tl.load(req_indices + batch)
    pack_start = tl.load(cu_k + batch)
    valid_count = tl.load(cu_k + batch + 1) - pack_start
    positions = tl.load(indices + batch * idx_stride + cols, mask=cols < topk, other=-1)
    valid = (cols < valid_count) & (positions >= 0) & (positions < length)
    slots = tl.load(
        req_to_token + req * req_stride + tl.where(valid, positions, 0),
        mask=valid,
        other=0,
    )
    src = slots[:, None] * heads * dim + head * dim + dims[None, :]
    dst = (pack_start + cols)[:, None] * heads * dim + head * dim + dims[None, :]
    mask = valid[:, None] & (dims[None, :] < dim)
    tl.store(out_k + dst, tl.load(k + src, mask=mask, other=0.0), mask=mask)
    tl.store(out_v + dst, tl.load(v + src, mask=mask, other=0.0), mask=mask)


def qwen_sparse_valid_counts_triton(seq_lens, indices, counts, batch, topk):
    """Valid-count pass alone, for consumers that need per-row lengths but
    not the packed cu_seqlens prefix sum (trtllm paged decode packs rows at
    a fixed page-aligned stride instead)."""
    _fa2_valid_counts[(batch,)](
        seq_lens,
        indices,
        counts,
        topk,
        indices.stride(0),
        BLOCK_TOPK=triton.next_power_of_2(topk),
        num_warps=_VALID_COUNTS_NUM_WARPS,
    )


def qwen_sparse_kv_extraction_compact_triton(
    k, v, req_to_token, req_indices, indices, seq_lens, cu_k, out_k, out_v, batch, topk
):
    _, heads, dim = k.shape
    block_topk = 16
    _compact_kv[(batch, heads, triton.cdiv(topk, block_topk))](
        k,
        v,
        req_to_token,
        req_indices,
        indices,
        seq_lens,
        cu_k,
        out_k,
        out_v,
        topk,
        heads,
        dim,
        req_to_token.stride(0),
        indices.stride(0),
        BLOCK_TOPK=block_topk,
        BLOCK_D=triton.next_power_of_2(dim),
        num_warps=_COMPACT_KV_NUM_WARPS,
    )


__all__ = [
    "qwen_sparse_fa2_cu_seqlens_triton",
    "qwen_sparse_valid_counts_triton",
    "qwen_sparse_kv_extraction_compact_triton",
    "sparse_gqa_fwd_interface_triton",
    "sparse_gqa_fwd_interface_triton_ck",
    "sparse_gqa_packed_decode_triton",
]
