"""Single-CTA direct-paged sparse attention for QSA decode."""

import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_gfx942_supported


@triton.jit
def _sparse_gqa_chunk_prefill(
    q,
    k,
    v,
    out,
    indices,
    page_table,
    row_to_page_table,
    sequence_lens,
    num_pages,
    num_page_table_rows,
    scale,
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
    si_n: tl.constexpr,
    spt_m: tl.constexpr,
    spt_n: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    TOPK: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Compute one decode row directly from logical paged-cache indices."""

    batch = tl.program_id(0)
    page_table_row = tl.load(row_to_page_table + batch).to(tl.int64)
    visible = tl.maximum(tl.load(sequence_lens + batch).to(tl.int64), 0)
    row_valid = (page_table_row >= 0) & (page_table_row < num_page_table_rows)
    safe_page_table_row = tl.where(row_valid, page_table_row, 0)
    complete_groups = tl.minimum(visible // COMPRESS_RATIO, BLOCK_TOPK)
    row_topk = tl.minimum(
        TOPK,
        complete_groups * COMPRESS_RATIO + visible % COMPRESS_RATIO,
    )
    row_topk = tl.where(row_valid, row_topk, 0)
    row_limit = tl.minimum(TOPK, ((row_topk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)

    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_values = tl.load(
        q + batch * sq_m + offs_h[:, None] * sq_h + offs_d[None, :] * sq_d,
        mask=(offs_h < GROUP_SIZE)[:, None],
        other=0.0,
    )
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    idx_row = indices + batch * si_m

    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in range(0, row_limit, BLOCK_N):
        current = start + offs_n
        logical = tl.load(
            idx_row + current * si_n,
            mask=current < row_topk,
            other=-1,
        ).to(tl.int64)
        logical_valid = (current < row_topk) & (logical >= 0) & (logical < visible)
        logical_page = logical // PAGE_SIZE
        page_offset = logical % PAGE_SIZE
        page_column_valid = logical_valid & (logical_page < PAGE_TABLE_WIDTH)
        physical_page = tl.load(
            page_table + safe_page_table_row * spt_m + logical_page * spt_n,
            mask=page_column_valid,
            other=-1,
        ).to(tl.int64)
        valid = page_column_valid & (physical_page >= 0) & (physical_page < num_pages)
        physical_token = physical_page * PAGE_SIZE + page_offset
        keys = tl.load(
            k + physical_token[None, :] * sk_n + offs_d[:, None] * sk_d,
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v + physical_token[:, None] * sv_n + offs_d[None, :] * sv_d,
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.where(valid[None, :], tl.dot(q_values, keys), -float("inf"))
        block_has_values = tl.sum(valid.to(tl.int32), axis=0) > 0
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        safe_next_max = tl.where(block_has_values, next_max, 0.0)
        safe_max_value = tl.where(block_has_values, max_value, 0.0)
        alpha = tl.math.exp2(safe_max_value - safe_next_max)
        probabilities = tl.where(
            valid[None, :],
            tl.math.exp2(scores - safe_next_max[:, None]),
            0.0,
        )
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, accumulator * alpha[:, None]
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = tl.where(block_has_values, next_max, max_value)

    has_values = normalizer > 0.0
    output = tl.where(has_values[:, None], accumulator / normalizer[:, None], 0.0)
    tl.store(
        out + batch * so_m + offs_h[:, None] * so_h + offs_d[None, :] * so_d,
        output,
        mask=(offs_h < GROUP_SIZE)[:, None],
    )


def is_sparse_gqa_direct_paged_one_cta_supported(
    q,
    k,
    v,
    indices,
    page_table,
    row_to_page_table,
    sequence_lens,
    *,
    full_kv_page_size,
    compress_ratio,
    block_topk,
):
    """Restrict direct one-CTA dispatch to the measured MI308X Qwen shape."""

    metadata = (page_table, row_to_page_table, sequence_lens)
    return (
        q.is_cuda
        and k.is_cuda
        and v.is_cuda
        and indices.is_cuda
        and q.device == k.device == v.device
        and indices.device == q.device
        and is_gfx942_supported()
        and "MI308X" in torch.cuda.get_device_name(q.device)
        and q.ndim == k.ndim == v.ndim == 3
        and indices.ndim == 2
        and indices.dtype == torch.int32
        and indices.shape == (q.shape[0], 2051)
        and 1 <= q.shape[0] <= 8
        and q.shape[1:] == (12, 256)
        and k.shape[1:] == (1, 256)
        and v.shape[1:] == (1, 256)
        and q.dtype == torch.bfloat16
        and k.dtype == q.dtype
        and v.dtype == q.dtype
        and all(tensor.is_cuda and tensor.device == q.device for tensor in metadata)
        and page_table.ndim == 2
        and row_to_page_table.ndim == sequence_lens.ndim == 1
        and row_to_page_table.shape[0] == q.shape[0]
        and sequence_lens.shape[0] == q.shape[0]
        and page_table.shape[0] > 0
        and page_table.shape[1] > 0
        and v.shape == k.shape
        and page_table.dtype == torch.int32
        and row_to_page_table.dtype == torch.int32
        and sequence_lens.dtype == torch.int32
        and q.is_contiguous()
        and k.is_contiguous()
        and v.is_contiguous()
        and indices.is_contiguous()
        and page_table.is_contiguous()
        and row_to_page_table.is_contiguous()
        and sequence_lens.is_contiguous()
        and full_kv_page_size == 64
        and k.shape[0] >= full_kv_page_size
        and k.shape[0] % full_kv_page_size == 0
        and compress_ratio == 4
        and block_topk == 512
    )


def sparse_gqa_direct_paged_decode_one_cta_triton(
    q,
    k,
    v,
    indices,
    page_table,
    row_to_page_table,
    sequence_lens,
    scale,
    *,
    full_kv_page_size,
    compress_ratio,
    block_topk,
):
    """Run one sparse-attention CTA per row directly over the full-KV cache."""

    if not is_sparse_gqa_direct_paged_one_cta_supported(
        q,
        k,
        v,
        indices,
        page_table,
        row_to_page_table,
        sequence_lens,
        full_kv_page_size=full_kv_page_size,
        compress_ratio=compress_ratio,
        block_topk=block_topk,
    ):
        raise ValueError("direct-paged one-CTA decode received unsupported inputs")
    out = torch.empty_like(q)
    _sparse_gqa_chunk_prefill[(q.shape[0],)](
        q,
        k,
        v,
        out,
        indices,
        page_table,
        row_to_page_table,
        sequence_lens,
        k.shape[0] // full_kv_page_size,
        page_table.shape[0],
        scale,
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
        indices.stride(1),
        page_table.stride(0),
        page_table.stride(1),
        PAGE_TABLE_WIDTH=page_table.shape[1],
        PAGE_SIZE=full_kv_page_size,
        COMPRESS_RATIO=compress_ratio,
        BLOCK_TOPK=block_topk,
        TOPK=indices.shape[1],
        GROUP_SIZE=q.shape[1],
        BLOCK_M=16,
        BLOCK_N=64,
        HEAD_DIM=q.shape[2],
        num_warps=4,
        num_stages=1,
        kpack=2,
    )
    return out


__all__ = [
    "is_sparse_gqa_direct_paged_one_cta_supported",
    "sparse_gqa_direct_paged_decode_one_cta_triton",
]
