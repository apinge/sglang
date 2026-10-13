"""Prepare continuous selected K/V and metadata for packed QSA decode."""

import torch
import triton
import triton.language as tl


@triton.jit
def _selected_paged_slots(
    indices,
    page_table,
    row,
    page_row,
    visible,
    row_topk,
    cols,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_PAGES: tl.constexpr,
):
    logical = tl.load(indices + row * TOPK + cols, mask=cols < row_topk, other=-1).to(
        tl.int64
    )
    logical_valid = (cols < row_topk) & (logical >= 0) & (logical < visible)
    logical_page = tl.where(logical_valid, logical // PAGE_SIZE, 0)
    page_valid = logical_valid & (logical_page < PAGE_TABLE_WIDTH)
    physical_page = tl.load(
        page_table + page_row * PAGE_TABLE_WIDTH + logical_page,
        mask=page_valid,
        other=-1,
    ).to(tl.int64)
    valid = page_valid & (physical_page >= 0) & (physical_page < NUM_PAGES)
    slots = tl.where(valid, physical_page * PAGE_SIZE + logical % PAGE_SIZE, 0)
    return slots, valid


@triton.jit
def _prepare_qsa_packed_kv(
    k,
    v,
    indices,
    page_table,
    row_to_page_table,
    sequence_lens,
    packed_k,
    packed_v,
    relative_indices,
    cu_q,
    cu_k,
    kv_lens,
    BATCH: tl.constexpr,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_PAGE_TABLE_ROWS: tl.constexpr,
    NUM_PAGES: tl.constexpr,
    KV_WIDTH: tl.constexpr,
    BLOCK_INDICES: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tile = tl.program_id(0)
    row = tl.program_id(1)
    page_row = tl.load(row_to_page_table + row).to(tl.int64)
    visible = tl.maximum(tl.load(sequence_lens + row).to(tl.int64), 0)
    row_valid = (page_row >= 0) & (page_row < NUM_PAGE_TABLE_ROWS)
    page_row = tl.where(row_valid, page_row, 0)
    complete_groups = tl.minimum(visible // COMPRESS_RATIO, BLOCK_TOPK)
    row_topk = tl.minimum(
        TOPK, complete_groups * COMPRESS_RATIO + visible % COMPRESS_RATIO
    )
    row_topk = tl.where(row_valid, row_topk, 0)
    packed_start = row.to(tl.int64) * (TOPK + 1)

    if tile == 0:
        metadata_cols = tl.arange(0, BLOCK_INDICES)
        _, metadata_valid = _selected_paged_slots(
            indices,
            page_table,
            row,
            page_row,
            visible,
            row_topk,
            metadata_cols,
            TOPK,
            PAGE_SIZE,
            PAGE_TABLE_WIDTH,
            NUM_PAGES,
        )
        count = tl.sum(metadata_valid.to(tl.int32), 0)
        if (count == row_topk) | (count == 0):
            relative = tl.where(metadata_cols < count, metadata_cols, -1)
            relative = tl.where((count == 0) & (metadata_cols == 0), TOPK, relative)
            tl.store(
                relative_indices + row * TOPK + metadata_cols,
                relative,
                mask=metadata_cols < TOPK,
            )
        else:
            # Stable partition gives every destination one writer. Initializing
            # the tail before scattering valid entries would race across waves.
            prefix = tl.cumsum(metadata_valid.to(tl.int32), 0)
            metadata_destination = tl.where(
                metadata_valid, prefix - 1, count + metadata_cols - prefix
            )
            relative = tl.where(metadata_valid, metadata_cols, -1)
            tl.store(
                relative_indices + row * TOPK + metadata_destination,
                relative,
                mask=metadata_cols < TOPK,
            )

        tl.store(cu_q + row, row)
        tl.store(cu_k + row, packed_start)
        tl.store(kv_lens + row, tl.maximum(count, 1))
        if row == BATCH - 1:
            tl.store(cu_q + BATCH, BATCH)
            tl.store(cu_k + BATCH, BATCH * (TOPK + 1))

        # The unchanged attention kernel requires at least one valid entry.
        # A zero sentinel makes graph padding and entirely invalid rows return
        # zero without introducing a special case in attention's softmax.
        sentinel_dims = tl.arange(0, BLOCK_D)
        sentinel = (packed_start + TOPK) * KV_WIDTH + sentinel_dims
        tl.store(packed_k + sentinel, 0.0, mask=sentinel_dims < KV_WIDTH)
        tl.store(packed_v + sentinel, 0.0, mask=sentinel_dims < KV_WIDTH)
    else:
        gather_cols = (tile - 1) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
        slots, gather_valid = _selected_paged_slots(
            indices,
            page_table,
            row,
            page_row,
            visible,
            row_topk,
            gather_cols,
            TOPK,
            PAGE_SIZE,
            PAGE_TABLE_WIDTH,
            NUM_PAGES,
        )
        gather_dims = tl.arange(0, BLOCK_D)
        source = slots[:, None] * KV_WIDTH + gather_dims[None, :]
        gather_destination = (
            packed_start + gather_cols[:, None]
        ) * KV_WIDTH + gather_dims[None, :]
        mask = gather_valid[:, None] & (gather_dims[None, :] < KV_WIDTH)
        tl.store(
            packed_k + gather_destination,
            tl.load(k + source, mask=mask, other=0.0),
            mask,
        )
        tl.store(
            packed_v + gather_destination,
            tl.load(v + source, mask=mask, other=0.0),
            mask,
        )


def prepare_qsa_packed_kv(
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
    """Gather paged selections and all packed-attention metadata in one launch.

    Each row owns a continuous ``TOPK + 1`` K/V segment. Valid selections retain
    their original slots, and relative indices list those slots in a compact
    prefix. Unreferenced K/V slots are unspecified. The final slot is a zero
    sentinel used only for empty rows, whose effective length is one.

    The caller selects the supported hardware and model shape. All inputs must
    be contiguous and on the same CUDA device; integer metadata must be int32.
    Allocations have fixed shapes and the kernel reads all mutable metadata on
    device, so the preparation can be captured and replayed in a CUDA graph.
    """

    if (
        k.ndim != 3
        or v.shape != k.shape
        or indices.ndim != 2
        or page_table.ndim != 2
        or row_to_page_table.ndim != 1
        or sequence_lens.ndim != 1
        or row_to_page_table.shape[0] != indices.shape[0]
        or sequence_lens.shape[0] != indices.shape[0]
        or indices.shape[0] == 0
        or page_table.shape[0] == 0
        or page_table.shape[1] == 0
        or full_kv_page_size <= 0
        or compress_ratio <= 0
        or block_topk <= 0
        or indices.shape[1] != block_topk * compress_ratio + compress_ratio - 1
        or k.shape[0] < full_kv_page_size
        or k.shape[0] % full_kv_page_size != 0
        or k.shape[1] <= 0
        or k.shape[2] <= 0
        or k.dtype != v.dtype
    ):
        raise ValueError("packed QSA preparation received incompatible input shapes")
    tensors = (k, v, indices, page_table, row_to_page_table, sequence_lens)
    if any(
        not tensor.is_cuda or tensor.device != k.device or not tensor.is_contiguous()
        for tensor in tensors
    ):
        raise ValueError("packed QSA preparation requires contiguous CUDA inputs")
    if any(tensor.dtype != torch.int32 for tensor in tensors[2:]):
        raise ValueError("packed QSA preparation requires int32 metadata")

    batch, topk = indices.shape
    packed_shape = (batch * (topk + 1), *k.shape[1:])
    packed_k = torch.empty(packed_shape, dtype=k.dtype, device=k.device)
    packed_v = torch.empty_like(packed_k)
    relative_indices = torch.empty_like(indices)
    cu_q = torch.empty(batch + 1, dtype=torch.int32, device=k.device)
    cu_k = torch.empty_like(cu_q)
    kv_lens = torch.empty(batch, dtype=torch.int32, device=k.device)
    block_tokens = 16
    _prepare_qsa_packed_kv[(triton.cdiv(topk, block_tokens) + 1, batch)](
        k,
        v,
        indices,
        page_table,
        row_to_page_table,
        sequence_lens,
        packed_k,
        packed_v,
        relative_indices,
        cu_q,
        cu_k,
        kv_lens,
        BATCH=batch,
        TOPK=topk,
        PAGE_SIZE=full_kv_page_size,
        COMPRESS_RATIO=compress_ratio,
        BLOCK_TOPK=block_topk,
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_PAGE_TABLE_ROWS=page_table.shape[0],
        NUM_PAGES=k.shape[0] // full_kv_page_size,
        KV_WIDTH=k.shape[1] * k.shape[2],
        BLOCK_INDICES=triton.next_power_of_2(topk),
        BLOCK_TOKENS=block_tokens,
        BLOCK_D=triton.next_power_of_2(k.shape[1] * k.shape[2]),
        num_warps=4,
        num_stages=1,
    )
    return packed_k, packed_v, relative_indices, cu_q, cu_k, kv_lens
