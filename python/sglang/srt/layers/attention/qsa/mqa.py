"""Weight-free MQA operators for the simple QSA indexer.

The TileLang kernels are reduced versions of the previously validated Qwen MQA
kernels: the per-head weight input and all unrelated feature branches are
removed. Decode also has a Triton kernel for builds without TileLang. Torch
implementations are kept as the only fallback and reference.
"""

import math
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.utils.common import is_hip

try:
    import flashinfer.comm  # noqa: F401
except ImportError:
    pass

try:
    import tilelang
    from tilelang import language as T

    HAS_TILELANG = True
except ImportError:
    tilelang = None
    T = None
    HAS_TILELANG = False


# Measured on gfx942 (MI308X, wave64) with a 64x128 key tile and 1024 pages;
# the page-group count tracks the 80 CUs. Re-tune when the tile shape changes.
_QSA_MQA_DECODE_NUM_WARPS = 4
_QSA_MQA_DECODE_PAGE_GROUPS = 64
_QSA_MQA_DECODE_BLOCK_FILL = 512


def _validate_q(q: torch.Tensor) -> None:
    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError(f"QSA requires q [tokens, heads, head_dim], got {q.shape}")


def _validate_k(k: torch.Tensor) -> None:
    if k.ndim != 3 or k.shape[1] != 1 or k.shape[2] <= 0:
        raise ValueError(f"QSA MQA requires k [tokens, 1, head_dim], got {k.shape}")


def torch_qsa_mqa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """Torch reference for packed, variable-length prefill MQA."""

    _validate_q(q)
    _validate_k(k)
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("QSA query and key head dimensions must match")
    scores = torch.einsum("mhd,nd->mnh", q.float(), k[:, 0].float())
    logits = torch.relu(scores).sum(dim=-1) / (score_scale or math.sqrt(q.shape[-1]))
    columns = torch.arange(k.shape[0], device=q.device).unsqueeze(0)
    valid = (columns >= row_starts.to(q.device).reshape(-1, 1)) & (
        columns < row_ends.to(q.device).reshape(-1, 1)
    )
    return logits.masked_fill(~valid, -float("inf"))


def _validate_decode_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
) -> None:
    _validate_q(q)
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError(
            "QSA decode cache must be [pages, page_size, 1, head_dim], "
            f"got {tuple(k_cache.shape)}"
        )
    if k_cache.shape[-1] != q.shape[-1]:
        raise ValueError("QSA query and key head dimensions must match")
    if page_table.ndim != 2 or page_table.shape[0] != q.shape[0]:
        raise ValueError("QSA decode page table must have one row per query")
    if context_lens.numel() != q.shape[0]:
        raise ValueError("QSA decode context lengths must have one entry per query")


def torch_qsa_mqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """Torch reference for variable-length paged decode MQA."""

    _validate_decode_inputs(q, k_cache, page_table, context_lens)
    batch = q.shape[0]
    page_size = k_cache.shape[1]
    total = page_table.shape[1] * page_size
    gathered = k_cache[page_table.long().clamp_min(0).reshape(-1), :, 0].reshape(
        batch, total, q.shape[-1]
    )
    scores = torch.einsum("bhd,bnd->bnh", q.float(), gathered.float())
    scores = torch.relu(scores).sum(dim=-1) / (score_scale or math.sqrt(q.shape[-1]))
    positions = torch.arange(total, device=q.device).unsqueeze(0)
    scores.masked_fill_(
        positions >= context_lens.to(q.device).reshape(-1, 1), -float("inf")
    )
    logits = torch.full(
        (batch, max_model_len), -float("inf"), dtype=torch.float32, device=q.device
    )
    copy_len = min(total, max_model_len)
    if copy_len:
        logits[:, :copy_len] = scores[:, :copy_len]
    return logits


@triton.jit
def _triton_qsa_mqa_decode_kernel(
    q,
    k_cache,
    page_table,
    context_lens,
    logits,
    scale,
    valid_limit,
    max_model_len,
    sq_b,
    sq_h,
    sq_d,
    sk_p,
    sk_t,
    sk_d,
    spt_b,
    sl_b,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    group = tl.program_id(0)
    batch = tl.program_id(1)
    groups = tl.num_programs(0)
    row = logits + batch * sl_b
    context_len = tl.minimum(tl.load(context_lens + batch).to(tl.int32), valid_limit)

    # Fill [context_len, max_model_len) with -inf in wide tiles: at a 64-wide
    # compressed page the tail is ~20x the scored region, and a page-wide store
    # moves only 256 B per program.
    fill_start = (context_len // BLOCK_F) * BLOCK_F
    offs_f = tl.arange(0, BLOCK_F)
    fill_tiles = (max_model_len - fill_start + BLOCK_F - 1) // BLOCK_F
    for tile in range(group, fill_tiles, groups):
        positions = fill_start + tile * BLOCK_F + offs_f
        tl.store(
            row + positions,
            float("-inf"),
            mask=(positions >= context_len) & (positions < max_model_len),
        )

    pages = (context_len + PAGE_SIZE - 1) // PAGE_SIZE
    if group < pages:
        offs_t = tl.arange(0, BLOCK_T)
        offs_h = tl.arange(0, BLOCK_H)
        offs_d = tl.arange(0, BLOCK_D)
        dim_mask = offs_d < HEAD_DIM
        q_values = tl.load(
            q + batch * sq_b + offs_h[:, None] * sq_h + offs_d[None, :] * sq_d,
            mask=(offs_h < HEADS)[:, None] & dim_mask[None, :],
            other=0.0,
        )
        for page in range(group, pages, groups):
            # A negative page id marks unused page-table padding; the torch
            # reference clamps it to page 0 and masks the row by context length.
            page_id = tl.maximum(tl.load(page_table + batch * spt_b + page), 0).to(
                tl.int64
            )
            keys = tl.load(
                k_cache
                + page_id * sk_p
                + offs_t[:, None] * sk_t
                + offs_d[None, :] * sk_d,
                mask=(offs_t < PAGE_SIZE)[:, None] & dim_mask[None, :],
                other=0.0,
            )
            # Zero-padded heads and dims contribute relu(0) = 0 to the head sum.
            scores = tl.maximum(tl.dot(keys, tl.trans(q_values)), 0.0)
            positions = page * PAGE_SIZE + offs_t
            tl.store(
                row + positions,
                tl.sum(scores, axis=1) / scale,
                mask=(offs_t < PAGE_SIZE) & (positions < context_len),
            )


def triton_qsa_mqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """Triton paged decode MQA, scoring one compressed page per program step."""

    _validate_decode_inputs(q, k_cache, page_table, context_lens)
    batch, heads, head_dim = q.shape
    page_size = int(k_cache.shape[1])
    logits = torch.empty((batch, max_model_len), dtype=torch.float32, device=q.device)
    if not batch or not max_model_len:
        return logits
    # Positions past the page table are -inf regardless of the context length.
    valid_limit = min(max_model_len, page_table.shape[1] * page_size)
    q_kernel = q.to(torch.bfloat16)
    groups = max(
        1, min(_QSA_MQA_DECODE_PAGE_GROUPS, triton.cdiv(max_model_len, page_size))
    )
    _triton_qsa_mqa_decode_kernel[(groups, batch)](
        q_kernel,
        k_cache,
        page_table,
        context_lens,
        logits,
        float(score_scale or math.sqrt(head_dim)),
        valid_limit,
        max_model_len,
        q_kernel.stride(0),
        q_kernel.stride(1),
        q_kernel.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(3),
        page_table.stride(0),
        logits.stride(0),
        HEADS=heads,
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        BLOCK_T=triton.next_power_of_2(page_size),
        # MFMA needs a 16-wide N dimension; CUDA MMA accepts 8 but 16 is legal too.
        BLOCK_H=max(16, triton.next_power_of_2(heads)),
        BLOCK_D=triton.next_power_of_2(head_dim),
        BLOCK_F=_QSA_MQA_DECODE_BLOCK_FILL,
        num_warps=_QSA_MQA_DECODE_NUM_WARPS,
        num_stages=1,
    )
    return logits


if HAS_TILELANG:

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        }
    )
    def _tilelang_qsa_mqa_prefill_kernel(
        heads: int,
        head_dim: int,
        block_n: int = 64,
        block_q: int = 32,
        num_stages: int = 3,
        threads: int = 512,
    ):
        rows = T.dynamic("rows")
        keys = T.dynamic("keys")

        @T.prim_func
        def kernel(
            Q: T.Tensor([rows * heads, head_dim], T.bfloat16),  # type: ignore
            K: T.Tensor([keys, head_dim], T.bfloat16),  # type: ignore
            Logits: T.Tensor([rows, keys], T.float32),  # type: ignore
            Starts: T.Tensor([rows], T.int32),  # type: ignore
            Ends: T.Tensor([rows], T.int32),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(rows, block_q), threads=threads) as bx:
                q_shared = T.alloc_shared([block_q * heads, head_dim], T.bfloat16)
                k_shared = T.alloc_shared([block_n, head_dim], T.bfloat16)
                scores = T.alloc_fragment([block_n, block_q * heads], T.float32)
                scores_3d = T.reshape(scores, (block_n, block_q, heads))
                reduced = T.alloc_fragment([block_n, block_q], T.float32)
                row_base = bx * block_q
                start_min = T.alloc_var(T.int32)
                end_max = T.alloc_var(T.int32)
                start_min = 2147483647
                end_max = -2147483648
                for qi in T.serial(block_q):
                    start_min = T.min(start_min, T.min(Starts[row_base + qi], keys))
                    end_max = T.max(end_max, T.min(Ends[row_base + qi], keys))

                T.copy(Q[row_base * heads, 0], q_shared)
                for ni in T.Pipelined(
                    T.ceildiv(end_max - start_min, block_n), num_stages=num_stages
                ):
                    T.copy(K[start_min + ni * block_n, 0], k_shared)
                    T.gemm(
                        k_shared,
                        q_shared,
                        scores,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
                    for n, qi, head in T.Parallel(block_n, block_q, heads):
                        scores_3d[n, qi, head] = T.max(scores_3d[n, qi, head], 0.0)
                    T.reduce_sum(scores_3d, reduced, dim=-1, clear=True)
                    for qi, n in T.Parallel(block_q, block_n):
                        Logits[row_base + qi, start_min + ni * block_n + n] = reduced[
                            n, qi
                        ]

        return kernel

    @tilelang.jit
    def _tilelang_qsa_mqa_mask_kernel(threads: int = 512, block_k: int = 4096):
        rows = T.dynamic("rows")
        keys = T.dynamic("keys")

        @T.prim_func
        def kernel(
            Logits: T.Tensor([rows, keys], T.float32),  # type: ignore
            Starts: T.Tensor([rows], T.int32),  # type: ignore
            Ends: T.Tensor([rows], T.int32),  # type: ignore
        ):
            with T.Kernel(rows, threads=threads) as bx:
                tx = T.thread_binding(0, threads, thread="threadIdx.x")
                for block in T.Pipelined(T.ceildiv(keys, block_k)):
                    for item in T.serial(block_k // threads):
                        column = block * block_k + item * threads + tx
                        if column < Starts[bx] or column >= Ends[bx]:
                            Logits[bx, column] = -T.infinity(T.float32)

        return kernel

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        }
    )
    def _tilelang_qsa_mqa_decode_kernel(
        heads: int,
        head_dim: int,
        page_size: int = 64,
        groups_per_cta: int = 1,
        num_stages: int = 3,
        threads: int = 128,
    ):
        # The validated MMA layout wants 64 GEMM rows; pages narrower than
        # that (full_page // ratio compressed views) are packed in sub-page
        # quadrants of one 64-row tile.
        GROUP = 64
        assert GROUP % page_size == 0, page_size
        sub_pages = GROUP // page_size
        batch = T.dynamic("batch")
        pages = T.dynamic("pages")
        max_pages = T.dynamic("max_pages")
        max_model_len = T.dynamic("max_model_len")

        @T.prim_func
        def kernel(
            Q: T.Tensor([batch, 1, heads, head_dim], T.bfloat16),  # type: ignore
            KCache: T.Tensor([pages, page_size, 1, head_dim], T.bfloat16),  # type: ignore
            PageTable: T.Tensor([batch, max_pages], T.int32),  # type: ignore
            ContextLens: T.Tensor([batch], T.int32),  # type: ignore
            Logits: T.Tensor([batch, max_model_len], T.float32),  # type: ignore
            Scale: T.float32,
        ):
            with T.Kernel(
                batch,
                T.ceildiv(T.ceildiv(max_pages, sub_pages), groups_per_cta),
                threads=threads,
            ) as (bx, group_block):
                q_shared = T.alloc_shared([heads, head_dim], T.bfloat16)
                k_shared = T.alloc_shared([GROUP, head_dim], T.bfloat16)
                scores = T.alloc_fragment([GROUP, heads], T.float32)
                reduced = T.alloc_fragment([GROUP], T.float32)
                T.copy(Q[bx, 0, :, :], q_shared)
                context_len = ContextLens[bx]

                for gi in T.Pipelined(groups_per_cta, num_stages=num_stages):
                    group = group_block * groups_per_cta + gi
                    if group * GROUP < context_len:
                        # Python-level unroll: sub_pages is a compile-time
                        # constant and TileLang's pipeliner rejects dynamic
                        # inner loops around shared-memory copies.
                        for sp in range(sub_pages):
                            if (group * sub_pages + sp) * page_size < context_len:
                                T.copy(
                                    KCache[
                                        PageTable[bx, group * sub_pages + sp],
                                        :,
                                        0,
                                        :,
                                    ],
                                    k_shared[sp * page_size : (sp + 1) * page_size, :],
                                )
                        T.gemm(
                            k_shared,
                            q_shared,
                            scores,
                            transpose_B=True,
                            clear_accum=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        for token, head in T.Parallel(GROUP, heads):
                            scores[token, head] = T.max(scores[token, head], 0.0)
                        T.reduce_sum(scores, reduced, dim=1, clear=True)
                        for token in T.Parallel(GROUP):
                            position = group * GROUP + token
                            if position < context_len:
                                Logits[bx, position] = reduced[token] / Scale

        return kernel


def tilelang_qsa_mqa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """Validated TileLang packed prefill kernel with weights removed."""

    if not HAS_TILELANG:
        raise RuntimeError("TileLang is unavailable")
    _validate_q(q)
    _validate_k(k)
    rows, keys = q.shape[0], k.shape[0]
    if not rows or not keys:
        logits = torch.zeros((rows, keys), dtype=torch.float32, device=q.device)
        return logits.masked_fill_(
            torch.ones_like(logits, dtype=torch.bool), -float("inf")
        )
    heads, head_dim = q.shape[1:]
    block_q = max(1, 128 // heads)
    padding = (-rows) % block_q
    padded_rows = rows + padding
    # Allocate the padded output once. Appending even a few padding rows with
    # torch.cat would allocate and copy the entire [rows, keys] FP32 matrix,
    # temporarily doubling the dominant prefill buffer for long contexts.
    logits = torch.zeros((padded_rows, keys), dtype=torch.float32, device=q.device)
    q_padded = q.to(torch.bfloat16).contiguous()
    starts = row_starts.to(device=q.device, dtype=torch.int32).contiguous()
    ends = row_ends.to(device=q.device, dtype=torch.int32).contiguous()
    if padding:
        q_padded = torch.cat([q_padded, q_padded.new_zeros(padding, heads, head_dim)])
        starts = torch.cat([starts, starts[-1:].expand(padding)])
        ends = torch.cat([ends, ends[-1:].expand(padding)])

    _tilelang_qsa_mqa_prefill_kernel(heads=heads, head_dim=head_dim, block_q=block_q)(
        q_padded.reshape(-1, head_dim),
        k[:, 0].to(torch.bfloat16).contiguous(),
        logits,
        starts,
        ends,
    )
    # A leading-dimension slice that retains every column is already
    # contiguous, so do not copy this large matrix again when removing padding.
    logits = logits[:rows]
    logits.div_(score_scale or math.sqrt(head_dim))
    _tilelang_qsa_mqa_mask_kernel()(logits, starts[:rows], ends[:rows])
    return logits


def tilelang_qsa_mqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """Validated TileLang paged decode kernel with weights removed."""

    if not HAS_TILELANG:
        raise RuntimeError("TileLang is unavailable")
    _validate_decode_inputs(q, k_cache, page_table, context_lens)
    page_size = int(k_cache.shape[1])
    if page_size < 8 or 64 % page_size != 0:
        raise ValueError(
            "TileLang QSA decode requires a compressed page size of "
            f"8/16/32/64 (64-row GEMM sub-page packing), got {page_size}"
        )
    logits = torch.full(
        (q.shape[0], max_model_len),
        -float("inf"),
        dtype=torch.float32,
        device=q.device,
    )
    if not q.shape[0] or not max_model_len:
        return logits
    # CUDA MMA accepts an eight-wide N dimension, while ROCm MFMA requires
    # sixteen. Zero-padding preserves the weight-free head sum on both paths.
    query_heads, head_dim = q.shape[1:]
    head_alignment = 16 if is_hip() else 8
    kernel_heads = max(
        head_alignment,
        ((query_heads + head_alignment - 1) // head_alignment) * head_alignment,
    )
    q_kernel = q.to(torch.bfloat16)
    if kernel_heads != query_heads:
        q_kernel = torch.cat(
            [
                q_kernel,
                q_kernel.new_zeros(q.shape[0], kernel_heads - query_heads, head_dim),
            ],
            dim=1,
        )
    _tilelang_qsa_mqa_decode_kernel(
        heads=kernel_heads, head_dim=head_dim, page_size=page_size
    )(
        q_kernel.unsqueeze(1).contiguous(),
        k_cache.to(torch.bfloat16).contiguous(),
        page_table.to(device=q.device, dtype=torch.int32).contiguous(),
        context_lens.to(device=q.device, dtype=torch.int32).contiguous(),
        logits,
        float(score_scale or math.sqrt(head_dim)),
    )
    return logits


def qsa_mqa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    if q.is_cuda and HAS_TILELANG:
        return tilelang_qsa_mqa_prefill(q, k, row_starts, row_ends, score_scale)
    return torch_qsa_mqa_prefill(q, k, row_starts, row_ends, score_scale)


def qsa_mqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    if q.is_cuda and HAS_TILELANG:
        return tilelang_qsa_mqa_decode(
            q, k_cache, page_table, context_lens, max_model_len, score_scale
        )
    if q.is_cuda:
        return triton_qsa_mqa_decode(
            q, k_cache, page_table, context_lens, max_model_len, score_scale
        )
    return torch_qsa_mqa_decode(
        q, k_cache, page_table, context_lens, max_model_len, score_scale
    )


__all__ = [
    "HAS_TILELANG",
    "qsa_mqa_decode",
    "qsa_mqa_prefill",
    "tilelang_qsa_mqa_decode",
    "tilelang_qsa_mqa_prefill",
    "triton_qsa_mqa_decode",
    "torch_qsa_mqa_decode",
    "torch_qsa_mqa_prefill",
]
