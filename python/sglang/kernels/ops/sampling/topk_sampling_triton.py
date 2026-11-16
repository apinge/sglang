"""Sort-free row sampling over large vocabularies (Triton; used by ROCm speculative verify).

Two kernels per call:
  1. ``_stats_kernel`` (rows x vocab chunks): per chunk, max and sum-exp of
     logits / T, plus (top-k mode) the maxima of GS-wide groups of logits.
  2. ``_sample_kernel`` (one program per row):
     * top-k mode: exact global top-K by hierarchical pruning -- the top-K
       elements lie in the K groups with the largest maxima, and those groups lie
       in the K super-groups (GS groups each) with the largest maxima -- then the
       joint top-k / top-p / min-p cut on the original softmax probabilities and
       an exponential-race draw;
     * temperature-only mode: exact two-stage inverse-CDF draw -- pick a chunk
       with probability proportional to its softmax mass, then a token inside it.

Semantics match ``top_k_top_p_min_p_sampling_from_probs_torch``: probabilities
are softmax(logits / T); a token is dropped if its rank >= top_k, if the mass of
strictly higher-ranked tokens exceeds top_p, or if p < min_p * p_max.
"""

import random
from typing import Optional

import torch
import triton
import triton.language as tl

_CHUNK = 8192
_GS = 32  # group size == super-group fan-in
MAX_TOP_K = 64
_TAIL_DRAWS = 8  # rejection draws for top-p rows whose support exceeds the top MAX_TOP_K


@triton.jit
def _stats_kernel(
    logits_ptr,
    stride_row,
    inv_t_ptr,
    chunk_max_ptr,
    chunk_sum_ptr,
    group_max_ptr,
    stride_group_row,
    vocab_size,
    n_chunks,
    TOPK: tl.constexpr,
    CHUNK: tl.constexpr,
    GS: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offs = chunk * CHUNK + tl.arange(0, CHUNK)
    valid = offs < vocab_size
    x = tl.load(
        logits_ptr + row.to(tl.int64) * stride_row + offs, mask=valid, other=float("-inf")
    ).to(tl.float32)
    y = x * tl.load(inv_t_ptr + row)
    m = tl.max(y, axis=0)
    s = tl.sum(tl.where(valid, tl.exp(y - m), 0.0), axis=0)
    s = tl.where(m == float("-inf"), 0.0, s)  # fully masked chunk (e.g. grammar): no mass, no NaN
    tl.store(chunk_max_ptr + row * n_chunks + chunk, m)
    tl.store(chunk_sum_ptr + row * n_chunks + chunk, s)
    if TOPK:
        G: tl.constexpr = CHUNK // GS
        gmax = tl.max(tl.reshape(x, [G, GS]), axis=1)
        tl.store(group_max_ptr + row * stride_group_row + chunk * G + tl.arange(0, G), gmax)


@triton.jit
def _topk_of(work, K: tl.constexpr, KP: tl.constexpr):
    """The K largest entries of a 1-D block in descending order, KP-padded.

    Returns (values, positions); a slot with no finite entry left gets position -1.
    """
    r = tl.arange(0, KP)
    j = tl.arange(0, work.shape[0])
    vals = tl.full([KP], float("-inf"), tl.float32)
    pos = tl.full([KP], -1, tl.int32)
    for k in range(K):
        v, i = tl.max(work, axis=0, return_indices=True)
        vals = tl.where(r == k, v, vals)
        pos = tl.where((r == k) & (v > float("-inf")), i, pos)
        work = tl.where(j == i, float("-inf"), work)
    return vals, pos


@triton.jit
def _pick(src, slot, KP: tl.constexpr):
    """src[slot] for a KP-vector src and KP-vector of slots (register gather)."""
    r = tl.arange(0, KP)
    return tl.sum(tl.where(r[None, :] == slot[:, None], src[None, :], 0), axis=1)


@triton.jit
def _draw_full(row_logits, w, n_chunks, vocab_size, inv_t, u1, u2, CHUNK: tl.constexpr):
    """Exact draw from softmax(logits / T) over the whole row by two-stage inverse CDF:
    a chunk with probability proportional to its mass ``w``, then a token inside it.
    Returns (token, logit / T of the token)."""
    cw = tl.cumsum(w, axis=0)
    ch = tl.minimum(tl.sum((cw <= u1 * tl.sum(w, axis=0)).to(tl.int32), axis=0), n_chunks - 1)
    i = tl.arange(0, CHUNK)
    offs = ch * CHUNK + i
    valid = offs < vocab_size
    y = tl.load(row_logits + offs, mask=valid, other=float("-inf")).to(tl.float32) * inv_t
    ex = tl.where(valid, tl.exp(y - tl.max(y, axis=0)), 0.0)
    ce = tl.cumsum(ex, axis=0)
    n_valid = tl.minimum(vocab_size - ch * CHUNK, CHUNK)
    t = tl.minimum(tl.sum((ce <= u2 * tl.sum(ex, axis=0)).to(tl.int32), axis=0), n_valid - 1)
    return ch * CHUNK + t, tl.sum(tl.where(i == t, y, 0.0), axis=0)


@triton.jit(do_not_specialize=["seed"])
def _sample_kernel(
    logits_ptr,
    stride_row,
    chunk_max_ptr,
    chunk_sum_ptr,
    group_max_ptr,
    stride_group_row,
    inv_t_ptr,
    top_k_ptr,
    top_p_ptr,
    min_p_ptr,
    out_ptr,
    lse_ptr,
    covered_ptr,
    top1_ptr,
    cand_tok_ptr,
    cand_y_ptr,
    cut_p_ptr,
    cut_idx_ptr,
    seed,
    vocab_size,
    n_chunks,
    n_groups,
    TOPK: tl.constexpr,
    TAIL: tl.constexpr,
    OUT_TOPK: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    NCP: tl.constexpr,
    NSP: tl.constexpr,
    CHUNK: tl.constexpr,
    GS: tl.constexpr,
    USE_MIN_P: tl.constexpr,
):
    row = tl.program_id(0)
    row_logits = logits_ptr + row.to(tl.int64) * stride_row
    inv_t = tl.load(inv_t_ptr + row)
    c = tl.arange(0, NCP)
    cm = c < n_chunks
    m = tl.load(chunk_max_ptr + row * n_chunks + c, mask=cm, other=float("-inf"))
    s = tl.load(chunk_sum_ptr + row * n_chunks + c, mask=cm, other=0.0)
    gmax = tl.max(m, axis=0)
    w = tl.where(cm, s * tl.exp(m - gmax), 0.0)  # chunk softmax mass, up to exp(gmax)
    if TOPK:
        lse = gmax + tl.log(tl.sum(w, axis=0))
        e = tl.arange(0, GS)
        # level 1: super-groups of GS consecutive groups
        gidx = tl.arange(0, NSP * GS)
        gm = tl.load(group_max_ptr + row * stride_group_row + gidx, mask=gidx < n_groups, other=float("-inf"))
        _sup_v, sup = _topk_of(tl.max(tl.reshape(gm, [NSP, GS]), axis=1), K, KP)
        # level 2: the K * GS groups inside the chosen super-groups
        g2 = sup[:, None] * GS + e[None, :]
        g2_ok = (sup[:, None] >= 0) & (g2 < n_groups)
        gm2 = tl.load(group_max_ptr + row * stride_group_row + g2, mask=g2_ok, other=float("-inf"))
        _g_v, p2 = _topk_of(tl.reshape(gm2, [KP * GS]), K, KP)
        r = tl.arange(0, KP)
        sel_grp = _pick(sup, p2 // GS, KP) * GS + p2 % GS
        # level 3: the K * GS logits inside the chosen groups
        tok = sel_grp[:, None] * GS + e[None, :]
        tok_ok = (p2[:, None] >= 0) & (tok < vocab_size)
        x3 = tl.load(row_logits + tok, mask=tok_ok, other=float("-inf")).to(tl.float32)
        vals, p3 = _topk_of(tl.reshape(x3, [KP * GS]), K, KP)
        tok_sel = _pick(sel_grp, p3 // GS, KP) * GS + p3 % GS

        p = tl.where(r < K, tl.exp(vals * inv_t - lse), 0.0)
        top_p = tl.load(top_p_ptr + row)
        do_cut = True
        if TAIL:
            # Unbounded top-k: the top-p support is inside the top K only if their mass
            # already exceeds top_p. Otherwise draw M candidates from the full softmax;
            # _tail_mass_kernel / _tail_pick_kernel accept the first one inside the support.
            covered = tl.sum(p, axis=0) > top_p
            tl.store(covered_ptr + row, covered.to(tl.int32))
            tl.store(lse_ptr + row, lse)
            tl.store(top1_ptr + row, tl.sum(tl.where(r == 0, tok_sel, 0), axis=0))
            if not covered:
                for mm in range(M):
                    tok_m, y_m = _draw_full(
                        row_logits, w, n_chunks, vocab_size, inv_t,
                        tl.rand(seed, (row * M + mm) * 2 + KP * 4096),
                        tl.rand(seed, (row * M + mm) * 2 + 1 + KP * 4096), CHUNK,
                    )
                    tl.store(cand_tok_ptr + row * M + mm, tok_m)
                    tl.store(cand_y_ptr + row * M + mm, y_m)
            do_cut = covered
        if do_cut:
            top_k = tl.load(top_k_ptr + row)
            if OUT_TOPK:
                # The cut support lies inside the top K iff top_k <= K or the top K alone
                # already carry more than top_p of the mass.
                full = (top_k <= K) | (tl.sum(p, axis=0) > top_p)
                tl.store(covered_ptr + row, full.to(tl.int32))
            drop = (r >= top_k) | ((tl.cumsum(p, axis=0) - p) > top_p)
            p = tl.where(drop, 0.0, p)
            if USE_MIN_P:
                p = tl.where(p < tl.max(p, axis=0) * tl.load(min_p_ptr + row), 0.0, p)
            if OUT_TOPK:
                # Emit the cut distribution (unnormalised, descending rank order) instead of a draw.
                tl.store(cut_p_ptr + row * KP + r, p)
                tl.store(cut_idx_ptr + row * KP + r, tok_sel)
            else:
                # exponential race: argmax p_i / E_i, E_i ~ Exp(1)
                u = tl.maximum(tl.rand(seed, row * KP + r), 1.0e-20)
                _best, win = tl.max(p / (-tl.log(u)), axis=0, return_indices=True)
                tl.store(out_ptr + row, tl.sum(tl.where(r == win, tok_sel, 0), axis=0).to(tl.int64))
    else:
        tok, _ = _draw_full(
            row_logits, w, n_chunks, vocab_size, inv_t,
            tl.rand(seed, row * 2), tl.rand(seed, row * 2 + 1), CHUNK,
        )
        tl.store(out_ptr + row, tok.to(tl.int64))


@triton.jit
def _tail_mass_kernel(
    logits_ptr,
    stride_row,
    inv_t_ptr,
    lse_ptr,
    covered_ptr,
    cand_y_ptr,
    partial_ptr,
    vocab_size,
    n_chunks,
    M: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """partial[row, chunk, m] = softmax mass in the chunk strictly above candidate m."""
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    if tl.load(covered_ptr + row) == 0:
        offs = chunk * CHUNK + tl.arange(0, CHUNK)
        valid = offs < vocab_size
        y = tl.load(
            logits_ptr + row.to(tl.int64) * stride_row + offs, mask=valid, other=float("-inf")
        ).to(tl.float32)
        y = y * tl.load(inv_t_ptr + row)
        p = tl.where(valid, tl.exp(y - tl.load(lse_ptr + row)), 0.0)
        for mm in range(M):
            thr = tl.load(cand_y_ptr + row * M + mm)
            tl.store(
                partial_ptr + (row * n_chunks + chunk) * M + mm,
                tl.sum(tl.where(y > thr, p, 0.0), axis=0),
            )


@triton.jit
def _tail_pick_kernel(
    covered_ptr,
    top_p_ptr,
    top1_ptr,
    cand_tok_ptr,
    partial_ptr,
    out_ptr,
    n_chunks,
    M: tl.constexpr,
    NCP: tl.constexpr,
):
    """Rejection step: candidate x is in the top-p support iff the mass strictly above it <= top_p."""
    row = tl.program_id(0)
    if tl.load(covered_ptr + row) == 0:
        c = tl.arange(0, NCP)
        mm = tl.arange(0, M)
        part = tl.load(
            partial_ptr + (row * n_chunks + c[:, None]) * M + mm[None, :],
            mask=(c < n_chunks)[:, None], other=0.0,
        )
        above = tl.sum(part, axis=0)  # (M,)
        ok = above <= tl.load(top_p_ptr + row)
        first = tl.min(tl.where(ok, mm, M), axis=0)
        cand = tl.load(cand_tok_ptr + row * M + mm)
        # All M draws outside the support (probability <= (1 - top_p) ** M): use the top-1 token.
        tok = tl.where(
            first < M, tl.sum(tl.where(mm == first, cand, 0), axis=0), tl.load(top1_ptr + row)
        )
        tl.store(out_ptr + row, tok.to(tl.int64))


def _launch(logits, temperatures, top_ks, top_ps, min_ps, k, top_p_tail=False, out_topk=False):
    rows, vocab = logits.shape
    if logits.stride(1) != 1:
        logits = logits.contiguous()
    dev = logits.device
    n_chunks = triton.cdiv(vocab, _CHUNK)
    ncp = triton.next_power_of_2(n_chunks)
    n_groups = n_chunks * (_CHUNK // _GS)
    nsp = triton.next_power_of_2(triton.cdiv(n_groups, _GS))
    topk = k is not None
    kk = k if topk else 1
    kp = triton.next_power_of_2(kk)
    inv_t = (1.0 / temperatures.view(-1).float()).contiguous()
    f32 = dict(dtype=torch.float32, device=dev)
    chunk_max = torch.empty(rows * n_chunks, **f32)
    chunk_sum = torch.empty_like(chunk_max)
    group_max = torch.empty((rows, nsp * _GS), **f32) if topk else chunk_max
    stride_g = group_max.stride(0) if topk else 0
    out = torch.empty(rows, dtype=torch.int64, device=dev) if not out_topk else inv_t
    if top_p_tail or out_topk:
        covered = torch.empty(rows, dtype=torch.int32, device=dev)
    else:
        covered = inv_t
    if top_p_tail:
        lse = torch.empty(rows, **f32)
        top1 = torch.empty(rows, dtype=torch.int32, device=dev)
        cand_tok = torch.empty(rows * _TAIL_DRAWS, dtype=torch.int32, device=dev)
        cand_y = torch.empty(rows * _TAIL_DRAWS, **f32)
    else:
        lse = top1 = cand_tok = cand_y = inv_t
    if out_topk:
        cut_p = torch.empty((rows, kp), **f32)
        cut_idx = torch.empty((rows, kp), dtype=torch.int32, device=dev)
    else:
        cut_p = cut_idx = inv_t
    _stats_kernel[(rows, n_chunks)](
        logits, logits.stride(0), inv_t, chunk_max, chunk_sum, group_max, stride_g,
        vocab, n_chunks, TOPK=topk, CHUNK=_CHUNK, GS=_GS, num_warps=8,
    )
    _sample_kernel[(rows,)](
        logits, logits.stride(0), chunk_max, chunk_sum, group_max, stride_g, inv_t,
        top_ks if topk else inv_t, top_ps if topk else inv_t,
        min_ps if min_ps is not None else inv_t,
        out, lse, covered, top1, cand_tok, cand_y, cut_p, cut_idx,
        random.getrandbits(31), vocab, n_chunks, n_groups,
        TOPK=topk, TAIL=top_p_tail, OUT_TOPK=out_topk, M=_TAIL_DRAWS, K=kk, KP=kp,
        NCP=ncp, NSP=nsp, CHUNK=_CHUNK, GS=_GS,
        USE_MIN_P=min_ps is not None, num_warps=4 if topk else 8,
    )
    if out_topk:
        return cut_p, cut_idx, covered
    if top_p_tail:
        partial = torch.empty(rows * n_chunks * _TAIL_DRAWS, **f32)
        _tail_mass_kernel[(rows, n_chunks)](
            logits, logits.stride(0), inv_t, lse, covered, cand_y, partial, vocab, n_chunks,
            M=_TAIL_DRAWS, CHUNK=_CHUNK, num_warps=8,
        )
        _tail_pick_kernel[(rows,)](
            covered, top_ps, top1, cand_tok, partial, out, n_chunks,
            M=_TAIL_DRAWS, NCP=ncp, num_warps=1,
        )
    return out


def sample_rows(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: Optional[torch.Tensor],
    top_ps: Optional[torch.Tensor],
    min_ps: Optional[torch.Tensor],
    k: Optional[int],
    top_p_tail: bool = False,
) -> torch.Tensor:
    """Sample one token per row of ``logits`` (rows, vocab), int64.

    * ``k=None``: temperature only.
    * ``k`` (<= MAX_TOP_K): the largest per-row top_k; top_ks (int32) / top_ps (/ min_ps) give each
      row's cut.
    * ``top_p_tail=True`` (with ``k=MAX_TOP_K``): rows with an unbounded top_k and a top_p cut, no min_p.
      Rows whose top-p support exceeds the top MAX_TOP_K tokens are drawn by rejection from the full
      softmax (M draws; with probability <= (1 - top_p) ** M all miss and the top-1 token is used).
    """
    return _launch(logits, temperatures, top_ks, top_ps, min_ps, k, top_p_tail=top_p_tail)


def topk_cut_probs(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: Optional[torch.Tensor],
    k: int,
):
    """The joint top-k / top-p (/ min-p) cut of softmax(logits / T) as a sparse distribution.

    Returns (p, idx, covered): p and idx are (rows, next_pow2(k)) -- probabilities of the kept tokens in
    descending order (unnormalised: they keep their full-softmax values; dropped / padded slots are 0)
    and token ids (int32); covered (rows,) int32 is 1 where the whole cut support lies inside the top k
    (top_k <= k, or the top k already carry more than top_p of the mass), i.e. where (p, idx) is the
    exact cut distribution. k <= MAX_TOP_K. Uses no RNG, so it is CUDA-graph safe.
    """
    return _launch(logits, temperatures, top_ks, top_ps, min_ps, k, out_topk=True)


__all__ = ["MAX_TOP_K", "sample_rows", "topk_cut_probs"]
