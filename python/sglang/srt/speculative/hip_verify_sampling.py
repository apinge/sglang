"""Stochastic EAGLE verification for ROCm without full-vocabulary sorts.

ROCm's sgl_kernel has no tree speculative-sampling or top-k/top-p renorm ops, so
the target-verify step used to commit argmax regardless of the request's sampling
parameters. This module restores sampling with the target-sampling form of
speculative verification:

  * draw one token y_i per verify row from the request's sampling distribution
    (temperature, then joint top-k / top-p, then min-p: the semantics of
    ``top_k_top_p_min_p_sampling_from_probs_torch``, the sampler ROCm uses
    outside speculative decoding);
  * run the existing greedy tree match with y in place of the argmax.

For deterministic draft tokens this is exact: a draft x is accepted with
probability p(x); on a mismatch the committed token is y conditioned on y != x,
i.e. the rejection-sampling residual (p - onehot(x))+ renormalised; the bonus
row is a plain sample from p. Sibling drafts of a tree are distinct tokens, so
the same argument holds node by node.

Row classes (decided on the host from ``reqs``, so dispatch adds no device sync):
  0. top_k <= MAX_TOP_K: fused Triton chunked top-k + log-sum-exp, cut and draw;
     MAX_TOP_K < top_k <= _MAX_TORCH_TOP_K: the same math with torch.topk.
  1. temperature only (no top-k / top-p / min-p cut): Triton two-stage inverse CDF.
  3. top-p only (unbounded top-k, no min-p): the class-0 kernel with K = MAX_TOP_K; rows
     whose top-p support is wider than that are drawn by rejection from the full softmax.
  2. anything else (e.g. min-p with unbounded top-k): the sort-based reference
     sampler, on those rows only.
Seeded (deterministic-inference) batches always use the reference sampler.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from sglang.kernels.ops.sampling.topk_sampling_triton import (
    MAX_TOP_K,
    sample_rows,
    topk_cut_probs,
)
from sglang.kernels.ops.speculative.sparse_chain_verify_triton import sparse_chain_verify
from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_probs_torch
from sglang.srt.sampling.sampling_params import TOP_K_ALL

_MAX_TORCH_TOP_K = 1024


def _sample_top_k_torch(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: Optional[torch.Tensor],
    k: int,
) -> torch.Tensor:
    """Rows with top_k <= k: the joint top-k/top-p/min-p support lies in the top k logits."""
    vals, idx = torch.topk(logits, k, dim=-1)  # T > 0 keeps the order of logits / T
    lse = torch.logsumexp(logits / temperatures, dim=-1, keepdim=True)
    probs = torch.exp(vals / temperatures - lse)  # the top-k entries of softmax(logits / T)
    rank = torch.arange(k, device=logits.device)
    drop = rank.unsqueeze(0) >= top_ks.unsqueeze(1)
    # Joint top-p: the cumulative mass is taken before the top-k cut, as the reference does.
    drop |= (probs.cumsum(dim=-1) - probs) > top_ps.unsqueeze(1)
    probs = probs.masked_fill(drop, 0.0)
    if min_ps is not None:
        probs.masked_fill_(probs < probs[:, :1] * min_ps.unsqueeze(1), 0.0)
    # Exponential race: argmax p_i / E_i with E_i ~ Exp(1) draws i with prob p_i / sum p.
    # exponential_ can return 0; keep the race finite so a dropped (p=0) entry never wins.
    race = torch.empty_like(probs).exponential_(1.0).clamp_min_(torch.finfo(torch.float32).tiny)
    choice = (probs / race).argmax(dim=-1, keepdim=True)
    return idx.gather(1, choice).squeeze(1)


def _sample_reference(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_min_p: bool,
    sampling_seed: Optional[torch.Tensor] = None,
    positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    probs = torch.softmax(logits / temperatures, dim=-1)
    return top_k_top_p_min_p_sampling_from_probs_torch(
        probs, top_ks, top_ps, min_ps, need_min_p, sampling_seed, positions
    ).to(torch.int64)


def _row_kind(sp) -> int:
    if sp.top_k <= _MAX_TORCH_TOP_K:
        return 0
    if sp.top_k >= TOP_K_ALL and sp.min_p <= 0.0:
        return 1 if sp.top_p >= 1.0 else 3
    return 2


def _sample_kind(kind, logits, temperatures, top_ks, top_ps, min_ps, need_min_p, k):
    if kind == 0:
        cut_min_p = min_ps if need_min_p else None
        if k <= MAX_TOP_K:
            return sample_rows(logits, temperatures, top_ks, top_ps, cut_min_p, k)
        return _sample_top_k_torch(logits, temperatures, top_ks, top_ps, cut_min_p, k)
    if kind == 1:
        return sample_rows(logits, temperatures, None, None, None, None)
    if kind == 3:
        return sample_rows(logits, temperatures, top_ks, top_ps, None, MAX_TOP_K, top_p_tail=True)
    return _sample_reference(logits, temperatures, top_ks, top_ps, min_ps, need_min_p)


def sample_verify_targets(
    next_token_logits: torch.Tensor,
    sampling_info,
    reqs: List,
    rows_per_req: int,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    """One sampled target token per verify row, shape (bs * rows_per_req,), int64."""

    def rep(t):
        return torch.repeat_interleave(t, rows_per_req, dim=0)

    temperatures = rep(sampling_info.temperatures)  # (rows, 1)
    top_ks = rep(sampling_info.top_ks)
    top_ps = rep(sampling_info.top_ps)
    min_ps = rep(sampling_info.min_ps)
    need_min_p = sampling_info.need_min_p_sampling

    if sampling_info.sampling_seed is not None:
        # Deterministic inference: seeded per (request, position) like the plain sampler.
        positions = rep(seq_lens) + torch.arange(
            rows_per_req, device=seq_lens.device
        ).repeat(len(reqs))
        return _sample_reference(
            next_token_logits, temperatures, top_ks, top_ps, min_ps, need_min_p,
            rep(sampling_info.sampling_seed), positions,
        )

    kinds = [_row_kind(r.sampling_params) for r in reqs]
    present = set(kinds)
    if len(present) == 1:
        kind = kinds[0]
        k = max(r.sampling_params.top_k for r in reqs) if kind == 0 else None
        return _sample_kind(
            kind, next_token_logits, temperatures, top_ks, top_ps, min_ps, need_min_p, k
        )

    # Mixed batch: sample each class on its own rows.
    out = torch.empty(
        next_token_logits.shape[0], dtype=torch.int64, device=next_token_logits.device
    )
    for kind in present:
        req_ids = [i for i, kk in enumerate(kinds) if kk == kind]
        rows = torch.tensor(
            [i * rows_per_req + j for i in req_ids for j in range(rows_per_req)],
            dtype=torch.int64,
        ).to(next_token_logits.device, non_blocking=True)
        k = max(reqs[i].sampling_params.top_k for i in req_ids) if kind == 0 else None
        out[rows] = _sample_kind(
            kind, next_token_logits[rows], temperatures[rows], top_ks[rows],
            top_ps[rows], min_ps[rows], need_min_p, k,
        )
    return out


# ---------------------------------------------------------------------------------------------
# Sampled drafts (speculative_use_rejection_sampling on ROCm)
#
# Requests with a top-k (2 <= top_k <= DRAFT_Q_K) or a top-p-only cut draw their draft tokens from
# the draft model's own cut distribution q (same temperature / top-k / top-p, at most DRAFT_Q_K
# tokens), so verification can accept with min(1, p/q) instead of only p(x_argmax). Every other
# request (greedy, temperature-only, larger top-k) keeps a greedy draft (q one-hot). q is kept
# sparse: per draft step and row, [q probs (DRAFT_Q_K) | token ids as float (DRAFT_Q_K)].
#
# Verification picks an exact rule per target row (sparse_chain_verify): standard speculative
# sampling where the cut target distribution fits in the top TARGET_SPARSE_K tokens, and
# "commit a target sample y, accept the draft iff it equals y" (exact for any proposal) elsewhere.
# ---------------------------------------------------------------------------------------------
DRAFT_Q_K = 32
DRAFT_Q_WIDTH = 2 * DRAFT_Q_K
TARGET_SPARSE_K = MAX_TOP_K


def _draft_k(top_ks: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    """Per-row proposal width (device; CUDA-graph safe): the request top_k, DRAFT_Q_K for a top-p-only
    cut, 1 (greedy) otherwise."""
    topk_req = (top_ks >= 2) & (top_ks <= DRAFT_Q_K)
    topp_only = (top_ks >= TOP_K_ALL) & (top_ps < 1.0)
    return torch.where(
        topk_req, top_ks, torch.where(topp_only, torch.full_like(top_ks, DRAFT_Q_K), 1)
    )


def _sparse_target_req(sp) -> bool:
    """Requests whose target rows may use speculative sampling on a sparse cut distribution."""
    if sp.min_p > 0.0:
        return False
    return 2 <= sp.top_k <= DRAFT_Q_K or (sp.top_k >= TOP_K_ALL and sp.top_p < 1.0)


def hip_draft_proposal(logits: torch.Tensor, sampling_info):
    """Draft proposal for one draft step: (packed q (bs, DRAFT_Q_WIDTH), q(x) (bs, 1), x (bs, 1)).

    Graph-safe: no host synchronisation, fixed shapes, RNG through torch's exponential_.
    """
    k_eff = _draft_k(sampling_info.top_ks, sampling_info.top_ps)
    p, idx, _ = topk_cut_probs(
        logits, sampling_info.temperatures, k_eff, sampling_info.top_ps, None, DRAFT_Q_K
    )
    q = p / p.sum(dim=-1, keepdim=True)
    race = torch.empty_like(q).exponential_(1.0).clamp_min_(torch.finfo(torch.float32).tiny)
    choice = (q / race).argmax(dim=-1, keepdim=True)
    x = idx.gather(1, choice).to(torch.int64)
    return torch.cat([q, idx.to(torch.float32)], dim=1), q.gather(1, choice), x


def verify_sampled_chain(
    next_token_logits: torch.Tensor,
    sampling_info,
    reqs: List,
    seq_lens: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    draft_q: torch.Tensor,
    predict: torch.Tensor,
    accept_index: torch.Tensor,
    accept_num: torch.Tensor,
) -> None:
    """Chain verification for a batch whose drafts came from hip_draft_proposal (topk == 1)."""
    bs, s1 = candidates.shape
    dev = next_token_logits.device
    sparse_req = [_sparse_target_req(r.sampling_params) for r in reqs]

    if any(sparse_req):
        bounded = [r.sampling_params.top_k for r, f in zip(reqs, sparse_req) if f]
        k = TARGET_SPARSE_K if max(bounded) >= TOP_K_ALL else max(bounded)
        sel = torch.tensor(sparse_req, dtype=torch.bool).to(dev, non_blocking=True)
        top_ks = torch.where(sel, sampling_info.top_ks, 1).to(torch.int32)
        target_p, target_idx, covered = topk_cut_probs(
            next_token_logits,
            torch.repeat_interleave(sampling_info.temperatures, s1, dim=0),
            torch.repeat_interleave(top_ks, s1, dim=0),
            torch.repeat_interleave(sampling_info.top_ps, s1, dim=0),
            None,
            k,
        )
        mode = covered * torch.repeat_interleave(sel.to(torch.int32), s1, dim=0)
    else:
        target_p = torch.zeros((1, 1), dtype=torch.float32, device=dev)
        target_idx = torch.zeros((1, 1), dtype=torch.int32, device=dev)
        mode = torch.zeros(bs * s1, dtype=torch.int32, device=dev)
    # Target samples for the rows verified by sample matching (everything outside the sparse rule,
    # including top-p rows whose support is wider than TARGET_SPARSE_K).
    if all(2 <= r.sampling_params.top_k <= DRAFT_Q_K and r.sampling_params.min_p <= 0.0 for r in reqs):
        y = torch.zeros(1, dtype=torch.int64, device=dev)
    else:
        y = sample_verify_targets(next_token_logits, sampling_info, reqs, s1, seq_lens)

    coins = torch.rand((bs, s1 - 1), dtype=torch.float32, device=dev)
    coin_final = torch.rand((bs,), dtype=torch.float32, device=dev)
    sparse_chain_verify(
        candidates, retrieve_index, target_p, target_idx, draft_q.contiguous(), y, mode,
        coins, coin_final, predict, accept_index, accept_num,
    )
