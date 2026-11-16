"""Chain (topk=1) speculative verification with sparse target / draft distributions (Triton).

Per request b with S draft tokens x_1..x_S (x_j drawn from the packed proposal q_j) and target
rows 0..S, each row j is verified by one of two exact rules, chosen per row by ``mode``:
  * mode 1 (the row's cut target distribution p_j is fully given by the sparse (p, idx)): accept
    x_j iff u_j * q_j(x_j) < p_j(x_j) (standard speculative sampling); on rejection commit a
    sample of the residual max(p_j - q_j, 0); after the last draft commit a sample of p_S.
  * mode 0: accept x_j iff x_j == y_j, where y_j ~ p_j was drawn by the caller; on mismatch (or
    at the bonus row) commit y_j. Exact for any proposal, used for rows without a sparse p.

p_j is given as the cut distribution of row j: (p, idx) of width KT, unnormalised (descending,
zeros for dropped slots). q_j is packed per step as [q probs (KQ) | token ids as float (KQ)].
Outputs follow verify_tree_greedy_func: predict (flat, indexed by retrieve_index), accept_index
(bs, S + 1), accept_token_num (bs,).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_chain_verify_kernel(
    candidates_ptr,  # (bs, S + 1) int64
    retrieve_index_ptr,  # (bs, S + 1) int64 / int32
    tp_ptr,  # (bs * (S + 1), KT) f32
    ti_ptr,  # (bs * (S + 1), KT) i32
    dq_ptr,  # (bs, S, 2 * KQ) f32
    y_ptr,  # (bs * (S + 1),) i64
    mode_ptr,  # (bs * (S + 1),) i32
    coins_ptr,  # (bs, S) f32
    coin_final_ptr,  # (bs,) f32
    predict_ptr,  # flat i32
    accept_index_ptr,  # (bs, S + 1) i32
    accept_num_ptr,  # (bs,) i32
    S: tl.constexpr,
    KT: tl.constexpr,
    KQ: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.arange(0, KT)
    qk = tl.arange(0, KQ)
    last = tl.load(retrieve_index_ptr + b * (S + 1)).to(tl.int32)
    tl.store(accept_index_ptr + b * (S + 1), last)
    n = 0
    going = 1
    for j in tl.static_range(S):
        if going != 0:
            row = b * (S + 1) + j
            x = tl.load(candidates_ptr + b * (S + 1) + j + 1)
            if tl.load(mode_ptr + row) != 0:
                tp = tl.load(tp_ptr + row * KT + t)
                ti = tl.load(ti_ptr + row * KT + t)
                px = tl.sum(tl.where(ti == x, tp, 0.0), axis=0) / tl.sum(tp, axis=0)
                qp = tl.load(dq_ptr + (b * S + j) * 2 * KQ + qk)
                qi = tl.load(dq_ptr + (b * S + j) * 2 * KQ + KQ + qk).to(tl.int64)
                qx = tl.sum(tl.where((qi == x) & (qp > 0.0), qp, 0.0), axis=0)
                ok = tl.load(coins_ptr + b * S + j) * qx < px
            else:
                ok = tl.load(y_ptr + row) == x
            if ok:
                tl.store(predict_ptr + last, x.to(tl.int32))
                n += 1
                last = tl.load(retrieve_index_ptr + b * (S + 1) + j + 1).to(tl.int32)
                tl.store(accept_index_ptr + b * (S + 1) + n, last)
            else:
                going = 0
    tl.store(accept_num_ptr + b, n)

    row = b * (S + 1) + n
    if tl.load(mode_ptr + row) != 0:
        tp = tl.load(tp_ptr + row * KT + t)
        ti = tl.load(ti_ptr + row * KT + t)
        p = tp / tl.sum(tp, axis=0)
        if n < S:
            # residual max(p - q_n, 0) on the support of p (it is 0 elsewhere since p is)
            qp = tl.load(dq_ptr + (b * S + n) * 2 * KQ + qk)
            qi = tl.load(dq_ptr + (b * S + n) * 2 * KQ + KQ + qk).to(tl.int32)
            q_at = tl.sum(
                tl.where((ti[:, None] == qi[None, :]) & (qp[None, :] > 0.0), qp[None, :], 0.0), axis=1
            )
            r = tl.maximum(p - q_at, 0.0)
            # p == q on the whole support (residual mass ~0) cannot reject in exact arithmetic;
            # fall back to p so a float-rounding rejection still commits a valid token.
            r = tl.where(tl.sum(r, axis=0) > 1.0e-6, r, p)
        else:
            r = p
        # inverse CDF: the first slot whose cumulative mass exceeds u has r > 0; a u that
        # rounds past the total is clamped to the last positive slot.
        cum = tl.cumsum(r, axis=0)
        u = tl.load(coin_final_ptr + b) * tl.sum(r, axis=0)
        k = tl.sum((cum <= u).to(tl.int32), axis=0)
        k = tl.minimum(k, tl.max(tl.where(r > 0.0, t, 0), axis=0))
        tok = tl.sum(tl.where(t == k, ti, 0), axis=0)
    else:
        tok = tl.load(y_ptr + row).to(tl.int32)
    tl.store(predict_ptr + last, tok)


def sparse_chain_verify(
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    target_p: torch.Tensor,
    target_idx: torch.Tensor,
    draft_q: torch.Tensor,
    y: torch.Tensor,
    mode: torch.Tensor,
    coins: torch.Tensor,
    coin_final: torch.Tensor,
    predict: torch.Tensor,
    accept_index: torch.Tensor,
    accept_num: torch.Tensor,
) -> None:
    bs, s1 = candidates.shape
    kq = draft_q.shape[-1] // 2
    _sparse_chain_verify_kernel[(bs,)](
        candidates, retrieve_index, target_p, target_idx, draft_q, y, mode, coins, coin_final,
        predict, accept_index, accept_num,
        S=s1 - 1, KT=target_p.shape[-1], KQ=kq, num_warps=1,
    )


__all__ = ["sparse_chain_verify"]
