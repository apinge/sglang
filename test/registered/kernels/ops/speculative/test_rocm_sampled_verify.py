"""Sort-free sampling kernels and sparse chain verification used by ROCm EAGLE verify.

* topk_cut_probs must reproduce the joint top-k / top-p cut of softmax(logits / T) that the
  sort-based sampler (top_k_top_p_min_p_sampling_from_probs_torch) applies, including when the
  strongest tokens sit in one group or straddle a vocabulary-chunk edge, and must flag rows whose
  cut support does not fit in k.
* sample_rows must never return a token outside the cut support.
* sparse_chain_verify must follow the speculative-sampling rules exactly for hand-built cases.
* hip_draft_proposal + verify_sampled_chain must commit tokens distributed as the target cut
  distribution (fixed seed, so the check is deterministic).
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.sampling.topk_sampling_triton import sample_rows, topk_cut_probs
from sglang.kernels.ops.speculative.sparse_chain_verify_triton import sparse_chain_verify
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, stage="jit-kernel-unit", runner_config="amd")

DEV = "cuda"
VOCAB = 248320


def _reference_cut(logits, temperature, top_k, top_p):
    """float64 cut of softmax(logits / T) with the sort-based sampler's semantics (unnormalised)."""
    probs = torch.softmax(logits.double() / temperature, dim=-1)
    s, i = probs.sort(descending=True)
    keep = torch.arange(s.numel(), device=s.device) < top_k
    keep &= ~((s.cumsum(0) - s) > top_p)
    out = torch.zeros_like(probs)
    out[i[keep]] = s[keep]
    return out


def _logits(seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(VOCAB, device=DEV, generator=g) * 0.5
    x[8180:8230] += torch.linspace(9, 7, 50, device=DEV)  # strong tokens across the 8192 chunk edge
    x[100000:100005] += 8.5  # and a few in one group far away
    return x


class TestTopkCutProbs(CustomTestCase):
    def test_matches_sort_reference(self):
        for seed, temperature, top_k, top_p in [
            (0, 1.0, 20, 0.95),
            (1, 0.7, 64, 1.0),
            (2, 1.3, 7, 0.5),
            (3, 1.0, TOP_K_ALL, 0.3),
        ]:
            logits = _logits(seed)
            rows = 3
            p, idx, covered = topk_cut_probs(
                logits.expand(rows, VOCAB).contiguous(),
                torch.full((rows, 1), temperature, device=DEV),
                torch.full((rows,), top_k, dtype=torch.int32, device=DEV),
                torch.full((rows,), top_p, device=DEV),
                None,
                64,
            )
            ref = _reference_cut(logits, temperature, top_k, top_p)
            self.assertTrue(bool(covered.all()))
            for r in range(rows):
                got = torch.zeros(VOCAB, dtype=torch.float64, device=DEV)
                got[idx[r].long()] += p[r].double()
                torch.testing.assert_close(got, ref, atol=1e-6, rtol=1e-4)

    def test_wide_top_p_is_not_covered(self):
        logits = torch.zeros(1, VOCAB, device=DEV)  # uniform: top-p 0.9 needs ~223K tokens
        _, _, covered = topk_cut_probs(
            logits,
            torch.ones(1, 1, device=DEV),
            torch.full((1,), TOP_K_ALL, dtype=torch.int32, device=DEV),
            torch.full((1,), 0.9, device=DEV),
            None,
            64,
        )
        self.assertEqual(covered.item(), 0)


class TestSampleRowsSupport(CustomTestCase):
    def test_samples_stay_in_cut_support(self):
        torch.manual_seed(0)
        logits = _logits(4)
        rows = 512
        for top_k, top_p, tail in [(20, 0.95, False), (1, 1.0, False), (TOP_K_ALL, 0.9, True)]:
            ref = _reference_cut(logits, 1.0, top_k, top_p)
            for _ in range(4):
                ids = sample_rows(
                    logits.expand(rows, VOCAB).contiguous(),
                    torch.ones(rows, 1, device=DEV),
                    torch.full((rows,), top_k, dtype=torch.int32, device=DEV),
                    torch.full((rows,), top_p, device=DEV),
                    None,
                    64 if tail else top_k,
                    top_p_tail=tail,
                )
                self.assertTrue(bool((ref[ids] > 0).all()), (top_k, top_p))


def _pack_q(probs_ids):
    """[(token, prob), ...] per step -> packed (1, S, 2 * 32) proposal."""
    steps = []
    for entries in probs_ids:
        q = torch.zeros(32)
        ids = torch.zeros(32)
        for k, (tok, pr) in enumerate(entries):
            q[k], ids[k] = pr, tok
        steps.append(torch.cat([q, ids]))
    return torch.stack(steps).unsqueeze(0).to(DEV)


def _run_chain(cand, tp, ti, dq, mode, coins, coin_final, y=None):
    s1 = len(cand)
    predict = torch.full((s1,), -1, dtype=torch.int32, device=DEV)
    accept_index = torch.full((1, s1), -1, dtype=torch.int32, device=DEV)
    accept_num = torch.empty(1, dtype=torch.int32, device=DEV)
    sparse_chain_verify(
        torch.tensor([cand], device=DEV),
        torch.arange(s1, device=DEV).view(1, s1),
        torch.tensor(tp, dtype=torch.float32, device=DEV),
        torch.tensor(ti, dtype=torch.int32, device=DEV),
        dq,
        torch.tensor(y if y is not None else [0] * s1, device=DEV),
        torch.tensor(mode, dtype=torch.int32, device=DEV),
        torch.tensor([coins], dtype=torch.float32, device=DEV),
        torch.tensor([coin_final], dtype=torch.float32, device=DEV),
        predict,
        accept_index,
        accept_num,
    )
    n = accept_num.item()
    return n, [predict[accept_index[0, j]].item() for j in range(n + 1)]


class TestSparseChainVerify(CustomTestCase):
    # target rows: p over tokens (10, 11, 12) with probabilities (0.5, 0.3, 0.2)
    TP = [[0.5, 0.3, 0.2, 0.0]] * 3
    TI = [[10, 11, 12, 0]] * 3

    def test_accepts_when_coin_below_ratio_then_samples_bonus(self):
        dq = _pack_q([[(10, 1.0)], [(11, 1.0)]])
        # p(10)=0.5 > 0.4*1, p(11)=0.3 > 0.2*1: both accepted; bonus coin 0.9 -> token 12
        n, toks = _run_chain([7, 10, 11], self.TP, self.TI, dq, [1, 1, 1], [0.4, 0.2], 0.9)
        self.assertEqual((n, toks), (2, [10, 11, 12]))

    def test_rejection_samples_residual(self):
        # q = (10: 0.8, 11: 0.2): residual max(p - q, 0) = (0, 0.1, 0.2) -> coin 0.2 lands on 11
        dq = _pack_q([[(10, 0.8), (11, 0.2)], [(11, 1.0)]])
        n, toks = _run_chain([7, 10, 11], self.TP, self.TI, dq, [1, 1, 1], [0.9, 0.0], 0.2)
        self.assertEqual((n, toks), (0, [11]))
        n, toks = _run_chain([7, 10, 11], self.TP, self.TI, dq, [1, 1, 1], [0.9, 0.0], 0.5)
        self.assertEqual((n, toks), (0, [12]))

    def test_out_of_support_draft_is_rejected(self):
        dq = _pack_q([[(99, 1.0)], [(11, 1.0)]])
        n, toks = _run_chain([7, 99, 11], self.TP, self.TI, dq, [1, 1, 1], [0.0, 0.0], 0.0)
        self.assertEqual((n, toks), (0, [10]))

    def test_sample_matching_rows(self):
        dq = _pack_q([[(10, 1.0)], [(11, 1.0)]])
        # mode 0 everywhere: accept iff draft == y; commit y at the stopping row
        n, toks = _run_chain([7, 10, 11], self.TP, self.TI, dq, [0, 0, 0], [0.0, 0.0], 0.0, y=[10, 12, 5])
        self.assertEqual((n, toks), (1, [10, 12]))


class TestSampledChainDistribution(CustomTestCase):
    def test_committed_tokens_follow_target(self):
        from sglang.srt.speculative.hip_verify_sampling import (
            hip_draft_proposal,
            verify_sampled_chain,
        )

        torch.manual_seed(0)
        vocab, steps, bs, k, top_p = 4096, 2, 4096, 20, 0.95
        tgt = [torch.randn(vocab, device=DEV) * 2.5 for _ in range(steps + 1)]
        drf = [t + torch.randn(vocab, device=DEV) for t in tgt]
        info = SimpleNamespace(
            temperatures=torch.ones(bs, 1, device=DEV),
            top_ks=torch.full((bs,), k, dtype=torch.int32, device=DEV),
            top_ps=torch.full((bs,), top_p, device=DEV),
            min_ps=torch.zeros(bs, device=DEV),
            sampling_seed=None,
            need_min_p_sampling=False,
        )
        reqs = [SimpleNamespace(sampling_params=SimpleNamespace(top_k=k, top_p=top_p, min_p=0.0))] * bs
        counts = torch.zeros(steps + 1, vocab, dtype=torch.float64, device=DEV)
        for _ in range(20):
            enc0, _, x1 = hip_draft_proposal(drf[0].expand(bs, vocab).contiguous(), info)
            enc1, _, x2 = hip_draft_proposal(drf[1].expand(bs, vocab).contiguous(), info)
            cand = torch.cat([torch.zeros(bs, 1, dtype=torch.int64, device=DEV), x1, x2], 1)
            predict = torch.zeros(bs * (steps + 1), dtype=torch.int32, device=DEV)
            accept_index = torch.full((bs, steps + 1), -1, dtype=torch.int32, device=DEV)
            accept_num = torch.empty(bs, dtype=torch.int32, device=DEV)
            verify_sampled_chain(
                torch.stack(tgt, 0).repeat(bs, 1), info, reqs,
                torch.zeros(bs, dtype=torch.int64, device=DEV), cand,
                torch.arange(bs * (steps + 1), device=DEV).view(bs, steps + 1),
                torch.stack([enc0, enc1], 1), predict, accept_index, accept_num,
            )
            for j in range(steps + 1):
                toks = predict[accept_index[accept_num >= j, j].long()].long()
                counts[j] += torch.bincount(toks, minlength=vocab).double()
        for j in range(steps + 1):
            ref = _reference_cut(tgt[j], 1.0, k, top_p)
            ref = ref / ref.sum()
            n = counts[j].sum()
            self.assertEqual(counts[j][ref == 0].sum().item(), 0)
            sup = ref > 0
            chi2 = ((counts[j][sup] - n * ref[sup]) ** 2 / (n * ref[sup])).sum().item()
            dof = int(sup.sum()) - 1
            # mean dof, sd sqrt(2 dof); 6 sd keeps a fixed-seed run far from flaky while
            # still catching a biased acceptance rule (which shows up as chi2/dof >> 1).
            self.assertLess(chi2, dof + 6 * (2 * dof) ** 0.5, (j, chi2, dof))


class TestDraftProposalGraphParameters(CustomTestCase):
    def test_replay_uses_live_top_k_and_top_p_for_mixed_rows(self):
        from sglang.srt.speculative.hip_verify_sampling import (
            DRAFT_Q_K,
            hip_draft_proposal,
        )

        logits = torch.stack([_logits(5), _logits(6)])
        info = SimpleNamespace(
            temperatures=torch.tensor([[1.0], [0.7]], device=DEV),
            top_ks=torch.full((2,), 20, dtype=torch.int32, device=DEV),
            top_ps=torch.ones(2, device=DEV),
        )
        with patch.dict("os.environ", {"SGLANG_ROCM_SAMPLED_DRAFT": "1"}):
            for _ in range(2):
                hip_draft_proposal(logits, info)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                packed, _, tokens = hip_draft_proposal(logits, info)

            # SamplingParams represents temperature=0 as T=1, top_k=1.
            # Replay must observe both changed request buffers without recapture.
            for ks, ps in [([1, 20], [1.0, 0.5]), ([20, 1], [0.6, 1.0])]:
                info.top_ks.copy_(torch.tensor(ks, device=DEV, dtype=torch.int32))
                info.top_ps.copy_(torch.tensor(ps, device=DEV))
                graph.replay()
                torch.cuda.synchronize()
                for row, temperature in enumerate([1.0, 0.7]):
                    ref = _reference_cut(logits[row], temperature, ks[row], ps[row])
                    ref = ref / ref.sum()
                    actual = torch.zeros(VOCAB, device=DEV, dtype=torch.float64)
                    actual.scatter_add_(
                        0, packed[row, DRAFT_Q_K:].long(), packed[row, :DRAFT_Q_K].double()
                    )
                    torch.testing.assert_close(actual, ref, atol=1e-6, rtol=1e-4)
                    self.assertGreater(ref[tokens[row, 0]].item(), 0)
                    if ks[row] == 1:
                        self.assertEqual(tokens[row, 0].item(), logits[row].argmax().item())


if __name__ == "__main__":
    unittest.main()
