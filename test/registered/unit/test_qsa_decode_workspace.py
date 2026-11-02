import math
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.qsa import mqa
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


def dense_reference(q, cache, table, lengths, max_len, scale=None):
    rows, _, dim = q.shape
    total = table.shape[1] * cache.shape[1]
    k = cache[table.long().clamp_min(0).reshape(-1), :, 0].reshape(rows, total, dim)
    scores = torch.einsum("bhd,bnd->bnh", q.float(), k.float())
    scores = scores.relu().sum(-1) / (scale or math.sqrt(dim))
    positions = torch.arange(total, device=q.device)
    scores.masked_fill_(positions[None, :] >= lengths[:, None], -float("inf"))
    out = torch.full((rows, max_len), -float("inf"), device=q.device)
    out[:, : min(total, max_len)] = scores[:, : min(total, max_len)]
    return out


class TestQsaDecodeWorkspace(CustomTestCase):
    def test_chunked_pages_match_reference_with_masks_and_partial_output(self):
        torch.manual_seed(42)
        q = torch.randn(3, 4, 128).bfloat16()
        cache = torch.randn(8, 16, 1, 128).bfloat16()
        table = torch.tensor([[2, 0, 7, 1, 3], [4, 1, 0, 6, 5], [7, 5, 2, 0, 1]])
        lengths = torch.tensor([0, 15, 77])
        # Force two-page chunks, including a last partial chunk/output.
        with patch.object(mqa, "_DECODE_FP32_KEY_BUDGET_BYTES", 3 * 128 * 4 * 32):
            for max_len in (47, 83):
                actual = mqa.torch_qsa_mqa_decode(q, cache, table, lengths, max_len, 3.25)
                expected = dense_reference(q, cache, table, lengths, max_len, 3.25)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_empty_capacity_and_empty_batch(self):
        cache = torch.zeros(1, 16, 1, 128)
        out = mqa.torch_qsa_mqa_decode(
            torch.ones(1, 4, 128), cache, torch.empty(1, 0, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32), 7,
        )
        self.assertTrue(torch.isneginf(out).all())
        out = mqa.torch_qsa_mqa_decode(
            torch.empty(0, 4, 128), cache, torch.empty(0, 1, dtype=torch.int32),
            torch.empty(0, dtype=torch.int32), 7,
        )
        self.assertEqual(out.shape, (0, 7))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/ROCm")
    def test_1m_capacity_graph_has_bounded_workspace_and_replays_new_inputs(self):
        device = torch.device("cuda", torch.cuda.current_device())
        rows, capacity, dim = 128, 262144, 128
        q = torch.randn(rows, 4, dim, device=device).bfloat16()
        cache = torch.randn(2, 16, 1, dim, device=device).bfloat16()
        table = (torch.arange(capacity // 16, device=device) % 2).int()
        table = table[None, :].expand(rows, -1).contiguous()
        lengths = torch.tensor([0, 1, 17, capacity], device=device).int().repeat(rows // 4)
        with torch.no_grad():
            warm = mqa.torch_qsa_mqa_decode(q, cache, table, lengths, capacity)
            torch.cuda.synchronize()
            del warm
            torch.cuda.empty_cache()
            start = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = mqa.torch_qsa_mqa_decode(q, cache, table, lengths, capacity)
            torch.cuda.synchronize()
            extra_peak = torch.cuda.max_memory_allocated() - start
            self.assertLess(extra_peak, 1024 * 1024 * 1024)

            def expected(period):
                small = torch.einsum("bhd,nd->bnh", q.float(), period.float())
                small = small.relu().sum(-1) / math.sqrt(dim)
                out = small.repeat(1, capacity // period.shape[0])
                pos = torch.arange(capacity, device=device)
                out.masked_fill_(pos[None, :] >= lengths[:, None], -float("inf"))
                return out

            graph.replay()
            torch.testing.assert_close(
                actual, expected(cache[:, :, 0].reshape(32, dim)), rtol=1e-5, atol=1e-5,
            )
            q.mul_(0.5)
            table.fill_(1)
            lengths.fill_(37)
            graph.replay()
            torch.testing.assert_close(
                actual, expected(cache[1, :, 0]), rtol=1e-5, atol=1e-5,
            )
            q.add_(0.25)
            cache.mul_(1.25)
            table.zero_()
            lengths.fill_(1)
            graph.replay()
            torch.testing.assert_close(
                actual, expected(cache[0, :, 0]), rtol=1e-5, atol=1e-5,
            )


if __name__ == "__main__":
    unittest.main()
