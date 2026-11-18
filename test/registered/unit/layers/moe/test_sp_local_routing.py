"""MoE routing on sequence-parallel local rows (sglang.srt.layers.moe.sp_local_routing):
the routing metadata round trip and the gate that keeps it equivalent to the block's forward."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.moe import sp_local_routing as lr
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKOutputFormat
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

TOPK, SHARED = 10, 1


class _MoeBlock:
    """The attributes of a Qwen2MoeSparseMoeBlock that sp_local_routing reads."""

    num_fused_shared_experts = SHARED
    shared_expert = None

    def __init__(self, ids, weights):
        self.topk = mock.Mock(
            return_value=StandardTopKOutput(ids[:, :TOPK], weights[:, :TOPK], None)
        )
        self.topk.topk_config = SimpleNamespace(top_k=TOPK, output_format=None)
        self._ids, self._weights = ids, weights
        self.experts = mock.Mock(return_value="experts output")

    def gate(self, rows):
        return None, None

    def _append_shared_to_topk_output(self, topk_output, rows):
        return StandardTopKOutput(self._weights, self._ids, None)


class TestSpLocalRouting(CustomTestCase):
    def test_metadata_round_trip_is_lossless(self):
        """Ids travel bitcast through f32; the experts must get the router's exact ids and
        weights, including ids whose bit patterns are f32 NaN / inf / denormals."""
        n = 37
        ids = torch.randint(0, 513, (n, TOPK + SHARED), dtype=torch.int32)
        ids[0, :4] = torch.tensor([-1, 0x7FC00000, 0x7F800000, 1], dtype=torch.int32)
        weights = torch.rand(n, TOPK + SHARED, dtype=torch.float32)
        block = _MoeBlock(ids, weights)
        rows = torch.zeros(n, 8)

        meta = lr.route_rows(block, rows)
        self.assertEqual(meta.shape, (n, lr.meta_width(block)))
        self.assertEqual(meta.shape[1] * 4 % 32, 0)
        self.assertEqual(lr.run_routed_experts(block, rows, meta), "experts output")

        (x, topk_output), _ = block.experts.call_args
        self.assertIs(x, rows)
        self.assertTrue(torch.equal(topk_output.topk_ids, ids))
        self.assertTrue(torch.equal(topk_output.topk_weights, weights))

    def test_gate_rejects_non_equivalent_blocks(self):
        """Local routing replaces the block's forward, so it must refuse blocks whose forward
        does more than router + experts or whose top-k output is not the standard one."""
        backend = SimpleNamespace(
            is_auto=lambda: True, is_aiter=lambda: False, is_triton=lambda: False
        )
        a2a = SimpleNamespace(is_none=lambda: True)
        block = _MoeBlock(
            torch.zeros(1, TOPK + SHARED, dtype=torch.int32),
            torch.zeros(1, TOPK + SHARED),
        )
        with (
            mock.patch.object(lr, "get_moe_runner_backend", lambda: backend),
            mock.patch.object(lr, "get_moe_a2a_backend", lambda: a2a),
        ):
            self.assertTrue(lr.local_routing_supported(block))

            block.shared_expert = torch.nn.Module()  # separate shared-expert MLP
            self.assertFalse(lr.local_routing_supported(block))
            block.shared_expert = None

            block.topk.topk_config.output_format = TopKOutputFormat.BYPASSED
            self.assertFalse(lr.local_routing_supported(block))
            block.topk.topk_config.output_format = None

            a2a.is_none = lambda: False  # a2a dispatch (DeepEP, MoRI, ...)
            self.assertFalse(lr.local_routing_supported(block))


if __name__ == "__main__":
    unittest.main()
