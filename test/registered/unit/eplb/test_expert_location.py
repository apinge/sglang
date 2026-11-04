"""Unit tests for ExpertLocationMetadata.logical_to_all_physical."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.eplb.expert_location import ExpertLocationMetadata
from sglang.test.test_utils import CustomTestCase


def _metadata(logical_to_all_physical_map):
    num_layers, _, num_physical = logical_to_all_physical_map.shape
    physical_to_logical_map = torch.zeros(num_layers, num_physical, dtype=torch.int64)
    return ExpertLocationMetadata(
        physical_to_logical_map=physical_to_logical_map,
        physical_to_logical_map_cpu=physical_to_logical_map,
        logical_to_all_physical_map=logical_to_all_physical_map,
        logical_to_all_physical_map_cpu=logical_to_all_physical_map,
        logical_to_all_physical_map_num_valid=(logical_to_all_physical_map != -1).sum(
            -1
        ),
        ep_size=1,
        logical_to_rank_dispatch_physical_map=None,
    )


class TestLogicalToAllPhysical(CustomTestCase):
    def test_follows_map_updates(self):
        cpu_map = torch.tensor([[[0, 2], [1, -1]], [[1, -1], [0, 2]]])
        metadata = _metadata(cpu_map)
        self.assertEqual(metadata.logical_to_all_physical(0, 0), [0, 2])
        self.assertEqual(metadata.logical_to_all_physical(0, 1), [1])
        self.assertEqual(metadata.logical_to_all_physical(1, 1), [0, 2])

        # update() writes the map in place.
        cpu_map[0, 1] = torch.tensor([2, 1])
        self.assertEqual(metadata.logical_to_all_physical(0, 1), [2, 1])

        metadata.logical_to_all_physical_map_cpu = torch.tensor(
            [[[2, -1], [0, 1]], [[1, -1], [0, 2]]]
        )
        self.assertEqual(metadata.logical_to_all_physical(0, 0), [2])


if __name__ == "__main__":
    unittest.main()
