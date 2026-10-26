"""GDN ReplaySSM extra_buffer slots start empty, including recycled slots.

Regression coverage for upstream #37837, using real pool methods with CPU
index buffers. No model allocation is needed.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.configs import BailingHybridConfig, KimiLinearConfig
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _pool(*, lazy=False, enabled=True):
    pool = object.__new__(HybridReqToTokenPool)
    pool.enable_mamba_extra_buffer_lazy = lazy
    pool.mamba_ping_pong_track_buffer_size = 2
    pool.mamba_pool = SimpleNamespace(
        replayssm_write_pos=torch.full((10,), 7, dtype=torch.int32) if enabled else None
    )
    pool.mamba_allocator = SimpleNamespace(
        alloc=Mock(side_effect=lambda n: torch.tensor([3, 5][:n]))
    )
    pool.req_index_to_mamba_ping_pong_track_buffer_mapping = torch.full(
        (3, 2), -1, dtype=torch.int64
    )
    req = SimpleNamespace(req_pool_idx=1, rid="replay-test")
    pool._alloc_ping_pong_buffer(req)
    return pool, req


class TestReplaySSMPingPong(unittest.TestCase):
    def test_recycled_tracking_slots_have_empty_rings(self):
        for lazy in (False, True):
            with self.subTest(lazy=lazy):
                pool, req = _pool(lazy=lazy)
                expected = torch.full((10,), 7, dtype=torch.int32)
                expected[3] = 0
                if not lazy:
                    expected[5] = 0
                torch.testing.assert_close(
                    pool.mamba_pool.replayssm_write_pos, expected
                )
                self.assertEqual(
                    req.mamba_ping_pong_track_buffer.tolist(), [3, -1 if lazy else 5]
                )

    def test_replacement_and_clear_do_not_corrupt_other_cursors(self):
        for value in (8, torch.tensor(8)):
            with self.subTest(tensor=isinstance(value, torch.Tensor)):
                pool, req = _pool()
                pool.set_mamba_ping_pong_slot(req, 0, value)
                self.assertEqual(pool.mamba_pool.replayssm_write_pos[8].item(), 0)
                self.assertEqual(pool.mamba_pool.replayssm_write_pos[9].item(), 7)
                pool.set_mamba_ping_pong_slot(req, 0, -1)
                self.assertEqual(pool.mamba_pool.replayssm_write_pos[9].item(), 7)
                self.assertEqual(
                    pool.req_index_to_mamba_ping_pong_track_buffer_mapping[1].tolist(),
                    [-1, 5],
                )

    def test_device_style_slot_id_is_not_read_as_python_bool(self):
        pool, req = _pool(lazy=True)
        # A Tensor-to-bool test would synchronize for a real GPU index.
        with patch.object(
            torch.Tensor, "__bool__", side_effect=AssertionError("host read")
        ):
            pool.set_mamba_ping_pong_slot(req, 1, torch.tensor(8))
        self.assertEqual(pool.mamba_pool.replayssm_write_pos[8].item(), 0)

    def test_donation_preserves_snapshot_id_and_resets_replacement(self):
        pool, req = _pool()
        donated = pool.donate_mamba_ping_pong_slot(req, torch.tensor([8]))
        self.assertEqual(donated.tolist(), [5])
        self.assertEqual(req.mamba_ping_pong_track_buffer.tolist(), [3, 8])
        self.assertEqual(pool.mamba_pool.replayssm_write_pos[5].item(), 0)
        self.assertEqual(pool.mamba_pool.replayssm_write_pos[8].item(), 0)
        self.assertEqual(pool.mamba_pool.replayssm_write_pos[2].item(), 7)
        torch.testing.assert_close(
            pool.req_index_to_mamba_ping_pong_track_buffer_mapping[1],
            req.mamba_ping_pong_track_buffer,
        )

    def test_replayssm_off_keeps_existing_slot_operations(self):
        pool, req = _pool(enabled=False)
        donated = pool.donate_mamba_ping_pong_slot(req, torch.tensor([8]))
        self.assertEqual(donated.tolist(), [5])
        self.assertIsNone(pool.mamba_pool.replayssm_write_pos)


class TestReplaySSMDecodeGuards(unittest.TestCase):
    def _check(self, config=None, **kwargs):
        args = ServerArgs(
            model_path="unused",
            enable_linear_replayssm=True,
            linear_attn_decode_backend="triton",
            mamba_radix_cache_strategy="extra_buffer",
            **kwargs,
        )
        with (
            patch("sglang.srt.server_args.is_sm100_supported", return_value=False),
            patch.object(
                ServerArgs,
                "get_model_config",
                return_value=SimpleNamespace(hf_config=config),
            ),
        ):
            args._handle_linear_attn_backend()

    def test_gdn_allows_radix_off_and_extra_buffer(self):
        for disabled in (False, True):
            self._check(disable_radix_cache=disabled)

    def test_kda_extra_buffer_remains_rejected(self):
        for config in (KimiLinearConfig(), BailingHybridConfig(use_kda=True)):
            with self.subTest(config=type(config).__name__):
                with self.assertRaisesRegex(ValueError, "KDA model requires"):
                    self._check(config)
                self._check(config, disable_radix_cache=True)

    def test_replay_decode_does_not_enable_speculation(self):
        with self.assertRaisesRegex(ValueError, "non-speculative"):
            self._check(speculative_algorithm="NEXTN")

    def test_invalid_length_and_double_enable_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be >= 1"):
            self._check(linear_replayssm_cache_len=0)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            self._check(enable_linear_replayssm_spec=True)


if __name__ == "__main__":
    unittest.main()
