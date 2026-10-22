"""gfx942 GDN prefill: TP2/TP4 shapes, indexed state, and radix reuse."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.srt.environ import envs
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.layers.attention.linear.kernels import gdn_aiter, gdn_triton
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, stage="stage-b", runner_config="1-gpu-large-amd")


@unittest.skipUnless(
    torch.version.hip is not None and torch.cuda.is_available(), "Requires ROCm"
)
class TestAiterGDNPrefill(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not gdn_aiter._is_gfx942(torch.device("cuda", 0)):
            raise unittest.SkipTest("Requires gfx942")
        if gdn_aiter._aiter_ops() is None:
            raise unittest.SkipTest("Requires AITER with FlyDSL K5 opt")

    def setUp(self):
        torch.manual_seed(42)
        self.kernel = TritonGDNKernel()

    def make_case(self, tp, lengths, zero_state=False):
        hg, h, t, n = 16 // tp, 48 // tp, sum(lengths), len(lengths)
        q = torch.randn(1, t, hg, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn(1, t, h, 128, device="cuda", dtype=torch.bfloat16)
        a = torch.randn(t, h, device="cuda", dtype=torch.bfloat16)
        b = torch.randn_like(a)
        A_log = torch.randn(h, device="cuda") * 0.1
        dt_bias = torch.randn(h, device="cuda", dtype=torch.bfloat16) * 0.1
        g, beta = fused_gdn_gating(A_log, a, b, dt_bias)
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        cu = torch.tensor(offsets, device="cuda", dtype=torch.int32)
        indices = torch.arange(1, 2 * n, 2, device="cuda", dtype=torch.int32)
        pool = torch.randn(2 * n + 2, h, 128, 128, device="cuda") * 0.1
        # V/K are equal in size: an asymmetric state catches silent transposes.
        row = torch.arange(128, device="cuda").float()
        pool += (row[:, None] - row[None, :] * 0.25) * 0.001
        if zero_state:
            pool[indices.long()] = 0
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            batch_size=n,
            _original_batch_size=None,
            input_ids=torch.zeros(t, device="cuda", dtype=torch.int64),
            extend_seq_lens_cpu=lengths,
        )
        md = ForwardMetadata(query_start_loc=cu, mamba_cache_indices=indices)
        meta = gdn_aiter.maybe_build_aiter_gdn_prefill_metadata(batch, md)
        self.assertIsNotNone(meta)
        args = dict(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            cache_indices=indices,
            query_start_loc=cu,
            aiter_gdn_prefill_metadata=meta,
        )
        return args, pool, batch, md, A_log, dt_bias

    def run_pair(self, args, pool):
        results = []
        for enabled in (False, True):
            state = pool.clone()
            with envs.SGLANG_USE_AITER_GDN_PREFILL.override(enabled):
                with patch.object(
                    gdn_aiter, "aiter_gdn_prefill", wraps=gdn_aiter.aiter_gdn_prefill
                ) as fly:
                    output, final, snapshots = self.kernel.extend(
                        **args, ssm_states=state
                    )
                    self.assertEqual(fly.call_count, int(enabled))
            self.assertIsNone(final)
            self.assertEqual(snapshots.dtype, torch.bfloat16)
            results.append((output, state, snapshots))
        for actual, expected in zip(results[1], results[0]):
            torch.testing.assert_close(
                actual.float(), expected.float(), atol=2e-2, rtol=2e-2
            )
        untouched = torch.ones(pool.shape[0], device="cuda", dtype=torch.bool)
        untouched[args["cache_indices"].long()] = False
        self.assertTrue(torch.equal(results[1][1][untouched], pool[untouched]))
        return results

    def compare_decode(self, args, pools, A_log, dt_bias):
        n, h = args["cache_indices"].numel(), A_log.numel()
        hg = args["q"].shape[2]
        decode_args = dict(
            q=torch.randn(1, n, hg, 128, device="cuda", dtype=torch.bfloat16),
            k=torch.randn(1, n, hg, 128, device="cuda", dtype=torch.bfloat16),
            v=torch.randn(1, n, h, 128, device="cuda", dtype=torch.bfloat16),
            a=torch.randn(n, h, device="cuda", dtype=torch.bfloat16),
            b=torch.randn(n, h, device="cuda", dtype=torch.bfloat16),
            A_log=A_log,
            dt_bias=dt_bias,
            cache_indices=args["cache_indices"],
            query_start_loc=torch.arange(n + 1, device="cuda", dtype=torch.int32),
        )
        outputs = [self.kernel.decode(**decode_args, ssm_states=pool) for pool in pools]
        torch.testing.assert_close(
            outputs[1].float(), outputs[0].float(), atol=2e-2, rtol=2e-2
        )
        torch.testing.assert_close(pools[1], pools[0], atol=2e-2, rtol=2e-2)

    def test_prefill_and_decode(self):
        for tp in (2, 4):
            for lengths, zero in (
                ([1], True),
                ([63, 64, 65, 127, 128, 129], False),
                ([8192], False),
            ):
                with self.subTest(tp=tp, lengths=lengths, zero=zero):
                    args, pool, _, _, A_log, dt_bias = self.make_case(tp, lengths, zero)
                    results = self.run_pair(args, pool)
                    self.compare_decode(args, [r[1] for r in results], A_log, dt_bias)

    def test_radix_tracking_and_restore(self):
        for tp in (2, 4):
            args, pool, batch, md, A_log, dt_bias = self.make_case(tp, [64, 65, 129])
            batch.extend_seq_lens = torch.tensor([64, 65, 129], device="cuda")
            batch.extend_prefix_lens = torch.tensor([0, 64, 128], device="cuda")
            batch.mamba_track_mask = torch.ones(3, device="cuda", dtype=torch.bool)
            batch.mamba_track_indices = torch.tensor([2, 4, 6], device="cuda")
            batch.mamba_track_seqlens = batch.extend_prefix_lens + batch.extend_seq_lens
            backend = object.__new__(GDNAttnBackend)
            backend.device = torch.device("cuda")
            backend._mamba_chunk_size = 64
            (
                md.track_ssm_h_src,
                md.track_ssm_h_dst,
                md.track_ssm_final_src,
                md.track_ssm_final_dst,
            ) = backend._init_track_ssm_indices(args["cache_indices"], batch)
            md.has_mamba_track_mask = True
            results = self.run_pair(args, pool)
            for _, state, snapshots in results:
                backend._track_mamba_state_extend(batch, snapshots, state, md)
            torch.testing.assert_close(
                results[1][1], results[0][1], atol=2e-2, rtol=2e-2
            )
            for _, state, _ in results:
                state[args["cache_indices"].long()] = state[batch.mamba_track_indices]
            self.compare_decode(args, [r[1] for r in results], A_log, dt_bias)

    def test_metadata_created_once_for_layers(self):
        args, pool, batch, md, _, _ = self.make_case(2, [65])
        backend = object.__new__(GDNAttnBackend)
        backend.kernel_dispatcher = SimpleNamespace(extend_kernel=self.kernel)
        backend.is_draft_worker = False
        backend.enable_unified_memory = False
        backend.forward_metadata = md
        with (
            envs.SGLANG_USE_AITER_GDN_PREFILL.override(True),
            patch.object(MambaAttnBackendBase, "init_forward_metadata"),
            patch(
                "sglang.srt.layers.attention.linear.gdn_backend.get_memory",
                return_value=SimpleNamespace(enable_page_major_kv_layout=False),
            ),
            patch.object(
                gdn_aiter,
                "maybe_build_aiter_gdn_prefill_metadata",
                wraps=gdn_aiter.maybe_build_aiter_gdn_prefill_metadata,
            ) as build,
        ):
            backend.init_forward_metadata(batch)
            args["aiter_gdn_prefill_metadata"] = md.aiter_gdn_prefill_metadata
            for _ in range(2):
                self.kernel.extend(**args, ssm_states=pool.clone())
            self.assertEqual(build.call_count, 1)
            self.assertIs(
                md.aiter_gdn_prefill_metadata.layout.cu_seqlens, md.query_start_loc
            )

    def test_padding_and_non_prefill_metadata_fallback(self):
        _, _, batch, md, _, _ = self.make_case(2, [64, 65])
        for mode in ForwardMode:
            if mode != ForwardMode.EXTEND:
                batch.forward_mode = mode
                self.assertIsNone(
                    gdn_aiter.maybe_build_aiter_gdn_prefill_metadata(batch, md)
                )
        batch.forward_mode = ForwardMode.EXTEND
        batch._original_batch_size = 1
        md.mamba_cache_indices[-1] = -1
        self.assertIsNone(gdn_aiter.maybe_build_aiter_gdn_prefill_metadata(batch, md))
        batch._original_batch_size = None
        batch.input_ids = torch.zeros(130, device="cuda", dtype=torch.int64)
        self.assertIsNone(gdn_aiter.maybe_build_aiter_gdn_prefill_metadata(batch, md))

    def test_tensor_and_disabled_fallback(self):
        args, pool, _, _, _, _ = self.make_case(2, [65])
        for change in (
            {"aiter_gdn_prefill_metadata": None},
            {"q": args["q"].to(torch.float16)},
            {"ssm_states": pool.transpose(-1, -2)},
            {"ssm_states": pool.to(torch.bfloat16)},
        ):
            call_args = dict(args, ssm_states=pool)
            call_args.update(change)
            with (
                envs.SGLANG_USE_AITER_GDN_PREFILL.override(True),
                patch.object(
                    gdn_triton,
                    "chunk_gated_delta_rule",
                    return_value=(None, None, None),
                ) as old,
                patch.object(gdn_aiter, "aiter_gdn_prefill") as fly,
            ):
                self.kernel.extend(**call_args)
                old.assert_called_once()
                fly.assert_not_called()

    def test_missing_aiter_falls_back(self):
        _, _, batch, md, _, _ = self.make_case(2, [65])
        with patch.object(gdn_aiter, "_aiter_ops", return_value=None):
            self.assertIsNone(
                gdn_aiter.maybe_build_aiter_gdn_prefill_metadata(batch, md)
            )

    def test_gathered_pool_keeps_triton(self):
        args, pool, batch, md, A_log, dt_bias = self.make_case(2, [65])
        md.aiter_gdn_prefill_metadata = args["aiter_gdn_prefill_metadata"]
        batch.extend_prefix_lens = torch.zeros(1, device="cuda", dtype=torch.int32)
        hg, h, t = args["q"].shape[2], args["v"].shape[2], args["q"].shape[1]
        channels = (2 * hg + h) * 128
        layer = SimpleNamespace(
            layer_id=0,
            q_dim=hg * 128,
            k_dim=hg * 128,
            v_dim=h * 128,
            num_q_heads=hg,
            num_k_heads=hg,
            num_v_heads=h,
            head_q_dim=128,
            head_k_dim=128,
            head_v_dim=128,
            conv_weights=None,
            bias=None,
            activation="silu",
            A_log=A_log,
            dt_bias=dt_bias,
        )
        cache = SimpleNamespace(
            conv=[
                torch.zeros(
                    pool.shape[0], channels, 3, device="cuda", dtype=torch.bfloat16
                )
            ],
            temporal=pool.transpose(-1, -2),
        )
        backend = object.__new__(GDNAttnBackend)
        backend.forward_metadata = md
        backend.req_to_token_pool = SimpleNamespace(mamba2_layer_cache=lambda _: cache)
        backend.kernel_dispatcher = SimpleNamespace(extend=self.kernel.extend)
        module = "sglang.srt.layers.attention.linear.gdn_backend"
        with (
            envs.SGLANG_USE_AITER_GDN_PREFILL.override(True),
            patch(module + ".causal_conv1d_fn", side_effect=lambda x, *a, **kw: x),
            patch(
                module + ".fused_qkv_split_gdn_prefill",
                return_value=(args["q"], args["k"], args["v"]),
            ),
            patch.object(
                gdn_triton,
                "chunk_gated_delta_rule",
                return_value=(args["v"], None, None),
            ) as old,
            patch.object(gdn_aiter, "aiter_gdn_prefill") as fly,
        ):
            backend.forward_extend(
                layer,
                batch,
                torch.zeros(t, channels, device="cuda", dtype=torch.bfloat16),
                torch.zeros(t, h, device="cuda", dtype=torch.bfloat16),
                torch.zeros(t, h, device="cuda", dtype=torch.bfloat16),
            )
            old.assert_called_once()
            fly.assert_not_called()


if __name__ == "__main__":
    unittest.main()
