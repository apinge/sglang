import argparse
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, sentinel

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
)
from sglang.srt.layers.attention.linear import gdn_backend
from sglang.srt.layers.attention.linear.gdn_backend import (
    GDNAttnBackend,
    GDNKernelDispatcher,
    _validate_gdn_linear_attn_backends,
    flashinfer_gdn_prefill_default,
)
from sglang.srt.layers.attention.linear.kernels import gdn_aiter
from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
    maybe_build_flashinfer_checkpoint_plan,
)
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.utils import (
    LinearAttnKernelBackend,
    resolve_linear_attn_backends,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _publish(testcase, **fields):
    """Install a published config for one case and restore on its cleanup --
    a failed case must not leave a partial publish for a later file in a
    monolithic local run."""
    from sglang.srt.runtime_context import get_context, get_server_args

    override = get_context().override_server_args(**fields)
    override.install()
    testcase.addCleanup(override.restore)
    return get_server_args()


def make_runner(
    testcase,
    *,
    state_dtype=torch.bfloat16,
    key_dim=128,
    value_dim=128,
    **arg_overrides,
):
    # The policy reads the published bags, so the fixture publishes the
    # configuration under test.
    fields = dict(
        linear_attn_backend="triton",
        linear_attn_prefill_backend=None,
        uses_mamba_radix_cache=False,
        enable_page_major_kv_layout=False,
        mamba_radix_cache_strategy="no_buffer",
        enable_dynamic_chunking=False,
        chunked_prefill_size=8192,
    )
    fields.update(arg_overrides)
    args = _publish(testcase, **fields)

    return SimpleNamespace(
        server_args=args,
        model_config=SimpleNamespace(),
        hybrid_gdn_config=SimpleNamespace(
            linear_key_head_dim=key_dim,
            linear_value_head_dim=value_dim,
        ),
        req_to_token_pool=SimpleNamespace(
            mamba_pool=SimpleNamespace(
                mamba_cache=SimpleNamespace(temporal=SimpleNamespace(dtype=state_dtype))
            )
        ),
    )


class TestFlashInferGDNPrefillBackendPolicy(CustomTestCase):
    def apply_policy(
        self,
        runner,
        *,
        cuda=True,
        capability=(10, 0),
        cuda_version="13.0",
        flashinfer_available=True,
    ):
        with (
            patch.object(
                gdn_backend,
                "hybrid_gdn_config",
                return_value=runner.hybrid_gdn_config,
            ),
            patch.object(gdn_backend, "is_cuda", return_value=cuda),
            patch.object(torch.cuda, "get_device_capability", return_value=capability),
            patch.object(torch.version, "cuda", cuda_version),
            patch(
                "sglang.srt.layers.attention.linear.kernels.gdn_flashinfer."
                "is_flashinfer_gdn_prefill_available",
                return_value=flashinfer_available,
            ),
        ):
            return flashinfer_gdn_prefill_default(runner)

    def test_selects_flashinfer_for_supported_sm100_gdn(self):
        self.assertEqual(self.apply_policy(make_runner(self)), "flashinfer")

    def test_selects_flashinfer_for_radix_cache_strategies(self):
        for strategy in ("no_buffer", "extra_buffer", "extra_buffer_lazy"):
            with self.subTest(strategy=strategy):
                runner = make_runner(
                    self,
                    uses_mamba_radix_cache=True,
                    mamba_radix_cache_strategy=strategy,
                )
                self.assertEqual(self.apply_policy(runner), "flashinfer")

    def test_declines_when_the_prefill_backend_is_explicit(self):
        for backend in ("triton", "flashinfer", "cutedsl", "aiter"):
            with self.subTest(backend=backend):
                runner = make_runner(self, linear_attn_prefill_backend=backend)
                self.assertIsNone(self.apply_policy(runner))

    def test_declines_when_deterministic_inference_is_enabled(self):
        """Batch-sensitive GDN prefill must not bypass deterministic inference."""
        runner = make_runner(
            self,
            state_dtype=torch.float32,
            enable_deterministic_inference=True,
        )

        self.assertIsNone(
            self.apply_policy(
                runner,
                capability=(9, 0),
                cuda_version="12.9",
            )
        )

    def test_rejects_explicit_flashinfer_prefill_in_deterministic_mode(self):
        """Explicit backend precedence must not bypass deterministic GDN startup."""
        cases = (
            {"linear_attn_prefill_backend": "flashinfer"},
            {"linear_attn_backend": "flashinfer"},
        )
        for fields in cases:
            with self.subTest(fields=fields):
                make_runner(
                    self,
                    enable_deterministic_inference=True,
                    **fields,
                )
                backends = resolve_linear_attn_backends()

                with self.assertRaisesRegex(
                    ValueError,
                    "FlashInfer GDN prefill is not supported with "
                    "--enable-deterministic-inference",
                ):
                    _validate_gdn_linear_attn_backends(backends)

    def test_rejects_unsupported_capability(self):
        cases = (
            ("non_cuda", {}, {"cuda": False}),
            ("hopper", {}, {"capability": (9, 0)}),
            ("future_sm", {}, {"capability": (12, 0)}),
            ("cuda_12", {}, {"cuda_version": "12.9"}),
            ("fp32_state", {"state_dtype": torch.float32}, {}),
            ("key_dim", {"key_dim": 64}, {}),
            ("value_dim", {"value_dim": 64}, {}),
            ("missing_api", {}, {"flashinfer_available": False}),
        )
        for name, runner_args, hardware in cases:
            with self.subTest(name=name):
                self.assertIsNone(
                    self.apply_policy(make_runner(self, **runner_args), **hardware)
                )

    def test_rejects_gdn_config_without_qwen_head_dims(self):
        runner = make_runner(self)
        runner.hybrid_gdn_config = SimpleNamespace()
        self.assertIsNone(self.apply_policy(runner))

    def test_rejects_unvalidated_runtime_modes(self):
        cases = (
            ("non_triton_base", {"linear_attn_backend": "cutedsl"}),
            ("page_major_kv", {"enable_page_major_kv_layout": True}),
            ("dynamic_chunk", {"enable_dynamic_chunking": True}),
            ("unchunked", {"chunked_prefill_size": -1}),
            ("unknown_chunk", {"chunked_prefill_size": None}),
            ("large_chunk", {"chunked_prefill_size": 8193}),
        )
        for name, runner_args in cases:
            with self.subTest(name=name):
                self.assertIsNone(self.apply_policy(make_runner(self, **runner_args)))

    def test_builds_compact_checkpoint_plan_for_packed_sequences(self):
        forward_batch = SimpleNamespace(
            extend_seq_lens=torch.tensor([63, 64, 65, 127, 128, 129]),
            mamba_track_mask=torch.tensor([False, True, True, True, True, True]),
            # 65 on the 128-token sequence represents an interior S64
            # boundary encoded as S64 + 1 by the scheduler.
            mamba_track_seqlens=torch.tensor([63, 64, 65, 127, 65, 129]),
            extend_prefix_lens=torch.zeros(6, dtype=torch.int64),
        )
        metadata = SimpleNamespace(
            track_ssm_h_src=torch.empty(4),
            track_ssm_h_dst=torch.empty(4),
        )

        # The chunk size is a derived config member; seed its private cache on a
        # published config rather than patching an import binding.
        override = get_context().override_server_args(_mamba_cache_chunk_size=64)
        override.install()
        self.addCleanup(override.restore)
        maybe_build_flashinfer_checkpoint_plan(forward_batch, metadata, "cpu")

        torch.testing.assert_close(
            metadata.state_checkpoint_cu_starts,
            torch.tensor([0, 0, 1, 2, 3, 5, 7]),
        )
        torch.testing.assert_close(metadata.track_ssm_h_src, torch.tensor([1, 2, 3, 6]))
        self.assertEqual(metadata.num_state_checkpoints, 7)
        self.assertEqual(metadata.state_checkpoint_every_n_tokens, 64)

    def test_decode_tracking_without_h_source_skips_checkpoint_plan(self):
        backend = object.__new__(GDNAttnBackend)
        backend.device = "cpu"
        backend.kernel_dispatcher = SimpleNamespace(
            extend_uses_state_checkpoints=True,
            # The test only requires that extend_kernel is *not* an AiterGDNKernel;
            # avoid constructing a real kernel object in a unit test.
            extend_kernel=sentinel.extend_kernel,
        )
        metadata = SimpleNamespace(has_mamba_track_mask=True, track_ssm_h_src=None)
        forward_batch = SimpleNamespace(
            mamba_track_mask=torch.tensor([True]),
            mamba_track_indices=torch.tensor([7]),
        )

        def init_base(instance, _forward_batch):
            instance.forward_metadata = metadata

        with patch.object(MambaAttnBackendBase, "init_forward_metadata", init_base):
            backend.init_forward_metadata(forward_batch)

        torch.testing.assert_close(metadata.conv_states_mask_indices, torch.tensor([7]))

    def test_tree_verify_uses_triton_kernel(self):
        flashinfer_kernel = MagicMock(supports_target_verify=True)
        with (
            patch.object(gdn_backend, "is_cuda", return_value=True),
            patch(
                "sglang.srt.layers.attention.linear.kernels.gdn_flashinfer."
                "FlashInferGDNKernel",
                return_value=flashinfer_kernel,
            ),
        ):
            dispatcher = GDNKernelDispatcher(
                LinearAttnKernelBackend.TRITON,
                LinearAttnKernelBackend.FLASHINFER,
            )

        self.assertIsInstance(dispatcher.tree_verify_kernel, TritonGDNKernel)

        tensor = sentinel.tensor
        with patch.object(
            dispatcher.tree_verify_kernel, "target_verify"
        ) as tree_verify:
            dispatcher.target_verify(
                *([tensor] * 7),
                ssm_states=tensor,
                cache_indices=tensor,
                query_start_loc=tensor,
                retrieve_parent_token=sentinel.parent_token,
            )

        tree_verify.assert_called_once()
        flashinfer_kernel.target_verify.assert_not_called()

    def test_helion_backend_reports_kda_only(self):
        cases = (
            (LinearAttnKernelBackend.HELION, LinearAttnKernelBackend.TRITON),
            (LinearAttnKernelBackend.TRITON, LinearAttnKernelBackend.HELION),
        )
        for decode_backend, prefill_backend in cases:
            with self.subTest(
                decode_backend=decode_backend,
                prefill_backend=prefill_backend,
            ):
                with self.assertRaisesRegex(ValueError, "supports KDA only"):
                    GDNKernelDispatcher(decode_backend, prefill_backend)


class TestAiterGDNPrefillBackendPolicy(CustomTestCase):
    def test_cli_accepts_aiter_only_for_prefill(self):
        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        args = parser.parse_args(
            ["--model-path", "unused", "--linear-attn-prefill-backend", "aiter"]
        )
        self.assertEqual(args.linear_attn_prefill_backend, "aiter")
        for flag in (
            "--linear-attn-backend",
            "--linear-attn-decode-backend",
            "--linear-attn-verify-backend",
        ):
            with self.subTest(flag=flag), self.assertRaises(SystemExit):
                parser.parse_args(["--model-path", "unused", flag, "aiter"])

    def test_prefill_override_keeps_triton_decode_and_verify(self):
        _publish(
            self, linear_attn_backend="triton", linear_attn_prefill_backend="aiter"
        )
        backends = resolve_linear_attn_backends()
        with (
            patch.object(gdn_backend, "_is_hip", True),
            patch.object(gdn_aiter, "_aiter_ops", return_value=None),
        ):
            dispatcher = GDNKernelDispatcher(
                backends.decode, backends.prefill, backends.verify
            )
        self.assertIsInstance(dispatcher.extend_kernel, gdn_aiter.AiterGDNKernel)
        self.assertIsInstance(dispatcher.decode_kernel, TritonGDNKernel)
        self.assertIs(dispatcher.verify_kernel, dispatcher.decode_kernel)
        self.assertIs(
            dispatcher.extend_kernel.fallback_kernel, dispatcher.decode_kernel
        )
        self.assertEqual(
            dispatcher.supports_packed_decode, TritonGDNKernel.supports_packed_decode
        )
        self.assertFalse(dispatcher.extend_uses_state_checkpoints)

    def test_triton_selection_and_non_rocm_do_not_load_aiter(self):
        with patch.object(gdn_aiter, "_aiter_ops") as load:
            GDNKernelDispatcher(
                LinearAttnKernelBackend.TRITON, LinearAttnKernelBackend.TRITON
            )
            with patch.object(gdn_backend, "_is_hip", False):
                dispatcher = GDNKernelDispatcher(
                    LinearAttnKernelBackend.TRITON, LinearAttnKernelBackend.AITER
                )
            self.assertIsInstance(dispatcher.extend_kernel, TritonGDNKernel)
            load.assert_not_called()

    def test_programmatic_decode_and_verify_selection_are_rejected(self):
        cases = (
            (LinearAttnKernelBackend.AITER, None),
            (LinearAttnKernelBackend.TRITON, LinearAttnKernelBackend.AITER),
        )
        for decode, verify in cases:
            with self.subTest(decode=decode, verify=verify):
                with self.assertRaisesRegex(ValueError, "supports prefill only"):
                    GDNKernelDispatcher(decode, LinearAttnKernelBackend.AITER, verify)

    def test_missing_dependencies_use_fallback_without_launch(self):
        fallback = MagicMock()
        with patch.object(gdn_aiter, "_aiter_ops", return_value=None):
            kernel = gdn_aiter.AiterGDNKernel(fallback)
        self.assertIsNone(
            kernel.build_prefill_metadata(sentinel.batch, sentinel.metadata)
        )
        args = dict(
            q=sentinel.q,
            k=sentinel.k,
            v=sentinel.v,
            g=sentinel.g,
            beta=sentinel.beta,
            ssm_states=sentinel.states,
            cache_indices=sentinel.indices,
            query_start_loc=sentinel.cu,
            state_checkpoint_cu_starts=sentinel.checkpoints,
        )
        with patch.object(gdn_aiter, "_run_aiter_gdn_prefill") as launch:
            self.assertIs(
                kernel.extend(**args, aiter_gdn_prefill_metadata=sentinel.metadata),
                fallback.extend.return_value,
            )
            fallback.extend.assert_called_once_with(**args)
            launch.assert_not_called()

    def test_aiter_is_not_a_kda_prefill_backend(self):
        from sglang.srt.layers.attention.linear.kda_backend import KDAKernelDispatcher

        with self.assertRaisesRegex(ValueError, "Unsupported KDA prefill backend"):
            KDAKernelDispatcher(
                LinearAttnKernelBackend.TRITON,
                LinearAttnKernelBackend.AITER,
                LinearAttnKernelBackend.TRITON,
            )

    def test_execution_failure_is_not_retried_after_state_mutation(self):
        fallback = MagicMock()
        with patch.object(gdn_aiter, "_aiter_ops", return_value=(None,) * 4):
            kernel = gdn_aiter.AiterGDNKernel(fallback)
        with (
            patch.object(gdn_aiter, "aiter_gdn_prefill_supported", return_value=True),
            patch.object(
                gdn_aiter,
                "_run_aiter_gdn_prefill",
                side_effect=RuntimeError("launch failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "launch failed"),
        ):
            kernel.extend(
                *([sentinel.tensor] * 5),
                ssm_states=sentinel.states,
                cache_indices=sentinel.indices,
                query_start_loc=sentinel.cu,
                aiter_gdn_prefill_metadata=sentinel.metadata,
            )
        fallback.extend.assert_not_called()

    def test_metadata_does_not_survive_backend_or_mode_changes(self):
        _publish(self, enable_page_major_kv_layout=False)
        with patch.object(gdn_aiter, "_aiter_ops", return_value=None):
            kernel = gdn_aiter.AiterGDNKernel(TritonGDNKernel())
        backend = object.__new__(GDNAttnBackend)
        backend.kernel_dispatcher = SimpleNamespace(extend_kernel=kernel)
        backend.is_draft_worker = False
        backend.enable_unified_memory = False
        metadata = SimpleNamespace(
            aiter_gdn_prefill_metadata=None, has_mamba_track_mask=False
        )
        backend.forward_metadata = metadata
        with (
            patch.object(gdn_backend, "_is_hip", True),
            patch.object(MambaAttnBackendBase, "init_forward_metadata"),
            patch.object(
                kernel,
                "build_prefill_metadata",
                side_effect=[sentinel.first, sentinel.second],
            ) as build,
        ):
            backend.init_forward_metadata(sentinel.batch1)
            self.assertIs(metadata.aiter_gdn_prefill_metadata, sentinel.first)
            backend.init_forward_metadata(sentinel.batch2)
            self.assertIs(metadata.aiter_gdn_prefill_metadata, sentinel.second)
            backend.is_draft_worker = True
            backend.init_forward_metadata(sentinel.draft)
            self.assertIsNone(metadata.aiter_gdn_prefill_metadata)
            backend.is_draft_worker = False
            backend.kernel_dispatcher.extend_kernel = kernel.fallback_kernel
            metadata.aiter_gdn_prefill_metadata = sentinel.stale
            backend.init_forward_metadata(sentinel.triton)
            self.assertIsNone(metadata.aiter_gdn_prefill_metadata)
            self.assertEqual(build.call_count, 2)


if __name__ == "__main__":
    unittest.main()
