import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.test.ci.ci_register import register_amd_ci


register_amd_ci(est_time=40, suite="stage-b-test-1-gpu-large-amd")


class TestAiterGDNBackendRegistration(unittest.TestCase):
    def test_aiter_backend_is_registered(self):
        self.assertEqual(LinearAttnKernelBackend.AITER.value, "aiter")

    def test_decode_backend_selection(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_aiter import (
            select_aiter_gdn_decode_backend,
        )

        self.assertEqual(
            select_aiter_gdn_decode_backend(
                2,
                8,
                hip_available=True,
                fly_available=True,
                batch_size=8,
            ),
            "flydsl",
        )
        self.assertEqual(
            select_aiter_gdn_decode_backend(
                2,
                8,
                hip_available=True,
                fly_available=True,
                batch_size=16,
            ),
            "flydsl",
        )
        self.assertEqual(
            select_aiter_gdn_decode_backend(
                2,
                8,
                hip_available=True,
                fly_available=True,
                batch_size=20,
            ),
            "flydsl",
        )
        self.assertEqual(
            select_aiter_gdn_decode_backend(
                2,
                8,
                hip_available=True,
                fly_available=True,
                batch_size=24,
            ),
            "hip",
        )
        self.assertEqual(
            select_aiter_gdn_decode_backend(
                2,
                8,
                hip_available=True,
                fly_available=True,
                batch_size=32,
            ),
            "hip",
        )
        self.assertEqual(
            select_aiter_gdn_decode_backend(
                16,
                48,
                hip_available=False,
                fly_available=True,
                batch_size=8,
            ),
            "flydsl",
        )
        self.assertEqual(
            select_aiter_gdn_decode_backend(
                8, 24, hip_available=False, fly_available=True
            ),
            "flydsl",
        )
        self.assertEqual(
            select_aiter_gdn_decode_backend(
                4,
                8,
                hip_available=True,
                fly_available=True,
                batch_size=64,
            ),
            "flydsl",
        )
        self.assertEqual(
            select_aiter_gdn_decode_backend(
                2,
                8,
                hip_available=False,
                fly_available=True,
                batch_size=24,
            ),
            "flydsl",
        )
        self.assertEqual(
            select_aiter_gdn_decode_backend(
                8, 24, hip_available=False, fly_available=False
            ),
            "triton",
        )

    def test_hip_decode_runtime_guard(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_aiter import (
            supports_hip_gdn_decode_runtime,
        )

        common = dict(
            local_num_k_heads=2,
            local_num_v_heads=8,
            q_dtype=torch.bfloat16,
            k_dtype=torch.bfloat16,
            v_dtype=torch.bfloat16,
            a_dtype=torch.bfloat16,
            b_dtype=torch.bfloat16,
            dt_bias_dtype=torch.bfloat16,
            state_dtype=torch.float32,
            head_k_dim=128,
            head_v_dim=128,
            state_shape=(64, 8, 128, 128),
        )
        self.assertTrue(supports_hip_gdn_decode_runtime(**common))
        self.assertFalse(
            supports_hip_gdn_decode_runtime(**{**common, "state_dtype": torch.bfloat16})
        )
        self.assertFalse(
            supports_hip_gdn_decode_runtime(
                **{**common, "local_num_v_heads": 24, "state_shape": (64, 24, 128, 128)}
            )
        )

    def test_decode_uses_hip_then_flydsl_then_triton(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_aiter import (
            AiterGDNKernel,
        )

        fallback = mock.Mock()
        fallback.decode.return_value = "triton"
        hip_decode = mock.Mock(return_value="hip")
        fly_decode = mock.Mock(return_value="flydsl")
        kernel = AiterGDNKernel(
            fallback_kernel=fallback,
            hip_decode=hip_decode,
            fly_decode=fly_decode,
            hip_arch_supported=True,
        )

        def inputs(k_heads, v_heads, batch_size=2, state_dtype=torch.float32):
            return dict(
                q=torch.empty(1, batch_size, k_heads, 128, dtype=torch.bfloat16),
                k=torch.empty(1, batch_size, k_heads, 128, dtype=torch.bfloat16),
                v=torch.empty(1, batch_size, v_heads, 128, dtype=torch.bfloat16),
                a=torch.empty(1, batch_size, v_heads, dtype=torch.bfloat16),
                b=torch.empty(1, batch_size, v_heads, dtype=torch.bfloat16),
                A_log=torch.empty(v_heads, dtype=torch.float32),
                dt_bias=torch.empty(v_heads, dtype=torch.bfloat16),
                ssm_states=torch.empty(
                    batch_size + 2, v_heads, 128, 128, dtype=state_dtype
                ),
                cache_indices=torch.arange(batch_size, dtype=torch.int32),
                query_start_loc=torch.arange(batch_size + 1, dtype=torch.int32),
            )

        self.assertEqual(kernel.decode(**inputs(2, 8, batch_size=32)), "hip")
        self.assertEqual(kernel.decode(**inputs(16, 48, batch_size=8)), "flydsl")

        fallback_only = AiterGDNKernel(
            fallback_kernel=fallback,
            hip_decode=hip_decode,
            fly_decode=None,
            hip_arch_supported=True,
        )
        self.assertEqual(
            fallback_only.decode(
                **inputs(2, 8, batch_size=24, state_dtype=torch.bfloat16)
            ),
            "triton",
        )
        self.assertEqual(
            fallback_only.decode(**inputs(2, 8), replayssm_d=torch.empty(1)),
            "triton",
        )

    def test_decode_sort_cache_reset_is_forwarded(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_aiter import (
            AiterGDNKernel,
        )

        reset = mock.Mock()
        kernel = AiterGDNKernel(
            fallback_kernel=mock.Mock(),
            hip_decode=mock.Mock(),
            fly_decode=None,
            reset_sort_cache=reset,
        )
        kernel.reset_decode_cache()
        reset.assert_called_once_with()

    def test_decode_ignores_cuda_graph_padding(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_aiter import (
            AiterGDNKernel,
        )

        fallback = mock.Mock()

        def hip_decode(**kwargs):
            return torch.ones_like(kwargs["v"])

        kernel = AiterGDNKernel(
            fallback_kernel=fallback,
            hip_decode=hip_decode,
            fly_decode=None,
            hip_arch_supported=True,
        )
        output = kernel.decode(
            q=torch.empty(1, 40, 2, 128, dtype=torch.bfloat16),
            k=torch.empty(1, 40, 2, 128, dtype=torch.bfloat16),
            v=torch.empty(1, 40, 8, 128, dtype=torch.bfloat16),
            a=torch.empty(1, 40, 8, dtype=torch.bfloat16),
            b=torch.empty(1, 40, 8, dtype=torch.bfloat16),
            A_log=torch.empty(8, dtype=torch.float32),
            dt_bias=torch.empty(8, dtype=torch.bfloat16),
            ssm_states=torch.empty(48, 8, 128, 128, dtype=torch.float32),
            cache_indices=torch.cat(
                [
                    torch.arange(32, dtype=torch.int32),
                    torch.full((8,), -1, dtype=torch.int32),
                ]
            ),
            query_start_loc=torch.cat(
                [
                    torch.arange(33, dtype=torch.int32),
                    torch.full((8,), 32, dtype=torch.int32),
                ]
            ),
            active_batch_size=32,
        )
        self.assertEqual(output.shape, (1, 40, 8, 128))
        torch.testing.assert_close(output[:, :32], torch.ones_like(output[:, :32]))
        torch.testing.assert_close(output[:, 32:], torch.zeros_like(output[:, 32:]))
        fallback.decode.assert_not_called()

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.version.hip is not None,
        "ROCm is required",
    )
    def test_decode_uses_hip_during_cuda_graph_capture(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_aiter import (
            AiterGDNKernel,
        )

        fallback = mock.Mock()
        hip_decode = mock.Mock(return_value="hip-graph")
        kernel = AiterGDNKernel(
            fallback_kernel=fallback,
            hip_decode=hip_decode,
            fly_decode=None,
            hip_arch_supported=True,
        )
        batch = 32
        with mock.patch("torch.cuda.is_current_stream_capturing", return_value=True):
            output = kernel.decode(
                q=torch.empty(1, batch, 2, 128, device="cuda", dtype=torch.bfloat16),
                k=torch.empty(1, batch, 2, 128, device="cuda", dtype=torch.bfloat16),
                v=torch.empty(1, batch, 8, 128, device="cuda", dtype=torch.bfloat16),
                a=torch.empty(1, batch, 8, device="cuda", dtype=torch.bfloat16),
                b=torch.empty(1, batch, 8, device="cuda", dtype=torch.bfloat16),
                A_log=torch.empty(8, device="cuda", dtype=torch.float32),
                dt_bias=torch.empty(8, device="cuda", dtype=torch.bfloat16),
                ssm_states=torch.empty(
                    batch + 2, 8, 128, 128, device="cuda", dtype=torch.float32
                ),
                cache_indices=torch.arange(batch, device="cuda", dtype=torch.int32),
                query_start_loc=torch.arange(
                    batch + 1, device="cuda", dtype=torch.int32
                ),
            )
        self.assertEqual(output, "hip-graph")
        fallback.decode.assert_not_called()
        hip_decode.assert_called_once()

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.version.hip is not None,
        "ROCm is required",
    )
    def test_packed_decode_uses_triton_below_hip_crossover(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_aiter import (
            AiterGDNKernel,
        )

        fallback = mock.Mock()
        fallback.packed_decode.return_value = "packed-triton"
        kernel = AiterGDNKernel(
            fallback_kernel=fallback,
            hip_decode=mock.Mock(),
            fly_decode=None,
            hip_arch_supported=True,
        )
        mixed_qkv = torch.empty(2, 1536, device="cuda", dtype=torch.bfloat16)
        kwargs = dict(
            mixed_qkv=mixed_qkv,
            a=torch.empty(2, 8, device="cuda", dtype=torch.bfloat16),
            b=torch.empty(2, 8, device="cuda", dtype=torch.bfloat16),
            A_log=torch.empty(8, device="cuda", dtype=torch.float32),
            dt_bias=torch.empty(8, device="cuda", dtype=torch.bfloat16),
            scale=128**-0.5,
            ssm_states=torch.empty(4, 8, 128, 128, device="cuda", dtype=torch.float32),
            cache_indices=torch.tensor([1, 2], device="cuda", dtype=torch.int32),
            num_v_heads=8,
            head_v_dim=128,
        )
        self.assertEqual(kernel.packed_decode(**kwargs), "packed-triton")
        with mock.patch("torch.cuda.is_current_stream_capturing", return_value=True):
            self.assertEqual(kernel.packed_decode(**kwargs), "packed-triton")
        self.assertEqual(fallback.packed_decode.call_count, 2)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.version.hip is not None,
        "ROCm is required",
    )
    def test_hip_decode_cuda_graph_replay_batch_192(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_aiter import (
            AiterGDNKernel,
        )
        from sglang.srt.layers.attention.linear.kernels.gdn_triton import (
            TritonGDNKernel,
        )

        torch.manual_seed(99)
        batch, k_heads, v_heads, dim, slots = 192, 2, 8, 128, 210
        q = torch.randn(1, batch, k_heads, dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn(1, batch, v_heads, dim, device="cuda", dtype=torch.bfloat16)
        a = torch.randn(1, batch, v_heads, device="cuda", dtype=torch.bfloat16)
        b = torch.randn_like(a)
        A_log = torch.randn(v_heads, device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(v_heads, device="cuda", dtype=torch.bfloat16)
        starts = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
        indices = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
        initial_state = torch.randn(
            slots, v_heads, dim, dim, device="cuda", dtype=torch.float32
        )
        state = initial_state.clone()
        triton = TritonGDNKernel()
        fallback = mock.Mock()
        fallback.decode.side_effect = AssertionError("unexpected Triton fallback")
        aiter = AiterGDNKernel(
            fallback_kernel=fallback,
            fly_decode=None,
        )

        aiter.decode(
            q,
            k,
            v,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=state,
            cache_indices=indices,
            query_start_loc=starts,
        )
        torch.cuda.synchronize()
        state.copy_(initial_state)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = aiter.decode(
                q,
                k,
                v,
                a,
                b,
                A_log=A_log,
                dt_bias=dt_bias,
                ssm_states=state,
                cache_indices=indices,
                query_start_loc=starts,
            )
        torch.cuda.synchronize()

        state.copy_(initial_state)
        indices.copy_(torch.arange(batch, 0, -1, device="cuda", dtype=torch.int32))
        state_ref = initial_state.clone()
        output_ref = triton.decode(
            q,
            k,
            v,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=state_ref,
            cache_indices=indices,
            query_start_loc=starts,
        )
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(output, output_ref, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(state, state_ref, rtol=1e-3, atol=1e-3)
        fallback.decode.assert_not_called()

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.version.hip is not None,
        "ROCm is required",
    )
    def test_flydsl_decode_matches_triton_for_27b_tp1(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_aiter import (
            AiterGDNKernel,
        )
        from sglang.srt.layers.attention.linear.kernels.gdn_triton import (
            TritonGDNKernel,
        )

        torch.manual_seed(101)
        batch, k_heads, v_heads, dim = 8, 16, 48, 128
        q = torch.randn(1, batch, k_heads, dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn(1, batch, v_heads, dim, device="cuda", dtype=torch.bfloat16)
        a = torch.randn(1, batch, v_heads, device="cuda", dtype=torch.bfloat16)
        b = torch.randn_like(a)
        A_log = torch.randn(v_heads, device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(v_heads, device="cuda", dtype=torch.bfloat16)
        indices = torch.arange(batch, device="cuda", dtype=torch.int32)
        starts = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
        initial = torch.randn(
            batch, v_heads, dim, dim, device="cuda", dtype=torch.float32
        )
        state_ref = initial.clone()
        state_actual = initial.clone()
        triton = TritonGDNKernel()
        fallback = mock.Mock()
        fallback.decode.side_effect = AssertionError("unexpected Triton fallback")
        aiter = AiterGDNKernel(
            fallback_kernel=fallback,
            hip_decode=None,
        )

        output_ref = triton.decode(
            q,
            k,
            v,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=state_ref,
            cache_indices=indices,
            query_start_loc=starts,
        )
        output = aiter.decode(
            q,
            k,
            v,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=state_actual,
            cache_indices=indices,
            query_start_loc=starts,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(output, output_ref, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(state_actual, state_ref, rtol=1e-3, atol=1e-3)

        state_actual.copy_(initial)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = aiter.decode(
                q,
                k,
                v,
                a,
                b,
                A_log=A_log,
                dt_bias=dt_bias,
                ssm_states=state_actual,
                cache_indices=indices,
                query_start_loc=starts,
            )
        torch.cuda.synchronize()
        state_actual.copy_(initial)
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(graph_output, output_ref, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(state_actual, state_ref, rtol=1e-3, atol=1e-3)
        fallback.decode.assert_not_called()

    def test_dispatcher_can_select_aiter_per_mode(self):
        from sglang.srt.layers.attention.linear.gdn_backend import (
            GDNKernelDispatcher,
        )
        from sglang.srt.layers.attention.linear.kernels.gdn_aiter import (
            AiterGDNKernel,
        )
        from sglang.srt.layers.attention.linear.kernels.gdn_triton import (
            TritonGDNKernel,
        )

        with mock.patch.object(
            AiterGDNKernel,
            "__init__",
            lambda self, fallback_kernel=None: setattr(
                self, "fallback_kernel", fallback_kernel
            ),
        ):
            dispatcher = GDNKernelDispatcher(
                LinearAttnKernelBackend.AITER,
                LinearAttnKernelBackend.AITER,
            )

        self.assertIsInstance(dispatcher.decode_kernel, AiterGDNKernel)
        self.assertIsInstance(dispatcher.extend_kernel, AiterGDNKernel)
        self.assertIsInstance(dispatcher.verify_kernel, TritonGDNKernel)
        self.assertTrue(dispatcher.supports_packed_decode)

    def test_dispatcher_forwards_decode_cache_reset(self):
        from sglang.srt.layers.attention.linear.gdn_backend import (
            GDNKernelDispatcher,
        )

        dispatcher = object.__new__(GDNKernelDispatcher)
        dispatcher.decode_kernel = mock.Mock()
        dispatcher.reset_decode_cache()
        dispatcher.decode_kernel.reset_decode_cache.assert_called_once_with()

    def test_backend_prepares_active_decode_batch_for_graph_padding(self):
        from sglang.srt.layers.attention.linear.gdn_backend import (
            GDNAttnBackend,
            _forward_batch_has_padding,
        )

        backend = object.__new__(GDNAttnBackend)
        backend.kernel_dispatcher = mock.Mock()
        forward_mode = mock.Mock()
        forward_mode.is_decode_or_idle.return_value = True
        forward_batch = SimpleNamespace(
            forward_mode=forward_mode,
            batch_size=8,
            num_padding=3,
        )
        backend._prepare_aiter_forward_metadata(forward_batch)
        self.assertEqual(backend._aiter_decode_active_batch_size, 5)
        backend.kernel_dispatcher.reset_decode_cache.assert_called_once_with()

        backend.kernel_dispatcher.reset_mock()
        backend._prepare_aiter_forward_metadata(
            SimpleNamespace(
                forward_mode=forward_mode,
                batch_size=8,
                num_padding=0,
                _original_batch_size=5,
            )
        )
        self.assertEqual(backend._aiter_decode_active_batch_size, 5)
        backend.kernel_dispatcher.reset_decode_cache.assert_called_once_with()

        self.assertTrue(
            _forward_batch_has_padding(
                SimpleNamespace(
                    batch_size=8,
                    num_padding=0,
                    _original_batch_size=5,
                )
            )
        )

    def test_backend_builds_reusable_aiter_prefill_metadata(self):
        from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend

        backend = object.__new__(GDNAttnBackend)
        backend._aiter_decode_active_batch_size = None
        backend._aiter_prefill_metadata = None
        backend._aiter_prefill_metadata_builder = mock.Mock(
            return_value="prefill-metadata"
        )
        backend.mamba_cache_chunk_size = 64
        query_start_loc = torch.tensor([0, 8192], dtype=torch.int32)
        backend.forward_metadata = SimpleNamespace(query_start_loc=query_start_loc)
        forward_mode = mock.Mock()
        forward_mode.is_decode_or_idle.return_value = False
        forward_mode.is_extend_without_speculative.return_value = True
        forward_batch = SimpleNamespace(
            forward_mode=forward_mode,
            batch_size=1,
            num_padding=0,
            extend_seq_lens_cpu=[8192],
        )

        backend._prepare_aiter_forward_metadata(forward_batch)

        self.assertEqual(backend._aiter_prefill_metadata, "prefill-metadata")
        backend._aiter_prefill_metadata_builder.assert_called_once_with(
            [8192],
            cu_seqlens=query_start_loc,
            chunk_size=64,
        )

    def test_prefill_uses_aiter_and_falls_back_when_intermediate_h_is_required(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_aiter import (
            AiterGDNKernel,
        )

        fallback = mock.Mock()
        fallback.extend.return_value = ("triton", None, "h")

        def prefill(**kwargs):
            self.assertTrue(kwargs["q"].is_contiguous())
            self.assertTrue(kwargs["k"].is_contiguous())
            self.assertTrue(kwargs["v"].is_contiguous())
            self.assertTrue(kwargs["g"].is_contiguous())
            self.assertTrue(kwargs["beta"].is_contiguous())
            return "aiter", "full-pool-state"

        prefill = mock.Mock(side_effect=prefill)
        kernel = AiterGDNKernel(
            fallback_kernel=fallback,
            hip_decode=None,
            fly_decode=None,
            prefill_vk=prefill,
            prefill_intermediate_ops=None,
            l2norm_qk=lambda q, k: (q, k),
        )
        kwargs = dict(
            q=torch.empty(1, 5, 2, 256, dtype=torch.bfloat16)[..., ::2],
            k=torch.empty(1, 5, 2, 256, dtype=torch.bfloat16)[..., ::2],
            v=torch.empty(1, 5, 8, 256, dtype=torch.bfloat16)[..., ::2],
            g=torch.empty(1, 5, 16, dtype=torch.float32)[..., ::2],
            beta=torch.empty(1, 5, 16, dtype=torch.float32)[..., ::2],
            ssm_states=torch.empty(4, 8, 128, 128, dtype=torch.float32),
            cache_indices=torch.tensor([1], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 5], dtype=torch.int32),
        )

        with mock.patch(
            "sglang.srt.layers.attention.linear.kernels.gdn_aiter.aiter_prefill_min_tokens",
            return_value=0,
        ):
            output, final_state, h = kernel.extend(
                **kwargs, return_intermediate_h=False
            )
        self.assertEqual(output, "aiter")
        self.assertIsNone(final_state)
        self.assertIsNone(h)
        prefill.assert_called_once()

        with mock.patch(
            "sglang.srt.layers.attention.linear.kernels.gdn_aiter.aiter_prefill_min_tokens",
            return_value=0,
        ):
            self.assertEqual(
                kernel.extend(**kwargs, return_intermediate_h=True),
                ("triton", None, "h"),
            )
        fallback.extend.assert_called_once()

    def test_prefill_uses_fused_qk_l2norm_before_high_level_vk(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_aiter import (
            AiterGDNKernel,
        )

        q = torch.empty(1, 5, 2, 128, dtype=torch.bfloat16)
        k = torch.empty_like(q)
        q_norm = torch.full_like(q, 1)
        k_norm = torch.full_like(k, 2)
        fused_l2norm_qk = mock.Mock(return_value=(q_norm, k_norm))
        prefill = mock.Mock(return_value=("aiter", None))
        prefill_metadata = object()
        kernel = AiterGDNKernel(
            fallback_kernel=mock.Mock(),
            hip_decode=None,
            fly_decode=None,
            prefill_vk=prefill,
            prefill_intermediate_ops=None,
            l2norm_qk=fused_l2norm_qk,
        )

        with mock.patch(
            "sglang.srt.layers.attention.linear.kernels.gdn_aiter.aiter_prefill_min_tokens",
            return_value=0,
        ):
            output, final_state, h = kernel.extend(
                q=q,
                k=k,
                v=torch.empty(1, 5, 8, 128, dtype=torch.bfloat16),
                g=torch.empty(1, 5, 8, dtype=torch.float32),
                beta=torch.empty(1, 5, 8, dtype=torch.float32),
                ssm_states=torch.empty(4, 8, 128, 128, dtype=torch.float32),
                cache_indices=torch.tensor([1], dtype=torch.int32),
                query_start_loc=torch.tensor([0, 5], dtype=torch.int32),
                return_intermediate_h=False,
                seq_lens_cpu=[5],
                prefill_metadata=prefill_metadata,
            )

        self.assertEqual(output, "aiter")
        self.assertIsNone(final_state)
        self.assertIsNone(h)
        fused_l2norm_qk.assert_called_once_with(q, k)
        call = prefill.call_args.kwargs
        self.assertIs(call["q"], q_norm)
        self.assertIs(call["k"], k_norm)
        self.assertEqual(call["seq_lens_cpu"], [5])
        self.assertIs(call["prefill_metadata"], prefill_metadata)
        self.assertFalse(call["use_qk_l2norm_in_kernel"])
        self.assertTrue(call["use_chunk_hip"])

    def test_prefill_fallback_skips_qk_norm_without_intermediate_ops(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_aiter import (
            AiterGDNKernel,
        )

        fallback = mock.Mock()
        fallback.extend.return_value = ("triton", None, "h")
        fused_l2norm_qk = mock.Mock(side_effect=lambda q, k: (q, k))
        kernel = AiterGDNKernel(
            fallback_kernel=fallback,
            hip_decode=None,
            fly_decode=None,
            prefill_vk=mock.Mock(),
            prefill_intermediate_ops=None,
            l2norm_qk=fused_l2norm_qk,
        )

        with mock.patch(
            "sglang.srt.layers.attention.linear.kernels.gdn_aiter.aiter_prefill_min_tokens",
            return_value=0,
        ):
            result = kernel.extend(
                q=torch.empty(1, 5, 2, 128, dtype=torch.bfloat16),
                k=torch.empty(1, 5, 2, 128, dtype=torch.bfloat16),
                v=torch.empty(1, 5, 8, 128, dtype=torch.bfloat16),
                g=torch.empty(1, 5, 8, dtype=torch.float32),
                beta=torch.empty(1, 5, 8, dtype=torch.float32),
                ssm_states=torch.empty(4, 8, 128, 128, dtype=torch.float32),
                cache_indices=torch.tensor([1], dtype=torch.int32),
                query_start_loc=torch.tensor([0, 5], dtype=torch.int32),
                return_intermediate_h=True,
            )

        self.assertEqual(result, ("triton", None, "h"))
        fallback.extend.assert_called_once()
        fused_l2norm_qk.assert_not_called()

    def test_prefill_falls_back_for_multi_sequence_batch(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_aiter import (
            AiterGDNKernel,
        )

        fallback = mock.Mock()
        fallback.extend.return_value = ("triton", None, "h")
        prefill = mock.Mock(return_value=("aiter", None))
        kernel = AiterGDNKernel(
            fallback_kernel=fallback,
            hip_decode=None,
            fly_decode=None,
            prefill_vk=prefill,
            prefill_intermediate_ops=None,
            l2norm_qk=lambda q, k: (q, k),
        )
        with mock.patch(
            "sglang.srt.layers.attention.linear.kernels.gdn_aiter.aiter_prefill_min_tokens",
            return_value=0,
        ):
            result = kernel.extend(
                q=torch.empty(1, 5, 2, 128, dtype=torch.bfloat16),
                k=torch.empty(1, 5, 2, 128, dtype=torch.bfloat16),
                v=torch.empty(1, 5, 8, 128, dtype=torch.bfloat16),
                g=torch.empty(1, 5, 8, dtype=torch.float32),
                beta=torch.empty(1, 5, 8, dtype=torch.float32),
                ssm_states=torch.empty(4, 8, 128, 128, dtype=torch.float32),
                cache_indices=torch.tensor([1, 2], dtype=torch.int32),
                query_start_loc=torch.tensor([0, 3, 5], dtype=torch.int32),
                return_intermediate_h=False,
            )

        self.assertEqual(result, ("triton", None, "h"))
        fallback.extend.assert_called_once()
        prefill.assert_not_called()

    def test_prefill_minimum_token_guard(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_aiter import (
            aiter_prefill_min_tokens,
        )

        self.assertEqual(aiter_prefill_min_tokens(8), 4096)
        self.assertEqual(aiter_prefill_min_tokens(48), 1024)

    def test_flydsl_decode_batch_guard(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_aiter import (
            should_use_flydsl_decode,
        )

        self.assertTrue(should_use_flydsl_decode(16, 48, 16))
        self.assertTrue(should_use_flydsl_decode(16, 48, 32))
        self.assertTrue(should_use_flydsl_decode(2, 8, 10))
        self.assertTrue(should_use_flydsl_decode(2, 8, 24))
        self.assertTrue(should_use_flydsl_decode(2, 8, 32))
        self.assertTrue(should_use_flydsl_decode(8, 24, 8))
        self.assertFalse(should_use_flydsl_decode(0, 8, 8))
        self.assertFalse(should_use_flydsl_decode(3, 8, 8))

    def test_decode_conv_split_returns_target_layout(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_aiter import (
            AiterGDNKernel,
        )

        def split(x, _state, _weight, key_dim, value_dim, **_kwargs):
            batch = x.shape[0]
            return (
                torch.empty(batch, key_dim, 1),
                torch.empty(batch, key_dim, 1),
                torch.empty(batch, value_dim, 1),
            )

        kernel = AiterGDNKernel(
            fallback_kernel=mock.Mock(),
            hip_decode=None,
            fly_decode=mock.Mock(),
            decode_conv_split=split,
            hip_arch_supported=False,
        )
        result = kernel.decode_conv_split(
            x=torch.empty(32, 1536, dtype=torch.bfloat16),
            conv_state=torch.empty(40, 1536, 3, dtype=torch.bfloat16),
            weight=torch.empty(1536, 4, dtype=torch.bfloat16),
            bias=None,
            activation="silu",
            conv_state_indices=torch.arange(32, dtype=torch.int32),
            key_dim=256,
            value_dim=1024,
            num_k_heads=2,
            num_v_heads=8,
            head_k_dim=128,
            head_v_dim=128,
        )
        self.assertIsNotNone(result)
        q, k, v = result
        self.assertEqual(q.shape, (1, 32, 2, 128))
        self.assertEqual(k.shape, (1, 32, 2, 128))
        self.assertEqual(v.shape, (1, 32, 8, 128))

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.version.hip is not None,
        "ROCm is required",
    )
    def test_real_decode_conv_split_matches_reference_without_qkv_copies(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_aiter import (
            AiterGDNKernel,
        )
        from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
            causal_conv1d_update,
        )

        torch.manual_seed(109)
        batch, k_heads, v_heads, dim = 4, 2, 8, 128
        key_dim, value_dim = k_heads * dim, v_heads * dim
        mixed_dim = 2 * key_dim + value_dim
        x = torch.randn(batch, mixed_dim, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(
            mixed_dim, 4, device="cuda", dtype=torch.bfloat16
        )
        bias = torch.randn(mixed_dim, device="cuda", dtype=torch.bfloat16)
        indices = torch.tensor([1, 3, 4, 5], device="cuda", dtype=torch.int32)
        initial_state = torch.randn(
            6, mixed_dim, 3, device="cuda", dtype=torch.bfloat16
        )
        state_actual = initial_state.clone()
        state_ref = initial_state.clone()

        kernel = AiterGDNKernel(
            fallback_kernel=mock.Mock(),
            hip_decode=None,
            fly_decode=mock.Mock(),
            hip_arch_supported=False,
        )
        actual = kernel.decode_conv_split(
            x,
            state_actual,
            weight,
            bias=bias,
            activation="silu",
            conv_state_indices=indices,
            key_dim=key_dim,
            value_dim=value_dim,
            num_k_heads=k_heads,
            num_v_heads=v_heads,
            head_k_dim=dim,
            head_v_dim=dim,
        )
        self.assertIsNotNone(actual)

        mixed_ref = causal_conv1d_update(
            x,
            state_ref,
            weight,
            bias=bias,
            activation="silu",
            conv_state_indices=indices,
        )
        q_ref, k_ref, v_ref = torch.split(
            mixed_ref, [key_dim, key_dim, value_dim], dim=-1
        )
        expected = (
            q_ref.view(batch, 1, k_heads, dim).transpose(0, 1),
            k_ref.view(batch, 1, k_heads, dim).transpose(0, 1),
            v_ref.view(batch, 1, v_heads, dim).transpose(0, 1),
        )

        for actual_tensor, expected_tensor in zip(actual, expected):
            torch.testing.assert_close(
                actual_tensor, expected_tensor, rtol=1e-2, atol=1e-2
            )
            self.assertTrue(actual_tensor.transpose(0, 1).is_contiguous())
        torch.testing.assert_close(state_actual, state_ref, rtol=1e-2, atol=1e-2)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.version.hip is not None,
        "ROCm is required",
    )
    def test_real_decode_conv_split_graph_replay_skips_padded_slot(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_aiter import (
            AiterGDNKernel,
        )
        from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
            causal_conv1d_update,
        )

        torch.manual_seed(113)
        batch, k_heads, v_heads, dim = 4, 2, 8, 128
        key_dim, value_dim = k_heads * dim, v_heads * dim
        mixed_dim = 2 * key_dim + value_dim
        x = torch.randn(batch, mixed_dim, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(
            mixed_dim, 4, device="cuda", dtype=torch.bfloat16
        )
        bias = torch.randn(mixed_dim, device="cuda", dtype=torch.bfloat16)
        initial_state = torch.randn(
            6, mixed_dim, 3, device="cuda", dtype=torch.bfloat16
        )
        state_actual = initial_state.clone()
        static_indices = torch.tensor(
            [0, 1, 2, 3], device="cuda", dtype=torch.int32
        )
        replay_indices = torch.tensor(
            [0, 1, -1, 3], device="cuda", dtype=torch.int32
        )
        valid_rows = torch.tensor([0, 1, 3], device="cuda")

        kernel = AiterGDNKernel(
            fallback_kernel=mock.Mock(),
            hip_decode=None,
            fly_decode=mock.Mock(),
            hip_arch_supported=False,
        )
        kernel.decode_conv_split(
            x,
            state_actual,
            weight,
            bias=bias,
            activation="silu",
            conv_state_indices=static_indices,
            key_dim=key_dim,
            value_dim=value_dim,
            num_k_heads=k_heads,
            num_v_heads=v_heads,
            head_k_dim=dim,
            head_v_dim=dim,
        )
        state_actual.copy_(initial_state)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_actual = kernel.decode_conv_split(
                x,
                state_actual,
                weight,
                bias=bias,
                activation="silu",
                conv_state_indices=static_indices,
                key_dim=key_dim,
                value_dim=value_dim,
                num_k_heads=k_heads,
                num_v_heads=v_heads,
                head_k_dim=dim,
                head_v_dim=dim,
                active_batch_size=3,
            )
        self.assertIsNotNone(graph_actual)

        state_actual.copy_(initial_state)
        static_indices.copy_(replay_indices)
        graph.replay()
        torch.cuda.synchronize()

        state_ref = initial_state.clone()
        mixed_ref = causal_conv1d_update(
            x,
            state_ref,
            weight,
            bias=bias,
            activation="silu",
            conv_state_indices=replay_indices,
        )
        q_ref, k_ref, v_ref = torch.split(
            mixed_ref, [key_dim, key_dim, value_dim], dim=-1
        )
        expected = (
            q_ref.view(batch, 1, k_heads, dim),
            k_ref.view(batch, 1, k_heads, dim),
            v_ref.view(batch, 1, v_heads, dim),
        )

        for actual_tensor, expected_tensor in zip(graph_actual, expected):
            torch.testing.assert_close(
                actual_tensor.transpose(0, 1).index_select(0, valid_rows),
                expected_tensor.index_select(0, valid_rows),
                rtol=1e-2,
                atol=1e-2,
            )
        torch.testing.assert_close(state_actual, state_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(state_actual[-1], initial_state[-1])

    def test_prefill_returns_aiter_intermediate_h_when_ops_are_available(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_aiter import (
            AiterGDNKernel,
        )

        fallback = mock.Mock()
        h_expected = torch.empty(1, 2, 8, 128, 128)
        ops = {
            "cumsum": mock.Mock(return_value=(torch.empty(1), torch.empty(1))),
            "solve": mock.Mock(return_value=(torch.empty(1), torch.empty(1))),
            "chunk_h": mock.Mock(return_value=(h_expected, torch.empty(1), None)),
            "chunk_o": mock.Mock(return_value="aiter-h"),
        }
        kernel = AiterGDNKernel(
            fallback_kernel=fallback,
            hip_decode=None,
            fly_decode=None,
            prefill_vk=None,
            prefill_intermediate_ops=ops,
            l2norm=lambda tensor: tensor,
            l2norm_qk=lambda q, k: (q, k),
        )
        with mock.patch(
            "sglang.srt.layers.attention.linear.kernels.gdn_aiter.aiter_prefill_min_tokens",
            return_value=0,
        ):
            output, final_state, h = kernel.extend(
                q=torch.empty(1, 5, 2, 128, dtype=torch.bfloat16),
                k=torch.empty(1, 5, 2, 128, dtype=torch.bfloat16),
                v=torch.empty(1, 5, 8, 128, dtype=torch.bfloat16),
                g=torch.empty(1, 5, 8, dtype=torch.float32),
                beta=torch.empty(1, 5, 8, dtype=torch.float32),
                ssm_states=torch.empty(4, 8, 128, 128, dtype=torch.float32),
                cache_indices=torch.tensor([1], dtype=torch.int32),
                query_start_loc=torch.tensor([0, 5], dtype=torch.int32),
                return_intermediate_h=True,
            )
        self.assertEqual(output, "aiter-h")
        self.assertIsNone(final_state)
        self.assertIs(h, h_expected)
        fallback.extend.assert_not_called()
        with mock.patch(
            "sglang.srt.layers.attention.linear.kernels.gdn_aiter.aiter_prefill_min_tokens",
            return_value=0,
        ):
            output, final_state, h = kernel.extend(
                q=torch.empty(1, 5, 2, 128, dtype=torch.bfloat16),
                k=torch.empty(1, 5, 2, 128, dtype=torch.bfloat16),
                v=torch.empty(1, 5, 8, 128, dtype=torch.bfloat16),
                g=torch.empty(1, 5, 8, dtype=torch.float32),
                beta=torch.empty(1, 5, 8, dtype=torch.float32),
                ssm_states=torch.empty(4, 8, 128, 128, dtype=torch.float32),
                cache_indices=torch.tensor([1], dtype=torch.int32),
                query_start_loc=torch.tensor([0, 5], dtype=torch.int32),
                return_intermediate_h=False,
            )
        self.assertEqual(output, "aiter-h")
        self.assertIsNone(final_state)
        self.assertIsNone(h)

    def test_final_state_tracking_does_not_require_intermediate_h(self):
        import sglang.srt.layers.attention.hybrid_linear_attn_backend as hybrid_backend
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            MambaAttnBackendBase,
        )

        backend = object.__new__(MambaAttnBackendBase)
        states = torch.arange(24, dtype=torch.float32).view(6, 4)
        metadata = SimpleNamespace(
            has_mamba_track_mask=True,
            track_ssm_h_src=torch.empty(0, dtype=torch.int64),
            track_ssm_h_dst=torch.empty(0, dtype=torch.int64),
            track_ssm_final_src=torch.tensor([1], dtype=torch.int64),
            track_ssm_final_dst=torch.tensor([4], dtype=torch.int64),
            track_ssm_h_trusted=True,
            track_ssm_final_trusted=True,
            track_ssm_final_disjoint=True,
        )

        with mock.patch.object(
            hybrid_backend, "copy_mamba_state_extend_rows"
        ) as copy_rows:
            backend._track_mamba_state_extend(
                SimpleNamespace(),
                None,
                states,
                metadata,
            )

        copy_rows.assert_called_once()
        call = copy_rows.call_args
        self.assertIsNone(call.args[0])
        self.assertIs(call.args[1], states)
        self.assertTrue(call.kwargs["h_indices_trusted"])
        self.assertTrue(call.kwargs["final_indices_trusted"])
        self.assertTrue(call.kwargs["final_state_disjoint"])

    def test_extend_metadata_derives_direct_copy_safety(self):
        import sglang.srt.layers.attention.hybrid_linear_attn_backend as hybrid_backend
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            MambaAttnBackendBase,
        )

        backend = object.__new__(MambaAttnBackendBase)
        backend.device = torch.device("cpu")
        backend.req_to_token_pool = SimpleNamespace(
            mamba_pool=SimpleNamespace(
                mamba_cache=SimpleNamespace(temporal=torch.empty(4, 64, 2, 128, 128))
            )
        )
        batch = SimpleNamespace(
            mamba_track_mask=torch.tensor([True, True]),
            extend_seq_lens=torch.tensor([17, 16]),
            mamba_track_indices=torch.tensor([33, 40]),
            mamba_track_seqlens=torch.tensor([17, 16]),
            extend_prefix_lens=torch.tensor([0, 0]),
        )

        with mock.patch.object(
            hybrid_backend,
            "get_global_server_args",
            return_value=SimpleNamespace(mamba_cache_chunk_size=16),
        ):
            result = backend._init_track_ssm_indices(torch.tensor([12, 20]), batch)

        self.assertEqual(len(result), 7)
        h_src, h_dst, final_src, final_dst, h_ok, final_ok, disjoint = result
        self.assertEqual(h_src.numel(), 1)
        self.assertEqual(final_src.numel(), 1)
        self.assertTrue(h_ok)
        self.assertTrue(final_ok)
        self.assertTrue(disjoint)
        self.assertTrue(torch.equal(h_dst, torch.tensor([33])))
        self.assertTrue(torch.equal(final_dst, torch.tensor([40])))

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.version.hip is not None,
        "ROCm is required",
    )
    def test_aiter_decode_matches_triton_for_397b_tp8(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_aiter import (
            AiterGDNKernel,
        )
        from sglang.srt.layers.attention.linear.kernels.gdn_triton import (
            TritonGDNKernel,
        )

        torch.manual_seed(7)
        batch, k_heads, v_heads, dim = 32, 2, 8, 128
        q = torch.randn(1, batch, k_heads, dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn(1, batch, v_heads, dim, device="cuda", dtype=torch.bfloat16)
        a = torch.randn(1, batch, v_heads, device="cuda", dtype=torch.bfloat16)
        b = torch.randn_like(a)
        A_log = torch.randn(v_heads, device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(v_heads, device="cuda", dtype=torch.bfloat16)
        indices = torch.arange(batch, device="cuda", dtype=torch.int32)
        starts = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
        state = torch.randn(
            batch + 2, v_heads, dim, dim, device="cuda", dtype=torch.float32
        )
        state_ref = state.clone()
        state_actual = state.clone()

        triton = TritonGDNKernel()
        fallback = mock.Mock()
        fallback.decode.side_effect = AssertionError("unexpected Triton fallback")
        output_ref = triton.decode(
            q,
            k,
            v,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=state_ref,
            cache_indices=indices,
            query_start_loc=starts,
        )
        output = AiterGDNKernel(
            fallback_kernel=fallback,
            fly_decode=None,
        ).decode(
            q,
            k,
            v,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=state_actual,
            cache_indices=indices,
            query_start_loc=starts,
        )
        torch.testing.assert_close(output, output_ref, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(state_actual, state_ref, rtol=1e-3, atol=1e-3)
        fallback.decode.assert_not_called()

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.version.hip is not None,
        "ROCm is required",
    )
    def test_aiter_prefill_matches_triton_for_397b_tp8(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_aiter import (
            AiterGDNKernel,
        )
        from sglang.srt.layers.attention.linear.kernels.gdn_triton import (
            TritonGDNKernel,
        )

        torch.manual_seed(11)
        tokens, k_heads, v_heads, dim = 65, 2, 8, 128
        q = torch.randn(1, tokens, k_heads, dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn(1, tokens, v_heads, dim, device="cuda", dtype=torch.bfloat16)
        g = -torch.nn.functional.softplus(
            torch.randn(1, tokens, v_heads, device="cuda", dtype=torch.float32)
        )
        beta = torch.sigmoid(torch.randn_like(g))
        indices = torch.tensor([1], device="cuda", dtype=torch.int32)
        starts = torch.tensor([0, 65], device="cuda", dtype=torch.int32)
        state = torch.randn(6, v_heads, dim, dim, device="cuda", dtype=torch.float32)
        state_ref = state.clone()
        state_actual = state.clone()

        triton = TritonGDNKernel()
        fallback = mock.Mock()
        fallback.extend.side_effect = AssertionError("unexpected Triton fallback")
        output_ref, _, _ = triton.extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=state_ref,
            cache_indices=indices,
            query_start_loc=starts,
        )
        with mock.patch(
            "sglang.srt.layers.attention.linear.kernels.gdn_aiter.aiter_prefill_min_tokens",
            return_value=0,
        ):
            output, _, h = AiterGDNKernel(
                fallback_kernel=fallback,
                hip_decode=None,
                fly_decode=None,
            ).extend(
                q,
                k,
                v,
                g,
                beta,
                ssm_states=state_actual,
                cache_indices=indices,
                query_start_loc=starts,
                return_intermediate_h=False,
            )
        self.assertIsNone(h)
        torch.testing.assert_close(output, output_ref, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(state_actual, state_ref, rtol=1e-3, atol=5e-3)
        fallback.extend.assert_not_called()


if __name__ == "__main__":
    unittest.main()
