"""Correctness test for the buffered output-only linear-attention decode kernel
(ReplaySSM Part A), implemented in
``python/sglang/srt/layers/attention/fla/fused_recurrent_linear_replayssm.py``.

One kernel, two gate granularities (selected by ``is_kda``):
  * GDN (``is_kda=False``): per-head SCALAR gate. Ground truth =
    ``fused_recurrent_gated_delta_rule_packed_decode`` (fused_recurrent.py:186).
  * KDA (``is_kda=True``):  per-K-channel gate. Ground truth =
    ``fused_recurrent_kda_packed_decode`` (fused_recurrent.py:406). The KDA
    baseline takes the per-K gate flattened (``a``=[B, HV*K], ``dt_bias``=[HV*K]);
    the ReplaySSM kernel takes it as [B, HV, K] / [HV, K]. Same memory layout,
    so we build [B, HV, K] / [HV, K] and reshape for the baseline.

We run a multi-step decode through BOTH kernels and assert:
  * per-step outputs ``y`` match, AND
  * the ground-truth full state matches the ReplaySSM checkpoint *after a flush*
    (folding the ring buffer into S0). At a non-flush step the ReplaySSM
    checkpoint legitimately lags (the update lives in the ring), so we only
    compare state at flush boundaries / after a forced flush.

L sweep:
  * L=1 (flush every step): the ring is always empty and ``write_pos == L-1``
    every step, so the reconstruction term is zero, the total decay is 1, and
    the kernel reduces to the packed decode. For GDN the only fp difference is a
    scalar ``alpha*(sum x)`` vs ``sum(alpha*x)`` reorder, so we assert ~bit-exact
    (atol 2e-6). For KDA the analogous reorder is per-K (``alpha[k]`` folded into
    q/k vs into S0), bounded by ~K*eps, so KDA uses the same (tighter-than-L>1)
    tensor-core tolerances rather than 2e-6.
  * L in {4,8,16}: algebraically equivalent but floating-point reordered (cumsum
    of gates, replay-decay scaling, the d@k reconstruction dot on tensor cores
    vs. the materialized recurrence). Tight fp tolerance, see ``_tols``.

Runnable as ``pytest`` and as ``__main__``.
"""

import unittest

import torch

# Mirror sibling GDN unittests: register for CUDA/AMD CI. This is a kernel-math
# unit test; it lives with the other linear-attention kernel correctness tests.
# The registry calls MUST be module-level (the CI collector / check-registered-
# tests hook parses them statically via AST and only scans top-level statements
# -- a try/except wrapper hides them and fails the hook).
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-large-amd")


def _build_inputs(B, H, HV, K, V, dtype, device, seed=0, is_kda=False):
    """Random packed-decode inputs for one step.

    Returns a packed ``mixed_qkv`` [B, 2*H*K + HV*V] plus the per-row gate
    inputs. ``b`` (beta) is [B, HV] for both gate types. ``a`` is [B, HV] for
    the GDN scalar gate and [B, HV, K] for the KDA per-K gate.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    qk_dim = 2 * H * K
    v_dim = HV * V
    mixed_qkv = torch.randn(B, qk_dim + v_dim, generator=g, device=device, dtype=dtype)
    # Keep gate input `a` modest so softplus / decay stay in a stable range.
    if is_kda:
        a = (
            torch.randn(B, HV, K, generator=g, device=device, dtype=dtype) * 0.5
        ).contiguous()
    else:
        a = torch.randn(B, HV, generator=g, device=device, dtype=dtype) * 0.5
    b = torch.randn(B, HV, generator=g, device=device, dtype=dtype)
    return mixed_qkv, a, b


def _tols(dtype):
    # L>1 reconstructs the state via a tensor-core tl.dot (TF32 for fp32 inputs,
    # bf16 tensor cores for bf16) -- the performance-critical path that keeps the
    # L-deep reconstruction hidden under the reduced memory traffic. TF32 carries
    # ~4e-4 error, bf16 ~1e-3, both benign end-to-end (ReplaySSM GSM8K parity);
    # vLLM's own ReplaySSM test uses fp32 atol 1e-4. GDN L=1 has an empty buffer
    # (dot == 0) and is asserted bit-exact separately.
    if dtype == torch.float32:
        return dict(atol=1e-4, rtol=1e-3)
    return dict(atol=1e-3, rtol=1e-2)  # bf16


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestLinearReplaySSMDecode(CustomTestCase):
    # Small config: HV=8 or 16 heads, K=V=128, batch B=4.
    CONFIGS = (
        dict(B=4, H=4, HV=8, K=128, V=128),
        dict(B=4, H=8, HV=16, K=128, V=128),
    )
    NUM_STEPS = 40  # > 16 -> multiple flush cycles for L in {4,8,16}
    L_SWEEP = (1, 4, 8, 16)

    def _run_one(
        self,
        cfg,
        L,
        dtype,
        force_flush_steps=(),
        is_kda=False,
        ring_dtype=None,
        nk=None,
        token_stride=1,
    ):
        from sglang.kernels.ops.attention.fla.fused_recurrent import (
            fused_recurrent_gated_delta_rule_packed_decode,
            fused_recurrent_kda_packed_decode,
        )
        from sglang.kernels.ops.attention.fla.fused_recurrent_linear_replayssm import (
            fused_recurrent_linear_replayssm_decode,
        )

        device = "cuda"
        B, H, HV, K, V = (cfg[x] for x in ("B", "H", "HV", "K", "V"))
        scale = K**-0.5

        # Shared static params across steps. A_log is per-head [HV] for both
        # gate types; dt_bias is per-head [HV] (GDN) or per-K [HV, K] (KDA).
        A_log = (torch.randn(HV, device=device, dtype=torch.float32) * 0.3).contiguous()
        if is_kda:
            dt_bias = (
                torch.randn(HV, K, device=device, dtype=torch.float32) * 0.1
            ).contiguous()
        else:
            dt_bias = (
                torch.randn(HV, device=device, dtype=torch.float32) * 0.1
            ).contiguous()

        # Two independent state pools (same init), one per kernel. One slot/req.
        num_slots = B
        # Seed the state init so the marginal bf16 flush-state element is
        # reproducible run-to-run (otherwise the pass/fail at the tolerance
        # boundary flickers with the global RNG).
        sg = torch.Generator(device=device).manual_seed(2024)
        ref_state = torch.randn(
            num_slots, HV, V, K, generator=sg, device=device, dtype=torch.float32
        )
        rep_state = ref_state.clone()
        cache_indices = torch.arange(B, device=device, dtype=torch.int32)

        # ReplaySSM ring tensors (caller-owned; this increment's caller == test).
        # g_cache is per-step scalar [HV, L] for GDN, per-K [HV, L, K] for KDA.
        ring_dtype = ring_dtype or dtype
        d_cache = torch.zeros(num_slots, HV, L, V, device=device, dtype=ring_dtype)
        k_cache = torch.zeros(num_slots, H, L, K, device=device, dtype=ring_dtype)
        if is_kda:
            g_cache = torch.zeros(
                num_slots, HV, L, K, device=device, dtype=torch.float32
            )
        else:
            g_cache = torch.zeros(num_slots, HV, L, device=device, dtype=torch.float32)
        write_pos = torch.zeros(B, device=device, dtype=torch.int32)

        tols = _tols(dtype)
        flush_step_seen = False

        for step in range(self.NUM_STEPS):
            mixed_qkv, a, b = _build_inputs(
                B * token_stride,
                H,
                HV,
                K,
                V,
                dtype,
                device,
                seed=1000 + step,
                is_kda=is_kda,
            )
            mixed_qkv, a, b = [x[::token_stride] for x in (mixed_qkv, a, b)]

            # ---- ground truth: packed decode, full-state write every step ----
            ref_out = mixed_qkv.new_empty(B, 1, HV, V)
            if is_kda:
                # KDA baseline takes the per-K gate flattened: a=[B, HV*K],
                # dt_bias=[HV*K] (same memory as our [B, HV, K] / [HV, K]).
                fused_recurrent_kda_packed_decode(
                    mixed_qkv=mixed_qkv,
                    a=a.reshape(B, HV * K).contiguous(),
                    b=b,
                    A_log=A_log,
                    dt_bias=dt_bias.reshape(HV * K).contiguous(),
                    scale=scale,
                    initial_state=ref_state,
                    out=ref_out,
                    ssm_state_indices=cache_indices,
                    use_qk_l2norm_in_kernel=True,
                )
            else:
                fused_recurrent_gated_delta_rule_packed_decode(
                    mixed_qkv=mixed_qkv,
                    a=a,
                    b=b,
                    A_log=A_log,
                    dt_bias=dt_bias,
                    scale=scale,
                    initial_state=ref_state,
                    out=ref_out,
                    ssm_state_indices=cache_indices,
                    use_qk_l2norm_in_kernel=True,
                )

            # ---- ReplaySSM buffered decode for this L ----
            rep_out = mixed_qkv.new_empty(B, 1, HV, V)
            # slice 2: force a flush at a radix track boundary even mid-buffer.
            force_now = step in force_flush_steps
            ff_tensor = (
                torch.ones(B, device=device, dtype=torch.int32) if force_now else None
            )
            is_flush = bool((write_pos[0].item()) == L - 1) or force_now
            flush_step_seen = flush_step_seen or is_flush
            fused_recurrent_linear_replayssm_decode(
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=scale,
                initial_state=rep_state,
                d_cache=d_cache,
                k_cache=k_cache,
                g_cache=g_cache,
                out=rep_out,
                ssm_state_indices=cache_indices,
                write_pos=write_pos,
                force_flush=ff_tensor,
                use_qk_l2norm_in_kernel=True,
                is_kda=is_kda,
                # nk=1 at L=1 keeps the K-reduction in a single tile (no
                # K-tiling), matching the reference's single-pass reduction;
                # nk=2 for L>1 exercises the K-tiled reconstruction path.
                nk=nk if nk is not None else (1 if L == 1 else 2),
            )

            # Output must match every step.
            if dtype == torch.float32 and L == 1 and not is_kda:
                # GDN L=1: essentially bit-exact -- only an alpha*(sum) vs
                # sum(alpha*) scalar-multiply reordering separates this from the
                # reference. (KDA L=1 has the analogous per-K reorder, bounded by
                # ~K*eps, so it uses the regular tensor-core tolerance below.)
                torch.testing.assert_close(
                    rep_out,
                    ref_out,
                    atol=2e-6,
                    rtol=1e-5,
                    msg=f"GDN L=1 step={step} out",
                )
            else:
                torch.testing.assert_close(
                    rep_out,
                    ref_out,
                    msg=f"is_kda={is_kda} L={L} step={step} out",
                    **tols,
                )

            # On a flush step the ReplaySSM checkpoint is brought fully
            # up-to-date and the ring logically clears -> compare states.
            if is_flush:
                if dtype == torch.float32 and L == 1 and not is_kda:
                    torch.testing.assert_close(
                        rep_state,
                        ref_state,
                        atol=2e-6,
                        rtol=1e-5,
                        msg=f"GDN L=1 step={step} state",
                    )
                else:
                    # The flush state is reconstructed via the tensor-core dot,
                    # so it carries tensor-core precision: ~tf32 (~4e-4, measured
                    # max_abs ~7e-4) for fp32, ~bf16 (measured max_abs ~1.9e-3)
                    # for bf16 -- looser than the per-step OUTPUT, which passes at
                    # the tighter output tol because the tf32/bf16 error partly
                    # cancels in the q-reduction. The per-step decay (alpha<1)
                    # attenuates old reconstruction error so it does not
                    # accumulate across flushes over a long decode; e2e accuracy
                    # is unaffected (ReplaySSM GSM8K parity).
                    state_tols = (
                        dict(atol=1.5e-3, rtol=1e-3)
                        if dtype == torch.float32
                        else dict(atol=3e-3, rtol=1e-2)
                    )
                    torch.testing.assert_close(
                        rep_state,
                        ref_state,
                        msg=f"is_kda={is_kda} L={L} step={step} state(flush)",
                        **state_tols,
                    )

            # Advance the ring cursor (caller's responsibility in this phase):
            # wrap to 0 after a flush (natural L-1 OR a forced track-boundary
            # flush), otherwise increment.
            if force_now:
                write_pos = torch.zeros_like(write_pos)
            else:
                write_pos = torch.where(
                    write_pos == L - 1,
                    torch.zeros_like(write_pos),
                    write_pos + 1,
                )

        # Explicit wrap/flush boundary must have been exercised for L>1; for
        # L=1 every step is a flush.
        self.assertTrue(flush_step_seen, f"flush boundary never hit for L={L}")

    # ---- GDN (scalar gate) ----
    def test_replayssm_matches_gdn_packed_decode_fp32(self):
        for cfg in self.CONFIGS:
            for L in self.L_SWEEP:
                with self.subTest(cfg=cfg, L=L, dtype="fp32"):
                    self._run_one(cfg, L, torch.float32)

    def test_replayssm_matches_gdn_packed_decode_bf16(self):
        for cfg in self.CONFIGS:
            for L in self.L_SWEEP:
                with self.subTest(cfg=cfg, L=L, dtype="bf16"):
                    self._run_one(cfg, L, torch.bfloat16)

    def test_qwen38_tp_decode_and_tracking(self):
        """Qwen3.8 TP2/TP4: HV/H=3, BF16 inputs, FP32 checkpoint AND ring.

        Exercise natural and forced flushes with the actual production pool
        dtype, which differs from the BF16 ring in the generic BF16 cases.
        """
        for H, HV in ((8, 24), (4, 12)):
            for B in (1, 3, 32):
                for L in (1, 8, 16):
                    with self.subTest(H=H, HV=HV, B=B, L=L):
                        torch.manual_seed(2026)
                        self._run_one(
                            dict(B=B, H=H, HV=HV, K=128, V=128),
                            L=L,
                            dtype=torch.bfloat16,
                            ring_dtype=torch.float32,
                            nk=2,
                        )
                with self.subTest(H=H, HV=HV, B=B, force_flush=True):
                    torch.manual_seed(2026)
                    self._run_one(
                        dict(B=B, H=H, HV=HV, K=128, V=128),
                        L=16,
                        dtype=torch.bfloat16,
                        ring_dtype=torch.float32,
                        force_flush_steps=(3, 5, 23),
                        nk=2,
                        token_stride=2,
                    )

    def test_qwen38_restore_flushed_checkpoint_in_graph(self):
        """A restored radix snapshot ignores the destination's stale ring.

        Both requests receive identical inputs; after a forced checkpoint and
        COW into a different physical slot they must continue identically under
        graph replay, including subsequent natural flushes.
        """
        from types import SimpleNamespace

        from sglang.kernels.ops.attention.fla.fused_recurrent_linear_replayssm import (
            fused_recurrent_gdn_replayssm_decode,
        )
        from sglang.srt.mem_cache.memory_pool import MambaPool

        for H, HV in ((8, 24), (4, 12)):
            with self.subTest(H=H, HV=HV):
                torch.manual_seed(2026)
                B, L, K, V = 2, 16, 128, 128
                state = torch.randn(3, HV, V, K, device="cuda")
                d = torch.randn(3, HV, L, V, device="cuda")
                k = torch.randn(3, H, L, K, device="cuda")
                g = torch.randn(3, HV, L, device="cuda")
                indices = torch.tensor([0, 2], device="cuda", dtype=torch.int32)
                cursor = torch.zeros(B, device="cuda", dtype=torch.int32)
                force = torch.zeros_like(cursor)
                packed, a, b = _build_inputs(1, H, HV, K, V, torch.bfloat16, "cuda")
                packed, a, b = [x.repeat(B, 1) for x in (packed, a, b)]
                A_log = torch.randn(HV, device="cuda") * 0.3
                dt_bias = torch.randn(HV, device="cuda") * 0.1
                out = torch.empty(B, 1, HV, V, device="cuda", dtype=torch.bfloat16)

                def forward():
                    fused_recurrent_gdn_replayssm_decode(
                        mixed_qkv=packed,
                        a=a,
                        b=b,
                        A_log=A_log,
                        dt_bias=dt_bias,
                        scale=K**-0.5,
                        initial_state=state,
                        d_cache=d,
                        k_cache=k,
                        g_cache=g,
                        out=out,
                        ssm_state_indices=indices,
                        write_pos=cursor,
                        force_flush=force,
                        use_qk_l2norm_in_kernel=True,
                    )

                # Materialize slot 0 at a non-periodic tracking boundary.
                for step in range(5):
                    cursor.fill_(step)
                    force.fill_(int(step == 4))
                    forward()
                pool = object.__new__(MambaPool)
                pool.debug_memory_pool = True
                pool._conv_fuse_ok = False
                pool.replayssm_write_pos = torch.tensor([0, 11, 5], device="cuda")
                pool.replayssm_cache_base = pool.replayssm_is_flush = None
                pool.mamba_cache = SimpleNamespace(temporal=state.unsqueeze(0), conv=[])
                pool.copy_from(
                    torch.tensor([0], device="cuda"), torch.tensor([1], device="cuda")
                )
                self.assertEqual(pool.replayssm_write_pos[1].item(), 0)
                indices.copy_(torch.tensor([0, 1], device="cuda", dtype=torch.int32))
                cursor.zero_()
                force.zero_()
                # Warm/capture only with a flush, then reset to the same state.
                snapshot = state.clone()
                force.fill_(1)
                forward()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    forward()
                state.copy_(snapshot)
                force.zero_()
                for step in range(40):
                    cursor.fill_(step % L)
                    graph.replay()
                    torch.testing.assert_close(out[0], out[1], rtol=0, atol=0)
                    if step % L == L - 1:
                        torch.testing.assert_close(state[0], state[1], rtol=0, atol=0)

    # ---- KDA (per-K gate) ----
    def test_replayssm_matches_kda_packed_decode_fp32(self):
        for cfg in self.CONFIGS:
            for L in self.L_SWEEP:
                with self.subTest(cfg=cfg, L=L, dtype="fp32"):
                    self._run_one(cfg, L, torch.float32, is_kda=True)

    def test_replayssm_matches_kda_packed_decode_bf16(self):
        for cfg in self.CONFIGS:
            for L in self.L_SWEEP:
                with self.subTest(cfg=cfg, L=L, dtype="bf16"):
                    self._run_one(cfg, L, torch.bfloat16, is_kda=True)

    # ---- forced (radix track-boundary) flush, both gate types ----
    def test_force_flush_matches_reference(self):
        """A radix track-boundary forces a flush mid-buffer (write_pos != L-1).
        The kernel must fold the partial ring + current token into the
        checkpoint so an external snapshot reads an up-to-date state; output
        every step and the forced-flush checkpoint must still match the running
        recurrence -- for both the GDN scalar gate and the KDA per-K gate."""
        cfg = dict(B=4, H=8, HV=16, K=128, V=128)
        for is_kda in (False, True):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(is_kda=is_kda, dtype=str(dtype)):
                    self._run_one(
                        cfg,
                        L=8,
                        dtype=dtype,
                        force_flush_steps=(3, 5, 11, 20),
                        is_kda=is_kda,
                    )

    def test_flush_boundary_state_exact_l1(self):
        """At L=1 (GDN) every step is a flush; the kernel is algebraically the
        packed decode. Each step reproduces the reference output AND the full
        state to atol 2e-6 / rtol 1e-5 (essentially bit-exact) in fp32. This also
        exercises the ``write_pos == L-1`` flush boundary on every step."""
        self._run_one(dict(B=4, H=4, HV=8, K=128, V=128), L=1, dtype=torch.float32)


if __name__ == "__main__":
    unittest.main()
