"""Tests for the gfx942 Triton skinny bf16 GEMM used by the lm_head."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.utils.common import is_gfx942_supported
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, stage="jit-kernel-unit", runner_config="amd")

pytestmark = pytest.mark.skipif(
    not is_gfx942_supported(), reason="Triton skinny GEMV is tuned for gfx942 only"
)


def _make(m, n, k, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.05
    return x, w


def _assert_close_to_reference(out, x, w):
    ref = x.float() @ w.float().t()
    blas_err = ((x @ w.t()).float() - ref).abs().max().item()
    err = (out.float() - ref).abs().max().item()
    assert out.shape == ref.shape and out.dtype == torch.bfloat16
    assert not torch.isnan(out).any()
    # fp32 accumulation with a bf16 store: no looser than hipBLASLt itself.
    assert err <= max(blas_err * 2.0, 1e-2), (err, blas_err)


# Kernel mechanics with a shipped config (TP8 shard entry): M covers both edges of
# each kernel family (1 | 2..16 | 17..32); N=31041, 4097 and 64 leave a partial
# last N tile for every BLOCK_N; K=512 is a single BLOCK_K=512 step.
@pytest.mark.parametrize("m", [1, 2, 3, 16, 17, 31, 32])
@pytest.mark.parametrize("n,k", [(31041, 2560), (4097, 4096), (64, 512)])
def test_kernel_matches_reference(m, n, k):
    from sglang.kernels.ops.gemm.rocm_bf16_gemv import _get_config, _launch

    x, w = _make(m, n, k)
    _assert_close_to_reference(_launch(x, w, _get_config(m, 31040, 2560)), x, w)


@pytest.mark.parametrize("m", [1, 8, 24])
def test_strided_rows(m):
    """Rows of x and w with a padded leading stride must not be read as packed."""
    from sglang.kernels.ops.gemm.rocm_bf16_gemv import (
        rocm_bf16_gemv,
        use_rocm_bf16_gemv,
    )

    n, k = 31040, 2560
    x_buf, w_buf = _make(m, n, k + 512)
    x, w = x_buf[:, :k], w_buf[:, :k]
    assert use_rocm_bf16_gemv(x, w)
    _assert_close_to_reference(rocm_bf16_gemv(x, w), x, w)


def test_predicate_rejects_unsupported():
    """Only tuned (N, K) shapes are routed; untuned ones measured slower than hipBLASLt."""
    from sglang.kernels.ops.gemm.rocm_bf16_gemv import use_rocm_bf16_gemv

    x, w = _make(32, 31040, 2560)
    assert use_rocm_bf16_gemv(x, w)
    assert not use_rocm_bf16_gemv(x, _make(1, 31041, 2560)[1])  # untuned N
    assert not use_rocm_bf16_gemv(x, _make(1, 4096, 2560)[1])  # small N
    x_k, w_k = _make(4, 31040, 3072)  # untuned K
    assert not use_rocm_bf16_gemv(x_k, w_k)
    assert not use_rocm_bf16_gemv(_make(33, 31040, 2560)[0], w)  # M > 32
    assert not use_rocm_bf16_gemv(x[:0], w)  # M == 0
    assert not use_rocm_bf16_gemv(x.half(), w.half())
    assert not use_rocm_bf16_gemv(x.float(), w)
    assert not use_rocm_bf16_gemv(x, w.t().contiguous().t())  # K not unit-stride


@pytest.mark.parametrize("m", [1, 4, 32])
def test_cuda_graph_replay(m):
    """Decode runs the lm_head inside a captured graph: launches must be capturable."""
    from sglang.kernels.ops.gemm.rocm_bf16_gemv import rocm_bf16_gemv

    n, k = 31040, 2560
    x, w = _make(m, n, k)
    rocm_bf16_gemv(x, w)  # compile outside capture, as CUDA graph warmup does
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = rocm_bf16_gemv(x, w)
    x.copy_(_make(m, n, k, seed=1)[0])
    graph.replay()
    torch.cuda.synchronize()
    _assert_close_to_reference(out, x, w)


def _shipped_buckets():
    from sglang.kernels.ops.gemm.rocm_bf16_gemv import _GFX942_CONFIGS

    cases = []
    for shape, buckets in _GFX942_CONFIGS.items():
        bounds = sorted(buckets)
        lower = [1] + [b + 1 for b in bounds[:-1]]
        for lo, hi in zip(lower, bounds):
            for m in sorted({lo, hi}):
                cases.append(pytest.param(shape, m, id=f"{shape}-M{m}"))
    return cases


@pytest.mark.parametrize("shape,m", _shipped_buckets())
def test_shipped_configs_match_reference(shape, m):
    """Every tuned entry, at both edges of its M bucket and at its own N."""
    from sglang.kernels.ops.gemm.rocm_bf16_gemv import rocm_bf16_gemv

    n, k = shape
    x, w = _make(m, n, k)
    _assert_close_to_reference(rocm_bf16_gemv(x, w), x, w)


def test_configs_cover_all_supported_m():
    """An entry whose largest bucket is below the predicate's M limit fails at runtime."""
    from sglang.kernels.ops.gemm.rocm_bf16_gemv import _GFX942_CONFIGS, _MAX_M

    for shape, buckets in _GFX942_CONFIGS.items():
        assert min(buckets) == 1 and max(buckets) >= _MAX_M, (shape, sorted(buckets))


class _LMHeadStub(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)


def _make_logits_processor(vocab_size):
    from sglang.srt.layers.logits_processor import LogitsProcessor
    from sglang.srt.runtime_context import get_context

    override = get_context().override_server_args(enable_fp32_lm_head=False)
    override.install()
    cfg = SimpleNamespace(vocab_size=vocab_size, final_logit_softcapping=None)
    try:
        return LogitsProcessor(cfg, skip_all_gather=True, logit_scale=None)
    finally:
        override.restore()


@pytest.mark.parametrize("enabled", [True, False])
def test_lm_head_routing(enabled):
    """The lm_head uses the Triton kernel on gfx942 and the env var opts out."""
    import sglang.srt.layers.logits_processor as lp
    from sglang.srt.environ import envs

    n, k = 31040, 2560
    x, w = _make(4, n, k)
    with envs.SGLANG_ENABLE_ROCM_LM_HEAD_GEMV.override(enabled):
        processor = _make_logits_processor(n)
    with patch.object(lp, "rocm_bf16_gemv", wraps=lp.rocm_bf16_gemv) as spy:
        logits = processor._compute_lm_head(x, _LMHeadStub(w))
        # Prefill with input logprobs can exceed the M bucket; must fall back.
        processor._compute_lm_head(_make(33, n, k)[0], _LMHeadStub(w))
    assert spy.call_count == (1 if enabled else 0)
    _assert_close_to_reference(logits, x, w)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
