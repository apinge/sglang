import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.layers.hc_mix_triton import (
    _FUSED_MIX_MAX_ROWS,
    fused_hc_mix,
    fused_hc_mix_supported,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")

HC_COUNT = 4
HIDDEN_SIZE = 2560
LOWRANK = 320
# Mirrors the decode CUDA-graph batch sizes the Qwen3.8 server captures.
DECODE_GRAPH_BUCKETS = (1, 2, 4, 8, 12, 16, 24, 32)


def _reference_mix(
    hyper_input_normed: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    hc: int,
    hs: int,
    compute_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Mirrors GatedResidual._mix_compute in hyperconnection.py."""
    x = hyper_input_normed.to(compute_dtype)
    t = F.silu(F.linear(x, w_down.to(compute_dtype)) / hc)
    u = torch.sigmoid(F.linear(t, w_up.to(compute_dtype)))
    return (u.unflatten(-1, (hc, hs)) * x.unflatten(-1, (hc, hs))).mean(dim=-2)


def _make_inputs(num_tokens: int, dtype: torch.dtype):
    torch.manual_seed(0)
    x = torch.randn(num_tokens, HC_COUNT * HIDDEN_SIZE, dtype=dtype, device="cuda")
    w_down = (
        torch.randn(LOWRANK, HC_COUNT * HIDDEN_SIZE, dtype=dtype, device="cuda") * 0.02
    )
    w_up = (
        torch.randn(HC_COUNT * HIDDEN_SIZE, LOWRANK, dtype=dtype, device="cuda") * 0.02
    )
    return x, w_down, w_up


_TOLERANCES = {
    torch.bfloat16: dict(rtol=1e-2, atol=5e-3),
    torch.float16: dict(rtol=2e-3, atol=1e-3),
}


# The decode CUDA-graph buckets, plus the row counts that sit just off a
# power-of-two boundary (the kernels pad the row tile to one).
_ROW_CASES = [1, 2, 3, 4, 7, 8, 9, 12, 15, 16, 17, 24, 31, _FUSED_MIX_MAX_ROWS]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_tokens", _ROW_CASES)
def test_fused_hc_mix_matches_reference(dtype, num_tokens):
    x, w_down, w_up = _make_inputs(num_tokens, dtype)
    assert fused_hc_mix_supported(x, w_down, w_up)
    out = fused_hc_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
    ref = _reference_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
    torch.testing.assert_close(out.to(torch.float64), ref, **_TOLERANCES[dtype])


def test_fused_hc_mix_no_less_accurate_than_eager():
    """The fused kernel (fp32 accumulation throughout) must not be farther
    from the fp64 reference than the eager bf16 chain it replaces."""
    x, w_down, w_up = _make_inputs(8, torch.bfloat16)
    ref = _reference_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
    fused = fused_hc_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
    eager = _reference_mix(
        x, w_down, w_up, HC_COUNT, HIDDEN_SIZE, compute_dtype=torch.bfloat16
    )
    fused_err = (fused.to(torch.float64) - ref).abs().max()
    eager_err = (eager.to(torch.float64) - ref).abs().max()
    assert fused_err <= eager_err * 1.5 + 1e-6


def test_fused_hc_mix_gate_rejects_prefill_rows():
    x, w_down, w_up = _make_inputs(_FUSED_MIX_MAX_ROWS + 1, torch.bfloat16)
    assert not fused_hc_mix_supported(x, w_down, w_up)


def test_fused_hc_mix_gate_accepts_every_decode_bucket():
    """Every decode CUDA-graph bucket must take the fused path; a row cap below
    the largest bucket silently routes that bucket to the torch.compile chain."""
    for num_tokens in DECODE_GRAPH_BUCKETS:
        x, w_down, w_up = _make_inputs(num_tokens, torch.bfloat16)
        assert fused_hc_mix_supported(x, w_down, w_up), num_tokens


@pytest.mark.parametrize("num_tokens", [1, 5, 16])
def test_fused_hc_mix_untuned_shape(num_tokens):
    """The tuned launch config only covers Qwen3.8's shape on gfx942; an
    unlisted shape falls back to the default config, which nothing else
    exercises."""
    hc, hs, lowrank = 4, 1024, 72
    torch.manual_seed(0)
    x = torch.randn(num_tokens, hc * hs, dtype=torch.bfloat16, device="cuda")
    w_down = torch.randn(lowrank, hc * hs, dtype=torch.bfloat16, device="cuda") * 0.02
    w_up = torch.randn(hc * hs, lowrank, dtype=torch.bfloat16, device="cuda") * 0.02
    assert fused_hc_mix_supported(x, w_down, w_up)
    out = fused_hc_mix(x, w_down, w_up, hc, hs)
    ref = _reference_mix(x, w_down, w_up, hc, hs)
    torch.testing.assert_close(
        out.to(torch.float64), ref, **_TOLERANCES[torch.bfloat16]
    )


@pytest.mark.parametrize("num_tokens", [1, 4, 7, 8, 16, 24, _FUSED_MIX_MAX_ROWS])
def test_fused_hc_mix_graph_replay(num_tokens):
    """A replay must observe the new input: no state may survive a call, and
    the scratch buffers must come from the graph's own pool."""
    x, w_down, w_up = _make_inputs(num_tokens, torch.bfloat16)
    for _ in range(3):
        fused_hc_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fused_hc_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)

    for _ in range(6):
        x.add_(0.03125)
        graph.replay()
        ref = _reference_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
        torch.testing.assert_close(
            out.to(torch.float64), ref, **_TOLERANCES[torch.bfloat16]
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
