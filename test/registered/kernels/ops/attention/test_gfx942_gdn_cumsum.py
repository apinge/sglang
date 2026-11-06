"""Cumsum launch tuning is independent of the state/output tile selection."""

import itertools

import pytest
import torch

from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, stage="stage-b", runner_config="1-gpu-small-amd")


@pytest.mark.skipif(torch.version.hip is None, reason="gfx942 cumsum tuning")
@pytest.mark.parametrize("lengths", [[1], [129], [12000], [333, 127, 512]])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("scale", [None, 0.5])
def test_cumsum_exact_and_graph_independent_of_prefill_tiles(lengths, reverse, scale):
    if not torch.cuda.get_device_properties().gcnArchName.startswith("gfx942"):
        pytest.skip("gfx942 cumsum tuning")
    torch.manual_seed(73)
    source = torch.randn(1, sum(lengths), 24, device="cuda") * 0.05
    cu = torch.tensor(
        [0, *itertools.accumulate(lengths)], device="cuda", dtype=torch.int64
    )

    def run(enabled):
        with envs.SGLANG_GFX942_GDN_CUMSUM.override(
            enabled
        ), envs.SGLANG_GFX942_GDN_PREFILL_TILES.override(False):
            return chunk_local_cumsum(
                source, chunk_size=64, reverse=reverse, scale=scale, cu_seqlens=cu
            )

    expected = run(False)
    torch.testing.assert_close(run(True), expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(True)
    source.mul_(0.5)
    graph.replay()
    torch.testing.assert_close(captured, run(False), rtol=0, atol=0)


@pytest.mark.skipif(torch.version.hip is None, reason="gfx942 cumsum tuning")
@pytest.mark.parametrize(
    "heads,chunk_size,head_first", [(8, 64, False), (24, 32, False), (24, 64, True)]
)
def test_cumsum_unsupported_layout_and_shape_fallback(heads, chunk_size, head_first):
    if not torch.cuda.get_device_properties().gcnArchName.startswith("gfx942"):
        pytest.skip("gfx942 cumsum tuning")
    shape = (1, heads, 129) if head_first else (1, 129, heads)
    source = torch.randn(shape, device="cuda")
    outputs = []
    for enabled in (False, True):
        with envs.SGLANG_GFX942_GDN_CUMSUM.override(enabled):
            outputs.append(
                chunk_local_cumsum(source, chunk_size=chunk_size, head_first=head_first)
            )
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
