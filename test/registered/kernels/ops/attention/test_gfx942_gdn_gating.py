"""Row-packed GDN gating preserves g and the BF16 sigmoid boundary exactly."""

import pytest
import torch

from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, stage="stage-b", runner_config="1-gpu-small-amd")


@pytest.mark.skipif(torch.version.hip is None, reason="gfx942 GDN gating")
@pytest.mark.parametrize("rows", [1, 129, 12000])
def test_gdn_gating_strided_and_graph(rows):
    if not torch.cuda.get_device_properties().gcnArchName.startswith("gfx942"):
        pytest.skip("gfx942 GDN gating")
    torch.manual_seed(68)
    a = torch.randn(rows, 48, device="cuda", dtype=torch.bfloat16)[:, 24:]
    b = torch.randn(rows, 48, device="cuda", dtype=torch.bfloat16)[:, :24]
    bias = torch.randn(24, device="cuda")
    log = torch.randn(24, device="cuda")

    def run(enabled):
        with envs.SGLANG_GFX942_GDN_GATING.override(enabled):
            return fused_gdn_gating(log, a, b, bias)

    for actual, expected in zip(run(True), run(False)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(True)
    a.mul_(0.5)
    b.mul_(0.5)
    graph.replay()
    for actual, expected in zip(captured, run(False)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
