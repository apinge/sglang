"""QKV split specialization preserves strided sources and graph replay."""

import pytest
import torch

from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    fused_qkv_split_gdn_prefill,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, stage="stage-b", runner_config="1-gpu-small-amd")


@pytest.mark.skipif(torch.version.hip is None, reason="gfx942 GDN QKV split")
@pytest.mark.parametrize("rows", [0, 32, 12000])
@pytest.mark.parametrize("strided", [False, True])
def test_qkv_split_exact_and_graph(rows, strided):
    if not torch.cuda.get_device_properties().gcnArchName.startswith("gfx942"):
        pytest.skip("gfx942 GDN QKV split")
    torch.manual_seed(65)
    source = torch.randn(
        rows, 5120 * (2 if strided else 1), device="cuda", dtype=torch.bfloat16
    )
    x = source[:, ::2] if strided else source

    def run():
        with envs.SGLANG_GFX942_GDN_QKV_SPLIT.override(True):
            return fused_qkv_split_gdn_prefill(x, 8, 8, 24, 128, 128, 128)

    def check(result):
        for actual, expected in zip(result, x.split([1024, 1024, 3072], dim=1)):
            torch.testing.assert_close(
                actual.reshape(rows, expected.shape[1]), expected, rtol=0, atol=0
            )

    check(run())
    if rows:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        x.mul_(0.5)
        graph.replay()
        check(captured)
