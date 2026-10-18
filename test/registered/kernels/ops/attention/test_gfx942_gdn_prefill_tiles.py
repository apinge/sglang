"""GDN tile changes must preserve prefill state and later chunk continuation."""

import itertools

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=20, stage="stage-b", runner_config="1-gpu-small-amd")


@pytest.mark.skipif(torch.version.hip is None, reason="gfx942 tile specialization")
@pytest.mark.parametrize("lengths", [[129], [4096], [12000], [333, 127, 512]])
def test_outputs_checkpoints_and_state_match(lengths):
    if not torch.cuda.get_device_properties().gcnArchName.startswith("gfx942"):
        pytest.skip("gfx942 tile specialization")
    torch.manual_seed(7)
    total = sum(lengths)
    q = F.normalize(
        torch.randn(1, total, 8, 128, device="cuda", dtype=torch.bfloat16), dim=-1
    )
    k = F.normalize(torch.randn_like(q), dim=-1)
    v = torch.randn(1, total, 24, 128, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(1, total, 24, device="cuda") * 0.05
    beta = torch.rand(1, total, 24, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor(
        [0, *itertools.accumulate(lengths)], device="cuda", dtype=torch.int64
    )
    slots = torch.arange(1, len(lengths) + 1, device="cuda", dtype=torch.int64)
    snapshot = torch.randn(len(lengths) + 2, 24, 128, 128, device="cuda") * 0.05
    results = []
    for enabled in [False, True]:
        states = snapshot.clone()
        with envs.SGLANG_GFX942_GDN_PREFILL_TILES.override(
            enabled
        ), envs.SGLANG_GFX942_GDN_CUMSUM.override(False):
            output, _, checkpoints = chunk_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                initial_state=states,
                initial_state_indices=slots,
                cu_seqlens=cu,
            )
            # Reuse the updated pool for a subsequent prefill chunk.
            continued, _, next_checkpoints = chunk_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                initial_state=states,
                initial_state_indices=slots,
                cu_seqlens=cu,
            )
        results.append((output, checkpoints, continued, next_checkpoints, states))
    for baseline, candidate in zip(*results):
        torch.testing.assert_close(candidate, baseline, rtol=0, atol=0)
