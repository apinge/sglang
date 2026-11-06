"""Verify tiles preserve cached states, accepted-state continuation and graphs."""

import pytest
import torch

from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, stage="stage-b", runner_config="1-gpu-small-amd")


@pytest.mark.skipif(torch.version.hip is None, reason="gfx942 verify tiles")
@pytest.mark.parametrize(
    "lengths", [[4], [4] * 2, [4] * 8, [4] * 16, [4] * 32, [1, 4, 2, 3]]
)
def test_verify_states_accepted_continuation_and_replay(lengths):
    if not torch.cuda.get_device_properties().gcnArchName.startswith("gfx942"):
        pytest.skip("gfx942 verify tiles")
    torch.manual_seed(56)
    n, total, heads, value_heads, dim = len(lengths), sum(lengths), 8, 24, 128
    q = torch.randn(1, total, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, total, value_heads, dim, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(total, value_heads, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    log = torch.randn(value_heads, device="cuda")
    bias = torch.randn(value_heads, device="cuda")
    states = torch.randn(n + 2, value_heads, dim, dim, device="cuda") * 0.01
    slots = torch.arange(1, n + 1, device="cuda", dtype=torch.int32)
    cu = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    cache = torch.empty(n + 2, 4, value_heads, dim, dim, device="cuda")

    def verify(enabled):
        with envs.SGLANG_GFX942_GDN_VERIFY_TILES.override(enabled):
            return fused_sigmoid_gating_delta_rule_update(
                log,
                a,
                bias,
                1.0,
                20.0,
                q,
                k,
                v,
                b,
                states,
                slots,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu,
                disable_state_update=True,
                intermediate_states_buffer=cache,
                intermediate_state_indices=slots,
                cache_steps=4,
            )

    expected = verify(False).clone()
    baseline_cache = cache.clone()
    snapshot = states.clone()
    actual = verify(True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(states, snapshot, rtol=0, atol=0)
    for slot, length in zip(slots, lengths):
        torch.testing.assert_close(
            cache[slot, :length], baseline_cache[slot, :length], rtol=0, atol=0
        )

    # Commit different accepted prefix lengths and decode one subsequent token.
    accepted = torch.tensor([max(0, length - 2) for length in lengths], device="cuda")
    after = []
    for source in [baseline_cache, cache]:
        committed = snapshot.clone()
        committed[slots] = source[slots, accepted]
        result = fused_sigmoid_gating_delta_rule_update(
            log,
            a[:n],
            bias,
            1.0,
            20.0,
            q[:, :n],
            k[:, :n],
            v[:, :n],
            b[:n],
            committed,
            slots,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=torch.arange(n + 1, device="cuda", dtype=torch.int32),
        )
        after.append((result, committed))
    for baseline, candidate in zip(*after):
        torch.testing.assert_close(candidate, baseline, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = verify(True)
    q.mul_(0.5)
    graph.replay()
    torch.testing.assert_close(captured, verify(False), rtol=0, atol=0)
