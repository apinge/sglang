"""ReplaySSM must commit accepted PLE states before its early return.

Execute the source commit/scatter functions in isolation so CPU CI does not
need speculative GPU kernels. GDN kernels are stubbed; PLE state writes and
accepted/tracking step calculations execute the real function bodies.
"""

import ast
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4] / "python" / "sglang" / "srt"
nodes = []
for filename, names in (
    (
        "speculative/spec_utils.py",
        {
            "_verify_commit_step_indices",
            "_commit_ple_states_after_replayssm_verify",
            "commit_mamba_states_after_verify",
        },
    ),
    (
        "layers/attention/hybrid_linear_attn_backend.py",
        {
            "_update_ple_state_after_mtp_verify",
            "_scatter_speculative_state_with_mask",
        },
    ),
):
    for node in ast.walk(ast.parse((ROOT / filename).read_text())):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            nodes.append(node)
namespace = {
    "torch": torch,
    "mambaish_config": lambda _: True,
    "mamba_track_grid": lambda _: 4,
}
module = ast.Module(
    body=[
        ast.ImportFrom(
            module="__future__",
            names=[ast.alias(name="annotations")],
            level=0,
        )
    ]
    + nodes,
    type_ignores=[],
)
exec(compile(ast.fix_missing_locations(module), "commit_regression", "exec"), namespace)


@pytest.mark.parametrize("mode", ["baseline", "compact", "fold"])
@pytest.mark.parametrize("accepted", [1, 2, 3, 4])
@pytest.mark.parametrize("track", [False, True])
def test_accepted_and_tracking_states(mode, accepted, track):
    calls = []
    source_context = torch.tensor([[[200 + 10 * t, 201 + 10 * t] for t in range(4)]])
    source_conv = torch.arange(4).reshape(1, 1, 4, 1).float() + 100
    pool = NS(
        ngram_pool=NS(
            context=torch.full((5, 2), -1), intermediate_context=source_context
        ),
        short_conv_pool=NS(
            conv_state=torch.full((1, 5, 1), -1.0), intermediate_conv_state=source_conv
        ),
    )
    # Use deliberately different virtual/physical ids to catch omitted translation.
    pool.translate_mamba_indices = lambda ids: torch.tensor(
        [{7: 1, 9: 3, -1: -1}[int(i)] for i in ids]
    )
    pool.get_mamba_indices = lambda rows: torch.tensor([7])
    pool.mamba_pool = NS(
        replayssm_spec_fold=mode == "fold",
        replayssm_is_kda=False,
        replayssm_cache_base=torch.tensor([0]) if mode == "compact" else None,
        replayssm_spec_write_pos=torch.tensor([0]),
        replayssm_is_flush=torch.tensor([0], dtype=torch.int8),
    )
    spec_state = NS(
        temporal=torch.zeros(1),
        conv=[torch.zeros(1)],
        intermediate_conv_window=[torch.zeros(1)],
        replayssm_d=torch.zeros((1, 1, 1, 16, 1)),
        replayssm_k=None,
        replayssm_g=None,
        replayssm_rawv=None,
        replayssm_rawk=None,
    )
    pool.get_speculative_mamba2_params_all_layers = lambda: spec_state

    class Backend:
        _scatter_speculative_state_with_mask = staticmethod(
            namespace["_scatter_speculative_state_with_mask"]
        )

        def _update_ple_state_after_mtp_verify(self, *args):
            calls.append("ple")
            namespace["_update_ple_state_after_mtp_verify"](self, *args)

        def update_mamba_state_after_mtp_verify(self, **kwargs):
            self._update_ple_state_after_mtp_verify(
                torch.tensor([1]),
                kwargs["last_correct_step_indices"],
                torch.tensor([3]) if track else None,
                kwargs["mamba_steps_to_track"],
            )

    backend = Backend()
    backend.linear_attn_backend = NS(req_to_token_pool=pool)
    runner = NS(
        model_config=True, req_to_token_pool=pool, attn_backend=backend, model=None
    )
    batch = NS(
        forward_mode=NS(is_idle=lambda: False),
        req_pool_indices=torch.tensor([0]),
        mamba_track_indices=torch.tensor([9]) if track else None,
        seq_lens=torch.tensor([2]),
        tree_cache=NS(page_size=4),
    )
    compact = ModuleType("sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode")
    compact.commit_gdn_replayssm_spec = lambda **kw: None
    compact.commit_gdn_replayssm_circular = lambda **kw: None
    fold = ModuleType("sglang.kernels.ops.attention.fla.gdn_replayssm_spec_fold")
    fold.commit_gdn_replayssm_fold_after_verify = lambda **kw: None
    scatter = ModuleType("sglang.kernels.ops.mamba.mamba_state_scatter_triton")
    scatter.fused_conv_window_scatter_with_mask = lambda *args: None
    with patch.dict("sys.modules", {m.__name__: m for m in [compact, fold, scatter]}):
        namespace["commit_mamba_states_after_verify"](
            NS(model_runner=runner),
            batch,
            torch.tensor([accepted]),
            torch.tensor([[0, 1, 2, 3]]),
            4,
        )
    assert calls == ["ple"]
    assert torch.equal(pool.ngram_pool.context[1], source_context[0, accepted - 1])
    assert torch.equal(
        pool.short_conv_pool.conv_state[:, 1], source_conv[:, 0, accepted - 1]
    )
    if track and accepted >= 2:
        assert torch.equal(pool.ngram_pool.context[3], source_context[0, 1])
        assert torch.equal(pool.short_conv_pool.conv_state[:, 3], source_conv[:, 0, 1])
    else:
        assert torch.all(pool.ngram_pool.context[3] == -1)
        assert torch.all(pool.short_conv_pool.conv_state[:, 3] == -1)
    assert torch.all(pool.ngram_pool.context[[0, 2, 4]] == -1)


def test_backend_without_ple_is_noop():
    namespace["_commit_ple_states_after_replayssm_verify"](
        NS(attn_backend=object()),
        torch.tensor([7]),
        torch.tensor([0]),
        None,
        None,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
