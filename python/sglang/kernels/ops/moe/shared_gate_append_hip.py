"""Native HIP shared-gate GEMV, sigmoid and top-k append for MI308X."""

import torch

from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def _shared_gate_append_module():
    return load_jit(
        "shared_gate_append_hip",
        cuda_files=["moe/shared_gate_append_hip.cuh"],
        cuda_wrappers=[("run", "SharedGateAppendHip::run")],
        extra_cuda_cflags=["-ffp-contract=off"],
    )


def can_use_shared_gate_append_hip(hidden, gate, ids, weights, shared):
    return (
        torch.version.hip is not None
        and hidden.is_cuda
        and hidden.ndim == 2
        and hidden.shape[1] == 4096
        and ids.ndim == weights.ndim == 2
        and gate.shape == (1, 4096)
        and hidden.dtype == gate.dtype == torch.bfloat16
        and ids.dtype == torch.int32
        and weights.dtype == torch.float32
        and ids.shape == weights.shape
        and ids.shape[0] == hidden.shape[0]
        and 0 < ids.shape[1] <= 256
        and 0 < shared <= 4
        and all(t.is_contiguous() for t in (hidden, gate, ids, weights))
        and hidden.data_ptr() % 16 == gate.data_ptr() % 16 == 0
        and all(t.device == hidden.device for t in (gate, ids, weights))
    )


def shared_gate_append_hip(hidden, gate, ids, weights, expert_base, shared, scale=1.0):
    assert can_use_shared_gate_append_hip(hidden, gate, ids, weights, shared)
    m, k = ids.shape
    out_ids = torch.empty((m, k + shared), dtype=ids.dtype, device=ids.device)
    out_weights = torch.empty(
        (m, k + shared), dtype=weights.dtype, device=weights.device
    )
    _shared_gate_append_module().run(
        hidden, gate, ids, weights, out_ids, out_weights, expert_base, shared, scale
    )
    return out_ids, out_weights
