"""Per-forward-call control context.

Owns ForwardContext — a frozen dataclass holding control configs the model
layer reads at depth via get_forward_context(). The only mandatory field
today is attn_backend; pool refs are derived from attn_backend.*
(every backend caches them at __init__), so a published ForwardContext
is enough to resolve the active pools without a separate global.

ModelRunner._forward_raw publishes a fresh ForwardContext for the
duration of each forward; callers that need a per-call override (PDmux
per-stream backend, frozen-KV MTP draft loop, TBO per-child dispatch) use
dataclasses.replace and wrap the override scope with forward_context().

Distinct from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph.TcPiecewiseForwardContext,
which collects compilation-time refs for the piecewise CUDA graph backend.

Concurrency: _current is a plain module-level global, not thread-local.
This matches the global_server_args precedent and is safe because each
forward runs synchronously on a single Python thread per worker process. If
worker threads ever share a process, migrate to contextvars.ContextVar.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool


@dataclass(frozen=True, slots=True)
class ForwardContext:
    """Per-forward-call control configs. Read via get_forward_context();
    extend by adding fields here. Frozen so accidental mutation raises at
    write time — use dataclasses.replace for per-call overrides."""

    attn_backend: AttentionBackend
    comm_phase: str | None = None


_current: ForwardContext | None = None


def set_forward_context(ctx: ForwardContext | None) -> ForwardContext | None:
    """Set the active context; return the previous one for explicit
    save/restore. Prefer the forward_context() context manager."""
    global _current
    prev, _current = _current, ctx
    return prev


def has_forward_context() -> bool:
    return _current is not None


def get_forward_context() -> ForwardContext:
    assert _current is not None, (
        "no forward context active — call forward_context(...) or set_forward_context(...) "
        "before reading get_forward_context()."
    )
    return _current


def get_attn_backend() -> AttentionBackend:
    return get_forward_context().attn_backend


def get_token_to_kv_pool() -> KVCache:
    return get_attn_backend().token_to_kv_pool


def get_req_to_token_pool() -> ReqToTokenPool:
    return get_attn_backend().req_to_token_pool


def get_communication_phase() -> str | None:
    return _current.comm_phase if _current is not None else None


def communication_phase_for_mode(mode) -> str:
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    if mode in (ForwardMode.EXTEND, ForwardMode.SPLIT_PREFILL):
        return "prefill"
    if mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY, ForwardMode.IDLE):
        return "decode"
    return "unsupported"


@contextmanager
def forward_context(ctx: ForwardContext, *, forward_mode=None):
    if forward_mode is not None:
        ctx = replace(ctx, comm_phase=communication_phase_for_mode(forward_mode))
    elif ctx.comm_phase is None and _current is not None:
        # Backend-only overrides (PDmux, etc.) retain the current phase.
        ctx = replace(ctx, comm_phase=_current.comm_phase)
    prev = set_forward_context(ctx)
    try:
        yield
    finally:
        set_forward_context(prev)
