"""Qwen3.5 MoE TP2 prefill: every layer all-reduce over SDMA, pipelined in row chunks.

The Qwen4-Exp sequence-parallel SDMA overlap (layers/sdma_reduce_scatter.py,
layers/sdma_mori.py) applied to Qwen3.5's plain residual stream. At TP2 an
all-reduce is "push the whole partial to the peer, then add", which moves the same
bytes per rank as reduce-scatter + all-gather, so the residual stays replicated and
nothing outside the decoder stack changes:

    out_proj / o_proj per row chunk -> push the chunk to the peer (SDMA)
    per chunk as the peer's rows land: rank 0 rows + rank 1 rows (same bits on both)
        -> residual add + RMSNorm -> MoE (partial, no all-reduce) -> push
    next layer: per landed chunk -> residual add + input RMSNorm

The copies run on the SDMA engines while the next chunk's GEMM / MoE runs on the
compute units, so most of the prefill all-reduce time leaves the critical path.
Shares the Qwen4 switches: SGLANG_QWEN4_SP_SDMA_RS (on by default),
SGLANG_QWEN4_SEQ_PARALLEL_MIN_TOKENS and SGLANG_QWEN4_SP_SDMA_RS_CHUNKS. Batches
that are not eager TP2 EXTEND batches of at least that many tokens keep the
all-reduce path unchanged.
"""

from __future__ import annotations

import logging
from typing import Iterator, List, Optional, Tuple

import torch
from torch import nn

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Measured on MI308X TP2 at 12k-token prefill: 4 chunks beat 2 and 8 by 5-12%.
_DEFAULT_CHUNKS = 4
_MOE_MODEL_TYPE = "qwen3_5_moe_text"


class _Plan:
    """Row chunks of one pipelined forward. `pending` hands the in-flight
    projection all-reduce from the patched out_proj / o_proj to the layer loop."""

    __slots__ = ("num_tokens", "bounds", "pending")

    def __init__(self, *, num_tokens: int, chunks: int):
        cuts = [num_tokens * i // chunks for i in range(chunks + 1)]
        self.num_tokens = num_tokens
        self.bounds: List[Tuple[int, int]] = [
            (a, b) for a, b in zip(cuts, cuts[1:]) if b > a
        ]
        self.pending: Optional[_PendingAllReduce] = None


class _PendingAllReduce:
    """This rank's partial rows are being pushed chunk by chunk; `chunks()` waits
    for each peer chunk and yields (r0, r1, summed rows)."""

    def __init__(self, *, ag, writer, plan: _Plan):
        self._ag = ag
        self._writer = writer
        self._plan = plan

    def chunks(self) -> Iterator[Tuple[int, int, torch.Tensor]]:
        num_tokens = self._plan.num_tokens
        buf = self._ag._buf[self._writer._slot, : 2 * num_tokens]
        first = self._writer._expected - len(self._plan.bounds)
        peer = 1 - self._ag._rank
        for i, (r0, r1) in enumerate(self._plan.bounds):
            self._ag._mori.wait(expected=first + i + 1, only=peer)
            # Rank 0 rows + rank 1 rows on both ranks keeps the replicas bit-identical.
            yield r0, r1, buf[r0:r1] + buf[num_tokens + r0 : num_tokens + r1]


class Qwen35SdmaAllReduce:
    """Owned by a Qwen3.5 MoE model whose decoder stack can take the SDMA path."""

    def __init__(self, *, hidden_size: int):
        self._hidden_size = hidden_size
        self._chunks = envs.SGLANG_QWEN4_SP_SDMA_RS_CHUNKS.get() or _DEFAULT_CHUNKS
        self._min_tokens = envs.SGLANG_QWEN4_SEQ_PARALLEL_MIN_TOKENS.get()
        self._ag = None
        self._usable: Optional[bool] = None
        self._active: Optional[_Plan] = None

    @staticmethod
    def maybe_create(config) -> Optional[Qwen35SdmaAllReduce]:
        from sglang.srt.runtime_context import get_parallel
        from sglang.srt.utils import is_hip

        par = get_parallel()
        if not (
            is_hip()
            and envs.SGLANG_QWEN4_SP_SDMA_RS.get()
            and config.model_type == _MOE_MODEL_TYPE
            and par.tp_size == 2
            and par.attn_tp_size == 2
        ):
            return None
        return Qwen35SdmaAllReduce(hidden_size=config.hidden_size)

    def wrap_projection(self, proj: nn.Module) -> None:
        """Under an active plan, the row-parallel projection runs per row chunk and
        pushes each chunk's partial; the layer loop reads it from `plan.pending`."""
        plain_forward = proj.forward

        def forward(x: torch.Tensor):
            plan = self._active
            if plan is None or x.dim() != 2 or x.shape[0] != plan.num_tokens:
                return plain_forward(x)
            writer = self._begin(plan)
            for r0, r1 in plan.bounds:
                writer.push(r0, r1, plain_forward(x[r0:r1])[0])
            plan.pending = _PendingAllReduce(ag=self._ag, writer=writer, plan=plan)
            # Unread: the caller only returns it up to the layer loop.
            return x.new_empty(x.shape[0], proj.output_size), None

        proj.forward = forward

    def eligible(self, *, model, forward_batch, num_tokens: int, deepstack) -> bool:
        from sglang.srt.layers.moe import get_moe_a2a_backend
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.model_executor.runner import get_is_capture_mode
        from sglang.srt.runtime_context import get_schedule

        if (
            forward_batch.forward_mode != ForwardMode.EXTEND
            or get_is_capture_mode()
            or torch.compiler.is_compiling()
            or num_tokens < self._min_tokens
            or num_tokens > get_schedule().max_prefill_tokens
            or not get_moe_a2a_backend().is_none()
            or not (model.pp_group.is_first_rank and model.pp_group.is_last_rank)
            or model.layers_to_capture
            or (deepstack is not None and deepstack.numel() > 0)
        ):
            return False
        if self._usable is None:
            self._usable = self._init_sdma()
        return self._usable

    def _init_sdma(self) -> bool:
        """Collective over the TP group: both ranks reach it on the same forward."""
        from sglang.srt.distributed.parallel_state import get_attn_tp_group
        from sglang.srt.layers.sdma_mori import sdma_available, sdma_group_usable
        from sglang.srt.layers.sdma_reduce_scatter import SdmaAllGather
        from sglang.srt.runtime_context import get_schedule

        cpu_group = get_attn_tp_group().cpu_group
        device = torch.device("cuda", torch.cuda.current_device())
        if not (sdma_available() and sdma_group_usable(cpu_group, device)):
            return False
        # Row r of rank k's partial lands at row k * T + r of every rank's buffer.
        self._ag = SdmaAllGather(
            capacity_tokens=2 * get_schedule().max_prefill_tokens,
            hidden_size=self._hidden_size,
            dtype=torch.bfloat16,
            device=device,
            cpu_group=cpu_group,
        )
        logger.info(
            "Qwen3.5 TP2 prefill: all-reduces over SDMA in %d row chunks "
            "(batches of at least %d tokens)",
            self._chunks,
            self._min_tokens,
        )
        return True

    def _begin(self, plan: _Plan):
        return self._ag.begin(
            num_tokens=2 * plan.num_tokens,
            local_rows=plan.num_tokens,
            pushes=len(plan.bounds),
        )

    def forward_layers(self, *, model, positions, hidden_states, forward_batch):
        """The decoder stack and the final norm with pipelined SDMA all-reduces.

        Double-buffer safety: a rank pushes all-reduce n only after it received all
        of n - 1, which its peer posts after it finished reading n - 2 (same slot)."""
        from sglang.srt.eplb.expert_distribution import (
            get_global_expert_distribution_recorder,
        )

        plan = _Plan(num_tokens=hidden_states.shape[0], chunks=self._chunks)
        residual = hidden_states.clone()
        x = model.layers[model.start_layer].input_layernorm(hidden_states)
        self._active = plan
        try:
            for layer_idx in range(model.start_layer, model.end_layer):
                layer = model.layers[layer_idx]
                with get_global_expert_distribution_recorder().with_current_layer(
                    layer_idx
                ):
                    attn_ar = self._attention(layer, plan, x, positions, forward_batch)
                    mlp_ar = self._moe(layer, plan, attn_ar, residual, forward_batch)
                next_norm = (
                    model.layers[layer_idx + 1].input_layernorm
                    if layer_idx + 1 < model.end_layer
                    else model.norm
                )
                x = torch.empty_like(residual)
                for r0, r1, summed in mlp_ar.chunks():
                    _add_norm_rows(next_norm, summed, residual, x, rows=slice(r0, r1))
        finally:
            self._active = None
        return x

    def _attention(self, layer, plan, x, positions, forward_batch) -> _PendingAllReduce:
        from .qwen3_5 import Qwen3_5LinearDecoderLayer

        plan.pending = None
        if isinstance(layer, Qwen3_5LinearDecoderLayer):
            layer.linear_attn(x, forward_batch)
        else:
            layer.self_attention(
                positions=positions, hidden_states=x, forward_batch=forward_batch
            )
        if plan.pending is None:
            raise RuntimeError("Qwen3.5 SDMA all-reduce: the projection did not push")
        return plan.pending

    def _moe(self, layer, plan, attn_ar, residual, forward_batch) -> _PendingAllReduce:
        from sglang.srt.runtime_context import get_forward

        normed = torch.empty_like(residual)
        writer = self._begin(plan)
        # The MoE returns its partial sum; the next chunks' pushes do the all-reduce.
        with get_forward().scoped(mlp_reduce_scatter=True):
            for r0, r1, summed in attn_ar.chunks():
                rows = slice(r0, r1)
                _add_norm_rows(
                    layer.post_attention_layernorm, summed, residual, normed, rows=rows
                )
                writer.push(r0, r1, layer.mlp(normed[rows], forward_batch))
        return _PendingAllReduce(ag=self._ag, writer=writer, plan=plan)


def _add_norm_rows(norm, summed, residual, out, *, rows: slice) -> None:
    """residual[rows] += summed; out[rows] = norm(residual[rows])."""
    res = residual[rows]
    normed, new_res = norm(summed, res)
    if new_res.data_ptr() != res.data_ptr():
        res.copy_(new_res)
    out[rows].copy_(normed)
