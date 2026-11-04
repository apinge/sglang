"""Sequence-parallel hyper-connection residual stream for TP prefill.

The hyper-connection ops between the TP-sharded blocks (combine, hc_norm, mix)
are per token, so with a replicated residual stream every TP rank repeats them on
all tokens. With a plan, each rank keeps only its slice of the token rows of the
residual stream: a block's partial output is reduce-scattered instead of
all-reduced, combined and mixed on the local rows, and the mixed block input is
all-gathered. The reduce-scatter plus all-gather move the same bytes as the
all-reduce they replace.
"""

from typing import Optional

import msgspec
import torch

from sglang.srt.distributed.parallel_state import get_attn_tp_group
from sglang.srt.utils import is_hip


class SeqParallelPlan(msgspec.Struct, frozen=True):
    num_tokens: int
    # Rows per rank; the last rank's slice is zero-padded past num_tokens.
    local_rows: int
    rank: int
    world_size: int

    def scatter(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's rows of a replicated [num_tokens, ...] tensor."""
        r0 = self.rank * self.local_rows
        local = x[r0 : r0 + self.local_rows]
        if local.shape[0] < self.local_rows:
            local = torch.nn.functional.pad(
                local, (0, 0, 0, self.local_rows - local.shape[0])
            )
        return local.contiguous()

    def reduce_scatter(self, partial: torch.Tensor) -> torch.Tensor:
        """This rank's rows of the sum over ranks of [num_tokens, hidden] partials."""
        padded = self.local_rows * self.world_size
        if partial.shape[0] < padded:
            partial = torch.nn.functional.pad(
                partial, (0, 0, 0, padded - partial.shape[0])
            )
        partial = partial.contiguous()
        out = partial.new_empty(self.local_rows, partial.shape[1])
        group = get_attn_tp_group()
        if not _aiter_reduce_scatter(group.ca_comm, partial, out):
            group.reduce_scatter_tensor(out, partial)
        return out

    def all_gather(self, local: torch.Tensor) -> torch.Tensor:
        """[num_tokens, ...] from every rank's local rows."""
        out = local.new_empty(self.local_rows * self.world_size, *local.shape[1:])
        get_attn_tp_group().all_gather_into_tensor(out, local.contiguous())
        return out[: self.num_tokens]


def _aiter_reduce_scatter(ca_comm, partial: torch.Tensor, out: torch.Tensor) -> bool:
    """Equal-chunk reduce-scatter on aiter's custom kernel when it takes this size.

    `GroupCoordinator.reduce_scatter_tensor` only uses it under
    SGLANG_DP_USE_REDUCE_SCATTER and otherwise falls back to RCCL.
    """
    if not is_hip() or ca_comm is None or ca_comm.disabled:
        return False
    try:
        from aiter.dist.device_communicators.custom_all_reduce import (
            CustomAllreduce as AiterCustomAllreduce,
        )
    except ImportError:
        return False

    if not isinstance(ca_comm, AiterCustomAllreduce) or not ca_comm.should_custom_ar(
        partial
    ):
        return False
    ca_comm.reduce_scatter(partial, out)
    return True


def plan_seq_parallel(
    *, num_tokens: int, rank: int, world_size: int, min_tokens: int
) -> Optional[SeqParallelPlan]:
    if world_size == 1 or num_tokens < min_tokens:
        return None
    return SeqParallelPlan(
        num_tokens=num_tokens,
        local_rows=-(-num_tokens // world_size),
        rank=rank,
        world_size=world_size,
    )
