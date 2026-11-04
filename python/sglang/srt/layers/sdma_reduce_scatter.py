"""Reduce-scatter over SDMA for the sequence-parallel prefill path (ROCm).

Rank r owns rows [r * L, (r + 1) * L) of the padded [N * L, hidden] partial sums.
Each rank produces the rows in row chunks, the blocks it does not own first and
its own block last. After computing a chunk of block b it pushes it with an SDMA
copy into rank b's IPC-shared receive buffer and then bumps a flag in rank b's
memory, both on a per-destination copy stream that only waits on local events.
The owner waits on the flags (hipStreamWaitValue32) on its compute stream right
before it sums the chunk; a wait on a side stream could block the compute
stream when HIP maps both onto one hardware queue.

With an `fp8` transport each pushed row is float8 e4m3fnuz values plus one bf16
scale per 32 (symmetric absmax); only peers' rows are quantized.

Flags carry a sequence number that grows by one per call, so they are never
reset. Receive buffers are double-buffered: callers must run a collective over
the group (the all-gather of the sequence-parallel layer) between two sends, so
that a rank pushes call i + 2 only after the owner consumed call i.
"""

import ctypes
import functools
import logging
from typing import Callable, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_HIP_IPC_LAZY_PEER = 0x1
# hipMemcpyDeviceToDeviceNoCU: force the SDMA engine instead of a blit kernel.
_HIP_MEMCPY_D2D_NO_CU = 1024
_HIP_STREAM_WAIT_GEQ = 0x0
_MAX_CHUNKS = 64
_TRANSPORTS = ("bf16", "fp8")
_QUANT_GROUP = 32
_GROUPS_PER_PROGRAM = 16


class _HipIpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


def _load_hip():
    hip = ctypes.CDLL("libamdhip64.so")
    hip.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(_HipIpcHandle), ctypes.c_void_p]
    hip.hipIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        _HipIpcHandle,
        ctypes.c_uint,
    ]
    hip.hipMemGetAddressRange.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
    ]
    hip.hipMemcpyAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    hip.hipStreamWaitValue32.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint,
        ctypes.c_uint32,
    ]
    return hip


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} failed with hipError {rc}")


class _RawPtr:
    """Pointer argument for Triton when there is no torch tensor to pass."""

    def __init__(self, ptr: int, dtype: torch.dtype):
        self._ptr = ptr
        self.dtype = dtype

    def data_ptr(self) -> int:
        return self._ptr


@triton.jit
def _signal_peer(flag_ptr, value):
    # System-scope release: the rows written by the preceding kernel are visible
    # to the peer's SDMA read once the peer observes the flag.
    tl.atomic_xchg(flag_ptr, value, sem="release", scope="sys")


@triton.jit
def _quantize_rows(
    x_ptr,
    x_row_stride,
    rec_ptr,
    rec_bytes,
    HIDDEN: tl.constexpr,
    GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
):
    # Record layout per row: HIDDEN fp8 values, then HIDDEN // GROUP bf16 scales.
    row = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1) * GROUPS + tl.arange(0, GROUPS)
    cols = group[:, None] * GROUP + tl.arange(0, GROUP)[None, :]
    x = tl.load(x_ptr + row * x_row_stride + cols).to(tl.float32)
    # 240 is the largest float8 e4m3fnuz value; quantize with the rounded bf16 scale.
    scale = (tl.max(tl.abs(x), axis=1) / 240.0).to(tl.bfloat16).to(tl.float32)
    inv = tl.where(scale > 0, 1.0 / scale, 0.0)
    v = tl.minimum(tl.maximum(x * inv[:, None], -240.0), 240.0)
    rec = rec_ptr + row * rec_bytes
    tl.store(rec + cols, v.to(tl.float8e4b8).to(tl.int8, bitcast=True))
    scale_ptr = (rec + HIDDEN).to(tl.pointer_type(tl.bfloat16))
    tl.store(scale_ptr + group, scale.to(tl.bfloat16))


@triton.jit
def _reduce_rows(
    local_ptr,
    local_row_stride,
    rec_ptr,
    rec_bytes,
    peer_stride,
    out_ptr,
    out_row_stride,
    PEERS: tl.constexpr,
    HIDDEN: tl.constexpr,
    GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    FP8: tl.constexpr,
):
    # out = local + sum of PEERS received rows (fp8 records or bf16), in fp32.
    row = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1) * GROUPS + tl.arange(0, GROUPS)
    cols = group[:, None] * GROUP + tl.arange(0, GROUP)[None, :]
    acc = tl.load(local_ptr + row * local_row_stride + cols).to(tl.float32)
    for p in tl.static_range(PEERS):
        rec = rec_ptr + p * peer_stride + row * rec_bytes
        if FP8:
            q = tl.load(rec + cols).to(tl.float8e4b8, bitcast=True).to(tl.float32)
            scale_ptr = (rec + HIDDEN).to(tl.pointer_type(tl.bfloat16))
            acc += q * tl.load(scale_ptr + group).to(tl.float32)[:, None]
        else:
            acc += tl.load(rec.to(tl.pointer_type(tl.bfloat16)) + cols).to(tl.float32)
    tl.store(out_ptr + row * out_row_stride + cols, acc.to(tl.bfloat16))


class PendingReduceScatter:
    """Reduce-scatter in flight; every chunk must be consumed before the next send."""

    def __init__(
        self,
        *,
        carrier: torch.Tensor,
        local_chunks: List[torch.Tensor],
        recv_records: torch.Tensor,
        rec_bytes: int,
        fp8: bool,
        bounds: List[int],
        wait_landed: Callable[[int], None],
    ):
        # [num_tokens, hidden] view used only to carry this handle through the model.
        self.carrier = carrier
        self._local_chunks = local_chunks
        self._recv_records = recv_records
        self._rec_bytes = rec_bytes
        self._fp8 = fp8
        self._bounds = bounds
        self._wait_landed = wait_landed

    @property
    def bounds(self) -> List[int]:
        """Chunk boundaries in this rank's local rows, [0, ..., local_rows]."""
        return self._bounds

    def chunks(self) -> Iterator[Tuple[int, int, torch.Tensor]]:
        """Yield (r0, r1, rows) in this rank's local row coordinates."""
        for j, local in enumerate(self._local_chunks):
            self._wait_landed(j)
            r0, r1 = self._bounds[j], self._bounds[j + 1]
            out = torch.empty_like(local)
            if r1 > r0:
                self._reduce_into(local, r0, out)
            yield r0, r1, out

    def result(self) -> torch.Tensor:
        return torch.cat([rows for _, _, rows in self.chunks()])

    def _reduce_into(self, local: torch.Tensor, r0: int, out: torch.Tensor) -> None:
        hidden = local.shape[1]
        _reduce_rows[(local.shape[0], hidden // (_QUANT_GROUP * _GROUPS_PER_PROGRAM))](
            local,
            local.stride(0),
            self._recv_records[:, r0 * self._rec_bytes :],
            self._rec_bytes,
            self._recv_records.stride(0),
            out,
            out.stride(0),
            PEERS=self._recv_records.shape[0],
            HIDDEN=hidden,
            GROUP=_QUANT_GROUP,
            GROUPS=_GROUPS_PER_PROGRAM,
            FP8=self._fp8,
        )


class SdmaReduceScatter:
    def __init__(
        self,
        *,
        capacity_tokens: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
        cpu_group: dist.ProcessGroup,
        transport: str = "bf16",
        backend: str = "hip",
    ):
        if transport not in _TRANSPORTS:
            raise ValueError(
                f"SDMA transport must be one of {_TRANSPORTS}, got {transport!r}"
            )
        if hidden_size % (_QUANT_GROUP * _GROUPS_PER_PROGRAM) != 0:
            raise ValueError(f"hidden size {hidden_size} is not a multiple of 512")
        self._hip = _load_hip()
        self._world = dist.get_world_size(cpu_group)
        self._rank = dist.get_rank(cpu_group)
        self._local_capacity = -(-capacity_tokens // self._world)
        self._row_bytes = hidden_size * torch.empty(0, dtype=dtype).element_size()
        self._fp8 = transport == "fp8" and dtype == torch.bfloat16
        self._rec_bytes = (
            hidden_size + hidden_size // _QUANT_GROUP * 2
            if self._fp8
            else self._row_bytes
        )
        # [slot, source (rank - owner - 1) mod N, row, hidden]; fp8 records are
        # smaller than bf16 rows, so they reuse these buffers.
        self._recv = torch.empty(
            2,
            self._world - 1,
            self._local_capacity,
            hidden_size,
            dtype=dtype,
            device=device,
        )
        # Staging for this rank's quantized rows, [slot, N * L, hidden].
        self._staging = (
            torch.empty(
                2,
                self._world * self._local_capacity,
                hidden_size,
                dtype=dtype,
                device=device,
            )
            if self._fp8
            else None
        )
        # [source rank, chunk of this rank's block]
        self._flags = torch.zeros(
            self._world, _MAX_CHUNKS, dtype=torch.int32, device=device
        )
        torch.cuda.synchronize(device)

        exported = [None] * self._world
        dist.all_gather_object(
            exported,
            (self._export(self._recv), self._export(self._flags)),
            group=cpu_group,
        )
        others = [p for p in range(self._world) if p != self._rank]
        self._peer_recv = {p: self._import(exported[p][0]) for p in others}
        self._peer_flags = {p: self._import(exported[p][1]) for p in others}
        self._seq = 0
        self._slot = 0
        self._mori = None
        if backend == "mori":
            from sglang.srt.layers.sdma_mori import get_mori_hub

            self._mori = get_mori_hub(cpu_group, device)
            self._copy_streams = {}
        else:
            self._copy_streams = {p: torch.cuda.Stream(device) for p in others}
        # [slot]: bf16 sources the copy engines may still read; a send two calls
        # later runs after the group collective, so every peer has received them.
        self._mori_sources: List[List[torch.Tensor]] = [[], []]

    def fits(self, local_rows: int) -> bool:
        return local_rows <= self._local_capacity

    def send(
        self,
        *,
        rows: Callable[[int, int], torch.Tensor],
        num_tokens: int,
        local_rows: int,
        chunks: int,
    ) -> PendingReduceScatter:
        """Reduce-scatter `rows(r0, r1)` (this rank's partial sums of rows
        [r0, r1) < num_tokens) over the padded [world * local_rows] rows."""
        if not 1 <= chunks <= _MAX_CHUNKS:
            raise ValueError(f"chunks must be in [1, {_MAX_CHUNKS}], got {chunks}")
        comp = torch.cuda.current_stream()
        for stream in self._copy_streams.values():
            comp.wait_stream(stream)
        self._seq += 1
        self._mori_sources[self._slot] = []
        bounds = [local_rows * j // chunks for j in range(chunks + 1)]
        local_chunks: List[Optional[torch.Tensor]] = [None] * chunks
        for i in range(1, self._world + 1):
            block = (self._rank + i) % self._world
            for j in range(chunks):
                r0 = block * local_rows + bounds[j]
                r1 = block * local_rows + bounds[j + 1]
                x = self._padded_rows(rows, r0, r1, num_tokens)
                if block == self._rank:
                    local_chunks[j] = x
                else:
                    self._push(x=x, owner=block, r0=bounds[j], row0=r0, chunk=j)
        slot = self._slot
        self._slot ^= 1
        recv = self._recv[slot]
        wait_landed = self._wait_landed
        if self._mori is not None:
            wait_landed = functools.partial(
                self._mori_wait_landed, received_before=self._mori.received
            )
            self._mori.received += chunks
        return PendingReduceScatter(
            carrier=self._recv.view(-1, self._recv.shape[-1])[:num_tokens],
            local_chunks=local_chunks,
            recv_records=recv.view(self._world - 1, -1).view(torch.int8),
            rec_bytes=self._rec_bytes,
            fp8=self._fp8,
            bounds=bounds,
            wait_landed=wait_landed,
        )

    def _push(self, *, x: torch.Tensor, owner: int, r0: int, row0: int, chunk: int):
        """Push rows [r0, r0 + len(x)) of owner's block, then flag them."""
        comp = torch.cuda.current_stream()
        if self._fp8:
            records = (
                self._staging[self._slot]
                .view(torch.int8)
                .reshape(-1)[row0 * self._rec_bytes :]
            )
            self._quantize(x, records)
            src = records.data_ptr()
        else:
            x = x.contiguous()
            src = x.data_ptr()
        source_index = (self._rank - owner - 1) % self._world
        dst = (
            self._peer_recv[owner]
            + ((self._slot * (self._world - 1) + source_index) * self._local_capacity)
            * self._row_bytes
            + r0 * self._rec_bytes
        )
        if self._mori is not None:
            if not self._fp8:
                self._mori_sources[self._slot].append(x)
            self._mori.push_one(
                peer=owner,
                dst_ptr=dst,
                src_ptr=src,
                nbytes=x.shape[0] * self._rec_bytes,
            )
            return
        stream = self._copy_streams[owner]
        stream.wait_stream(comp)
        x.record_stream(stream)
        _check(
            self._hip.hipMemcpyAsync(
                dst,
                src,
                x.shape[0] * self._rec_bytes,
                _HIP_MEMCPY_D2D_NO_CU,
                stream.cuda_stream,
            ),
            "hipMemcpyAsync",
        )
        with torch.cuda.stream(stream):
            _signal_peer[(1,)](
                _RawPtr(
                    self._peer_flags[owner] + 4 * (self._rank * _MAX_CHUNKS + chunk),
                    torch.int32,
                ),
                self._seq,
            )

    def _mori_wait_landed(self, chunk: int, *, received_before: int) -> None:
        # Each peer pushes chunks in order on one queue, so chunk j has landed
        # once that peer's counter reaches received_before + j + 1.
        self._mori.wait(expected=received_before + chunk + 1)

    def _wait_landed(self, chunk: int) -> None:
        stream = torch.cuda.current_stream().cuda_stream
        for peer in self._peer_recv:
            _check(
                self._hip.hipStreamWaitValue32(
                    stream,
                    self._flags[peer, chunk].data_ptr(),
                    self._seq,
                    _HIP_STREAM_WAIT_GEQ,
                    0xFFFFFFFF,
                ),
                "hipStreamWaitValue32",
            )

    def _padded_rows(self, rows, r0, r1, num_tokens) -> torch.Tensor:
        if r1 <= num_tokens:
            return rows(r0, r1)
        out = self._recv.new_zeros(r1 - r0, self._recv.shape[-1])
        if r0 < num_tokens:
            out[: num_tokens - r0] = rows(r0, num_tokens)
        return out

    def _quantize(self, x: torch.Tensor, records: torch.Tensor) -> None:
        hidden = x.shape[1]
        _quantize_rows[(x.shape[0], hidden // (_QUANT_GROUP * _GROUPS_PER_PROGRAM))](
            x,
            x.stride(0),
            records,
            self._rec_bytes,
            HIDDEN=hidden,
            GROUP=_QUANT_GROUP,
            GROUPS=_GROUPS_PER_PROGRAM,
        )

    def _export(self, t: torch.Tensor):
        # The caching allocator hands out sub-ranges; IPC handles cover whole allocations.
        base, size = ctypes.c_void_p(), ctypes.c_size_t()
        _check(
            self._hip.hipMemGetAddressRange(
                ctypes.byref(base), ctypes.byref(size), t.data_ptr()
            ),
            "hipMemGetAddressRange",
        )
        handle = _HipIpcHandle()
        _check(
            self._hip.hipIpcGetMemHandle(ctypes.byref(handle), base),
            "hipIpcGetMemHandle",
        )
        return ctypes.string_at(ctypes.addressof(handle), 64), t.data_ptr() - base.value

    def _import(self, exported) -> int:
        # Opened on the local device; mapping on the owning device faults here.
        raw, offset = exported
        handle = _HipIpcHandle()
        ctypes.memmove(ctypes.addressof(handle), raw, 64)
        ptr = ctypes.c_void_p()
        _check(
            self._hip.hipIpcOpenMemHandle(
                ctypes.byref(ptr), handle, _HIP_IPC_LAZY_PEER
            ),
            "hipIpcOpenMemHandle",
        )
        return ptr.value + offset


_BACKEND: Optional[str] = None


def resolve_sdma_backend() -> str:
    """SGLANG_QWEN4_SP_SDMA_BACKEND, falling back to "hip" (with one warning) when
    mori cannot run in this process."""
    global _BACKEND
    if _BACKEND is None:
        wanted = envs.SGLANG_QWEN4_SP_SDMA_BACKEND.get()
        if wanted not in ("mori", "hip"):
            raise ValueError(
                f"SGLANG_QWEN4_SP_SDMA_BACKEND must be mori or hip, got {wanted!r}"
            )
        _BACKEND = wanted
        if wanted == "mori":
            from sglang.srt.layers.sdma_mori import mori_unavailable_reason

            reason = mori_unavailable_reason()
            if reason is not None:
                logger.warning(
                    "SDMA backend mori unavailable (%s); using hip copy streams", reason
                )
                _BACKEND = "hip"
    return _BACKEND


_INSTANCE: Optional[SdmaReduceScatter] = None


def get_sdma_reduce_scatter(
    *,
    capacity_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    cpu_group: dist.ProcessGroup,
    transport: str,
) -> SdmaReduceScatter:
    """Process-wide instance; the first call is collective over the group."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = SdmaReduceScatter(
            capacity_tokens=capacity_tokens,
            hidden_size=hidden_size,
            dtype=dtype,
            device=torch.device("cuda", torch.cuda.current_device()),
            cpu_group=cpu_group,
            transport=transport,
            backend=resolve_sdma_backend(),
        )
    return _INSTANCE


class PendingAllGather:
    """All-gather in flight: this rank's rows are local, peers' rows land in
    `gathered` once their flags are set. `wait_block(src)` on the consuming
    stream before reading block src; after waiting every block, `gathered` is
    the complete [num_tokens, hidden] tensor."""

    def __init__(
        self,
        *,
        local: torch.Tensor,
        gathered: torch.Tensor,
        rank: int,
        world_size: int,
        local_rows: int,
        wait_block: Callable[[int], None],
    ):
        self.local = local
        self.gathered = gathered
        self.rank = rank
        self.world_size = world_size
        self.local_rows = local_rows
        self._wait_block = wait_block
        self._waited = False

    def blocks(self) -> Iterator[Tuple[int, int, torch.Tensor]]:
        """Yield (r0, r1, rows): this rank's rows first, then each peer's block
        once it has landed (rows past num_tokens are dropped)."""
        num_tokens = self.gathered.shape[0]
        for i in range(self.world_size):
            src = (self.rank + i) % self.world_size
            r0 = src * self.local_rows
            r1 = min(r0 + self.local_rows, num_tokens)
            if i > 0:
                self._wait_block(src)
            if r1 > r0:
                yield r0, r1, (
                    self.local[: r1 - r0] if i == 0 else self.gathered[r0:r1]
                )
        self._waited = True

    def wait_all(self) -> torch.Tensor:
        if not self._waited:
            for src in range(self.world_size):
                if src != self.rank:
                    self._wait_block(src)
            self._waited = True
        return self.gathered


class SdmaAllGather:
    """All-gather over SDMA for the sequence-parallel layout (ROCm).

    Each rank copies its rows into its own slot and pushes them with SDMA into
    every peer's IPC-shared buffer at its row offset, then bumps a flag in the
    peer's memory, all on per-destination copy streams that only wait on local
    events. Consumers wait on the flags on their compute stream block by block
    (`PendingAllGather.blocks()`), so the next projection can start on the
    local rows while peers' rows are in flight.

    Buffers are double-buffered under the same contract as SdmaReduceScatter:
    a group collective runs between two sends.
    """

    def __init__(
        self,
        *,
        capacity_tokens: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
        cpu_group: dist.ProcessGroup,
        backend: str = "hip",
    ):
        self._hip = _load_hip()
        self._world = dist.get_world_size(cpu_group)
        self._rank = dist.get_rank(cpu_group)
        self._local_capacity = -(-capacity_tokens // self._world)
        self._buf = torch.empty(
            2,
            self._world * self._local_capacity,
            hidden_size,
            dtype=dtype,
            device=device,
        )
        self._row_bytes = hidden_size * self._buf.element_size()
        # [source rank]
        self._flags = torch.zeros(self._world, dtype=torch.int32, device=device)
        torch.cuda.synchronize(device)

        exported = [None] * self._world
        dist.all_gather_object(
            exported,
            (self._export(self._buf), self._export(self._flags)),
            group=cpu_group,
        )
        others = [p for p in range(self._world) if p != self._rank]
        self._peer_buf = {p: self._import(exported[p][0]) for p in others}
        self._peer_flags = {p: self._import(exported[p][1]) for p in others}
        self._seq = 0
        self._slot = 0
        self._mori = None
        if backend == "mori":
            from sglang.srt.layers.sdma_mori import get_mori_hub

            self._mori = get_mori_hub(cpu_group, device)
            slot_bytes = self._world * self._local_capacity * self._row_bytes
            # [slot, peer]: base address of the peer's receive slot.
            self._mori_tables = torch.tensor(
                [
                    [
                        self._peer_buf.get(p, 0) + slot * slot_bytes
                        for p in range(self._world)
                    ]
                    for slot in range(2)
                ],
                dtype=torch.int64,
                device=device,
            )
            self._copy_streams = {}
        else:
            self._copy_streams = {p: torch.cuda.Stream(device) for p in others}

    _export = SdmaReduceScatter._export
    _import = SdmaReduceScatter._import

    def fits(self, local_rows: int) -> bool:
        return local_rows <= self._local_capacity

    def send(
        self, *, local: torch.Tensor, num_tokens: int, local_rows: int
    ) -> PendingAllGather:
        """Start gathering `local` ([local_rows, hidden], this rank's rows)."""
        comp = torch.cuda.current_stream()
        for stream in self._copy_streams.values():
            comp.wait_stream(stream)
        self._seq += 1
        slot = self._slot
        self._slot ^= 1
        local = local.contiguous()
        buf = self._buf[slot, : self._world * local_rows]
        r0 = self._rank * local_rows
        buf[r0 : r0 + local_rows].copy_(local)
        wait_block = self._wait_block
        if self._mori is not None:
            # Push from the own slot, which stays untouched until the call after next.
            self._mori.push_all(
                dst_table=self._mori_tables[slot],
                src_ptr=buf[r0].data_ptr(),
                dst_off=r0 * self._row_bytes,
                nbytes=local_rows * self._row_bytes,
            )
            self._mori.received += 1
            wait_block = functools.partial(
                self._mori_wait_block, expected=self._mori.received
            )
        for peer, stream in self._copy_streams.items():
            stream.wait_stream(comp)
            local.record_stream(stream)
            dst = (
                self._peer_buf[peer]
                + (slot * self._world * self._local_capacity + r0) * self._row_bytes
            )
            _check(
                self._hip.hipMemcpyAsync(
                    dst,
                    local.data_ptr(),
                    local_rows * self._row_bytes,
                    _HIP_MEMCPY_D2D_NO_CU,
                    stream.cuda_stream,
                ),
                "hipMemcpyAsync",
            )
            with torch.cuda.stream(stream):
                _signal_peer[(1,)](
                    _RawPtr(self._peer_flags[peer] + 4 * self._rank, torch.int32),
                    self._seq,
                )
        return PendingAllGather(
            local=local,
            gathered=buf[:num_tokens],
            rank=self._rank,
            world_size=self._world,
            local_rows=local_rows,
            wait_block=wait_block,
        )

    def begin(
        self, *, num_tokens: int, local_rows: int, pushes: int
    ) -> "AllGatherWriter":
        """Start an all-gather whose local rows are pushed in `pushes` row ranges
        as the producer finishes them (the same count on every rank)."""
        comp = torch.cuda.current_stream()
        for stream in self._copy_streams.values():
            comp.wait_stream(stream)
        self._seq += 1
        slot = self._slot
        self._slot ^= 1
        expected = 0
        if self._mori is not None:
            self._mori.received += pushes
            expected = self._mori.received
        return AllGatherWriter(
            ag=self,
            slot=slot,
            num_tokens=num_tokens,
            local_rows=local_rows,
            pushes=pushes,
            mori_expected=expected,
        )

    def _push_rows(self, *, slot: int, row0: int, rows: torch.Tensor) -> None:
        """Copy this rank's rows into its own slot at row0 and push them to every peer."""
        own = self._buf[slot, row0 : row0 + rows.shape[0]]
        own.copy_(rows)
        nbytes = rows.shape[0] * self._row_bytes
        if self._mori is not None:
            self._mori.push_all(
                dst_table=self._mori_tables[slot],
                src_ptr=own.data_ptr(),
                dst_off=row0 * self._row_bytes,
                nbytes=nbytes,
            )
            return
        comp = torch.cuda.current_stream()
        for peer, stream in self._copy_streams.items():
            stream.wait_stream(comp)
            dst = (
                self._peer_buf[peer]
                + (slot * self._world * self._local_capacity + row0) * self._row_bytes
            )
            _check(
                self._hip.hipMemcpyAsync(
                    dst,
                    own.data_ptr(),
                    nbytes,
                    _HIP_MEMCPY_D2D_NO_CU,
                    stream.cuda_stream,
                ),
                "hipMemcpyAsync",
            )

    def _signal_all(self) -> None:
        # Copies to a peer run in order on its copy stream, so one flag after the
        # last range covers every range pushed before it.
        for peer, stream in self._copy_streams.items():
            with torch.cuda.stream(stream):
                _signal_peer[(1,)](
                    _RawPtr(self._peer_flags[peer] + 4 * self._rank, torch.int32),
                    self._seq,
                )

    def _wait_all_landed(self, mori_expected: int) -> None:
        if self._mori is not None:
            self._mori.wait(expected=mori_expected)
            return
        for src in self._peer_buf:
            self._wait_block(src)

    def _mori_wait_block(self, src: int, *, expected: int) -> None:
        self._mori.wait(expected=expected, only=src)

    def _wait_block(self, src: int) -> None:
        _check(
            self._hip.hipStreamWaitValue32(
                torch.cuda.current_stream().cuda_stream,
                self._flags[src].data_ptr(),
                self._seq,
                _HIP_STREAM_WAIT_GEQ,
                0xFFFFFFFF,
            ),
            "hipStreamWaitValue32",
        )


class AllGatherWriter:
    """An all-gather filled range by range: `push(r0, r1, rows)` for this rank's
    rows [r0, r1) as soon as they are computed, then `finish()` waits for every
    peer's ranges and returns the gathered [num_tokens, hidden] tensor."""

    def __init__(
        self,
        *,
        ag: SdmaAllGather,
        slot: int,
        num_tokens: int,
        local_rows: int,
        pushes: int,
        mori_expected: int,
    ):
        self._ag = ag
        self._slot = slot
        self._num_tokens = num_tokens
        self._local_rows = local_rows
        self._pushes = pushes
        self._pushed = 0
        self._mori_expected = mori_expected

    def push(self, r0: int, r1: int, rows: torch.Tensor) -> None:
        if self._pushed >= self._pushes:
            raise RuntimeError(f"more than the {self._pushes} announced pushes")
        self._ag._push_rows(
            slot=self._slot, row0=self._ag._rank * self._local_rows + r0, rows=rows
        )
        self._pushed += 1
        if self._pushed == self._pushes:
            self._ag._signal_all()

    def finish(self) -> torch.Tensor:
        if self._pushed != self._pushes:
            raise RuntimeError(f"pushed {self._pushed} of {self._pushes} ranges")
        self._ag._wait_all_landed(self._mori_expected)
        ag = self._ag
        return ag._buf[self._slot, : ag._world * self._local_rows][: self._num_tokens]


_AG_INSTANCE: Optional[SdmaAllGather] = None


def get_sdma_all_gather(
    *,
    capacity_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    cpu_group: dist.ProcessGroup,
) -> SdmaAllGather:
    """Process-wide instance; the first call is collective over the group."""
    global _AG_INSTANCE
    if _AG_INSTANCE is None:
        _AG_INSTANCE = SdmaAllGather(
            capacity_tokens=capacity_tokens,
            hidden_size=hidden_size,
            dtype=dtype,
            device=torch.device("cuda", torch.cuda.current_device()),
            cpu_group=cpu_group,
            backend=resolve_sdma_backend(),
        )
    return _AG_INSTANCE
