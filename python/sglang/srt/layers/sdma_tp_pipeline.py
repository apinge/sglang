"""TP2 all-reduce over SDMA, pipelined over row chunks on ROCm.

Each rank writes the rows to reduce chunk by chunk into its IPC-shared send
buffer and bumps a flag in the peer's memory; the peer's comm stream waits on
that flag (hipStreamWaitValue32) and pulls the chunk with an SDMA copy, so the
transfer runs on the copy engine while the CUs keep computing (the next
chunk's GEMM, or the previous chunk's consumer). `PendingReduce` hands out each
chunk's sum (local + received) once its pull has landed.

With an `int8` or `fp8` transport, the send buffer holds each row as 8-bit
values (int8, or float8 e4m3fnuz) plus one bf16 scale per 32 values (symmetric,
absmax), roughly halving the bytes on the link; the receiver adds the
dequantized peer rows to its own bf16 rows.

Flags carry monotonically increasing sequence numbers, so they are never
reset. Send buffers are double-buffered: a rank only reuses a slot after it
observed the peer's next-call flags, which the peer writes after its pulls of
the previous call completed.
"""

import ctypes
from typing import Callable, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import triton
import triton.language as tl

_HIP_IPC_LAZY_PEER = 0x1
# hipMemcpyDeviceToDeviceNoCU: force the SDMA engine instead of a blit kernel.
_HIP_MEMCPY_D2D_NO_CU = 1024
_HIP_STREAM_WAIT_GEQ = 0x0
_MAX_CHUNKS = 64


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


@triton.jit
def _signal_peer(flag_ptr, value):
    # System-scope release: the chunk written by the preceding kernel is visible
    # to the peer's SDMA read once the peer observes the flag.
    tl.atomic_xchg(flag_ptr, value, sem="release", scope="sys")


_TRANSPORTS = ("bf16", "int8", "fp8")
_QUANT_GROUP = 32
_GROUPS_PER_PROGRAM = 16


@triton.jit
def _quantize_rows(
    x_ptr,
    x_row_stride,
    rec_ptr,
    rec_bytes,
    HIDDEN: tl.constexpr,
    GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    FP8: tl.constexpr,
):
    # Record layout per row: HIDDEN 8-bit values, then HIDDEN // GROUP bf16 scales.
    row = tl.program_id(0)
    group = tl.program_id(1) * GROUPS + tl.arange(0, GROUPS)
    cols = group[:, None] * GROUP + tl.arange(0, GROUP)[None, :]
    x = tl.load(x_ptr + row * x_row_stride + cols).to(tl.float32)
    # 240 is the largest float8 e4m3fnuz value.
    qmax = 240.0 if FP8 else 127.0
    # Quantize with the rounded bf16 scale the receiver will multiply by.
    scale = (tl.max(tl.abs(x), axis=1) / qmax).to(tl.bfloat16).to(tl.float32)
    inv = tl.where(scale > 0, 1.0 / scale, 0.0)
    v = x * inv[:, None]
    rec = rec_ptr + row * rec_bytes
    if FP8:
        v = tl.minimum(tl.maximum(v, -qmax), qmax)
        tl.store(rec + cols, v.to(tl.float8e4b8).to(tl.int8, bitcast=True))
    else:
        q = tl.where(v >= 0, v + 0.5, v - 0.5).to(tl.int32)
        q = tl.minimum(tl.maximum(q, -127), 127)
        tl.store(rec + cols, q.to(tl.int8))
    scale_ptr = (rec + HIDDEN).to(tl.pointer_type(tl.bfloat16))
    tl.store(scale_ptr + group, scale.to(tl.bfloat16))


@triton.jit
def _dequant_add_rows(
    local_ptr,
    local_row_stride,
    rec_ptr,
    rec_bytes,
    out_ptr,
    out_row_stride,
    HIDDEN: tl.constexpr,
    GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    FP8: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1) * GROUPS + tl.arange(0, GROUPS)
    cols = group[:, None] * GROUP + tl.arange(0, GROUP)[None, :]
    rec = rec_ptr + row * rec_bytes
    raw = tl.load(rec + cols)
    if FP8:
        q = raw.to(tl.float8e4b8, bitcast=True).to(tl.float32)
    else:
        q = raw.to(tl.float32)
    scale_ptr = (rec + HIDDEN).to(tl.pointer_type(tl.bfloat16))
    scale = tl.load(scale_ptr + group).to(tl.float32)
    local = tl.load(local_ptr + row * local_row_stride + cols).to(tl.float32)
    y = local + q * scale[:, None]
    tl.store(out_ptr + row * out_row_stride + cols, y.to(tl.bfloat16))


class _RawPtr:
    """Pointer argument for Triton when there is no torch tensor to pass."""

    def __init__(self, ptr: int, dtype: torch.dtype):
        self._ptr = ptr
        self.dtype = dtype

    def data_ptr(self) -> int:
        return self._ptr


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} failed with hipError {rc}")


class PendingReduce:
    """All-reduce in flight, consumed per row chunk.

    Valid until the pipeline's next `send`; every chunk must be consumed
    (via `chunks()` or `sum()`) before that.
    """

    def __init__(
        self,
        *,
        partial: torch.Tensor,
        local_chunks: Optional[List[torch.Tensor]],
        recv: torch.Tensor,
        recv_records: Optional[torch.Tensor],
        rec_bytes: int,
        fp8: bool,
        bounds: List[int],
        chunk_done: List[torch.cuda.Event],
    ):
        # [num_tokens, hidden] view of the send buffer: this rank's partial sum
        # with bf16 transport, raw int8 records otherwise. Only its identity and
        # shape are used as the carrier of this handle.
        self.partial = partial
        self._local_chunks = local_chunks
        self._recv = recv
        self._recv_records = recv_records
        self._rec_bytes = rec_bytes
        self._fp8 = fp8
        self._bounds = bounds
        self._chunk_done = chunk_done

    def chunks(self) -> Iterator[Tuple[int, int, torch.Tensor]]:
        """Yield (r0, r1, reduced_rows), each once its pull has landed."""
        comp = torch.cuda.current_stream()
        for k, done in enumerate(self._chunk_done):
            comp.wait_event(done)
            r0, r1 = self._bounds[k], self._bounds[k + 1]
            out = torch.empty(
                r1 - r0,
                self.partial.shape[1],
                dtype=self.partial.dtype,
                device=self.partial.device,
            )
            self._reduce_into(k, out)
            yield r0, r1, out

    def sum(self) -> torch.Tensor:
        comp = torch.cuda.current_stream()
        out = torch.empty_like(self.partial)
        for k, done in enumerate(self._chunk_done):
            comp.wait_event(done)
            self._reduce_into(k, out[self._bounds[k] : self._bounds[k + 1]])
        return out

    def _reduce_into(self, k: int, out: torch.Tensor) -> None:
        r0, r1 = self._bounds[k], self._bounds[k + 1]
        if self._local_chunks is None:
            torch.add(self.partial[r0:r1], self._recv[r0:r1], out=out)
            return
        local = self._local_chunks[k]
        hidden = local.shape[1]
        _dequant_add_rows[(r1 - r0, hidden // (_QUANT_GROUP * _GROUPS_PER_PROGRAM))](
            local,
            local.stride(0),
            self._recv_records[r0 * self._rec_bytes :],
            self._rec_bytes,
            out,
            out.stride(0),
            HIDDEN=hidden,
            GROUP=_QUANT_GROUP,
            GROUPS=_GROUPS_PER_PROGRAM,
            FP8=self._fp8,
        )


class SdmaTp2Pipeline:
    def __init__(
        self,
        *,
        capacity_tokens: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
        cpu_group: dist.ProcessGroup,
        transport: str = "bf16",
    ):
        if transport not in _TRANSPORTS:
            raise ValueError(
                f"SDMA transport must be one of {_TRANSPORTS}, got {transport!r}"
            )
        self._hip = _load_hip()
        self._capacity = capacity_tokens
        self._row_bytes = hidden_size * torch.empty(0, dtype=dtype).element_size()
        # An int8 row record is smaller than the bf16 row, so the records
        # reuse the bf16-sized buffers.
        self._quantized = (
            transport != "bf16"
            and dtype == torch.bfloat16
            and hidden_size % (_QUANT_GROUP * _GROUPS_PER_PROGRAM) == 0
        )
        self._rec_bytes = (
            hidden_size + hidden_size // _QUANT_GROUP * 2
            if self._quantized
            else self._row_bytes
        )
        self._fp8 = self._quantized and transport == "fp8"
        self._send = torch.empty(
            2, capacity_tokens, hidden_size, dtype=dtype, device=device
        )
        self._recv = torch.empty(
            capacity_tokens, hidden_size, dtype=dtype, device=device
        )
        self._flags = torch.zeros(_MAX_CHUNKS, dtype=torch.int32, device=device)
        torch.cuda.synchronize(device)

        rank = dist.get_rank(cpu_group)
        exported = [None, None]
        dist.all_gather_object(
            exported,
            (self._export(self._send), self._export(self._flags)),
            group=cpu_group,
        )
        peer_send, peer_flags = exported[1 - rank]
        self._peer_send = self._import(peer_send)
        self._peer_flags = self._import(peer_flags)
        self._comm = torch.cuda.Stream(device)
        # Reused across calls: a stream wait captures the event's state when it
        # is enqueued, and the next send starts after the consumer's waits.
        self._chunk_done = [torch.cuda.Event() for _ in range(_MAX_CHUNKS)]
        self._seq = 0
        self._slot = 0

    def fits(self, num_tokens: int) -> bool:
        return num_tokens <= self._capacity

    def run(
        self,
        *,
        project: Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        chunks: int,
    ) -> torch.Tensor:
        pending = self.send(
            rows=lambda r0, r1: project(x[r0:r1]), num_tokens=x.shape[0], chunks=chunks
        )
        return pending.sum()

    def send(
        self,
        *,
        rows: Callable[[int, int], torch.Tensor],
        num_tokens: int,
        chunks: int,
    ) -> "PendingReduce":
        """Start the all-reduce of `rows(r0, r1)` over row chunks; consume the
        result through the returned handle before the next `send`."""
        comp = torch.cuda.current_stream()
        self._comm.wait_stream(comp)
        send = self._send[self._slot]
        send_records = send.view(torch.int8).reshape(-1)
        recv_records = self._recv.view(torch.int8).reshape(-1)
        peer_send = self._peer_send + self._slot * self._capacity * self._row_bytes
        bounds = [num_tokens * k // chunks for k in range(chunks + 1)]
        local_chunks = [] if self._quantized else None
        for k in range(chunks):
            r0, r1 = bounds[k], bounds[k + 1]
            local = rows(r0, r1)
            if self._quantized:
                self._quantize(local, send_records[r0 * self._rec_bytes :])
                local_chunks.append(local)
            else:
                send[r0:r1].copy_(local)
            self._seq += 1
            _signal_peer[(1,)](
                _RawPtr(self._peer_flags + 4 * k, torch.int32), self._seq
            )
            self._pull(
                dst=recv_records[r0 * self._rec_bytes :].data_ptr(),
                src=peer_send + r0 * self._rec_bytes,
                nbytes=(r1 - r0) * self._rec_bytes,
                flag=self._flags[k],
            )
            self._chunk_done[k].record(self._comm)
        self._slot ^= 1
        return PendingReduce(
            partial=send[:num_tokens],
            local_chunks=local_chunks,
            recv=self._recv[:num_tokens],
            recv_records=recv_records if self._quantized else None,
            fp8=self._fp8,
            rec_bytes=self._rec_bytes,
            bounds=bounds,
            chunk_done=self._chunk_done[:chunks],
        )

    def _quantize(self, rows: torch.Tensor, records: torch.Tensor) -> None:
        hidden = rows.shape[1]
        _quantize_rows[(rows.shape[0], hidden // (_QUANT_GROUP * _GROUPS_PER_PROGRAM))](
            rows,
            rows.stride(0),
            records,
            self._rec_bytes,
            HIDDEN=hidden,
            GROUP=_QUANT_GROUP,
            GROUPS=_GROUPS_PER_PROGRAM,
            FP8=self._fp8,
        )

    def _pull(self, *, dst: int, src: int, nbytes: int, flag: torch.Tensor) -> None:
        stream = self._comm.cuda_stream
        _check(
            self._hip.hipStreamWaitValue32(
                stream, flag.data_ptr(), self._seq, _HIP_STREAM_WAIT_GEQ, 0xFFFFFFFF
            ),
            "hipStreamWaitValue32",
        )
        _check(
            self._hip.hipMemcpyAsync(dst, src, nbytes, _HIP_MEMCPY_D2D_NO_CU, stream),
            "hipMemcpyAsync",
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


_PIPELINE: Optional[SdmaTp2Pipeline] = None


def get_sdma_tp2_pipeline(
    *,
    capacity_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    cpu_group: dist.ProcessGroup,
    transport: str = "bf16",
) -> SdmaTp2Pipeline:
    """Process-wide pipeline; the first call is collective over the TP group."""
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = SdmaTp2Pipeline(
            capacity_tokens=capacity_tokens,
            hidden_size=hidden_size,
            dtype=dtype,
            device=torch.device("cuda", torch.cuda.current_device()),
            cpu_group=cpu_group,
            transport=transport,
        )
    return _PIPELINE
