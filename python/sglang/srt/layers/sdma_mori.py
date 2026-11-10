"""Device-posted SDMA pushes through mori CCO (ROCm).

One-wave FlyDSL kernels on the compute stream write the SDMA copy packets
(`Sdma.put_ptr`) and return; the copy engine bumps the receiver's mori signal
counter for this sender once the bytes have landed. Consumers wait with a
one-wave kernel that spins on those counters, also on the compute stream. No
side streams or cross-stream events are involved.

Only mori's SDMA queues and signal pool are used. Destinations are addresses
the caller already mapped (hipIpc), not mori's symmetric windows: on hosts whose
kernel lacks CONFIG_DMABUF_MOVE_NOTIFY those windows are pinned in host memory.

Requirements (checked by `mori_unavailable_reason`): mori built with
BUILD_CCO_SDMA=ON importable, MORI_ENABLE_SDMA=1, and the process must load the
same libdrm/libdrm_amdgpu as the HIP runtime (with ROCm's rocm_sysdeps copies,
LD_PRELOAD them), otherwise mori's KFD context fails to acquire the GPU VM and
exits the process.
"""

import functools
import logging
import os
from typing import Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_WAVE = 64


def mori_unavailable_reason() -> Optional[str]:
    """Why the mori backend cannot run in this process, or None if it can.
    Must run before any mori queue is created: a failing KFD setup exits."""
    if os.getenv("MORI_ENABLE_SDMA", "").lower() not in ("1", "true", "on"):
        return "MORI_ENABLE_SDMA is not set"
    try:
        import flydsl.compiler  # noqa: F401
        import mori.cco  # noqa: F401
    except ImportError as e:
        return f"cannot import mori / flydsl: {e}"
    # torch's HIP may load ROCm's private libdrm copy; mori links the system one
    # unless the private copy is preloaded, and two copies cannot share the GPU VM.
    with open("/proc/self/maps") as f:
        hip_uses_sysdeps_drm = "librocm_sysdeps_drm_amdgpu" in f.read()
    if hip_uses_sysdeps_drm and "rocm_sysdeps_drm_amdgpu" not in os.getenv(
        "LD_PRELOAD", ""
    ):
        return "LD_PRELOAD lacks ROCm's librocm_sysdeps_drm(_amdgpu)"
    return None


@functools.lru_cache(maxsize=None)
def sdma_available() -> bool:
    """Whether the SDMA overlap can run in this process; warns once when it cannot,
    in which case callers keep the aiter reduce-scatter / all-gather."""
    reason = mori_unavailable_reason()
    if reason is not None:
        logger.warning(
            "SP SDMA overlap disabled (%s); using the aiter reduce-scatter / all-gather",
            reason,
        )
        return False
    return True


def _build_kernels(rank: int, world: int):
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    import mori.cco.device.flydsl as cco
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm, scf
    from flydsl.expr import gpu as fgpu
    from flydsl.expr.typing import Int32, Int64
    from mori.cco.device.flydsl.signals import i32_type, signal_ptr

    def load_i64(addr):
        return llvm.load(ir.IntegerType.get_signless(64), signal_ptr(addr), alignment=8)

    def load_u32_system(ptr):
        return llvm.load(
            i32_type(),
            ptr,
            alignment=4,
            ordering=llvm.AtomicOrdering.monotonic,
            syncscope="",
        )

    def spin_until(ptr, flag):
        first = load_u32_system(ptr)
        i32 = i32_type()
        loop = scf.WhileOp([i32], [first])
        cond = ir.Block.create_at_start(loop.before, [i32])
        body = ir.Block.create_at_start(loop.after, [i32])
        with ir.InsertionPoint(cond):
            cur = fx.Int32(cond.arguments[0])
            scf.ConditionOp(
                ((cur - fx.Int32(flag)) < fx.Int32(0)).ir_value(), [cond.arguments[0]]
            )
        with ir.InsertionPoint(body):
            scf.YieldOp([load_u32_system(ptr)])

    # Push the same source to every peer; dst_table holds one base address per peer.
    @flyc.kernel(known_block_size=[_WAVE, 1, 1])
    def push_all(
        dev_comm: Int64, dst_table: Int64, src: Int64, dst_off: Int64, nbytes: Int64
    ):
        tid = fx.thread_idx.x
        sdma = cco.DevComm(dev_comm).sdma()
        if tid < world:
            if tid != rank:
                base = fx.Int64(
                    load_i64(fx.Int64(dst_table) + fx.Int64(tid) * fx.Int64(8))
                )
                sdma.put_ptr(
                    tid,
                    base + fx.Int64(dst_off),
                    fx.Int64(src),
                    fx.Int64(nbytes),
                    0,
                    remote_signal=True,
                )
        fgpu.barrier()

    @flyc.jit
    def push_all_run(
        dev_comm: Int64,
        dst_table: Int64,
        src: Int64,
        dst_off: Int64,
        nbytes: Int64,
        stream: fx.Stream = fx.Stream(None),
    ):
        push_all(dev_comm, dst_table, src, dst_off, nbytes).launch(
            grid=(1, 1, 1), block=[_WAVE, 1, 1], stream=stream
        )

    @flyc.kernel(known_block_size=[_WAVE, 1, 1])
    def push_one(dev_comm: Int64, dst: Int64, src: Int64, nbytes: Int64, peer: Int32):
        tid = fx.thread_idx.x
        sdma = cco.DevComm(dev_comm).sdma()
        if tid == peer:
            sdma.put_ptr(
                tid,
                fx.Int64(dst),
                fx.Int64(src),
                fx.Int64(nbytes),
                0,
                remote_signal=True,
            )
        fgpu.barrier()

    @flyc.jit
    def push_one_run(
        dev_comm: Int64,
        dst: Int64,
        src: Int64,
        nbytes: Int64,
        peer: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        push_one(dev_comm, dst, src, nbytes, peer).launch(
            grid=(1, 1, 1), block=[_WAVE, 1, 1], stream=stream
        )

    @flyc.kernel(known_block_size=[_WAVE, 1, 1])
    def wait_all(dev_comm: Int64, expected: Int32):
        tid = fx.thread_idx.x
        sdma = cco.DevComm(dev_comm).sdma()
        if tid < world:
            if tid != rank:
                spin_until(
                    signal_ptr(fx.Int64(sdma.signal_slot(tid, 0))), fx.Int32(expected)
                )
        fgpu.barrier()

    @flyc.jit
    def wait_all_run(
        dev_comm: Int64, expected: Int32, stream: fx.Stream = fx.Stream(None)
    ):
        wait_all(dev_comm, expected).launch(
            grid=(1, 1, 1), block=[_WAVE, 1, 1], stream=stream
        )

    @flyc.kernel(known_block_size=[_WAVE, 1, 1])
    def wait_one(dev_comm: Int64, expected: Int32, src_rank: Int32):
        tid = fx.thread_idx.x
        sdma = cco.DevComm(dev_comm).sdma()
        if tid == src_rank:
            spin_until(
                signal_ptr(fx.Int64(sdma.signal_slot(tid, 0))), fx.Int32(expected)
            )
        fgpu.barrier()

    @flyc.jit
    def wait_one_run(
        dev_comm: Int64,
        expected: Int32,
        src_rank: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        wait_one(dev_comm, expected, src_rank).launch(
            grid=(1, 1, 1), block=[_WAVE, 1, 1], stream=stream
        )

    return flyc, fx, push_all_run, push_one_run, wait_all_run, wait_one_run


class MoriSdmaHub:
    """Shared mori SDMA queues + signal counters for every SDMA collective in the process.

    mori's counter for a sender counts every signalled put from it on queue 0, in
    issue order, across all users of the hub. Every rank issues the same sequence
    of collectives, so `received` (puts each peer has sent me so far) is the same
    for every peer and is what a wait compares against.
    """

    def __init__(self, cpu_group: dist.ProcessGroup, device: torch.device):
        from mori.cco import (
            GDA_CONNECTION_NONE,
            CCODevCommRequirements,
            Communicator,
            UniqueId,
        )

        self.rank = dist.get_rank(cpu_group)
        self.world = dist.get_world_size(cpu_group)
        self.device = device
        uid = [bytes(Communicator.get_unique_id()) if self.rank == 0 else None]
        dist.broadcast_object_list(
            uid, src=dist.get_global_rank(cpu_group, 0), group=cpu_group
        )
        self._comm = Communicator.init(
            self.world, self.rank, UniqueId.from_bytes(uid[0]), per_rank_vmm=64 << 20
        )
        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        reqs.sdma_queue_count = 1
        self._dev_comm = self._comm.create_dev_comm(reqs)
        # 0 when mori was built without BUILD_CCO_SDMA or no peer has SDMA; puts
        # on such a communicator move no bytes, so waits would never return.
        self.sdma_queues = self._dev_comm.sdma_num_queue
        (
            self._flyc,
            self._fx,
            self._push_all_jit,
            self._push_one_jit,
            self._wait_all_jit,
            self._wait_one_jit,
        ) = _build_kernels(self.rank, self.world)
        self._compiled = {}
        self.received = 0

    def _launch(self, name, jit_fn, *args) -> None:
        stream = self._fx.Stream(torch.cuda.current_stream())
        fn = self._compiled.get(name)
        # flyc.compile also runs the launcher once; that run is this call.
        if fn is None:
            self._compiled[name] = self._flyc.compile(jit_fn, *args, stream)
        else:
            fn(*args, stream)

    def push_all(
        self, *, dst_table: torch.Tensor, src_ptr: int, dst_off: int, nbytes: int
    ) -> None:
        self._launch(
            "push_all",
            self._push_all_jit,
            self._dev_comm.ptr,
            dst_table.data_ptr(),
            src_ptr,
            dst_off,
            nbytes,
        )

    def push_one(self, *, peer: int, dst_ptr: int, src_ptr: int, nbytes: int) -> None:
        self._launch(
            "push_one",
            self._push_one_jit,
            self._dev_comm.ptr,
            dst_ptr,
            src_ptr,
            nbytes,
            peer,
        )

    def wait(self, *, expected: int, only: Optional[int] = None) -> None:
        """Wait until every peer (or only peer `only`) has `expected` puts landed here."""
        if only is None:
            self._launch("wait_all", self._wait_all_jit, self._dev_comm.ptr, expected)
        else:
            self._launch(
                "wait_one", self._wait_one_jit, self._dev_comm.ptr, expected, only
            )


_HUB: Optional[MoriSdmaHub] = None
_GROUP_USABLE: Optional[bool] = None


def get_mori_hub(cpu_group: dist.ProcessGroup, device: torch.device) -> MoriSdmaHub:
    """Process-wide hub; the first call is collective over the group."""
    global _HUB
    if _HUB is None:
        _HUB = MoriSdmaHub(cpu_group, device)
    return _HUB


def sdma_group_usable(cpu_group: dist.ProcessGroup, device: torch.device) -> bool:
    """Whether the SDMA collectives can run over this group; the first call is
    collective over the group and warns once when they cannot."""
    global _GROUP_USABLE
    if _GROUP_USABLE is None:
        reason = _group_unusable_reason(cpu_group, device)
        if reason is not None:
            logger.warning(
                "SP SDMA overlap disabled (%s); using the aiter reduce-scatter / all-gather",
                reason,
            )
        _GROUP_USABLE = reason is None
    return _GROUP_USABLE


def _group_unusable_reason(
    cpu_group: dist.ProcessGroup, device: torch.device
) -> Optional[str]:
    from sglang.srt.distributed.parallel_state import in_the_same_node_as

    # The SDMA destinations are HIP IPC buffers, which map only within one node.
    if not all(in_the_same_node_as(cpu_group, source_rank=0)):
        return "the group spans nodes"
    queues = torch.tensor([get_mori_hub(cpu_group, device).sdma_queues])
    dist.all_reduce(queues, op=dist.ReduceOp.MIN, group=cpu_group)
    if queues.item() < 1:
        return "mori set up no SDMA queue (built without BUILD_CCO_SDMA?)"
    return None
