"""Stream-ordered control words for ROCm GPU IPC feature pools."""

import torch
import triton
import triton.language as tl

from sglang.srt.multimodal.transport.ipc_wait_watchdog import get_ipc_wait_watchdog
from sglang.srt.runtime_context import get_device


@triton.jit
def _wait_value32(address, value):
    pointer = tl.cast(address, tl.pointer_type(tl.uint32))
    while tl.atomic_add(pointer, 0, sem="acquire", scope="sys") < value:
        pass


@triton.jit
def _write_value32(address, value):
    pointer = tl.cast(address, tl.pointer_type(tl.uint32))
    tl.atomic_xchg(pointer, value, sem="release", scope="sys")


def wait_value32(address: int, value: int, transport_name: str = "CUDA IPC") -> None:
    # IPC reconstruction is eager. A captured event would monitor only one
    # graph execution and could leave later replays' waits unsupervised.
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "ROCm IPC readiness waits cannot run during CUDA graph capture"
        )
    watchdog = get_ipc_wait_watchdog()
    wait = watchdog.arm(
        device_id=torch.cuda.current_device(),
        address=address,
        generation=value,
        transport_name=transport_name,
        timeout=get_device().watchdog_timeout,
    )
    enqueued = False
    try:
        event = torch.cuda.Event()
        _wait_value32[(1,)](address, value, num_warps=1)
        enqueued = True
        event.record(torch.cuda.current_stream())
        watchdog.launched(wait, event)
    except Exception:
        # If event recording fails after launch, keep supervising the wait:
        # it may already be spinning even though the caller sees an error.
        if not enqueued:
            watchdog.cancel(wait)
        raise


def write_value32(address: int, value: int) -> None:
    _write_value32[(1,)](address, value, num_warps=1)
