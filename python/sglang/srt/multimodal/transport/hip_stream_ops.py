"""Stream-ordered control words for ROCm GPU IPC feature pools."""

import triton
import triton.language as tl


@triton.jit
def _wait_value32(address, value):
    pointer = tl.cast(address, tl.pointer_type(tl.uint32))
    while tl.atomic_add(pointer, 0, sem="acquire", scope="sys") < value:
        pass


@triton.jit
def _write_value32(address, value):
    pointer = tl.cast(address, tl.pointer_type(tl.uint32))
    tl.atomic_xchg(pointer, value, sem="release", scope="sys")


def wait_value32(address: int, value: int) -> None:
    _wait_value32[(1,)](address, value, num_warps=1)


def write_value32(address: int, value: int) -> None:
    _write_value32[(1,)](address, value, num_warps=1)
