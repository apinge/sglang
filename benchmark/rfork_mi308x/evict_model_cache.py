"""Report and optionally drop the page-cache residency of one model's files.

Only the given directory's files are touched (posix_fadvise DONTNEED), unlike a
global /proc/sys/vm/drop_caches. Pages mapped or pinned by a live process may stay.

    python evict_model_cache.py /models/<model> [--evict]
"""

import argparse
import ctypes
import mmap
import os

_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_long,
]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
_PAGE = os.sysconf("SC_PAGE_SIZE")
_MAP_FAILED = ctypes.c_void_p(-1).value
# Bytes inspected per mincore call; arbitrary, only bounds the vector size.
_CHUNK = 1 << 30


def resident_bytes(path: str) -> int:
    size = os.path.getsize(path)
    resident_pages = 0
    fd = os.open(path, os.O_RDONLY)
    try:
        for offset in range(0, size, _CHUNK):
            length = min(_CHUNK, size - offset)
            addr = _libc.mmap(None, length, mmap.PROT_READ, mmap.MAP_SHARED, fd, offset)
            if addr == _MAP_FAILED:
                raise OSError(ctypes.get_errno(), "mmap failed")
            try:
                vec = ctypes.create_string_buffer((length + _PAGE - 1) // _PAGE)
                if _libc.mincore(addr, length, vec) != 0:
                    raise OSError(ctypes.get_errno(), "mincore failed")
                resident_pages += sum(b & 1 for b in vec.raw)
            finally:
                _libc.munmap(addr, length)
    finally:
        os.close(fd)
    return min(resident_pages * _PAGE, size)


def evict(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--evict", action="store_true")
    args = parser.parse_args()
    files = sorted(
        os.path.join(args.model_dir, name)
        for name in os.listdir(args.model_dir)
        if os.path.isfile(os.path.join(args.model_dir, name))
    )
    if args.evict:
        for path in files:
            evict(path)
    total = sum(os.path.getsize(p) for p in files)
    resident = sum(resident_bytes(p) for p in files)
    print(
        f"{'after evict' if args.evict else 'current'}: resident "
        f"{resident / 2**30:.1f} GiB / {total / 2**30:.1f} GiB "
        f"({100 * resident / total:.1f}%)"
    )


if __name__ == "__main__":
    main()
