"""Minimal repro: Mooncake TransferEngine (protocol=hip) reading a tensor that does
not start at its caching-allocator segment base.

Seed (GPU A) allocates a, b, c (filled 1, 2, 3) inside one allocator segment and
registers them with SGLang's register_memory_region_v2 (or, with --raw-register,
directly with engine.register_memory). Client (GPU B) reads b's
address with batch_transfer_sync_read. Correct result: every byte == 2.

    python mooncake_hip_offset_repro.py --layout offset --raw-register  # WRONG DATA
    python mooncake_hip_offset_repro.py --layout offset                 # OK
"""

import argparse
import multiprocessing as mp
import socket

import torch

# 6.25 MiB, the size of one PyHIP GR read weight in Qwen3.8-Flash-Next.
_N = 6553600


def _make_engine(gpu: int):
    from mooncake.engine import TransferEngine

    torch.cuda.set_device(gpu)
    engine = TransferEngine()
    host = socket.gethostbyname(socket.gethostname())
    assert engine.initialize(host, "P2PHANDSHAKE", "hip", "") == 0
    return engine, f"{host}:{engine.get_rpc_port()}"


def _seed(gpu: int, layout: str, raw_register: bool, conn) -> None:
    from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
        register_memory_region_v2,
    )

    engine, session = _make_engine(gpu)
    pad = None
    if layout == "views":
        big = torch.empty(3 * _N, dtype=torch.uint8, device="cuda")
        tensors = [big[i * _N : (i + 1) * _N] for i in range(3)]
    else:
        # Free a large block first so a, b, c are carved out of one segment.
        scratch = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
        del scratch
        if layout == "offset":
            # A live non-weight allocation ahead of the weights, so the registered
            # region does not start at the segment base.
            pad = torch.zeros(_N, dtype=torch.uint8, device="cuda")
        tensors = [torch.empty(_N, dtype=torch.uint8, device="cuda") for _ in range(3)]
    for value, t in enumerate(tensors, start=1):
        t.fill_(value)
    torch.cuda.synchronize()
    module = torch.nn.Module()
    for name, t in zip("abc", tensors):
        module.register_parameter(name, torch.nn.Parameter(t, requires_grad=False))
    if raw_register:
        # Register exactly the weight span, bypassing SGLang's ROCm workaround.
        assert engine.register_memory(tensors[0].data_ptr(), 3 * _N) == 0
        info = {n: (t.data_ptr(), t.numel(), 1) for n, t in zip("abc", tensors)}
    else:
        info = register_memory_region_v2(module, engine)
    segments = [
        (s["address"], s["total_size"]) for s in torch.cuda.memory.memory_snapshot()
    ]
    conn.send((session, info, segments))
    conn.recv()
    del pad


def _client(gpu: int, seed_session: str, seed_ptr: int) -> torch.Tensor:
    engine, _ = _make_engine(gpu)
    dst = torch.zeros(_N, dtype=torch.uint8, device="cuda")
    assert engine.register_memory(dst.data_ptr(), _N) == 0
    ret = engine.batch_transfer_sync_read(
        seed_session, [dst.data_ptr()], [seed_ptr], [_N]
    )
    torch.cuda.synchronize()
    print(f"batch_transfer_sync_read ret={ret}")
    return dst


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-gpu", type=int, default=4)
    parser.add_argument("--client-gpu", type=int, default=6)
    parser.add_argument(
        "--layout", choices=["blocks", "views", "offset"], default="blocks"
    )
    parser.add_argument(
        "--raw-register",
        action="store_true",
        help="register the weight span directly instead of via SGLang",
    )
    args = parser.parse_args()
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    proc = ctx.Process(
        target=_seed, args=(args.seed_gpu, args.layout, args.raw_register, child)
    )
    proc.start()
    session, info, segments = parent.recv()
    ptrs = {name: info[name][0] for name in "abc"}
    seg = next((s for s in segments if s[0] <= ptrs["a"] < s[0] + s[1]), None)
    print(
        f"layout={args.layout} raw_register={args.raw_register} seed segment base={seg[0]:#x} size={seg[1]}"
    )
    for name in "abc":
        print(f"  {name}: ptr={ptrs[name]:#x} offset_in_segment={ptrs[name] - seg[0]}")
    dst = _client(args.client_gpu, session, ptrs["b"])
    values = torch.unique(dst).tolist()
    print(f"read b -> unique byte values {values[:8]} (expected [2])")
    print("RESULT:", "OK" if values == [2] else "WRONG DATA")
    parent.send("done")
    proc.join()


if __name__ == "__main__":
    main()
