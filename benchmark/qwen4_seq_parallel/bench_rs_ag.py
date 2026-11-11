"""Qwen4-Exp sequence parallelism: aiter all-reduce vs reduce-scatter + all-gather of a [T, 2560] bf16 partial.

The sequence-parallel layout replaces each prefill all-reduce with a reduce-scatter and an all-gather;
this checks that the pair moves the same data in about the same time and gives bit-identical results.

    WORLD=4 HIP_VISIBLE_DEVICES=0,1,2,3 python benchmark/qwen4_seq_parallel/bench_rs_ag.py
"""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

N, H = int(os.environ.get("WORLD", "4")), 2560


def bench(fn, iters=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    t = torch.tensor([a.elapsed_time(b) / iters])
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def worker(rank):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29621")
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("gloo", rank=rank, world_size=N)
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    ca = CustomAllreduce(dist.group.WORLD, dev)
    for T in (12000, 6000, 3000):
        x = torch.randn(
            T,
            H,
            device=dev,
            dtype=torch.bfloat16,
            generator=torch.Generator(device=dev).manual_seed(rank),
        )
        rs_out = torch.empty(T // N, H, device=dev, dtype=torch.bfloat16)
        ca.reduce_scatter(x, rs_out)
        ag = ca.custom_all_gather(rs_out)
        ar = ca.all_reduce(x)
        exact = torch.equal(ag, ar)
        t_ar = bench(lambda: ca.all_reduce(x))
        t_rs = bench(lambda: ca.reduce_scatter(x, rs_out))
        t_ag = bench(lambda: ca.custom_all_gather(rs_out))
        if rank == 0:
            print(
                f"T={T:5d} ({T * H * 2 / 1e6:.0f} MB): AR {t_ar:.3f} ms | "
                f"RS {t_rs:.3f} + AG {t_ag:.3f} = {t_rs + t_ag:.3f} ms, "
                f"AG(RS(x)) == AR(x): {exact}"
            )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, nprocs=N)
