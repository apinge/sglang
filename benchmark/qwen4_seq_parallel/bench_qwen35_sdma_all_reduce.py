"""Qwen3.5 TP2 SDMA all-reduce (models/qwen3_5_sdma_all_reduce.py): correctness, and a
layer-like timing against RCCL.

Layer: o_proj FP8 GEMM [T, 2048] -> all-reduce -> MoE stand-in (two FP8 GEMMs) -> all-reduce.

    HIP_VISIBLE_DEVICES=0,1 python benchmark/qwen4_seq_parallel/bench_qwen35_sdma_all_reduce.py

Needs mori and its runtime environment (see README: MORI_ENABLE_SDMA, the libdrm LD_PRELOAD).
Timed with a hot-L2 loop: every call moves 50 MB or more (several times the 4 MB L2 per XCD).
"""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

H = 2048
CHUNKS = (1, 2, 4, 8)


def bench(fn, group, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    dist.barrier(group=group)
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    t = torch.tensor([a.elapsed_time(b) / iters])
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    return t.item()


def make_all_reduce(rank, gloo, T):
    from sglang.srt.layers.sdma_reduce_scatter import SdmaAllGather
    from sglang.srt.models.qwen3_5_sdma_all_reduce import (
        Qwen35SdmaAllReduce,
        _PendingAllReduce,
        _Plan,
    )

    ar = Qwen35SdmaAllReduce(hidden_size=H)
    ar._ag = SdmaAllGather(
        capacity_tokens=2 * T,
        hidden_size=H,
        dtype=torch.bfloat16,
        device=torch.device("cuda", rank),
        cpu_group=gloo,
    )

    def start(num_tokens, chunks, rows):
        plan = _Plan(num_tokens=num_tokens, chunks=chunks)
        writer = ar._begin(plan)
        for r0, r1 in plan.bounds:
            writer.push(r0, r1, rows(r0, r1))
        return _PendingAllReduce(ag=ar._ag, writer=writer, plan=plan)

    return start


def check_correctness(rank, gloo, dev, start):
    worst = 0.0
    for it, T in enumerate([12000, 11999, 4097, 32768, 2048]):
        g = torch.Generator(device=dev).manual_seed(it * 10 + rank)
        x = torch.randn(T, H, device=dev, dtype=torch.bfloat16, generator=g)
        ref = x.float()
        dist.all_reduce(ref)
        # A slow producer exposes a missing wait.
        pending = start(
            T,
            CHUNKS[it % 4],
            lambda r0, r1: (torch.cuda._sleep(2_000_000), x[r0:r1] * 1)[1],
        )
        out = torch.empty_like(x)
        for r0, r1, rows in pending.chunks():
            out[r0:r1] = rows
        peer = out.clone()
        dist.broadcast(peer, src=0)
        assert torch.equal(peer, out), f"T={T}: ranks differ"
        err = ((out.float() - ref).abs().max() / ref.abs().max()).item()
        worst = max(worst, err)
        assert err < 1e-2, (T, err)
    if rank == 0:
        print(f"correct: ranks bit-identical, worst rel err vs fp32 {worst:.1e}")


def time_layer(rank, gloo, dev, start, T):
    f8, bf, one = torch.float8_e4m3fnuz, torch.bfloat16, torch.tensor(1.0, device=dev)
    g = lambda s: torch.Generator(device=dev).manual_seed(s)
    a = (torch.randn(T, 2048, device=dev, generator=g(5)) * 0.1).to(f8)
    w_o = (torch.randn(H, 2048, device=dev, generator=g(7)) * 0.1).to(f8).t()
    w_up = (torch.randn(6144, H, device=dev, generator=g(9)) * 0.1).to(f8).t()
    w_dn = (torch.randn(H, 6144, device=dev, generator=g(10)) * 0.01).to(f8).t()
    mm = lambda x, w: torch._scaled_mm(
        x.to(f8), w, scale_a=one, scale_b=one, out_dtype=bf
    )
    proj = lambda rows: mm(rows, w_o)
    moe = lambda h: mm(mm(h, w_up), w_dn)

    def rccl():
        p = proj(a)
        dist.all_reduce(p)
        m = moe(p)
        dist.all_reduce(m)
        return m

    def sdma(chunks):
        attn = start(T, chunks, lambda r0, r1: proj(a[r0:r1]))
        moe_out = {}
        for r0, r1, rows in attn.chunks():
            moe_out[r0] = moe(rows)
        mlp = start(T, chunks, lambda r0, r1: moe_out[r0])
        out = torch.empty(T, H, device=dev, dtype=bf)
        for r0, r1, rows in mlp.chunks():
            out[r0:r1] = rows
        return out

    t_compute = bench(lambda: moe(proj(a)), gloo)
    t_rccl = bench(rccl, gloo)
    rows = [(c, bench(lambda: sdma(c), gloo)) for c in CHUNKS]
    if rank == 0:
        print(f"T={T}: compute only {t_compute:.3f} ms, RCCL layer {t_rccl:.3f} ms")
        for c, t in rows:
            print(f"  SDMA chunks={c}: {t:.3f} ms ({t_rccl - t:+.3f} vs RCCL)")


def worker(rank):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29667")
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=2, device_id=dev)
    gloo = dist.new_group(backend="gloo")
    start = make_all_reduce(rank, gloo, 32768)
    check_correctness(rank, gloo, dev, start)
    for T in (12000, 32768):
        time_layer(rank, gloo, dev, start, T)
    dist.barrier(group=gloo)
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, nprocs=2)
