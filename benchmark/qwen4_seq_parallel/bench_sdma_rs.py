"""Qwen4-Exp sequence parallelism: SdmaReduceScatter correctness, and a layer-like timing vs aiter.

Layer: out_proj FP8 GEMM [T, 6144/N] -> [T, 2560] partial -> reduce-scatter -> hc_combine on own rows.

    WORLD=2 HIP_VISIBLE_DEVICES=0,1 python benchmark/qwen4_seq_parallel/bench_sdma_rs.py
    WORLD=4 HIP_VISIBLE_DEVICES=0,1,2,3 python benchmark/qwen4_seq_parallel/bench_sdma_rs.py
"""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

N = int(os.environ.get("WORLD", "4"))
H, HC = 2560, 4


def bench(fn, iters=20):
    for _ in range(3):
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
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29623")
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("gloo", rank=rank, world_size=N)
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    from sglang.kernels.ops.elementwise.hc_combine import hc_combine
    from sglang.srt.layers.sdma_reduce_scatter import SdmaReduceScatter

    ca = CustomAllreduce(dist.group.WORLD, dev)
    bf = torch.bfloat16

    # ---- correctness
    for transport, tol in (("bf16", 1e-2), ("fp8", 4e-2)):
        rs = SdmaReduceScatter(
            capacity_tokens=16384,
            hidden_size=H,
            dtype=bf,
            device=dev,
            cpu_group=dist.group.WORLD,
            transport=transport,
        )
        worst = 0.0
        for it, T in enumerate([12000, 11998, 4097, 16384, 12000, 11998]):
            L = -(-T // N)
            x = torch.randn(
                T,
                H,
                device=dev,
                dtype=bf,
                generator=torch.Generator(device=dev).manual_seed(it * 10 + rank),
            )
            full = x.float().cpu()
            dist.all_reduce(full)
            full = torch.nn.functional.pad(full, (0, 0, 0, L * N - T))
            ref = full[rank * L : (rank + 1) * L].to(dev)
            # slow producer exposes missing waits
            pending = rs.send(
                rows=lambda r0, r1: (torch.cuda._sleep(2_000_000), x[r0:r1] * 1)[1],
                num_tokens=T,
                local_rows=L,
                chunks=2 + it % 3,
            )
            got = (
                pending.result()
                if it % 2
                else torch.cat([r for _, _, r in pending.chunks()])
            )
            # a collective between sends (the model's all-gather) per the contract
            ca.all_reduce(torch.ones(1024, device=dev, dtype=bf))
            err = ((got.float() - ref).abs().max() / ref.abs().max()).item()
            worst = max(worst, err)
            assert err < tol, (transport, T, err)
        if rank == 0:
            print(f"[N={N}] {transport}: correct, worst rel err {worst:.1e}")

    # ---- layer-like timing
    T = 12000
    L = T // N
    K = 6144 // N
    g = torch.Generator(device=dev).manual_seed(rank)
    a = (torch.randn(T, K, device=dev, generator=g) * 0.1).to(torch.float8_e4m3fnuz)
    w = (
        (
            torch.randn(
                H, K, device=dev, generator=torch.Generator(device=dev).manual_seed(7)
            )
            * 0.1
        )
        .to(torch.float8_e4m3fnuz)
        .t()
    )
    one = torch.tensor(1.0, device=dev)
    proj = lambda rows: torch._scaled_mm(
        rows, w, scale_a=one, scale_b=one, out_dtype=bf
    )
    g2 = torch.Generator(device=dev).manual_seed(11 + rank)
    residual = torch.randn(L, HC * H, device=dev, dtype=bf, generator=g2)
    normed = torch.randn(L, HC * H, device=dev, dtype=bf, generator=g2)
    weight = torch.randn(HC, HC * H, device=dev, dtype=bf, generator=g2) * 0.01
    combine = lambda y, out=None, rows=slice(None): hc_combine(
        y, residual[rows], normed[rows], weight, HC, H, out=out
    )
    rs_out = torch.empty(L, H, device=dev, dtype=bf)
    barrier_buf = torch.ones(1024, device=dev, dtype=bf)

    def base():
        ca.reduce_scatter(proj(a), rs_out)
        out = combine(rs_out)
        ca.all_reduce(
            barrier_buf
        )  # stands in for the all-gather that follows in the model
        return out

    t_gemm = bench(lambda: proj(a))
    t_rs = bench(lambda: ca.reduce_scatter(proj(a), rs_out)) - t_gemm
    t_base = bench(base)
    ref = base().float()
    if rank == 0:
        print(
            f"[N={N}] layer: GEMM {t_gemm:.3f} ms, aiter RS {t_rs:.3f} ms, GEMM+RS+combine(+barrier) {t_base:.3f} ms"
        )
    for transport in ("bf16", "fp8"):
        rs = SdmaReduceScatter(
            capacity_tokens=16384,
            hidden_size=H,
            dtype=bf,
            device=dev,
            cpu_group=dist.group.WORLD,
            transport=transport,
        )
        for chunks in (1, 2, 4):

            def sdma():
                pending = rs.send(
                    rows=lambda r0, r1: proj(a[r0:r1]),
                    num_tokens=T,
                    local_rows=L,
                    chunks=chunks,
                )
                out = torch.empty_like(residual)
                for r0, r1, rows in pending.chunks():
                    combine(rows, out=out[r0:r1], rows=slice(r0, r1))
                ca.all_reduce(barrier_buf)
                return out

            t = bench(sdma)
            err = ((sdma().float() - ref).abs().max() / ref.abs().max()).item()
            if rank == 0:
                print(
                    f"  SDMA RS {transport:4s} chunks={chunks}: {t:.3f} ms ({t_base - t:+.3f} vs aiter), err {err:.1e}"
                )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, nprocs=N)
