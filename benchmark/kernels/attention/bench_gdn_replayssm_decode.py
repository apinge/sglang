"""Microbenchmark: buffered output-only GDN decode (ReplaySSM Part A) vs. the
existing packed GDN decode kernel.

Compares per-step decode latency of
``fused_recurrent_gated_delta_rule_packed_decode`` (writes the full recurrent
state S every step) against ``fused_recurrent_gdn_replayssm_decode`` at
L in {1, 8, 16} (writes the full state only every L steps) across batch sizes
{1, 16, 64, 256} for a realistic GDN config (HV=32, K=V=128).

The win is per-step HBM *state* traffic: the packed kernel reads + writes S
(~2 * num_slots * HV * V * K * 4 bytes / step for an fp32 state), while the
ReplaySSM kernel reads S every step but writes it only 1-in-L steps, plus a
small ring append (d:[HV,V], k:[H,K], g:[HV] per step). The amortized state
traffic ratio is reported per L.

Default measurement captures a complete 0..L-1 cursor cycle in a CUDA/HIP
graph and divides latency by L. The baseline runs L packed updates in the
same way. Cursor tensors are preallocated, so this is kernel-only timing,
excluding scheduler metadata and radix copies. ``--mode phases`` retains
the midpoint/non-flush plus flush estimate from upstream #38399.

Inputs default to BF16 and both checkpoint and ring to FP32, matching the
ordinary decode pool. Speculative ReplaySSM has a different ring protocol.

Run::

    python benchmark/kernels/attention/bench_gdn_replayssm_decode.py

Requires a GPU (Triton).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import triton

from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
)
from sglang.kernels.ops.attention.fla.fused_recurrent_linear_replayssm import (
    fused_recurrent_gdn_replayssm_decode,
)


def _make_static(B, H, HV, K, V, dtype, device):
    qk_dim = 2 * H * K
    v_dim = HV * V
    mixed_qkv = torch.randn(B, qk_dim + v_dim, device=device, dtype=dtype)
    a = torch.randn(B, HV, device=device, dtype=dtype) * 0.5
    b = torch.randn(B, HV, device=device, dtype=dtype)
    A_log = (torch.randn(HV, device=device, dtype=torch.float32) * 0.3).contiguous()
    dt_bias = (torch.randn(HV, device=device, dtype=torch.float32) * 0.1).contiguous()
    return mixed_qkv, a, b, A_log, dt_bias


def _state_bytes_per_step(B, HV, K, V, L, state_dtype):
    """Checkpoint-only traffic estimate; excludes ring reads/writes."""
    state_bytes = state_dtype.itemsize
    state_elems = B * HV * V * K  # one record per active request slot
    packed = (state_elems * state_bytes) * 2
    replay = (state_elems * state_bytes) * (1 + 1.0 / L)
    return packed, replay


def _amortize(t_nonflush, t_flush, L):
    """Mean per-step latency over one full L-phase cycle.

    `write_pos` walks 0, 1, ... L-1 and wraps, so a steady-state cycle is L-1
    non-flush steps and one flush step. The non-flush steps are not identical --
    each replays one more ring entry than the last -- so the midpoint phase is
    sampled as their representative rather than phase 0, which replays nothing.
    Use cycle mode when a measured complete cycle is needed.
    """
    if L == 1:
        return t_flush
    return ((L - 1) * t_nonflush + t_flush) / L


def _bench_cfg(
    B,
    H,
    HV,
    K,
    V,
    Ls,
    dtype,
    device,
    nk=2,
    num_slots=None,
    warmup=25,
    rep=100,
    state_dtype=torch.float32,
    ring_dtype=torch.float32,
    mode="cycle",
):
    num_slots = num_slots or B
    mixed_qkv, a, b, A_log, dt_bias = _make_static(B, H, HV, K, V, dtype, device)
    scale = K**-0.5
    cache_indices = torch.arange(B, device=device, dtype=torch.int32)

    # packed decode
    state = torch.randn(num_slots, HV, V, K, device=device, dtype=state_dtype)
    out = mixed_qkv.new_empty(B, 1, HV, V)

    def run_packed():
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=state,
            out=out,
            ssm_state_indices=cache_indices,
            use_qk_l2norm_in_kernel=True,
        )

    t_packed = (
        triton.testing.do_bench(run_packed, warmup=warmup, rep=rep)
        if mode == "phases"
        else None
    )

    rows = []
    for L in Ls:
        rstate = state.clone()
        d_cache = torch.zeros(num_slots, HV, L, V, device=device, dtype=ring_dtype)
        k_cache = torch.zeros(num_slots, H, L, K, device=device, dtype=ring_dtype)
        g_cache = torch.zeros(num_slots, HV, L, device=device, dtype=torch.float32)
        write_pos = torch.empty(B, device=device, dtype=torch.int32)
        rout = mixed_qkv.new_empty(B, 1, HV, V)

        def run_replay(cursor=write_pos):
            fused_recurrent_gdn_replayssm_decode(
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=scale,
                initial_state=rstate,
                d_cache=d_cache,
                k_cache=k_cache,
                g_cache=g_cache,
                out=rout,
                ssm_state_indices=cache_indices,
                write_pos=cursor,
                use_qk_l2norm_in_kernel=True,
                nk=nk,
            )

        def time_at(pos):
            write_pos.fill_(pos)
            return triton.testing.do_bench(run_replay, warmup=warmup, rep=rep)

        # The kernel branches on write_pos: it appends to the ring on every step
        # but folds the ring into the checkpoint only when write_pos == L-1. Both
        # phases have to be timed, or the reported cost is the cheap one repeated
        # (see `_amortize`). L == 1 is all flush, so it has no non-flush phase.
        if mode == "phases":
            t_flush = time_at(L - 1)
            t_nonflush = time_at((L - 1) // 2) if L > 1 else None
            t_replay = _amortize(t_nonflush, t_flush, L)
            t_reference = t_packed
        else:
            cursors = [torch.full_like(write_pos, pos) for pos in range(L)]

            def replay_cycle():
                for cursor in cursors:
                    run_replay(cursor)

            def packed_cycle():
                for _ in range(L):
                    run_packed()

            t_reference = triton.testing.do_bench_cudagraph(packed_cycle, rep=rep) / L
            t_replay = triton.testing.do_bench_cudagraph(replay_cycle, rep=rep) / L
            t_nonflush = t_flush = None

        packed_bytes, replay_bytes = _state_bytes_per_step(B, HV, K, V, L, state_dtype)
        rows.append(
            dict(
                batch_size=B,
                cache_len=L,
                packed_ms=t_reference,
                replay_ms=t_replay,
                nonflush_ms=t_nonflush,
                flush_ms=t_flush,
                speedup=t_reference / t_replay,
                checkpoint_traffic_ratio=replay_bytes / packed_bytes,
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hv", type=int, default=32, help="num value heads")
    parser.add_argument("--h", type=int, default=16, help="num key/query heads")
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--v", type=int, default=128)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16, 64, 256])
    parser.add_argument("--ls", type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--state-dtype", choices=["bf16", "fp32"], default="fp32")
    parser.add_argument(
        "--ring-dtype",
        choices=["bf16", "fp32"],
        default=None,
        help="Defaults to state dtype, as in the decode pool.",
    )
    parser.add_argument("--mode", choices=["cycle", "phases"], default="cycle")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--rep", type=int, default=100, help="Timing duration in ms")
    parser.add_argument(
        "--nk", type=int, default=2, help="K-chunks the reconstruction walks"
    )
    args = parser.parse_args()
    if any(L < 1 for L in args.ls):
        parser.error("all cache lengths must be positive")
    if args.h < 1 or args.hv % args.h:
        parser.error("HV must be a positive multiple of H")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA / Triton required for this microbenchmark.")
    device = "cuda"
    dtypes = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    args.ring_dtype = args.ring_dtype or args.state_dtype

    print(
        f"GDN ReplaySSM decode microbench  HV={args.hv} H={args.h} "
        f"K={args.k} V={args.v} input={args.dtype} state={args.state_dtype} "
        f"ring={args.ring_dtype} nk={args.nk} mode={args.mode}\n"
        "Per-step kernel latency in ms; speedup = packed/replay. "
        "Checkpoint traffic estimate excludes ring traffic."
    )
    results = []
    for B in args.batch_sizes:
        rows = _bench_cfg(
            B,
            args.h,
            args.hv,
            args.k,
            args.v,
            args.ls,
            dtypes[args.dtype],
            device,
            nk=args.nk,
            state_dtype=dtypes[args.state_dtype],
            ring_dtype=dtypes[args.ring_dtype],
            mode=args.mode,
            rep=args.rep,
        )
        results.extend(rows)
        for row in rows:
            print(
                f"B={B:<4d} L={row['cache_len']:<3d} "
                f"packed={row['packed_ms']:.4f} replay={row['replay_ms']:.4f} "
                f"speedup={row['speedup']:.2f}x"
            )
    if args.json_output:
        config = {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        }
        args.json_output.write_text(
            json.dumps(dict(config=config, results=results), indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
