"""Tune the gfx942 Triton skinny bf16 GEMM (sglang.kernels.ops.gemm.rocm_bf16_gemv).

Prints a "(N, K): {M bound: config}" entry to paste into _GFX942_CONFIGS in
rocm_bf16_gemv.py, plus per-bucket timings. Run one process per N on an idle GPU:

    HIP_VISIBLE_DEVICES=2 python benchmark/kernels/rocm_bf16_gemv/tune_rocm_bf16_gemv.py \
        --n 124160 --k 2560

Search is coarse-to-fine: tile shape first (BLOCK_N, BLOCK_K, num_warps) with neutral
knobs, then waves_per_eu / num_stages / cache_modifier on the best tiles.
Every candidate is checked against an fp32 reference before it is timed. Only add an
entry when the tuned config beats hipBLASLt in every bucket: listed shapes are the
only ones routed to these kernels.
"""

import argparse
import itertools
import json
import statistics

import torch

from sglang.kernels.ops.gemm.rocm_bf16_gemv import _GFX942_CONFIGS, _launch

M_BOUNDS = (1, 2, 4, 8, 16, 24, 32)


def _time_us(fn, iters):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000


def _tile_candidates(m, k):
    if m == 1:
        for bn, bk, warps in itertools.product(
            (8, 16, 32, 64), (128, 256, 512), (1, 2, 4, 8)
        ):
            if bn * bk <= 32768 and k % bk == 0:
                yield dict(
                    BLOCK_SIZE_N=bn,
                    BLOCK_SIZE_K=bk,
                    num_warps=warps,
                    num_stages=2,
                    waves_per_eu=0,
                    cache_modifier=None,
                )
        return
    block_m = 16 if m <= 16 else 32
    for bn, bk, warps in itertools.product(
        (16, 32, 64, 128, 256), (64, 128, 256, 512), (1, 2, 4, 8)
    ):
        if block_m * bk + bn * bk <= 65536 and k % bk == 0:
            yield dict(
                BLOCK_SIZE_M=block_m,
                BLOCK_SIZE_N=bn,
                BLOCK_SIZE_K=bk,
                num_warps=warps,
                num_stages=1,
                waves_per_eu=0,
                matrix_instr_nonkdim=16,
                kpack=2,
                cache_modifier=None,
            )


def _knob_candidates(tile):
    for wpe, stages, cache in itertools.product((0, 1, 2, 3, 4), (1, 2), (None, ".cg")):
        yield dict(tile, waves_per_eu=wpe, num_stages=stages, cache_modifier=cache)


def _screen(x, w, ref, tol, candidates, iters):
    timed = []
    for cfg in candidates:
        try:
            out = _launch(x, w, cfg)
            torch.cuda.synchronize()
        except Exception:  # compile failure (registers / LDS) rules the config out
            continue
        if (out.float() - ref).abs().max().item() > tol:
            continue
        timed.append((_time_us(lambda: _launch(x, w, cfg), iters), cfg))
    timed.sort(key=lambda item: item[0])
    return timed


def _final_time(fn, repeats=3, iters=200):
    return statistics.median(_time_us(fn, iters) for _ in range(repeats))


def tune_bucket(w, m):
    n, k = w.shape
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    ref = x.float() @ w.float().t()
    blas_err = ((x @ w.t()).float() - ref).abs().max().item()
    tol = max(blas_err * 2.0, 1e-2)

    tiles = _screen(x, w, ref, tol, _tile_candidates(m, k), iters=20)[:8]
    knobs = [c for _, tile in tiles for c in _knob_candidates(tile)]
    finalists = [cfg for _, cfg in _screen(x, w, ref, tol, knobs, iters=20)[:5]]

    t_blas = _final_time(lambda: torch.matmul(x, w.t()))
    best = min(
        ((_final_time(lambda: _launch(x, w, cfg)), cfg) for cfg in finalists),
        key=lambda item: item[0],
    )
    timing = dict(tuned_us=round(best[0], 1), hipblaslt_us=round(t_blas, 1))
    # Compare with the shipped entry so a re-tune only replaces it when it wins.
    current = _GFX942_CONFIGS.get((n, k))
    if current is not None:
        cfg = current[min(b for b in current if m <= b)]
        timing["current_us"] = round(_final_time(lambda: _launch(x, w, cfg)), 1)
    return best[1], timing


def _format_entry(n, k, configs):
    lines = [f"    ({n}, {k}): {{"]
    for bound, c in configs.items():
        knobs = (
            f"{c['BLOCK_SIZE_N']}, {c['BLOCK_SIZE_K']}, {c['num_warps']}, "
            f"{c['num_stages']}, {c['waves_per_eu']}, {c['cache_modifier']!r}"
        )
        if "BLOCK_SIZE_M" in c:
            lines.append(f"        {bound}: _mfma({c['BLOCK_SIZE_M']}, {knobs}),")
        else:
            lines.append(f"        {bound}: _gemv({knobs}),")
    lines.append("    },")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--m-bounds", type=int, nargs="+", default=list(M_BOUNDS))
    args = parser.parse_args()

    torch.manual_seed(0)
    w = torch.randn(args.n, args.k, dtype=torch.bfloat16, device="cuda") * 0.05
    configs = {}
    for bound in args.m_bounds:
        cfg, timing = tune_bucket(w, bound)
        configs[bound] = cfg
        print(f"N={args.n} K={args.k} M<={bound}: {json.dumps(timing)}", flush=True)
    print(_format_entry(args.n, args.k, configs))


if __name__ == "__main__":
    main()
