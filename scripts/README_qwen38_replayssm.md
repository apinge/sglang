# Qwen3.8 Flash Next non-MTP ReplaySSM on MI308X

This fork supports ordinary GDN ReplaySSM decode with radix caching either
disabled or enabled through `extra_buffer`. The validated configuration uses
ROCm MI308X (`gfx942`), pure TP2 or TP4, the
`Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16` checkpoint, and FP32 SSM state.

ReplaySSM remains **opt-in**. The measured kernel savings did not translate to
lower serving TPOT in the tested workloads: TPOT increased by approximately
3%–15%. This change provides radix compatibility, correct memory accounting,
and a reproducible comparison; it does not enable a new default fast path.

## Implementation and provenance

- Backport the final effect of upstream [#37837](https://github.com/sgl-project/sglang/pull/37837):
  allow GDN `extra_buffer` and reset decode cursors when recycled slots enter
  ping-pong tracking buffers. Both source commits are required:
  `958e507d1e8c576a1002140656e597cc9e5ffd3f` and
  `df944d5f762a37f31c29d1a3bc4b41ce69d4a556`.
- Map upstream's `attention_hook.py` guard to this fork's `server_args.py`,
  and retain this fork's direct `Req` fields. KDA extra-buffer stays rejected.
- Charge ordinary decode D/K/G rings and one shared int32 cursor per physical
  slot to the memory solver, including its padding slot. Decode D/K use the
  SSM dtype; the speculative raw-input window uses a different layout.
- Reject ordinary ReplaySSM with speculative decoding. The MTP/spec protocol
  and the existing PLE side-state copy/restore implementation are unchanged.
- Backport the benchmark correction from
  [#38399](https://github.com/sgl-project/sglang/pull/38399), commit
  `5ae4ccb0fd97d2a903c07e52dea7949f429d1f42`, then add complete-cycle graph
  timing and independently selected input, state, and ring dtypes.

The existing decode kernel force-flushes pending updates before a tracked
checkpoint is saved. A restored checkpoint starts with an empty logical ring;
stale destination ring contents are ignored after resetting its cursor. All
GDN layers consume the same step's cursor snapshot.

## Launch and compare

Run from the repository root with the existing ROCm/AITER/PyHIP environment.
The launch script imports Python code from its own checkout. Set `MODEL_PATH`
if your checkpoint is not at its default location
`/models/Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16`.

```bash
LAUNCHER=scripts/launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh

# TP2, ReplaySSM on, radix off.
HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1 \
TP_SIZE=2 DISABLE_RADIX_CACHE=1 ENABLE_LINEAR_REPLAYSSM=1 \
bash "$LAUNCHER"

# Stop the previous server before running the other configuration.
# TP2, ReplaySSM on, radix extra_buffer on.
HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1 \
TP_SIZE=2 DISABLE_RADIX_CACHE=0 ENABLE_LINEAR_REPLAYSSM=1 \
bash "$LAUNCHER"
```

For TP4, select four visible GPUs and set `TP_SIZE=4`.
`LINEAR_REPLAYSSM_CACHE_LEN` defaults to 16. Enabling ReplaySSM selects Triton
linear decode, FP32 SSM, and `extra_buffer` when radix is enabled. Full
attention/QSA and MoE retain the launcher's AITER/PyHIP configuration.

Use `ENABLE_LINEAR_REPLAYSSM=0` for the baseline, keeping TP, radix, workload,
and all other settings fixed. Remove any manually supplied ReplaySSM enable
flag as well. Leave the launcher's speculative arguments commented out.
The model requires QSA page size 64; `no_buffer` is not a substitute for
`extra_buffer` here.

## Validation

Validation on 2026-10-26 used PyTorch
`2.12.0+rocm7.14.0.git6bbd260`, ROCm `7.14.60850`, Triton 3.8.0,
max-running-requests 32, and decode graph max batch 32.

- 32 CPU tests passed across slot lifecycle, argument guards, accounting,
  existing dispatcher behavior, and retraction backup plumbing.
- Six GPU tests passed with 46 subtests. Target shapes are TP2 H/HV=8/24 and
  TP4 H/HV=4/12, K=V=128, BF16 inputs and FP32 SSM/rings. Coverage includes
  strided input rows, periodic/forced flushes, and graph checkpoint restore.
- All eight TP/radix/ReplaySSM configurations completed generation and full
  GSM8K: 1,314 scored examples, five demonstrations, temperature zero,
  thinking disabled, 4,096 maximum output tokens, and 32 client threads.
- All configurations used graph decode without runtime tracebacks or HTTP
  500 responses. Radix-on runs reused 3,584-token prefill prefixes and
  3,840-token decode checkpoint prefixes. Radix-off hit counts were zero.
- ReplaySSM-on configurations also completed inputs longer than the 16,384
  prefill chunk followed by 128 generated tokens, for both TP sizes and
  radix settings.

| TP | Radix | GSM8K, ReplaySSM off | GSM8K, ReplaySSM on |
|---|---|---|---|
| 2 | Off | 96.73% | 96.80% |
| 2 | Extra buffer | 96.96% | 96.73% |
| 4 | Off | 96.73% | 96.58% |
| 4 | Extra buffer | 96.58% | 96.42% |

The largest paired decrease was 0.23 percentage points. Repeated greedy text
also varied in the original FP8/AITER baseline; full-text equality alone was
not used as the numerical correctness criterion.

Serving TPOT below is the median over three rounds of requests with 128
output tokens. Different TP groups ran on separate GPUs of the same host;
shared host resources limit conclusions about small timing differences.

| TP | Radix | Concurrency 1, off → on | Concurrency 32, off → on |
|---|---|---|---|
| 2 | Off | 7.01 → 7.91 ms | 15.65 → 16.31 ms |
| 2 | Extra buffer | 7.16 → 7.73 ms | 16.26 → 16.82 ms |
| 4 | Off | 6.39 → 7.33 ms | 13.06 → 13.46 ms |
| 4 | Extra buffer | 6.43 → 7.19 ms | 13.09 → 13.68 ms |

## Reproduce focused checks

```bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
  test/registered/unit/mem_cache/test_replayssm_ping_pong.py \
  test/registered/unit/mem_cache/test_replayssm_ring_accounting.py \
  test/registered/unit/mem_cache/test_mamba_donated_alloc_ratio.py

HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
  test/registered/attention/unittests/gdn/test_linear_replayssm_decode.py \
  -k 'gdn_packed_decode or qwen38 or force_flush or flush_boundary_state_exact_l1'

HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 python \
  benchmark/kernels/attention/bench_gdn_replayssm_decode.py \
  --h 8 --hv 24 --batch-sizes 1 2 4 8 16 32 --ls 1 8 16 \
  --dtype bf16 --state-dtype fp32 --mode cycle \
  --json-output /tmp/qwen38_tp2_replayssm_kernel.json
```

For TP4 kernel measurements, use `--h 4 --hv 12`. Cycle mode includes every
cursor phase and the periodic flush, but excludes scheduler metadata and
radix copies. `--mode phases` retains the upstream midpoint/flush estimate.
Kernel timing and checkpoint-traffic estimates must not be interpreted as
end-to-end serving speedups.

MTP, TP8, extra-buffer-lazy serving, PD disaggregation, unified memory,
multimodal/offloaded-PLE combinations, and long-duration soak tests are
outside this validation scope.
