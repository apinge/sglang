# Validated gfx942 linear configurations

This reproduces the completed PTPC FP8, CK decode and BF16 ViT tuning work for
Qwen3.8 TP2 on MI308X (`gfx942`, 80 CUs). It uses the existing AITER dispatcher and
tuners; it does not vendor AITER or change its GEMM algorithms.

The recorded runtime was AITER `514c166e9abad59acde74309923c2d923a89eee0`, Torch
2.12 / ROCm 7.14 / Triton 3.8. Solution indexes and CK kernel IDs belong to that
runtime. Keep the version and hardware identity with each measurement.

## Prepare complete runtime tables

`configs/ck_accepted_rows.csv` contains the one validated CK override;
`configs/tgemm_accepted_rows.csv` contains the four validated FP8/BF16 overrides.
They are **deltas**, not complete runtime tables. Preserve all other baseline
entries by merging them before setting the AITER environment variables:

```bash
python3 benchmark/rocm_tgemm/prepare_configs.py \
  --ck-base /path/to/recorded/ck_base.csv \
  --tgemm-base /path/to/recorded/tgemm_base.csv \
  --output-dir /path/to/fresh/merged-configs

export AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE=/path/to/fresh/merged-configs/ck_accepted.csv
export AITER_CONFIG_GEMM_BF16=/path/to/fresh/merged-configs/tgemm_accepted.csv
export SGLANG_USE_AITER=1
export SGLANG_ROCM_USE_AITER_LINEAR_FP8HIPB=1
export SGLANG_ROCM_USE_AITER_LINEAR_SHUFFLE=1
```

Supply the baseline tables from the verified AITER runtime. The helper records
input/overlay/output hashes, preserves unmatched entries and rejects duplicate
shape keys or input/output overwrites. It also copies the two baseline files for
the before measurement. Complete baseline snapshots and raw results belong to
the separate PR evidence branch, rather than this source directory.

## Reproduce the native comparison

Choose a free reserved GPU and inspect utilization, VRAM and process ownership
before running. The following uses the recorded five eager and three graph
samples per shape, 10 warmups / 101 iterations, 16 rotating tensor sets, and two
changed-input graph replays. Numerical output is checked against dense FP32
arithmetic. All 16 shapes are retained, including unchanged-path controls.

```bash
export HIP_VISIBLE_DEVICES=0
export AITER_USE_SYSTEM_TRITON=1
python3 benchmark/rocm_tgemm/benchmark_gemm.py \
  --configs-dir /path/to/fresh/merged-configs \
  --phase base \
  --output /path/to/results/base.json
python3 benchmark/rocm_tgemm/benchmark_gemm.py \
  --configs-dir /path/to/fresh/merged-configs \
  --phase accepted \
  --output /path/to/results/accepted.json
```

These are operator measurements, not model TTFT/TPOT/ETE or GSM8K. Use the same
original AITER source and prepared kernels for both phases; do not compare one
prebuilt cache with a differently patched rebuilt runtime.

## Reproduce all six regular CI cases

Native `qwen4_ci.py` cleans inherited environment variables. Use this thin wrapper
so the two flags and prepared table paths reach the **server**; client environment,
workloads and accuracy gates remain native and unchanged. The before/accepted
phase selects tables, not a source checkout: run the matching source revision in
each comparison arm and record that revision separately.

```bash
python3 benchmark/rocm_tgemm/run_ci.py \
  --configs-dir /path/to/fresh/merged-configs \
  --phase accepted \
  run \
  --case Qwen3.8-Flash-TP2-Base,Qwen3.8-Flash-TP2-MTP,Qwen3.5-35B-TP4-Base,Qwen3.5-35B-TP1-DFlash,Qwen3.5-397B-TP8-Base,Qwen3.5-397B-TP8-DFlash \
  --models-root /models \
  --datasets-root /datasets \
  --client-root /ci/client \
  --gpus 0,1,2,3,4,5,6,7 \
  --output /path/to/results/ci-accepted
```

Use `--phase base` with another output directory for the baseline tables. This
runs the existing 128-request / 200-example regular CI protocol; it does not alter
the native case selection, GPU mask, benchmark specifications or accuracy gates.

## Tuner entry points and accepted scope

The input files contain nine CK decode shapes (`M=1/4/16`) and seven tgemm shapes
(three FP8 prefill shapes plus four BF16 ViT shapes). Existing AITER commands were:

```bash
python3 /path/to/aiter/csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py \
  -i benchmark/rocm_tgemm/configs/ck_decode_shapes.csv \
  -o /path/to/results/ck_decode_tuned.csv \
  --profile_file /path/to/results/ck_candidates.csv \
  --libtype ck \
  --splitK \
  --mp 1 \
  --warmup 10 \
  --iters 101 \
  --errRatio 0.01
python3 /path/to/aiter/csrc/gemm_a16w16/gemm_a16w16_tune.py \
  -i benchmark/rocm_tgemm/configs/tgemm_shapes.csv \
  -o /path/to/results/tgemm_tuned.csv \
  --profile_file /path/to/results/tgemm_candidates.csv \
  --libtype hipblaslt,asm,torch \
  --with-hipblaslt \
  --mp 1 \
  --errRatio 0.01
```

The historical AITER search needed two temporary tuning-only fixes: BF16
`scaleAB=False` must supply no scale tensors, and unsupported hipBLASLt solutions
must raise instead of aborting the worker. Those patches/checkpoint recovery are
retained as external tuning evidence. Apply them only in an isolated AITER tuning
checkout if reproducing that exact search, and restore the original source before
production-interface verification. This source kit applies already validated
rows; it never patches the installed library or automatically accepts tuner output.

The selected changes are one CK row `(4,2560,3072)`; two FP8 rows
`(8192,8192,2560)` / `(8192,6656,2560)`; and two BF16 rows
`(256,1152,2152)` / `(256,1152,576)`. The other candidates were not accepted.
Selection required at least 3% lower median latency in both eager and graph
comparison, followed by numerical and changed-input replay validation. Re-running
a tuner produces new experimental data, not automatically validated replacements.
