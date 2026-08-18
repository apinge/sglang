# Qwen3.5 CI Launch Comparison

This compares the current GitHub workflow launch path with the three local launch scripts:

```text
.github/workflows/sglang_benchmark_workflow.yaml
scripts/ci/sglang_benchmark_workflow.sh

/opt/evaluation7/launch_qwen3.5-397B-fp8_tp8_prefix_cache_origin_ali2.sh
/opt/evaluation7/launch_qwen3.5-35B-fp8_tp4_origin.sh
/opt/evaluation7/launch_qwen3.5_27B_FP8_tp.sh
```

## CI Baseline

The workflow resolves model config in `.github/workflows/sglang_benchmark_workflow.yaml`, then launches through:

```bash
bash scripts/ci/sglang_benchmark_workflow.sh launch "${MODEL_NAME}" "${MODEL_PATH}" "${MODEL_TP}" 1 45
```

The shared CI launch script always sets:

```bash
export SGLANG_USE_AITER=1
export SGLANG_ROCM_USE_AITER_LINEAR_SHUFFLE=1
export SGLANG_ROCM_USE_AITER_LINEAR_FP8HIPB=1
export SGLANG_USE_AITER_NEW_CA=false
export USE_AITER_COMM=1
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
```

For non-27B models, CI also sets:

```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT6
```

For the 27B case, CI leaves `AITER_QUICK_REDUCE_QUANTIZATION` unset.

And launches with:

```text
--attention-backend aiter
--mm-attention-backend aiter_attn
--linear-attn-backend aiter
--linear-attn-decode-backend aiter
--linear-attn-prefill-backend aiter
--chunked-prefill-size 32768
--max-prefill-tokens 32768
--max-running-requests 128
--cuda-graph-max-bs 128
--kv-cache-dtype fp8_e4m3
--disable-radix-cache
--watchdog-timeout 1200
```

CI currently adds `--disable-custom-all-reduce` for the default 27B case only.

## 397B

Local script:

```text
/opt/evaluation7/launch_qwen3.5-397B-fp8_tp8_prefix_cache_origin_ali2.sh
```

Overall: mostly aligned.

| Item | Local 397B | CI |
| --- | --- | --- |
| Model path | `/models/Qwen/Qwen3.5-397B-A17B-PTPC-FP8` | `/models/offical_qwen3p5_397B_ptpc` |
| TP | `8` | `8` |
| AITER | enabled | enabled |
| `SGLANG_USE_AITER_NEW_CA` | `false` | `false` |
| `USE_AITER_COMM` | `1` | `1` |
| `AITER_QUICK_REDUCE_QUANTIZATION` | `INT6` | `INT6` |
| `--disable-custom-all-reduce` | not set | not set |
| `--disable-radix-cache` | set | set |
| `--watchdog-timeout` | `1200` | `1200` |
| `--cuda-graph-max-bs` | not set | `128` |

Notes:

1. Communication mode matches the intended custom-reduce path.
2. CI adds `--cuda-graph-max-bs 128`, while the local 397B script does not.
3. CI uses a different model directory alias, but it appears to point to the same 397B PTPC FP8 model family.

## 35B

Local script:

```text
/opt/evaluation7/launch_qwen3.5-35B-fp8_tp4_origin.sh
```

Overall: mostly aligned for custom-reduce mode.

| Item | Local 35B | CI |
| --- | --- | --- |
| Model path | `/models/Qwen/Qwen3.5-35B-A3B-PTPC-FP8` | `/models/Qwen3.5-35B-A3B-PTPC-compressor` |
| TP | `4` | `4` |
| AITER | enabled | enabled |
| `SGLANG_USE_AITER_NEW_CA` | `false` | `false` |
| `USE_AITER_COMM` | `1` | `1` |
| `AITER_QUICK_REDUCE_QUANTIZATION` | `INT6` | `INT6` |
| `--disable-custom-all-reduce` | not set | not set |
| `--disable-radix-cache` | set | set |
| `--cuda-graph-max-bs` | not set | `128` |
| `--watchdog-timeout` | not set | `1200` |
| `HIP_GDN_SORT_IDX_BS` | `32768` | not set |
| `TVM_FFI_DISABLE_TORCH_C_DLPACK` | `1` | `1` |

Notes:

1. CI covers 35B custom-reduce mode, not 35B NCCL/RCCL fallback mode.
2. CI uses the `PTPC-compressor` model directory, while the local script uses `PTPC-FP8`.
3. CI adds `--cuda-graph-max-bs 128` and `--watchdog-timeout 1200`.
4. Local script sets `HIP_GDN_SORT_IDX_BS=32768`; CI does not.

## 27B

Local script:

```text
/opt/evaluation7/launch_qwen3.5_27B_FP8_tp.sh
```

Overall: **not aligned**.

| Item | Local 27B | CI |
| --- | --- | --- |
| Model path | `/models/Qwen/Qwen3.5-27B-PTPC-FP8` | `/models/Qwen3.5-27B-PTPC-compressor` |
| TP | `2` | `1` |
| AITER | enabled | enabled |
| `SGLANG_USE_AITER_NEW_CA` | `false` | `false` |
| `USE_AITER_COMM` | not set | `1` |
| `AITER_QUICK_REDUCE_QUANTIZATION` | not set, `INT8` commented out | not set |
| `--disable-custom-all-reduce` | set | set |
| `--disable-radix-cache` | not set | set |
| `--mamba-scheduler-strategy` | `extra_buffer` | not set |
| `--page-size` | `64` | not set |
| `--mem-fraction-static` | `0.8` | `0.9` |
| `--watchdog-timeout` | not set | `1200` |
| `HIP_GDN_SORT_IDX_BS` | `32768` | not set |
| `SGLANG_DISABLE_AITER_FUSED_AR_RMSNORM` | set in local script | not set in CI |

Notes:

1. CI now passes `--disable-custom-all-reduce` for 27B, but still uses TP1.
2. Local 27B uses TP2 with `--disable-custom-all-reduce`, so CI does not validate the multi-GPU all-reduce fallback shape.
3. The local `SGLANG_DISABLE_AITER_FUSED_AR_RMSNORM` env is stale after the repo-side env gate was removed; with current code it should not affect behavior.
4. CI leaves `AITER_QUICK_REDUCE_QUANTIZATION` unset for 27B, matching the local script.

## Current Coverage Summary

| Scenario | Covered by current CI? | Notes |
| --- | --- | --- |
| 397B custom reduce, TP8 | Yes | Mostly matches local 397B script. |
| 35B custom reduce, TP4 | Yes | Mostly matches local 35B script, with model path/env differences. |
| 35B NCCL/RCCL fallback | No | No default CI case passes `--disable-custom-all-reduce` for 35B. |
| 27B NCCL/RCCL fallback, TP2 | Partial | CI passes `--disable-custom-all-reduce`, but runs 27B as TP1. |

## Suggested CI Follow-Up

To match the intended coverage for this PR, CI should distinguish communication modes instead of only model names:

```text
397B custom_reduce TP8
35B custom_reduce TP4
35B nccl_fallback TP4 --disable-custom-all-reduce
27B nccl_fallback TP2 --disable-custom-all-reduce
```

The remaining custom-reduce/fallback coverage gaps are 35B NCCL/RCCL fallback and the local 27B TP2 fallback shape.
