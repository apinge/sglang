# SDMA all-reduce pipeline for TP2 (ROCm)

Opt-in prefill path for Qwen4-Exp (Qwen3.8-Flash-Next) at TP2 on MI300-class GPUs.
The per-layer all-reduces after the attention output projection and after the MoE
are moved from CU kernels to the SDMA copy engine and pipelined over row chunks:
each rank pulls the peer's partial sums chunk by chunk while the CUs keep working,
and the hyper-connection combine consumes each chunk as soon as it has landed.
Optionally the peer's partial sums are sent as FP8 (or INT8) with one bf16 scale
per 32 values, roughly halving the bytes on the link.

Recommended configuration:

```bash
SGLANG_QWEN4_SDMA_PROJ_PIPELINE=1 \
SGLANG_QWEN4_SDMA_PIPELINE_CONSUMER=1 \
SGLANG_QWEN4_SDMA_PIPELINE_MOE=1 \
SGLANG_QWEN4_SDMA_PIPELINE_TRANSPORT=fp8 \
  bash scripts/launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh
```

| Knob | Default | Effect |
|---|---|---|
| `SGLANG_QWEN4_SDMA_PROJ_PIPELINE` | 0 | attention o_proj / GDN out_proj: row-chunked GEMM + SDMA reduce |
| `SGLANG_QWEN4_SDMA_PROJ_PIPELINE_CHUNKS` | 4 | row chunks per reduce |
| `SGLANG_QWEN4_SDMA_PROJ_PIPELINE_MIN_TOKENS` | 4096 | smaller batches (decode, short prefill) keep the regular all-reduce |
| `SGLANG_QWEN4_SDMA_PIPELINE_CONSUMER` | 0 | combine each chunk as it lands instead of waiting for the full sum |
| `SGLANG_QWEN4_SDMA_PIPELINE_MOE` | 0 | also send the MoE output all-reduce over SDMA |
| `SGLANG_QWEN4_SDMA_PIPELINE_MOE_CHUNKS` | 1 | >1 splits the MoE by tokens (slower on MI308X) |
| `SGLANG_QWEN4_SDMA_PIPELINE_TRANSPORT` | bf16 | `bf16` (exact), `int8` or `fp8` (lossy, ~half the bytes) |

Decode, CUDA graph capture and other TP sizes keep the original all-reduce.

## Scripts

| Script | What it measures |
|---|---|
| `bench_sdma_overlap.py` | Peer copy concurrent with GEMM. SDMA overlaps fully; with `HSA_ENABLE_SDMA=0` (blit kernels) it does not. |
| `bench_proj_pipeline.py` | FP8 o_proj GEMM + aiter all-reduce vs `SdmaTp2Pipeline`; `--combine` adds the hyper-connection combine, `--transport` picks the wire format. Includes a numerical check. |

## Results (8x MI308X, TP2, 12000 in / 350 out)

`bench_sdma_overlap.py`, 29 MiB pull vs bf16 GEMMs: SDMA 0.71-0.74 ms (41-43 GB/s) with overlap 1.0;
blit kernels 0.77 ms with overlap 0.31.

`bench_proj_pipeline.py --combine`, one attention layer (12000 x 3072 -> 2560 FP8 GEMM + reduce + combine):

| | ms | max abs err / max ref |
|---|---|---|
| GEMM + aiter all-reduce + combine | 2.55 | - |
| pipeline, bf16, 8 chunks | 1.64 | 4.0e-3 (rounding) |
| pipeline, int8, 8 chunks | 1.63 | 8.0e-3 |
| pipeline, fp8, 8 chunks | 1.61 | 1.6e-2 |

Prefill step (12k tokens) and serving throughput (`bench_serving`, interleaved rounds):

| | prefill step | req/s c16 | req/s c32 |
|---|---|---|---|
| baseline | 745 ms | - | - |
| SDMA pipeline, bf16 | 688 ms | +6.1-6.5% | +6.7-7.0% |
| SDMA pipeline, int8 | 667-670 ms | +7.9-8.7% | +9.3-9.5% |
| SDMA pipeline, fp8 | 667 ms | +8.7% | +10.0% |

gsm8k and a long-context probe (passkey + counting, 12-16k tokens) are unchanged in all
configurations, and greedy outputs on the probe match the baseline.
