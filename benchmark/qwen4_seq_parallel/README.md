# Sequence-parallel hyper-connection residual for Qwen4-Exp TP prefill

Qwen4-Exp (Qwen3.8-Flash-Next) keeps a 4-lane hyper-connection residual stream. Between the TP-sharded
blocks, three per-token ops run on it: `hc_combine` (inject the block output), `hc_norm`
(grouped Gemma RMSNorm, variance per token and per 2560-wide lane) and the low-rank `mix` that builds the
next block's input. With the usual layout every TP rank repeats them on all tokens; in a 12k-token
prefill step that is ~165 ms per rank (31% of the step at TP4, 22% at TP2).

With `SGLANG_QWEN4_SEQ_PARALLEL=1` (the default) each rank keeps only its 1/TP slice of the token rows of the residual
stream during prefill:

```
partial block output [T, 2560]
  -> reduce-scatter            (this rank's T/TP rows, summed)
  -> hc_combine, hc_norm, mix  (on T/TP rows)
  -> all-gather                (mixed input, [T, 2560])
  -> attention / GDN / MoE     (unchanged, on all T rows; MoE skips its own all-reduce)
```

The reduce-scatter plus all-gather (aiter custom kernels) move the same bytes as the all-reduce they
replace, and AG(RS(x)) is bit-identical to AR(x). The residual is sharded after the last PLE layer
(PLE's short conv mixes neighbouring tokens). At the end the final mixer runs on the local rows and only
its [T, 2560] output is gathered; the [T, 4 x 2560] residual (HC state) is gathered only when the forward
captures hidden states (speculative decoding, `return_hidden_states`).

| Knob | Default | Effect |
|---|---|---|
| `SGLANG_QWEN4_SEQ_PARALLEL` | 1 | sequence-parallel residual for prefill; 0 keeps the all-reduce |
| `SGLANG_QWEN4_SEQ_PARALLEL_MIN_TOKENS` | 2048 | smaller batches keep the all-reduce |

The path covers eager EXTEND / MIXED / TARGET_VERIFY forwards with at least
`SGLANG_QWEN4_SEQ_PARALLEL_MIN_TOKENS` tokens. Decode, CUDA-graph-captured batches (including captured
target verify), draft-model forwards, DP attention and EP all-to-all keep the original path.

## Scripts

| Script | What it measures |
|---|---|
| `bench_rs_ag.py` | aiter all-reduce vs reduce-scatter + all-gather, and their bit-identity |

## Results (8x MI308X, 12000 in / 350 out, `scripts/run_pure_text_bench.sh`)

12k-token prefill step (mean of 4): TP4 542.0 -> 434.9 ms, TP2 748.7 -> 691.8 ms. Serving req/s, two
interleaved rounds (TP4 and TP2 run one after the other):

| conc | TP4 base | TP4 SP | TP2 base | TP2 SP |
|---|---|---|---|---|
| 1 | 0.342 / 0.339 | 0.349 / 0.353 | 0.302 / 0.302 | 0.310 / 0.309 |
| 4 | 0.776 / 0.769 | 0.821 / 0.836 | 0.594 / 0.597 | 0.619 / 0.616 |
| 8 | 1.003 / 1.012 | 1.087 / 1.122 | 0.798 / 0.792 | 0.834 / 0.826 |
| 16 | 1.211 / 1.205 | 1.359 / 1.373 | 0.930 / 0.931 | 0.991 / 0.993 |
| 32 | 1.353 / 1.347 | 1.541 / 1.553 | 1.008 / 1.006 | 1.079 / 1.081 |

Greedy outputs of a long-context probe are bit-identical to the baseline; the long-context check is
48/48 and gsm8k stays within run-to-run noise.

## SDMA overlap (ROCm)

On top of sequence parallelism, each of its communications moves to the SDMA copy engines and runs
under compute (`layers/sdma_reduce_scatter.py`, `layers/sdma_mori.py`):

```
RS       o_proj / out_proj in row chunks, peers' blocks first, own block last;
         each chunk is pushed to its owner as soon as it is computed       -> overlaps the next chunk's GEMM
         (the MoE output, computed whole, is pushed in the same chunks)
attn AG  mix -> push this rank's rows; in_proj runs on the local rows,
         then on each peer's block as it lands                             -> overlaps in_proj on local rows
MoE AG   combine / hc_norm / mix in row ranges, each range pushed as
         soon as it is mixed; the MoE waits for every range                -> overlaps the next range's HC ops
```

With the `mori` backend (default) the copies are posted by one-wave kernels on the compute stream, which
write the SDMA packets directly (mori CCO `put_ptr`); the copy engine bumps the receiver's counter when the
bytes land and the consumer waits with a one-wave spin kernel. No side streams or cross-stream events are
involved, so fine-grained pipelining stays cheap. The `hip` backend uses `hipMemcpyAsync` on per-peer copy
streams with an event per push and a flag kernel; it is used automatically when mori cannot run.

| Knob | Default | Effect |
|---|---|---|
| `SGLANG_QWEN4_SP_SDMA_RS` | 1 | reduce-scatters over SDMA |
| `SGLANG_QWEN4_SP_SDMA_AG` | 1 | attention / GDN input all-gather over SDMA, consumed per block by the input projection |
| `SGLANG_QWEN4_SP_SDMA_AG_MOE` | 1 | MoE input all-gather over SDMA, pushed per HC row range |
| `SGLANG_QWEN4_SP_SDMA_BACKEND` | mori | `mori` or `hip`; an unusable mori falls back to `hip` with a warning |
| `SGLANG_QWEN4_SP_SDMA_RS_TRANSPORT` | bf16 | `bf16` (exact) or `fp8` (peers' rows as fp8 + a bf16 scale per 32; lossy) |
| `SGLANG_QWEN4_SP_SDMA_RS_CHUNKS` | auto | row chunks per owner block; auto = 2 at TP2, 1 above |
| `SGLANG_QWEN4_SP_SDMA_AG_MOE_CHUNKS` | auto | minimum HC row ranges; auto = 4 at TP2, 2 above |

All of them act only with `SGLANG_QWEN4_SEQ_PARALLEL=1`.

mori requirements: mori built with `BUILD_CCO_SDMA=ON`, `MORI_ENABLE_SDMA=1`, and ROCm's private libdrm
copies preloaded so mori and the HIP runtime share the GPU VM (the MI308X launch script does both). On
MI308X, mori needs its SDMA engine selection not to put two peers of one GPU on the same engine
(`MORI_SDMA_SPREAD_MI308X=1` with the engine-spread patch); without it TP4 copies to two peers serialize.
mori's own symmetric windows are not used (destinations are hipIpc mappings), so hosts whose kernel lacks
`CONFIG_DMABUF_MOVE_NOTIFY` work too.

| Script | What it measures |
|---|---|
| `bench_sdma_rs.py` | `SdmaReduceScatter` vs an fp32 reference (bf16 / fp8, uneven T), and GEMM + RS + combine vs aiter |
