# Sequence-parallel hyper-connection residual for Qwen4-Exp TP prefill

Qwen4-Exp (Qwen3.8-Flash-Next) keeps a 4-lane hyper-connection residual stream. Between the TP-sharded
blocks, three per-token ops run on it: `hc_combine` (inject the block output), `hc_norm`
(grouped Gemma RMSNorm, variance per token and per 2560-wide lane) and the low-rank `mix` that builds the
next block's input. With the usual layout every TP rank repeats them on all tokens; in a 12k-token
prefill step that is ~165 ms per rank (31% of the step at TP4, 22% at TP2).

With `SGLANG_QWEN4_SEQ_PARALLEL=1` each rank keeps only its 1/TP slice of the token rows of the residual
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
(PLE's short conv mixes neighbouring tokens) and gathered back before the final mixer.

| Knob | Default | Effect |
|---|---|---|
| `SGLANG_QWEN4_SEQ_PARALLEL` | 0 | enable the sequence-parallel residual for prefill |
| `SGLANG_QWEN4_SEQ_PARALLEL_MIN_TOKENS` | 2048 | smaller batches keep the all-reduce |

Decode, CUDA graph capture, speculative modes, DP attention and EP all-to-all keep the original path.

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
