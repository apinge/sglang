# Qwen3.5 DFlash acceptance verification

These entry points bring the packaged Qwen3.5-397B acceptance benchmarks into
`apinge/sglang:qwen4_v0.5.21` for [issue #49](https://github.com/apinge/sglang/issues/49).
The development baseline is
[`apinge/sglang:qwen4_v0.5.21`](https://github.com/apinge/sglang/tree/qwen4_v0.5.21)
at `c0a27e4810`, which already contains the unified CI from PR #65.
The workload drivers originate from [PR #13](https://github.com/apinge/sglang/pull/13).
The [DFlash release benchmark](https://huggingface.co/z-lab/Qwen3.5-397B-A17B-DFlash/tree/main/benchmark)
is the reference requested by the issue. Its B200/Modal results use different
hardware, software and workloads; the scripts here retain the local packaged
prompts and sampling rather than claiming to reproduce those numbers.

## Setup and execution

Use the ROCm/SGLang/AITER environment described in the
[Qwen evaluation guide](../../evaluation/Qwen4_Evaluation_Guide.md).
The server entry point targets eight MI308X GPUs, the PTPC-FP8 target model and
the DFlash draft model. Adjust the `/models/...` paths in the shell commands to
match the installed weights.

From the repository root, download the public datasets once:

```bash
python3 benchmark/dflash/download_data.py
```

The downloader writes separate `benchmark/dflash/data/<workload>.jsonl` files,
validates their required fields, and prints row counts and SHA256 hashes.
It decompresses HumanEval and prevents GSM8K/MATH-500 from sharing a cached
`test.jsonl` filename. Workload drivers also validate a supplied `--data-path`.

Start the server in one terminal:

```bash
bash evaluation/dflash/launch_qwen3.5-397B-fp8_tp8_DFlash.sh
```

Run the desired tests in another terminal after the service is ready:

```bash
bash evaluation/dflash/run_math500.sh
bash evaluation/dflash/run_humaneval.sh
bash evaluation/dflash/run_mbpp.sh
bash evaluation/dflash/run_mtbench.sh
bash evaluation/dflash/run_gsm8k_acceptance.sh
```

The shell entry points use an existing server at `http://127.0.0.1:9080`.
The launch command sets TP8, draft16, AITER full attention, Triton linear and draft
attention, FP8 KV, page64, radix cache disabled, prefill32768 and
max-running-requests128. It uses the rebased names
`--cuda-graph-max-bs-decode` and `--mamba-radix-cache-strategy`.
The rebased branch does not accept the old AITER linear-attention backend;
these entry points explicitly select its supported Triton backend for GDN.
Stop the manually started service when the tests finish.

| Acceptance workload | Selected prompts | Concurrency | Warmup excluded | Measured requests | Maximum output tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| GSM8K | 200 | 64 | 64 | 136 | 4096 |
| MATH-500 | 128 | 1 | 1 | 127 | 2048 |
| HumanEval | 164 | 1 | 1 | 163 | 1024 |
| MBPP | 128 | 1 | 1 | 127 | 1024 |
| MT-Bench | 80 | 1 | 1 | 79 | 1024 |

The acceptance drivers use the model's chat template with `enable_thinking=False`
and greedy sampling (`temperature=0`, `top_p=1`, `top_k=1`). Each run sends an
untimed short startup probe, flushes the server cache and excludes the first
full concurrency batch from all measured counters, timing and scoring.
The remaining requests use client concurrency; `--batch-requests` switches to
batched `/generate` requests with the same warmup exclusion.

The original five-shot GSM8K accuracy entry point is retained separately:

```bash
bash evaluation/dflash/run_gsm8k.sh
```

It calls the repository's existing `benchmark/gsm8k/bench_sglang.py`, uses 200
questions, five shots and parallel64, and reports accuracy and throughput.
It has a different prompt protocol from the zero-shot GSM8K acceptance driver;
compare only runs using the same protocol. Its acceptance counters remain in
the server log.

## Output and definitions

Generation logs are written under `logs/dflash/<workload>/`. The workload's
metrics are saved in `results/dflash/<workload>/<workload>.json`. The response
recorder additionally writes:

```text
results/dflash/<workload>/acceptance/
  acceptance.json
  run1_responses.jsonl
```

`acceptance.json` contains one entry per configuration, including measured
request count, excluded warmup count, sampling, total output tokens, total verify
steps, draft counters, finish reasons, generation throughput and elapsed time.
It also records an allowlist of the actual resolved `/server_info` configuration
and the reported SGLang version; client `--tp-sizes` and backend flags are only
labels in external mode. Record the server's source commit, model revisions,
GPU identities and software versions alongside results when comparing runs.

- **Strict draft acceptance rate:**
  `sum(spec_num_correct_drafts) / sum(spec_num_proposed_drafts)`.
  This excludes bonus tokens and uses server counters directly.
- **Verification-step-weighted acceptance length:**
  `sum(completion_tokens) / sum(spec_verify_ct)`.
  This includes emitted bonus tokens and matches SGLang's acceptance-length
  definition. It differs from the arithmetic mean of per-request lengths.
- **Generation throughput:** measured completion tokens divided by the timed
  generation interval. Code execution for HumanEval/MBPP occurs after timing.

The recorder checks the actual server is configured for DFLASH and requires
complete, nonnegative integer draft and verify counters. It checks its captured
response, token and verify totals against the workload's metrics. Missing
counters cause the DFlash run to fail rather than produce a partial acceptance
rate. Length-truncated requests are included and counted in `finish_reason_counts`.

MATH-500 uses lightweight normalized boxed-answer string comparison, not symbolic
equivalence. HumanEval and MBPP execute generated code in child processes with
CPU limits and wall timeouts; run them in a disposable benchmark environment.
MBPP uses the first selected rows of the upstream 974-item file, as in the
packaged benchmark; this is not an official held-out-split accuracy evaluation.
MT-Bench uses the first turn only and reports no judge-based quality score.

## Validation scope

Run the CPU regression checks with:

```bash
python3 -m unittest discover -s benchmark/dflash -p 'test_*.py' -v
```

They exercise all five workload entry points against a local HTTP fixture in
both concurrent and batched modes, validate warmup exclusion and counter
aggregation, and reject incomplete counters, invalid counters, incorrect dataset
schemas and an EAGLE server labeled as DFlash. They do not load model weights.

The earlier packaging check on 2026-10-16 used four MI308X GPUs, TP4/draft16 and
max-running-requests32 on the older `qwen3_5_v0.5.15_dflash_th` source based on
[`cc6888def7`](https://github.com/apinge/sglang/commit/cc6888def73c3eb3ae64bc6b3be107d2ce460015).
Its MATH-500 run recorded 127 measured requests, strict acceptance rate
48.8183% (105843/216810), weighted length 8.3125 (120149/14454), and 17
length-truncated responses. These are historical packaging results; the rebased
TP8 entry points and the new GSM8K acceptance entry point require fresh model
execution before drawing conclusions about issue #49.
