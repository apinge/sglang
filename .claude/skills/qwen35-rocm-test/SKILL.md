---
name: qwen35-rocm-test
description: Run and document the customized Qwen3.5 ROCm SGLang functional and pressure-test workflow, including cache validation, customer benchmark scripts, logging, and crash attribution.
---

# Qwen3.5 ROCm SGLang Test Runbook

Use this skill for the customized Qwen3.5 ROCm SGLang model-suite test flow. This is not a generic benchmark skill. It captures the local workflow, script boundaries, cache expectations, logging rules, and crash-handling conventions for this test suite.

## Required Context

At the start of every run, read the latest user instructions and any run-specific test plan or report path the user provides before launching anything. This skill is the common runbook; do not depend on private historical markdown files to understand the workflow.

- Launch scripts usually live in the SGLang repo's `evaluation/` directory.
- Functional client scripts are external/local test assets. Their directory name is not part of the contract; use the path from the user's run plan when provided.
- Customer pressure scripts are external/local assets. Their directory name is not part of the contract; use the path from the user's run plan when provided.

If the user provides a markdown report path, use that file for live progress and final records. If no report path is provided, ask for the report location before starting tests. The user's latest message overrides older notes in any report or plan. If the user says not to start testing yet, discuss only.

## Path Discovery

Do not hard-code private directory names as requirements. Directories like `/opt/evaluation7/`, `/opt/evaluation8/`, or `/opt/benchmark_example/` are environment-specific staging locations, not the meaning of the workflow.

Resolve paths in this order:

1. Use explicit paths from the user's latest message or run-specific markdown.
2. Use paths encoded in the assigned launch/client/pressure command if the user supplied one.
3. Search likely local roots for exact script names, such as the current SGLang repo, `/opt`, and user-provided work directories.
4. If there are multiple plausible matches or no match, ask the user to identify the correct script or root before running tests.

The stable identifiers are the script purposes and common filenames, not their parent directories:

- Smoke client: `check_acc_long.py`
- Taobao Xiaomi client: `run_acc_tb_27b.sh`
- Five-image client: `req3.sh`, often under an `ali_uc/` subdirectory
- GSM8K client: `run_gsm8k.sh`
- FP8 pressure entrypoint: `run_pressure_test_fp8.sh`
- BF16 pressure entrypoint: `run_pressure_test_bf16.sh`

## Hard Boundaries

- Do not upload customer pressure scripts or copy their contents into reports, commits, or skill files.
- Do not modify customer pressure scripts. Treat customer-provided pressure entrypoints, commonly named `run_pressure_test_fp8.sh` and `run_pressure_test_bf16.sh`, as immutable black-box entrypoints wherever they are located.
- Do not modify original launch scripts unless the user explicitly approves. If a change is needed, copy the launch script and edit the copy.
- For copied launch scripts, the normally allowed edits are only `--tp-size`, port, visible GPU assignment, and log path or `tee` output name. Ask before changing model path, dtype, scheduler, DFlash/speculative arguments, tokenizer, quantization, cache behavior, or other serving semantics.
- If a crash, OOM, segmentation fault, killed process, traceback, or core dump appears during functional testing, stop the test flow and ask the user before continuing.
- For pressure tests, stop the affected pressure run when the benchmark naturally stops or when a serious abnormal condition appears. If parallel pressure tests are running and one side crashes, stop both sides, record attribution evidence, and wait for user instruction.
- Core dumps may be several GB. Do not open, parse, strings-scan, or copy core dump files. Only detect their existence, timestamp, size, and likely owner process/case.

## Assigned Cases And Resource Convention

Do not assume the agent owns the full test matrix. Each person or agent may be assigned a different subset of cases. Execute only the cases explicitly assigned in the active markdown and the latest user messages.

For each assigned case, infer the actual model precision, TP size, DFlash/no-DFlash mode, cache mode, port, and GPU visibility from the launch script and the user's latest notes. The script name usually encodes these properties, but verify them from the script and server args in `launch_stdout.log`.

When multiple cases are run in parallel:

- Keep ports, GPU visibility, log directories, result directories, and process groups isolated per case.
- Use the GPU groups and ports already defined by the user's launch scripts unless the user instructs otherwise.
- If a launch script does not set GPU visibility, remember that the machine normally exposes GPU IDs from `0` upward, so TP placement may start at GPU `0`.
- Confirm actual ports from both the launch script and client script before running.

## Cache Rules

Default functional/origin tests use cache-disabled launches unless the active test plan explicitly says otherwise.

Run with cache disabled for:

- Smoke tests
- Taobao Xiaomi tests
- Five-image tests
- GSM8K
- Pressure tests, unless the active plan explicitly requests a with-cache pressure test

Run with radix/prefix cache enabled only for dedicated cache validation cases, or when the current plan explicitly names a `with radix cache` or `with prefix cache` case.

For the Qwen3.5 ROCm model suite, enabling radix cache requires both serving arguments:

```bash
--mamba-scheduler-strategy extra_buffer
--page-size 64
```

Cache-disabled scripts should include the project-specific disable flag used by the script, for example `--disable-radix-cache` or the no-DFlash `disable_prefix_cache` launch variant. The exact flag name may differ by script version, so verify from server args in the log:

- Cache disabled evidence: `disable_radix_cache=True` and request logs showing `#cached-token: 0`
- Cache enabled evidence: `disable_radix_cache=False`, `page_size=64`, `mamba_radix_cache_strategy='extra_buffer'`, and request logs showing nonzero `#cached-token`

The dedicated cache hit check is:

1. Start the cache-enabled launch script.
2. Run the resolved Taobao Xiaomi client script twice against the same port.
3. In the second run, find the server log line containing nonzero `#cached-token`.
4. Copy the single relevant server log line into the markdown as evidence.

Do not count client-side reuse alone as cache evidence. Use the server-side `#cached-token` line.

## Functional Test Flow

For each requested functional case:

1. Create a case-specific log directory under the user-provided or run-specific log root. Do not assume a fixed parent directory; use a stable structure like `<log_root>/<run_name>/<case_name>/`.
2. Start the server from the requested launch script and save `launch_stdout.log` in the case directory.
3. Wait until the service port is ready before running clients.
4. Run the smoke test twice. Record both logs, but judge the case by the second result.
5. Run Taobao Xiaomi four times. Save each run as `tb_1.log` through `tb_4.log`.
6. Run the five-image test once and record the actual model answer or concise answer summary in the markdown.
7. Run GSM8K from `/opt/sglang`; record accuracy, invalid rate, latency, output throughput, and log path.
8. If this is a cache-enabled case, run the cache hit check and record server-side `#cached-token` evidence.
9. Scan logs for abnormal keywords and record the result.
10. Stop the server and verify process/GPU cleanup before moving to the next case unless the user requested parallel execution.

Standard client entry names:

- Smoke: `check_acc_long.py`
- Taobao Xiaomi: `run_acc_tb_27b.sh`
- Five-image: `req3.sh`
- GSM8K: `run_gsm8k.sh`

Client scripts may be copied for a run if ports or tokenizer paths need to differ. Only change the port by default. For GSM8K, it is also acceptable to adjust `--tokenizer-path` to the matching Qwen3.5 model required by the current case.

Expected functional records:

- Smoke: second run should answer that it is a Qwen or Qwen3.5 model.
- Taobao Xiaomi: store every run log path and summarize `finish_reason`, `completion_tokens`, and `reasoning_tokens`.
- Five-image: record whether the answer is coherent and not hallucinated; include a concise description of the recognized images.
- GSM8K: record `Accuracy`, `Invalid`, `Latency`, and `Output throughput`.
- Cache validation: record the exact server-side `#cached-token` evidence line.

## Pressure Test Flow

Only run pressure tests after the user explicitly approves the pressure matrix.

Customer pressure script entrypoints are conventionally named by precision:

- FP8: `run_pressure_test_fp8.sh`
- BF16: `run_pressure_test_bf16.sh`

The scripts commonly contain a fixed port, ramp logic, dataset shape, and tee log name. Check these values locally before the run without modifying the script. Examples of fields to verify:

- `LLM_HTTP_PORT`
- Ramp logic: `LLM_REQ_RATE=0.1`, `LLM_REQ_RATE_DURATION=12`, `LLM_REQ_RATE_STEP=0.05`
- Dataset shape: `LLM_AUTO_DATASET_CONF=tk_in:250,tk_out:20`
- Script tee log name, often like `pressure-test_35b_*.log`

Do not rely on these values without checking the local script. Do not modify the script when checking it.

Recommended trigger pattern:

1. Create a separate pressure case directory for each model/case.
2. Start each server with its requested launch script and save `launch_stdout.log`.
3. Wait for every required port to become ready.
4. Run each customer pressure script from inside its own isolated case directory or from an isolated copied benchmark package so generated `results/` JSON and tee logs do not overwrite another case.
5. Capture stdout/stderr to `pressure.log` while preserving the script's own tee log.
6. Periodically append progress to the active markdown.
7. Let the customer script's threshold and supplement logic finish naturally unless there is a serious abnormal condition.
8. After completion, copy or preserve result JSON under the case directory and record the final paths.

Pressure tests may run in parallel when GPU groups, ports, log directories, and result directories are isolated. If parallel cases use the same customer script package, use separate working directories or isolated copied benchmark packages so `results/` JSON and tee logs do not collide.

If TP1 or TP2 DFlash cases show memory pressure, reduce memory and concurrency conservatively. Prior working TP1 DFlash settings were:

```bash
--mem-fraction-static 0.7
--max-running-requests 32
--cuda-graph-max-bs 32
```

When using `--max-running-requests 32`, pressure-script request rates can still exceed 32. That is not automatically unsafe and does not mean the client cannot issue load. Interpret it using failures, TPOT, server-side queue, and crash signals. Client `Running Tasks P90` is a benchmark-client statistic and is not the same as the server's raw `#running-req`.

Pressure monitoring fields:

- Current request rate
- `Failed`
- TPOT, TTFT, ITL, E2EL
- Throughput
- Client-side `Running Tasks P90`
- Server-side `#running-req` and `#queue-req`
- GPU memory and utilization
- OOM/core/segfault/killed/traceback keywords

Primary threshold:

- TPOT crossing `100ms` is the normal customer-script stop condition.
- Record the last regular passing QPS, the triggering QPS, and the supplement interval or best supplement result.
- If wrapper summary and per-QPS JSON disagree, prefer the per-QPS JSON for exact TPOT boundary analysis and note the discrepancy.

## Crash And Core Dump Attribution

The responsibility is to determine whether a core dump happened and which test case most likely caused it. Do not analyze core file contents.

Use metadata only:

- Record the test start timestamp in UTC before launching pressure tests.
- Record server PID, process group, port, GPU group, launch script, and case directory.
- Check `coredumpctl --since '<UTC timestamp>' --no-pager` when systemd-coredump is configured.
- Inspect `/proc/sys/kernel/core_pattern` to know where core dumps go.
- Metadata-scan likely locations such as `/var/lib/systemd/coredump`, the run log root, the SGLang repo, and `/tmp` for new files named like `core`, `core.<pid>`, or `*.core`.
- Attribute using timestamp, PID, executable name, port, GPU group, and nearby server log messages.

Never run commands that read the full core dump body. Avoid `strings`, `gdb`, `cat`, compression, upload, or checksum of multi-GB core files unless the user explicitly asks and approves.

## Markdown Recording Standard

Append progress and final results directly to the user-provided run report during the run. Do not wait until the end. If the user has not provided a report path, ask for one before launching tests.

For each functional case, record:

- Case name, launch script, port, GPU group, and log directory
- Server args evidence for TP/cache/DFlash mode
- Smoke logs and second-run pass/fail
- Taobao Xiaomi four log paths and brief finish/token summary
- Five-image answer summary
- GSM8K metrics and log path
- Cache evidence if applicable
- Abnormal keyword scan result

For each pressure case, record:

- Case name, launch script, customer pressure script, port, GPU group, and case directory
- How the pressure script was triggered
- Periodic QPS progress and health observations
- Final last-passing QPS, trigger QPS, supplement result, and result JSON paths
- Server queue/running observations
- OOM/core/segfault/killed/traceback scan result
- Cleanup status for processes and GPU memory

Use concise, auditable records with absolute paths. Preserve enough detail that another agent can resume from the markdown without rerunning completed work.

## Cleanup

After each case or parallel batch:

- Stop launched SGLang server processes and benchmark wrappers.
- Check no relevant `sglang.launch_server`, benchmark wrapper, multi-worker, or run-pressure process remains.
- Check GPU utilization and memory have returned to idle or expected baseline.
- Leave unrelated user processes alone.
