"""DFLASH vs baseline MT-Bench sweep (performance only).

This is a *benchmark script* (not a CI test): it can take a long time because it
launches servers for multiple (attention_backend, tp_size) configs and runs an
MT-Bench workload for each (concurrency, num_questions) setting.

MT-Bench is an open-ended, LLM-judged benchmark. This script does NOT run a judge:
it measures throughput and DFLASH acceptance length only (no quality score). Each
MT-Bench item's first-turn prompt is used as a single-turn generation request.

Example usage:
  ./venv/bin/python benchmark/dflash/bench_dflash_mtbench_sweep.py
  ./venv/bin/python benchmark/dflash/bench_dflash_mtbench_sweep.py --skip-baseline --concurrencies 32 --tp-sizes 8
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

WORKLOAD = "mt-bench"

# Environment matching evaluation/launch_qwen3.5-397B-fp8_tp8_with_prefix_cache_DFlash.sh.
# Applied via setdefault (caller's explicit exports win) BEFORE importing torch
# and sglang: e.g. TVM_FFI_DISABLE_TORCH_C_DLPACK avoids a libtorch_cuda.so load
# failure on this ROCm build during the sglang import chain.
_LAUNCH_ENV = {
    "SGLANG_DISABLE_CUDNN_CHECK": "1",
    "SGLANG_USE_CUDA_IPC_TRANSPORT": "1",
    "SGLANG_VLM_CACHE_SIZE_MB": "8192",
    "SGLANG_USE_AITER": "1",
    "SGLANG_ROCM_USE_AITER_LINEAR_SHUFFLE": "1",
    "SGLANG_ROCM_USE_AITER_LINEAR_FP8HIPB": "1",
    "AITER_QUICK_REDUCE_QUANTIZATION": "INT6",
    "USE_AITER_COMM": "1",
    "SGLANG_USE_AITER_NEW_CA": "false",
    "SGLANG_USE_IPC_POOL_HANDLE_CACHE": "1",
    "HIP_GDN_SORT_IDX_BS": "32768",
    "TVM_FFI_DISABLE_TORCH_C_DLPACK": "1",
}

for _k, _v in _LAUNCH_ENV.items():
    os.environ.setdefault(_k, _v)

import requests  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from sglang.srt.utils import get_device_sm, kill_process_tree  # noqa: E402
from sglang.test.test_utils import (  # noqa: E402
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    find_available_port,
    popen_launch_server,
)
from sglang.utils import download_and_cache_file, read_jsonl  # noqa: E402


def _parse_int_csv(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def _filter_attention_backends(backends: list[str], *, device_sm: int) -> list[str]:
    if not (80 <= device_sm <= 90):
        backends = [b for b in backends if b != "fa3"]
    if device_sm < 100:
        backends = [b for b in backends if b not in ("fa4", "trtllm_mha")]
    return backends or ["flashinfer"]


def _maybe_download_mtbench(data_path: str) -> str:
    url = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"
    if os.path.isfile(data_path):
        return data_path
    return download_and_cache_file(url)


def _flush_cache(base_url: str) -> None:
    resp = requests.get(base_url + "/flush_cache", timeout=60)
    resp.raise_for_status()


def _send_generate(
    base_url: str,
    text: str | list[str],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    timeout_s: int,
) -> list[dict]:
    if isinstance(text, list) and not text:
        return []
    sampling_params: dict = {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "max_new_tokens": int(max_new_tokens),
    }
    resp = requests.post(
        base_url + "/generate",
        json={
            "text": text,
            "sampling_params": sampling_params,
        },
        timeout=int(timeout_s),
    )
    resp.raise_for_status()
    out = resp.json()
    if isinstance(text, list):
        if not isinstance(out, list):
            raise RuntimeError(
                "Expected a list response for batched /generate, but got "
                f"type={type(out).__name__}."
            )
        if len(out) != len(text):
            raise RuntimeError(
                "Batched /generate output length mismatch: "
                f"got {len(out)} outputs for {len(text)} prompts."
            )
        return out

    if isinstance(out, list):
        raise RuntimeError(
            "Expected an object response for single /generate, but got "
            f"type={type(out).__name__}."
        )
    return [out]


@dataclass(frozen=True)
class BenchMetrics:
    latency_s: float
    output_tokens: int
    output_toks_per_s: float
    spec_accept_length: Optional[float]
    spec_accept_length_weighted: Optional[float]
    spec_verify_ct_sum: int


def _run_requests(
    base_url: str,
    *,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    concurrency: int,
    batch_requests: bool,
    timeout_s: int,
    expect_dflash: bool,
) -> BenchMetrics:
    # Drop the first batch from metrics to exclude one-time JIT/cuda-graph overhead
    # that often happens immediately after /flush_cache for large batch sizes.
    bs = max(int(concurrency), 1)
    if len(prompts) > bs:
        warmup_prompts = prompts[:bs]
        if batch_requests:
            _send_generate(
                base_url,
                warmup_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                timeout_s=timeout_s,
            )
        else:
            with ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
                futures = [
                    pool.submit(
                        _send_generate,
                        base_url=base_url,
                        text=prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        timeout_s=timeout_s,
                    )
                    for prompt in warmup_prompts
                ]
                for fut in as_completed(futures):
                    outs = fut.result()
                    if len(outs) != 1:
                        raise RuntimeError(
                            "Expected exactly one output for single /generate warmup request."
                        )

        prompts = prompts[bs:]

    start = time.perf_counter()
    total_tokens = 0
    spec_verify_ct_sum = 0
    spec_accept_lengths: list[float] = []

    def _handle_output(out: dict) -> None:
        nonlocal total_tokens, spec_verify_ct_sum
        meta = out.get("meta_info", {}) or {}
        total_tokens += int(meta.get("completion_tokens", 0))
        spec_verify_ct_sum += int(meta.get("spec_verify_ct", 0))
        if "spec_accept_length" in meta:
            try:
                spec_accept_lengths.append(float(meta["spec_accept_length"]))
            except (TypeError, ValueError):
                pass

    if batch_requests:
        bs = max(int(concurrency), 1)
        for start_idx in range(0, len(prompts), bs):
            chunk_prompts = prompts[start_idx : start_idx + bs]
            outs = _send_generate(
                base_url,
                chunk_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                timeout_s=timeout_s,
            )
            for out in outs:
                _handle_output(out)
    else:
        with ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
            futures = [
                pool.submit(
                    _send_generate,
                    base_url=base_url,
                    text=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    timeout_s=timeout_s,
                )
                for prompt in prompts
            ]
            for fut in as_completed(futures):
                outs = fut.result()
                if len(outs) != 1:
                    raise RuntimeError(
                        "Expected exactly one output for single /generate request."
                    )
                _handle_output(outs[0])

    latency = time.perf_counter() - start
    toks_per_s = total_tokens / max(latency, 1e-6)

    if expect_dflash and spec_verify_ct_sum <= 0:
        raise RuntimeError(
            "DFLASH sanity check failed: did not observe any `spec_verify_ct` in responses "
            "(DFLASH may not have been enabled)."
        )

    spec_accept_length = (
        float(statistics.mean(spec_accept_lengths)) if spec_accept_lengths else None
    )
    # Token-weighted global accept length: total completion tokens / total verify
    # steps (each request weighted by its length, unlike the per-request mean above).
    spec_accept_length_weighted = (
        float(total_tokens) / float(spec_verify_ct_sum)
        if spec_verify_ct_sum > 0
        else None
    )

    return BenchMetrics(
        latency_s=float(latency),
        output_tokens=int(total_tokens),
        output_toks_per_s=float(toks_per_s),
        spec_accept_length=spec_accept_length,
        spec_accept_length_weighted=spec_accept_length_weighted,
        spec_verify_ct_sum=int(spec_verify_ct_sum),
    )


def _format_table(
    *,
    tp_sizes: list[int],
    concurrencies: list[int],
    values: dict[tuple[int, int], Optional[float]],
    float_fmt: str,
) -> str:
    header = ["tp\\conc"] + [str(c) for c in concurrencies]
    rows: list[list[str]] = [header]
    for tp in tp_sizes:
        row = [str(tp)]
        for c in concurrencies:
            v = values.get((tp, c), None)
            row.append("N/A" if v is None else format(v, float_fmt))
        rows.append(row)

    col_widths = [
        max(len(row[col_idx]) for row in rows) for col_idx in range(len(rows[0]))
    ]

    lines: list[str] = []
    lines.append("  ".join(cell.rjust(col_widths[i]) for i, cell in enumerate(rows[0])))
    lines.append("  ".join("-" * w for w in col_widths))
    for row in rows[1:]:
        lines.append("  ".join(cell.rjust(col_widths[i]) for i, cell in enumerate(row)))
    return "\n".join(lines)


def _build_common_server_args(
    args: argparse.Namespace, *, backend: str, tp: int
) -> list[str]:
    # Mirror evaluation/launch_qwen3.5-397B-fp8_tp8_with_prefix_cache_DFlash.sh (baseline flags).
    common_server_args: list[str] = [
        "--tp-size",
        str(tp),
        "--reasoning-parser",
        "qwen3",
        "--tool-call-parser",
        "qwen3_coder",
        "--enable-multimodal",
        "--trust-remote-code",
        "--chunked-prefill-size",
        "32768",
        "--mem-fraction-static",
        str(args.mem_fraction_static if args.mem_fraction_static is not None else 0.9),
        "--max-prefill-tokens",
        "32768",
        "--max-running-requests",
        str(args.max_running_requests),
        "--cuda-graph-max-bs",
        "128",
        "--linear-attn-backend",
        "aiter",
        "--linear-attn-decode-backend",
        "aiter",
        "--linear-attn-prefill-backend",
        "aiter",
        "--watchdog-timeout",
        "1200",
        "--attention-backend",
        backend,
        "--mm-attention-backend",
        "aiter_attn",
        "--mamba-scheduler-strategy",
        str(args.mamba_scheduler_strategy),
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--page-size",
        str(int(args.page_size)),
    ]
    if args.dtype is not None:
        common_server_args += ["--dtype", str(args.dtype)]
    if args.disable_radix_cache:
        common_server_args.append("--disable-radix-cache")
    return common_server_args


def _build_mode_runs(
    args: argparse.Namespace, common_server_args: list[str]
) -> list[tuple[str, str, list[str], bool]]:
    mode_runs: list[tuple[str, str, list[str], bool]] = []
    if not args.skip_baseline:
        mode_runs.append(("baseline", "baseline", common_server_args, False))
    mode_runs.append(
        (
            "dflash",
            "DFLASH",
            [
                *common_server_args,
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-draft-model-path",
                args.draft_model,
                "--speculative-num-draft-tokens",
                str(int(args.speculative_num_draft_tokens)),
                *(
                    [
                        "--speculative-dflash-draft-window-size",
                        str(int(args.speculative_dflash_draft_window_size)),
                    ]
                    if args.speculative_dflash_draft_window_size is not None
                    else []
                ),
                *(
                    [
                        "--speculative-draft-attention-backend",
                        args.speculative_draft_attention_backend,
                    ]
                    if args.speculative_draft_attention_backend
                    else []
                ),
            ],
            True,
        )
    )
    return mode_runs


def _collect_metric(
    *,
    results: dict[tuple[str, int, int, str], BenchMetrics],
    backend: str,
    tp_sizes: list[int],
    concurrencies: list[int],
    mode: str,
    field: str,
) -> dict[tuple[int, int], Optional[float]]:
    out: dict[tuple[int, int], Optional[float]] = {}
    for tp in tp_sizes:
        for conc in concurrencies:
            metrics = results.get((backend, tp, conc, mode), None)
            out[(tp, conc)] = None if metrics is None else getattr(metrics, field)
    return out


def _compute_speedup(
    baseline: dict[tuple[int, int], Optional[float]],
    dflash: dict[tuple[int, int], Optional[float]],
) -> dict[tuple[int, int], Optional[float]]:
    return {
        key: None if (b is None or d is None or b <= 0) else (d / b)
        for key, b in baseline.items()
        for d in [dflash.get(key, None)]
    }


def _print_kv_lines(items: list[tuple[str, object]]) -> None:
    for key, value in items:
        print(f"{key}={value}")


def _run_mode_for_backend_tp(
    *,
    mode_label: str,
    model_path: str,
    base_url: str,
    server_args: list[str],
    expect_dflash: bool,
    prompts: list[str],
    concurrencies: list[int],
    num_questions_by_conc: dict[int, int],
    args: argparse.Namespace,
    launch_server: bool = True,
) -> dict[int, BenchMetrics]:
    print(f"\n=== {mode_label} ===")
    proc = None
    if launch_server:
        server_start_timeout_s = int(
            max(DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, args.timeout_s)
        )
        proc = popen_launch_server(
            model_path,
            base_url,
            timeout=server_start_timeout_s,
            other_args=server_args,
        )
    else:
        print(f"[external] reusing already-running server at {base_url}.")
    try:
        _send_generate(
            base_url,
            "Hello",
            max_new_tokens=8,
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            top_k=int(args.top_k),
            timeout_s=min(int(args.timeout_s), 300),
        )

        metrics_by_conc: dict[int, BenchMetrics] = {}
        for conc in concurrencies:
            max_n = len(prompts) - int(conc)
            if max_n <= 0:
                raise RuntimeError(
                    f"concurrency={conc} exceeds available prompts={len(prompts)}; "
                    "reduce --concurrencies or increase --max-questions-per-config."
                )
            n = min(num_questions_by_conc[conc], max_n)
            _flush_cache(base_url)
            print(
                f"[warmup] run 1 warmup batch (size={conc}) after /flush_cache; excluded from metrics."
            )
            metrics = _run_requests(
                base_url,
                prompts=prompts[: n + conc],
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                top_k=int(args.top_k),
                concurrency=int(conc),
                batch_requests=bool(args.batch_requests),
                timeout_s=int(args.timeout_s),
                expect_dflash=expect_dflash,
            )
            metrics_by_conc[conc] = metrics
            line = (
                f"[{mode_label}] conc={conc:>2} n={n:<4} "
                f"toks/s={metrics.output_toks_per_s:,.2f} "
                f"latency={metrics.latency_s:.1f}s "
                f"out_tokens={metrics.output_tokens}"
            )
            if expect_dflash or metrics.spec_verify_ct_sum > 0:
                accept_len = (
                    "N/A"
                    if metrics.spec_accept_length is None
                    else f"{metrics.spec_accept_length:.3f}"
                )
                accept_len_w = (
                    "N/A"
                    if metrics.spec_accept_length_weighted is None
                    else f"{metrics.spec_accept_length_weighted:.3f}"
                )
                line += (
                    f" accept_len={accept_len} "
                    f"accept_len_weighted={accept_len_w} "
                    f"spec_verify_ct_sum={metrics.spec_verify_ct_sum}"
                )
            print(line)
        return metrics_by_conc
    finally:
        if proc is not None:
            kill_process_tree(proc.pid)
            try:
                proc.wait(timeout=30)
            except Exception:
                pass


def _dump_result_json(
    path: str,
    *,
    args: argparse.Namespace,
    results: dict[tuple[str, int, int, str], BenchMetrics],
) -> None:
    payload = {
        "workload": WORKLOAD,
        "target_model": args.target_model,
        "draft_model": args.draft_model,
        "speculative_num_draft_tokens": int(args.speculative_num_draft_tokens),
        "max_new_tokens": args.max_new_tokens,
        "rows": [
            {
                "backend": backend,
                "tp": tp,
                "concurrency": conc,
                "mode": mode,
                "output_toks_per_s": m.output_toks_per_s,
                "accuracy": getattr(m, "accuracy", None),
                "spec_accept_length": m.spec_accept_length,
                "spec_accept_length_weighted": m.spec_accept_length_weighted,
            }
            for (backend, tp, conc, mode), m in sorted(results.items())
        ],
    }
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {path}")


def _print_summary(
    *,
    args: argparse.Namespace,
    attention_backends: list[str],
    tp_sizes: list[int],
    concurrencies: list[int],
    device_sm: int,
    results: dict[tuple[str, int, int, str], BenchMetrics],
) -> None:
    print("\n=== DFLASH MT-Bench Sweep Summary (performance only; no judge) ===")
    if args.base_url:
        _print_kv_lines(
            [
                ("base_url", args.base_url),
                ("external_mode", args.external_mode),
                (
                    "note",
                    "server-side flags below were NOT applied; the running server's own "
                    "config was measured (tp/backend are labels only)",
                ),
            ]
        )
    _print_kv_lines(
        [
            ("target_model", args.target_model),
            ("draft_model", args.draft_model),
            ("max_new_tokens", args.max_new_tokens),
            (
                "sampling",
                f"temperature:{args.temperature}, top_p:{args.top_p}, top_k:{args.top_k}",
            ),
            ("attention_backends", ",".join(attention_backends)),
            (
                "speculative_draft_attention_backend",
                args.speculative_draft_attention_backend,
            ),
            (
                "speculative_dflash_draft_window_size",
                args.speculative_dflash_draft_window_size,
            ),
            ("tp_sizes", ",".join(str(x) for x in tp_sizes)),
            ("concurrencies", ",".join(str(x) for x in concurrencies)),
            (
                "questions_per_concurrency_base",
                args.questions_per_concurrency_base,
            ),
            ("device_sm", device_sm),
            ("skip_baseline", bool(args.skip_baseline)),
        ]
    )

    section_fields = [
        ("Baseline output tok/s", "baseline", "output_toks_per_s", ",.2f"),
        ("DFLASH output tok/s", "dflash", "output_toks_per_s", ",.2f"),
        (
            "DFLASH acceptance length (mean spec_accept_length)",
            "dflash",
            "spec_accept_length",
            ".3f",
        ),
        (
            "DFLASH acceptance length (token-weighted: total_tokens/verify_ct)",
            "dflash",
            "spec_accept_length_weighted",
            ".3f",
        ),
    ]

    for backend in attention_backends:
        print(f"\n=== Backend: {backend} ===")
        metrics_map = {
            (mode, field): _collect_metric(
                results=results,
                backend=backend,
                tp_sizes=tp_sizes,
                concurrencies=concurrencies,
                mode=mode,
                field=field,
            )
            for _, mode, field, _ in section_fields
        }
        sections: list[tuple[str, dict[tuple[int, int], Optional[float]], str]] = [
            (title, metrics_map[(mode, field)], fmt)
            for title, mode, field, fmt in section_fields
        ]
        sections.insert(
            2,
            (
                "Speedup (DFLASH / baseline)",
                _compute_speedup(
                    metrics_map[("baseline", "output_toks_per_s")],
                    metrics_map[("dflash", "output_toks_per_s")],
                ),
                ".3f",
            ),
        )

        for title, values, fmt in sections:
            print(f"\n{title}")
            print(
                _format_table(
                    tp_sizes=tp_sizes,
                    concurrencies=concurrencies,
                    values=values,
                    float_fmt=fmt,
                )
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="mt_bench_question.jsonl")
    parser.add_argument(
        "--target-model", default="/shared/xiaomi/Qwen3.5-397B-A17B-PTPC-FP8/"
    )
    parser.add_argument(
        "--draft-model", default="/shared/xiaomi/Qwen3.5-397B-A17B-DFlash/"
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "Benchmark an already-running server at this URL (e.g. http://127.0.0.1:9080) "
            "instead of launching one server per config. All server-side flags "
            "(--tp-size, --attention-backends, DFLASH flags, ...) are then ignored: the "
            "running server's own configuration is what gets measured, and the first "
            "--tp-sizes / --attention-backends value is used only as a summary label."
        ),
    )
    parser.add_argument(
        "--external-mode",
        default="auto",
        choices=["auto", "baseline", "dflash"],
        help=(
            "Which summary row a --base-url run fills in. `auto` classifies by whether "
            "the server reports spec_verify_ct; `dflash` additionally enforces the "
            "DFLASH sanity check."
        ),
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip running the baseline (target-only) sweep; only run DFLASH and report N/A for baseline/speedup.",
    )
    parser.add_argument(
        "--batch-requests",
        action="store_true",
        help="Send prompts as server-side batched /generate requests (batch size = concurrency) instead of client-side concurrent requests.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=3600,
        help=(
            "Timeout in seconds for benchmarked /generate calls and server startup "
            "health checks."
        ),
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=0.9,
        help="Server --mem-fraction-static (default: 0.9, matches the 397B launch script).",
    )
    parser.add_argument(
        "--disable-radix-cache",
        action="store_true",
        help="Pass --disable-radix-cache to the server. Off by default, since "
        "the 397B DFlash launch script runs with the prefix cache on.",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="Optional server --dtype override. Unset matches the 397B launch script, "
        "which lets the server infer dtype from the checkpoint.",
    )
    parser.add_argument(
        "--speculative-num-draft-tokens",
        type=int,
        default=16,
        help="DFLASH verify window length (matches the 397B launch script).",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=64,
        help="Server --page-size for both baseline and DFLASH runs "
        "(default: 64, matches the 397B launch script).",
    )
    parser.add_argument(
        "--max-running-requests",
        type=int,
        default=128,
        help="Server --max-running-requests (default: 128, matches the 397B launch script).",
    )
    parser.add_argument(
        "--mamba-scheduler-strategy",
        default="extra_buffer",
        help=(
            "Server --mamba-scheduler-strategy value to pass through to benchmark "
            "runs, e.g. `no_buffer` or `extra_buffer`."
        ),
    )
    parser.add_argument("--tp-sizes", default="8")
    parser.add_argument("--concurrencies", default="1")
    parser.add_argument(
        "--questions-per-concurrency-base",
        type=int,
        default=80,
        help="num_questions = base * concurrency (default matches the sweep plan).",
    )
    parser.add_argument(
        "--max-questions-per-config",
        type=int,
        default=80,
        help="Cap num_questions per (tp, concurrency) run (MT-Bench has 80 items).",
    )
    parser.add_argument("--attention-backends", default="aiter")
    parser.add_argument(
        "--speculative-draft-attention-backend",
        default="triton",
        help="Optional server --speculative-draft-attention-backend override for DFLASH runs.",
    )
    parser.add_argument(
        "--speculative-dflash-draft-window-size",
        type=int,
        default=None,
        help="Optional server --speculative-dflash-draft-window-size override for DFLASH runs.",
    )
    parser.add_argument(
        "--result-json",
        default=None,
        help="Write per-run metrics to this path for make_table.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # With --base-url the server runs elsewhere (or was launched by hand), so this
    # process is a pure HTTP client and needs no local GPU.
    external = bool(args.base_url)
    if not external and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this sweep.")
    if args.temperature < 0.0:
        raise RuntimeError(f"--temperature must be >= 0, got {args.temperature}.")
    if not (0.0 < args.top_p <= 1.0):
        raise RuntimeError(f"--top-p must be in (0, 1], got {args.top_p}.")
    if args.top_k == 0 or args.top_k < -1:
        raise RuntimeError(f"--top-k must be -1 (all vocab) or >= 1, got {args.top_k}.")
    if args.timeout_s <= 0:
        raise RuntimeError(f"--timeout-s must be > 0, got {args.timeout_s}.")

    tp_sizes = _parse_int_csv(args.tp_sizes)
    if external:
        # Labels only: nothing is launched, so there is no GPU count to validate against.
        tp_sizes = tp_sizes[:1] or [0]
    else:
        visible_gpus = int(torch.cuda.device_count())
        tp_sizes = [tp for tp in tp_sizes if tp >= 1 and tp <= visible_gpus]
        if not tp_sizes:
            raise RuntimeError(
                f"No tp sizes are runnable with visible_gpus={visible_gpus}. "
                "Set CUDA_VISIBLE_DEVICES accordingly."
            )

    concurrencies = _parse_int_csv(args.concurrencies)
    concurrencies = [c for c in concurrencies if c >= 1]
    if not concurrencies:
        raise RuntimeError("No concurrencies specified.")

    num_questions_by_conc = {
        c: min(
            int(args.questions_per_concurrency_base) * int(c),
            int(args.max_questions_per_config),
        )
        for c in concurrencies
    }
    max_questions = max(num_questions_by_conc.values())

    attention_backends = [
        s.strip() for s in args.attention_backends.split(",") if s.strip()
    ]
    if external:
        # Label only; the running server already picked its backend.
        device_sm = 0
        attention_backends = attention_backends[:1] or ["external"]
    else:
        device_sm = get_device_sm()
        attention_backends = _filter_attention_backends(
            attention_backends, device_sm=device_sm
        )

    data_path = _maybe_download_mtbench(args.data_path)
    lines = list(read_jsonl(data_path))
    if len(lines) < max_questions:
        raise RuntimeError(
            f"MT-Bench file only has {len(lines)} lines, but need {max_questions}."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)

    prompts: list[str] = []
    for i in range(max_questions):
        # MT-Bench items store a list of turns; use the first-turn prompt.
        turns = lines[i]["turns"]
        user_content = turns[0]
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )

    # Results indexed by (backend, tp, concurrency, mode).
    results: dict[tuple[str, int, int, str], BenchMetrics] = {}
    # Baseline metrics are backend-agnostic in this sweep; run once per TP and reuse.
    baseline_cache_by_tp: dict[int, dict[int, BenchMetrics]] = {}

    if external:
        base_url = args.base_url.rstrip("/")
        backend = attention_backends[0]
        tp = tp_sizes[0]
        mode_metrics = _run_mode_for_backend_tp(
            mode_label=f"external server {base_url} (mode={args.external_mode})",
            model_path=args.target_model,
            base_url=base_url,
            server_args=[],
            expect_dflash=(args.external_mode == "dflash"),
            prompts=prompts,
            concurrencies=concurrencies,
            num_questions_by_conc=num_questions_by_conc,
            args=args,
            launch_server=False,
        )
        for conc, metrics in mode_metrics.items():
            mode_key = args.external_mode
            if mode_key == "auto":
                mode_key = "dflash" if metrics.spec_verify_ct_sum > 0 else "baseline"
            results[(backend, tp, conc, mode_key)] = metrics

        _print_summary(
            args=args,
            attention_backends=attention_backends,
            tp_sizes=tp_sizes,
            concurrencies=concurrencies,
            device_sm=device_sm,
            results=results,
        )

        if args.result_json:
            _dump_result_json(args.result_json, args=args, results=results)
        return

    for backend_idx, backend in enumerate(attention_backends):
        for tp in tp_sizes:
            port_base = find_available_port(20000)
            common_server_args = _build_common_server_args(args, backend=backend, tp=tp)
            mode_runs = _build_mode_runs(args, common_server_args)

            for idx, (
                mode_key,
                mode_name,
                mode_server_args,
                expect_dflash,
            ) in enumerate(mode_runs):
                if (
                    mode_key == "baseline"
                    and not args.skip_baseline
                    and backend_idx > 0
                    and tp in baseline_cache_by_tp
                ):
                    mode_metrics = baseline_cache_by_tp[tp]
                else:
                    mode_metrics = _run_mode_for_backend_tp(
                        mode_label=f"backend={backend} tp={tp} ({mode_name})",
                        model_path=args.target_model,
                        base_url=f"http://127.0.0.1:{find_available_port(port_base + idx)}",
                        server_args=mode_server_args,
                        expect_dflash=expect_dflash,
                        prompts=prompts,
                        concurrencies=concurrencies,
                        num_questions_by_conc=num_questions_by_conc,
                        args=args,
                    )
                    if mode_key == "baseline" and not args.skip_baseline:
                        baseline_cache_by_tp[tp] = mode_metrics

                for conc, metrics in mode_metrics.items():
                    results[(backend, tp, conc, mode_key)] = metrics

    _print_summary(
        args=args,
        attention_backends=attention_backends,
        tp_sizes=tp_sizes,
        concurrencies=concurrencies,
        device_sm=device_sm,
        results=results,
    )

    if args.result_json:
        _dump_result_json(args.result_json, args=args, results=results)


if __name__ == "__main__":
    main()
