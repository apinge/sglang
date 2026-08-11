#!/usr/bin/env python3
"""Run the official DFlash benchmark harness on ROCm/Triton with FP8 KV and GSM8K accuracy.

This wrapper imports the official benchmark harness from the DFlash model bundle,
overrides the server launch config for this MI350X/ROCm environment, and adds a
sidecar accuracy recorder for GSM8K measured samples.

The upstream harness reports throughput and DFlash accept length, but it does
not score GSM8K correctness. This wrapper keeps the same prompt construction,
warmup/warmdown behavior, and response-metadata accept-length metric, while
recording generated texts and numeric GSM8K correctness for the measured pass.

For the common single-draft-model path, this script can also generate tiny
per-block DFlash config overlays from the draft model config. That keeps Triton
target attention and DFlash block-size sweeps in one entry point.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


HARNESS_PATH = Path(
    "/models/Qwen3.5-397B-A17B-DFlash/benchmark/run_benchmark.py"
)

DEFAULT_ACCURACY_OUTPUT_NAME = "official_harness_rocm_triton_accuracy.jsonl"
DEFAULT_ACCURACY_CSV_NAME = "official_harness_rocm_triton_accuracy.csv"

ACCURACY_RECORDS: list[dict[str, Any]] = []
ACCURACY_SUMMARY: list[dict[str, Any]] = []
ACCURACY_LOCK = threading.Lock()
ACCURACY_OUTPUT_PATH: Path | None = None
ACCURACY_CSV_OUTPUT_PATH: Path | None = None
DFLASH_DRAFT_MODEL_MAP: dict[int, str] = {}
DFLASH_DRAFT_CONFIG_FILE_MAP: dict[int, str] = {}
TARGET_ATTENTION_BACKEND = "triton"
TARGET_MM_ATTENTION_BACKEND = "triton_attn"
ENABLE_AITER_ENV = False
TARGET_TP_SIZE = 8
TARGET_MAX_RUNNING_REQUESTS = 1
TARGET_CUDA_GRAPH_MAX_BS_DECODE = 1
TARGET_MEM_FRACTION_STATIC = 0.9
TARGET_PAGE_SIZE = 64
TARGET_LINEAR_ATTN_BACKEND = "triton"
TARGET_MAMBA_SSM_DTYPE = "bfloat16"
TARGET_EXTRA_SERVER_ARGS: list[str] = []


def _pop_cli_value(argv: list[str], flag: str) -> str | None:
    eq_prefix = flag + "="
    for idx, arg in enumerate(list(argv)):
        if arg.startswith(eq_prefix):
            del argv[idx]
            return arg[len(eq_prefix) :]
        if arg == flag:
            if idx + 1 >= len(argv):
                raise SystemExit(f"{flag} requires a value")
            value = argv[idx + 1]
            del argv[idx : idx + 2]
            return value
    return None


def _pop_cli_flag(argv: list[str], flag: str) -> bool:
    if flag not in argv:
        return False
    argv.remove(flag)
    return True


def _get_cli_value(argv: list[str], flag: str) -> str | None:
    eq_prefix = flag + "="
    for idx, arg in enumerate(argv):
        if arg.startswith(eq_prefix):
            return arg[len(eq_prefix) :]
        if arg == flag and idx + 1 < len(argv):
            return argv[idx + 1]
    return None


def _pop_cli_int(argv: list[str], flag: str, default: int) -> int:
    value = _pop_cli_value(argv, flag)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise SystemExit(f"{flag} must be an integer, got {value!r}") from exc


def _pop_cli_float(argv: list[str], flag: str, default: float) -> float:
    value = _pop_cli_value(argv, flag)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise SystemExit(f"{flag} must be a float, got {value!r}") from exc


def _pop_cli_json_list(argv: list[str], flag: str) -> list[str]:
    value = _pop_cli_value(argv, flag)
    if value is None:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{flag} must be a JSON list of strings") from exc
    if not isinstance(parsed, list) or not all(
        isinstance(item, str) for item in parsed
    ):
        raise SystemExit(f"{flag} must be a JSON list of strings")
    return list(parsed)


def _derive_output_paths(argv: list[str]) -> tuple[Path, Path]:
    accuracy_output = _pop_cli_value(argv, "--accuracy-output")
    accuracy_csv_output = _pop_cli_value(argv, "--accuracy-csv-output")

    csv_output = None
    if "--csv-output" in argv:
        idx = argv.index("--csv-output")
        if idx + 1 < len(argv):
            csv_output = Path(argv[idx + 1])

    if accuracy_output is None:
        parent = csv_output.parent if csv_output is not None else Path.cwd()
        accuracy_output_path = parent / DEFAULT_ACCURACY_OUTPUT_NAME
    else:
        accuracy_output_path = Path(accuracy_output)

    if accuracy_csv_output is None:
        parent = csv_output.parent if csv_output is not None else Path.cwd()
        accuracy_csv_path = parent / DEFAULT_ACCURACY_CSV_NAME
    else:
        accuracy_csv_path = Path(accuracy_csv_output)

    return accuracy_output_path, accuracy_csv_path


def _parse_dflash_draft_model_map(value: str | None) -> dict[int, str]:
    if value is None or not value.strip():
        return {}
    parsed: dict[int, str] = {}
    text = value.strip()
    if text.startswith("{"):
        raw = json.loads(text)
        if not isinstance(raw, dict):
            raise SystemExit("--dflash-draft-model-map JSON must be an object")
        items = raw.items()
    else:
        items = []
        for part in text.split(","):
            if not part.strip():
                continue
            if "=" not in part:
                raise SystemExit(
                    "--dflash-draft-model-map entries must be BLOCK=PATH"
                )
            block, path = part.split("=", 1)
            items.append((block.strip(), path.strip()))
    for block, path in items:
        try:
            block_int = int(block)
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"--dflash-draft-model-map block key must be an int, got {block!r}"
            ) from exc
        if block_int <= 0:
            raise SystemExit(
                f"--dflash-draft-model-map block key must be positive, got {block_int}"
            )
        if not str(path):
            raise SystemExit(f"--dflash-draft-model-map path is empty for {block_int}")
        parsed[block_int] = str(path)
    return parsed


def _parse_dflash_draft_config_file_map(value: str | None) -> dict[int, str]:
    if value is None or not value.strip():
        return {}
    parsed: dict[int, str] = {}
    text = value.strip()
    if text.startswith("{"):
        raw = json.loads(text)
        if not isinstance(raw, dict):
            raise SystemExit("--dflash-draft-config-file-map JSON must be an object")
        items = raw.items()
    else:
        items = []
        for part in text.split(","):
            if not part.strip():
                continue
            if "=" not in part:
                raise SystemExit(
                    "--dflash-draft-config-file-map entries must be BLOCK=PATH"
                )
            block, path = part.split("=", 1)
            items.append((block.strip(), path.strip()))
    for block, path in items:
        try:
            block_int = int(block)
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"--dflash-draft-config-file-map block key must be an int, got {block!r}"
            ) from exc
        if block_int <= 0:
            raise SystemExit(
                f"--dflash-draft-config-file-map block key must be positive, got {block_int}"
            )
        if not str(path):
            raise SystemExit(
                f"--dflash-draft-config-file-map path is empty for {block_int}"
            )
        parsed[block_int] = str(path)
    return parsed


def _parse_block_sizes(value: str | None) -> list[int]:
    if value is None:
        return []
    text = value.strip()
    if not text or text == "default":
        return []

    blocks: list[int] = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            block = int(raw)
        except ValueError as exc:
            raise SystemExit(
                f"--dflash-block-sizes must be comma-separated integers, got {raw!r}"
            ) from exc
        if block <= 0:
            raise SystemExit(
                f"--dflash-block-sizes values must be positive, got {block}"
            )
        blocks.append(block)
    return blocks


def _default_overlay_dir(argv: list[str]) -> Path:
    csv_output = _get_cli_value(argv, "--csv-output")
    if csv_output:
        return Path(csv_output).resolve().parent / "dflash_block_config_overrides"
    return Path.cwd() / "dflash_block_config_overrides"


def _load_dflash_base_config(
    draft_model: str, base_config_file: str | None
) -> dict[str, Any]:
    config_path = (
        Path(base_config_file) if base_config_file else Path(draft_model) / "config.json"
    )
    if not config_path.is_file():
        raise SystemExit(
            "Cannot find DFlash draft config. Pass a local --dflash-draft-model "
            "directory or set --dflash-base-config-file. Missing: "
            f"{config_path}"
        )

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise SystemExit(f"Expected JSON object in {config_path}")
    if not isinstance(config.get("dflash_config"), dict):
        raise SystemExit(
            f"{config_path} does not contain a dflash_config object; cannot "
            "generate block-size config overlays."
        )
    return config


def _write_dflash_config_overlays(
    *,
    base_config: dict[str, Any],
    blocks: list[int],
    overlay_dir: Path,
) -> dict[int, str]:
    overlay_dir.mkdir(parents=True, exist_ok=True)
    config_file_map: dict[int, str] = {}

    for block in blocks:
        config = json.loads(json.dumps(base_config))
        dflash_config = dict(config["dflash_config"])
        dflash_config["block_size"] = int(block)
        config["dflash_config"] = dflash_config

        out = overlay_dir / f"config_dflash_block_{block}.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            f.write("\n")
        config_file_map[int(block)] = str(out)

    return config_file_map


def _maybe_generate_dflash_config_file_map(
    *,
    argv: list[str],
    existing_config_file_map: dict[int, str],
    existing_model_map: dict[int, str],
    overlay_dir_arg: str | None,
    base_config_file: str | None,
    disable_auto_overlays: bool,
) -> dict[int, str]:
    if existing_config_file_map or existing_model_map or disable_auto_overlays:
        return existing_config_file_map

    draft_model = _get_cli_value(argv, "--dflash-draft-model")
    block_sizes = _parse_block_sizes(_get_cli_value(argv, "--dflash-block-sizes"))
    if not draft_model or not block_sizes:
        return existing_config_file_map

    overlay_dir = (
        Path(overlay_dir_arg).resolve()
        if overlay_dir_arg
        else _default_overlay_dir(argv)
    )
    base_config = _load_dflash_base_config(draft_model, base_config_file)
    generated = _write_dflash_config_overlays(
        base_config=base_config,
        blocks=block_sizes,
        overlay_dir=overlay_dir,
    )
    print(f"[config] generated DFlash block config overlays: {overlay_dir}")
    return generated


def load_harness():
    spec = importlib.util.spec_from_file_location(
        "official_dflash_run_benchmark_fp8kv_accuracy", HARNESS_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load official harness from {HARNESS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _normalize_numeric_answer(text: str) -> Decimal | None:
    text = str(text).replace(",", "")
    boxed = re.findall(r"\\boxed\s*\{([^{}]+)\}", text)
    candidates: list[str] = []
    if boxed:
        candidates.extend(boxed)
    candidates.extend(re.findall(r"[-+]?\d+(?:\.\d+)?", text))
    for raw in reversed(candidates):
        raw = raw.strip()
        match = re.search(r"[-+]?\d+(?:\.\d+)?", raw)
        if not match:
            continue
        try:
            return Decimal(match.group(0))
        except InvalidOperation:
            continue
    return None


def _load_gsm8k_items(h) -> list[dict[str, Any]]:
    path = h._download_to_cache(h.GSM8K_TEST_URL, "gsm8k_test.jsonl")
    rows = h._read_jsonl(path)
    items: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        prompt = (
            row["question"]
            + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        )
        label = _normalize_numeric_answer(row["answer"])
        if label is None:
            raise RuntimeError(f"Could not parse GSM8K label for row {idx}: {row}")
        items.append(
            {
                "sample_index": idx,
                "turns": [prompt],
                "question": row["question"],
                "answer": row["answer"],
                "label": str(label),
            }
        )
    return items


def _generation_turn_count_items(items: list[dict[str, Any]]) -> int:
    return sum(len(item["turns"]) for item in items)


def _take_items(
    items: list[dict[str, Any]], *, start: int, count: int
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    if not items:
        raise RuntimeError("Cannot take benchmark samples from an empty workload.")
    return [items[(start + i) % len(items)] for i in range(count)]


def _take_items_for_min_generation_turns(
    items: list[dict[str, Any]], *, start: int, min_generation_turns: int
) -> list[dict[str, Any]]:
    if min_generation_turns <= 0:
        return []
    if not items:
        raise RuntimeError("Cannot take benchmark samples from an empty workload.")
    out: list[dict[str, Any]] = []
    generation_turns = 0
    idx = 0
    while generation_turns < int(min_generation_turns):
        item = items[(start + idx) % len(items)]
        out.append(item)
        generation_turns += len(item["turns"])
        idx += 1
    return out


def _build_measured_items(
    items: list[dict[str, Any]],
    *,
    num_samples: int | None,
    min_generation_turns: int,
    concurrency: int,
) -> list[dict[str, Any]]:
    if num_samples is not None:
        return _take_items(items, start=0, count=int(num_samples))
    if int(concurrency) == 1:
        return list(items)
    source_generation_turns = _generation_turn_count_items(items)
    repeats = max(1, (int(min_generation_turns) + source_generation_turns - 1) // source_generation_turns)
    return items * repeats


def _build_item_plan(h, source_items: list[dict[str, Any]], *, concurrency: int, methodology):
    measured_items = _build_measured_items(
        source_items,
        num_samples=methodology.num_samples,
        min_generation_turns=int(methodology.min_generation_turns_per_config),
        concurrency=int(concurrency),
    )
    warmup_min_generation_turns = max(
        int(methodology.min_warmup_generation_turns), 2 * int(concurrency)
    )
    warmup_items = _take_items_for_min_generation_turns(
        measured_items,
        start=0,
        min_generation_turns=warmup_min_generation_turns,
    )
    warmdown_items = _take_items(
        measured_items,
        start=len(warmup_items),
        count=int(concurrency),
    )
    return measured_items, warmup_items, warmdown_items


def _append_accuracy_record(record: dict[str, Any]) -> None:
    with ACCURACY_LOCK:
        ACCURACY_RECORDS.append(record)
        if ACCURACY_OUTPUT_PATH is not None:
            ACCURACY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with ACCURACY_OUTPUT_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _run_item(
    h,
    *,
    base_url: str,
    item: dict[str, Any],
    tokenizer,
    sampling,
    timeout_s: int,
    record_accuracy: bool,
    job,
):
    messages: list[dict[str, str]] = []
    total_tokens = 0
    spec_verify_ct_sum = 0
    turn_accept_lengths: list[float] = []
    generated_texts: list[str] = []
    meta_infos: list[dict[str, Any]] = []

    tic = time.perf_counter()
    for turn_idx, user_content in enumerate(item["turns"]):
        messages.append({"role": "user", "content": user_content})
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(sampling.enable_thinking),
        )
        out = h._send_generate(
            base_url=base_url,
            text=prompt,
            max_new_tokens=sampling.max_new_tokens,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
            timeout_s=timeout_s,
        )
        generated_text = h._extract_generated_text(out)
        generated_texts.append(generated_text)
        meta = out.get("meta_info", {}) or {}
        meta_infos.append(meta)

        output_tokens, spec_verify_ct, turn_accept_length = h._extract_generate_stats(out)
        total_tokens += output_tokens
        spec_verify_ct_sum += spec_verify_ct
        if spec_verify_ct > 0:
            turn_accept_lengths.append(float(output_tokens) / float(spec_verify_ct))
        elif turn_accept_length is not None:
            turn_accept_lengths.append(turn_accept_length)

        if turn_idx + 1 < len(item["turns"]):
            messages.append({"role": "assistant", "content": generated_text})

    latency_s = time.perf_counter() - tic

    if record_accuracy:
        label = Decimal(str(item["label"]))
        pred = _normalize_numeric_answer(generated_texts[-1] if generated_texts else "")
        correct = pred is not None and pred == label
        key = job.key
        _append_accuracy_record(
            {
                "workload": key.workload,
                "backend": key.backend,
                "tp": key.tp,
                "concurrency": key.concurrency,
                "mode": key.mode,
                "dflash_block_size": job.deployment.dflash_block_size,
                "run_index": job.run_index,
                "sample_index": item["sample_index"],
                "label": str(label),
                "prediction": None if pred is None else str(pred),
                "correct": bool(correct),
                "invalid": pred is None,
                "latency_s": latency_s,
                "output_tokens": int(total_tokens),
                "spec_verify_ct": int(spec_verify_ct_sum),
                "accept_length": (
                    None
                    if not turn_accept_lengths
                    else float(sum(turn_accept_lengths) / len(turn_accept_lengths))
                ),
                "question": item["question"],
                "answer": item["answer"],
                "output": generated_texts[-1] if generated_texts else "",
                "meta_info": meta_infos[-1] if meta_infos else {},
            }
        )

    return h.SampleMetrics(
        generation_turn_count=len(item["turns"]),
        output_tokens=int(total_tokens),
        spec_verify_ct_sum=int(spec_verify_ct_sum),
        spec_accept_lengths=tuple(turn_accept_lengths),
    )


def _run_unmeasured_items(
    h,
    *,
    base_url: str,
    items: list[dict[str, Any]],
    tokenizer,
    sampling,
    concurrency: int,
    timeout_s: int,
    job,
) -> None:
    if not items:
        return
    with ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
        futures = [
            pool.submit(
                _run_item,
                h,
                base_url=base_url,
                item=item,
                tokenizer=tokenizer,
                sampling=sampling,
                timeout_s=timeout_s,
                record_accuracy=False,
                job=job,
            )
            for item in items
        ]
        for fut in as_completed(futures):
            fut.result()


def _run_measured_items(
    h,
    *,
    base_url: str,
    items: list[dict[str, Any]],
    warmdown_items: list[dict[str, Any]],
    tokenizer,
    sampling,
    concurrency: int,
    timeout_s: int,
    expect_spec: bool,
    job,
):
    start = time.perf_counter()
    total_tokens = 0
    spec_verify_ct_sum = 0
    generation_turn_count = 0
    turn_accept_lengths: list[float] = []
    measured_completed = 0
    latency = None

    with ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
        measured_futures = [
            pool.submit(
                _run_item,
                h,
                base_url=base_url,
                item=item,
                tokenizer=tokenizer,
                sampling=sampling,
                timeout_s=timeout_s,
                record_accuracy=True,
                job=job,
            )
            for item in items
        ]
        measured_future_set = set(measured_futures)
        warmdown_futures = [
            pool.submit(
                _run_item,
                h,
                base_url=base_url,
                item=item,
                tokenizer=tokenizer,
                sampling=sampling,
                timeout_s=timeout_s,
                record_accuracy=False,
                job=job,
            )
            for item in warmdown_items
        ]
        consumed_warmdown_futures = set()

        for fut in as_completed([*measured_futures, *warmdown_futures]):
            if fut in measured_future_set:
                sample_metrics = fut.result()
                total_tokens += sample_metrics.output_tokens
                spec_verify_ct_sum += sample_metrics.spec_verify_ct_sum
                generation_turn_count += sample_metrics.generation_turn_count
                turn_accept_lengths.extend(sample_metrics.spec_accept_lengths)
                measured_completed += 1
                if measured_completed == len(measured_futures):
                    latency = time.perf_counter() - start
                    break
            else:
                consumed_warmdown_futures.add(fut)
                fut.result()

        for fut in warmdown_futures:
            if fut not in consumed_warmdown_futures:
                fut.result()

    if latency is None:
        latency = time.perf_counter() - start
    toks_per_s = total_tokens / max(latency, 1e-6)

    if expect_spec and spec_verify_ct_sum <= 0:
        raise RuntimeError(
            "Speculative decoding sanity check failed: did not observe any "
            "`spec_verify_ct` in responses (speculative decoding may not have been enabled)."
        )

    spec_accept_length = (
        float(sum(turn_accept_lengths) / len(turn_accept_lengths))
        if turn_accept_lengths
        else None
    )

    records = [
        record
        for record in ACCURACY_RECORDS
        if record["workload"] == job.key.workload
        and record["mode"] == job.key.mode
        and record["run_index"] == job.run_index
        and record["concurrency"] == job.key.concurrency
    ]
    correct_count = sum(1 for record in records if record["correct"])
    invalid_count = sum(1 for record in records if record["invalid"])
    accuracy = correct_count / len(records) if records else None
    invalid_rate = invalid_count / len(records) if records else None
    summary = {
        "workload": job.key.workload,
        "backend": job.key.backend,
        "tp": job.key.tp,
        "concurrency": job.key.concurrency,
        "mode": job.key.mode,
        "dflash_block_size": job.deployment.dflash_block_size,
        "draft_model": (
            getattr(job.deployment.mode_config, "draft_model", None)
            if job.deployment.expect_spec
            else None
        ),
        "run_index": job.run_index,
        "sample_count": len(records),
        "correct_count": correct_count,
        "invalid_count": invalid_count,
        "accuracy": accuracy,
        "invalid_rate": invalid_rate,
        "latency_s": float(latency),
        "output_tokens": int(total_tokens),
        "output_toks_per_s": float(toks_per_s),
        "spec_accept_length": spec_accept_length,
        "spec_verify_ct_sum": int(spec_verify_ct_sum),
    }
    with ACCURACY_LOCK:
        ACCURACY_SUMMARY.append(summary)

    return h.BenchMetrics(
        sample_count=len(items),
        generation_turn_count=int(generation_turn_count),
        latency_s=float(latency),
        output_tokens=int(total_tokens),
        output_toks_per_s=float(toks_per_s),
        spec_accept_length=spec_accept_length,
        spec_verify_ct_sum=int(spec_verify_ct_sum),
    )


def _run_benchmark_job_with_accuracy(h, job):
    from sglang.test.test_utils import (
        DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH as SGLANG_DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        find_available_port,
        popen_launch_server,
    )
    from transformers import AutoTokenizer

    if job.workload != "gsm8k":
        raise RuntimeError("This accuracy wrapper currently supports only GSM8K.")

    key = job.key
    print(f"\n=== {job.label} {job.run_label} ===")
    source_items = _load_gsm8k_items(h)
    source_sample_count = len(source_items)
    source_generation_turn_count = _generation_turn_count_items(source_items)
    measured_items, warmup_items, warmdown_items = _build_item_plan(
        h,
        source_items,
        concurrency=job.concurrency,
        methodology=job.methodology,
    )
    if len(measured_items) > source_sample_count:
        print(
            "[config] measured sample count exceeds workload size; "
            "repeating whole workload copies with radix cache enabled."
        )

    base_url = f"http://127.0.0.1:{find_available_port(20000)}"
    tokenizer = AutoTokenizer.from_pretrained(job.target_model)
    server_start_timeout_s = int(
        max(SGLANG_DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, job.methodology.timeout_s)
    )
    server_env = h._server_env_for_job(job)
    if ENABLE_AITER_ENV:
        server_env.update(
            {
                "SGLANG_USE_AITER": "1",
                "SGLANG_USE_CUDA_IPC_TRANSPORT": "1",
                "SGLANG_USE_IPC_POOL_HANDLE_CACHE": "1",
            }
        )
    print(
        "server_env="
        + ",".join(f"{k}:{v}" for k, v in sorted(server_env.items()))
    )
    proc = popen_launch_server(
        job.target_model,
        base_url,
        timeout=server_start_timeout_s,
        other_args=job.deployment.server_args,
        env=server_env,
    )
    try:
        h._send_generate(
            base_url,
            "Hello",
            max_new_tokens=8,
            temperature=job.sampling.temperature,
            top_p=job.sampling.top_p,
            top_k=job.sampling.top_k,
            timeout_s=min(job.methodology.timeout_s, 300),
        )

        h._flush_cache(base_url)
        print(
            f"[warmup {job.run_label}] run {len(warmup_items)} samples / "
            f"{_generation_turn_count_items(warmup_items)} generation turns after "
            "/flush_cache; excluded from metrics."
        )
        _run_unmeasured_items(
            h,
            base_url=base_url,
            items=warmup_items,
            tokenizer=tokenizer,
            sampling=job.sampling,
            concurrency=job.concurrency,
            timeout_s=job.methodology.timeout_s,
            job=job,
        )
        h._flush_cache(base_url)
        print(
            f"[warmup {job.run_label}] flushed cache after warmup; "
            "starting measured workload."
        )
        metrics = _run_measured_items(
            h,
            base_url=base_url,
            items=measured_items,
            warmdown_items=warmdown_items,
            tokenizer=tokenizer,
            sampling=job.sampling,
            concurrency=job.concurrency,
            timeout_s=job.methodology.timeout_s,
            expect_spec=job.deployment.expect_spec,
            job=job,
        )
        summary = next(
            s
            for s in reversed(ACCURACY_SUMMARY)
            if s["workload"] == key.workload
            and s["mode"] == key.mode
            and s["run_index"] == job.run_index
        )
        line = (
            f"[{job.label} {job.run_label}] samples={len(measured_items):<4} "
            f"turns={_generation_turn_count_items(measured_items):<4} "
            f"toks/s={metrics.output_toks_per_s:,.2f} "
            f"latency={metrics.latency_s:.1f}s "
            f"accuracy={summary['accuracy']:.3f} "
            f"invalid={summary['invalid_rate']:.3f} "
            f"warmup_turns={_generation_turn_count_items(warmup_items)} "
            f"warmdown_turns={_generation_turn_count_items(warmdown_items)}"
        )
        if job.deployment.expect_spec:
            accept_len = (
                "N/A"
                if metrics.spec_accept_length is None
                else f"{metrics.spec_accept_length:.3f}"
            )
            line += (
                f" accept_len_mean={accept_len} "
                f"spec_verify_ct_sum={metrics.spec_verify_ct_sum}"
            )
        print(line)
        return h.JobResult(
            key=key,
            deployment=job.deployment,
            source_sample_count=source_sample_count,
            source_generation_turn_count=source_generation_turn_count,
            warmup_generation_turn_count=_generation_turn_count_items(warmup_items),
            warmdown_generation_turn_count=_generation_turn_count_items(warmdown_items),
            run_index=job.run_index,
            metrics=metrics,
        )
    finally:
        h._shutdown_server(
            proc,
            base_url,
            drain_timeout_s=job.methodology.server_shutdown_drain_timeout_s,
            kill_timeout_s=job.methodology.server_shutdown_timeout_s,
        )


def _write_accuracy_summary_csv() -> None:
    if ACCURACY_CSV_OUTPUT_PATH is None:
        return
    ACCURACY_CSV_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "workload",
        "backend",
        "tp",
        "concurrency",
        "mode",
        "dflash_block_size",
        "draft_model",
        "run_index",
        "sample_count",
        "correct_count",
        "invalid_count",
        "accuracy",
        "invalid_rate",
        "latency_s",
        "output_tokens",
        "output_toks_per_s",
        "spec_accept_length",
        "spec_verify_ct_sum",
    ]
    with ACCURACY_CSV_OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in ACCURACY_SUMMARY:
            writer.writerow(row)
    print(f"[accuracy-csv] wrote {len(ACCURACY_SUMMARY)} rows to {ACCURACY_CSV_OUTPUT_PATH}")


def main() -> None:
    global ACCURACY_OUTPUT_PATH, ACCURACY_CSV_OUTPUT_PATH
    global DFLASH_DRAFT_MODEL_MAP, DFLASH_DRAFT_CONFIG_FILE_MAP
    global TARGET_ATTENTION_BACKEND, TARGET_MM_ATTENTION_BACKEND, ENABLE_AITER_ENV
    global TARGET_TP_SIZE, TARGET_MAX_RUNNING_REQUESTS
    global TARGET_CUDA_GRAPH_MAX_BS_DECODE, TARGET_MEM_FRACTION_STATIC
    global TARGET_PAGE_SIZE, TARGET_LINEAR_ATTN_BACKEND, TARGET_MAMBA_SSM_DTYPE
    global TARGET_EXTRA_SERVER_ARGS

    argv = sys.argv
    ACCURACY_OUTPUT_PATH, ACCURACY_CSV_OUTPUT_PATH = _derive_output_paths(argv)
    overlay_dir_arg = _pop_cli_value(argv, "--dflash-config-overlay-dir")
    base_config_file = _pop_cli_value(argv, "--dflash-base-config-file")
    disable_auto_overlays = _pop_cli_flag(argv, "--disable-auto-dflash-config-overlays")
    TARGET_ATTENTION_BACKEND = (
        _pop_cli_value(argv, "--target-attention-backend") or "triton"
    )
    TARGET_MM_ATTENTION_BACKEND = (
        _pop_cli_value(argv, "--target-mm-attention-backend") or "triton_attn"
    )
    ENABLE_AITER_ENV = _pop_cli_flag(argv, "--enable-aiter-env") or (
        TARGET_ATTENTION_BACKEND == "aiter"
        or TARGET_MM_ATTENTION_BACKEND == "aiter_attn"
    )
    TARGET_TP_SIZE = _pop_cli_int(argv, "--target-tp-size", 8)
    TARGET_MAX_RUNNING_REQUESTS = _pop_cli_int(
        argv, "--target-max-running-requests", 1
    )
    TARGET_CUDA_GRAPH_MAX_BS_DECODE = _pop_cli_int(
        argv, "--target-cuda-graph-max-bs-decode", 1
    )
    TARGET_MEM_FRACTION_STATIC = _pop_cli_float(
        argv, "--target-mem-fraction-static", 0.9
    )
    TARGET_PAGE_SIZE = _pop_cli_int(argv, "--target-page-size", 64)
    TARGET_LINEAR_ATTN_BACKEND = (
        _pop_cli_value(argv, "--target-linear-attn-backend") or "triton"
    )
    TARGET_MAMBA_SSM_DTYPE = (
        _pop_cli_value(argv, "--target-mamba-ssm-dtype") or "bfloat16"
    )
    TARGET_EXTRA_SERVER_ARGS = _pop_cli_json_list(argv, "--target-extra-server-args")
    DFLASH_DRAFT_MODEL_MAP = _parse_dflash_draft_model_map(
        _pop_cli_value(argv, "--dflash-draft-model-map")
    )
    DFLASH_DRAFT_CONFIG_FILE_MAP = _parse_dflash_draft_config_file_map(
        _pop_cli_value(argv, "--dflash-draft-config-file-map")
    )
    DFLASH_DRAFT_CONFIG_FILE_MAP = _maybe_generate_dflash_config_file_map(
        argv=argv,
        existing_config_file_map=DFLASH_DRAFT_CONFIG_FILE_MAP,
        existing_model_map=DFLASH_DRAFT_MODEL_MAP,
        overlay_dir_arg=overlay_dir_arg,
        base_config_file=base_config_file,
        disable_auto_overlays=disable_auto_overlays,
    )
    if DFLASH_DRAFT_MODEL_MAP:
        print(
            "[config] DFlash draft model map: "
            + ", ".join(
                f"block{block}={path}"
                for block, path in sorted(DFLASH_DRAFT_MODEL_MAP.items())
            )
        )
    if DFLASH_DRAFT_CONFIG_FILE_MAP:
        print(
            "[config] DFlash draft config file map: "
            + ", ".join(
                f"block{block}={path}"
                for block, path in sorted(DFLASH_DRAFT_CONFIG_FILE_MAP.items())
            )
        )
    print(
        "[config] target attention backend: "
        f"{TARGET_ATTENTION_BACKEND}, target mm attention backend: "
        f"{TARGET_MM_ATTENTION_BACKEND}, enable_aiter_env={ENABLE_AITER_ENV}"
    )
    print(
        "[config] target server config: "
        f"tp={TARGET_TP_SIZE}, max_running={TARGET_MAX_RUNNING_REQUESTS}, "
        f"cuda_graph_max_bs_decode={TARGET_CUDA_GRAPH_MAX_BS_DECODE}, "
        f"mem_fraction={TARGET_MEM_FRACTION_STATIC}, page_size={TARGET_PAGE_SIZE}, "
        f"linear_attn={TARGET_LINEAR_ATTN_BACKEND}, "
        f"mamba_ssm_dtype={TARGET_MAMBA_SSM_DTYPE}"
    )
    if TARGET_EXTRA_SERVER_ARGS:
        print("[config] target extra server args: " + " ".join(TARGET_EXTRA_SERVER_ARGS))
    for path in (ACCURACY_OUTPUT_PATH, ACCURACY_CSV_OUTPUT_PATH):
        if path.exists():
            path.unlink()

    h = load_harness()

    h.BASE_SHARED_SERVER_CONFIG = replace(
        h.BASE_SHARED_SERVER_CONFIG,
        tp_size=TARGET_TP_SIZE,
        attention_backend=TARGET_ATTENTION_BACKEND,
        dtype="auto",
        max_running_requests=TARGET_MAX_RUNNING_REQUESTS,
        cuda_graph_max_bs=TARGET_CUDA_GRAPH_MAX_BS_DECODE,
        mem_fraction_static=TARGET_MEM_FRACTION_STATIC,
        page_size=TARGET_PAGE_SIZE,
        mamba_scheduler_strategy="extra_buffer",
        mamba_ssm_dtype=TARGET_MAMBA_SSM_DTYPE,
        linear_attn_backend=TARGET_LINEAR_ATTN_BACKEND,
        enable_piecewise_cuda_graph=False,
        enable_flashinfer_allreduce_fusion=False,
    )

    original_shared_to_args = h.SharedServerConfig.to_args

    def shared_to_args_rocm_fp8kv(self):
        args = list(original_shared_to_args(self))
        args.extend(
            [
                "--mm-attention-backend",
                TARGET_MM_ATTENTION_BACKEND,
                "--kv-cache-dtype",
                "fp8_e4m3",
            ]
        )
        args.extend(TARGET_EXTRA_SERVER_ARGS)
        return args

    h.SharedServerConfig.to_args = shared_to_args_rocm_fp8kv

    @dataclass(frozen=True)
    class DFlashConfigWithConfigFile(h.DFlashConfig):
        draft_config_file: str | None = None

        def to_args(self) -> list[str]:
            args = list(super().to_args())
            if self.draft_config_file:
                args.extend(
                    ["--decrypted-draft-config-file", self.draft_config_file]
                )
            return args

    def build_deployments_rocm(shared_config, sweep):
        deployments = []
        if sweep.include_baseline:
            deployments.append(
                h.ServerDeployment(
                    shared_config=shared_config,
                    mode_config=h.BaselineConfig(),
                )
            )

        for spec_mode in sweep.spec_modes:
            if spec_mode == "mtp":
                for mtp_num_steps in sweep.mtp_num_steps:
                    deployments.append(
                        h.ServerDeployment(
                            shared_config=shared_config,
                            mode_config=h.MTPConfig(num_steps=int(mtp_num_steps)),
                        )
                    )
            elif spec_mode == "dflash":
                if sweep.dflash_draft_model is None:
                    raise RuntimeError("DFlash deployment requires a draft model.")
                for block_size in sweep.dflash_block_sizes:
                    draft_model = sweep.dflash_draft_model
                    draft_config_file = None
                    if block_size is not None:
                        draft_model = DFLASH_DRAFT_MODEL_MAP.get(
                            int(block_size), draft_model
                        )
                        draft_config_file = DFLASH_DRAFT_CONFIG_FILE_MAP.get(
                            int(block_size)
                        )
                    deployments.append(
                        h.ServerDeployment(
                            shared_config=shared_config,
                            mode_config=DFlashConfigWithConfigFile(
                                draft_model=draft_model,
                                block_size=block_size,
                                draft_attention_backend="triton",
                                draft_config_file=draft_config_file,
                            ),
                        )
                    )
            else:
                raise ValueError(f"Unknown speculative mode: {spec_mode}")
        return deployments

    h._build_deployments = build_deployments_rocm
    h._run_benchmark_job = lambda job: _run_benchmark_job_with_accuracy(h, job)

    try:
        h.main()
    finally:
        if ACCURACY_SUMMARY:
            _write_accuracy_summary_csv()


if __name__ == "__main__":
    main()
