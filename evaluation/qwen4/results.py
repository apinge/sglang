"""Execution validity is mandatory; numerical regression gates are opt-in."""

import math
import statistics
from pathlib import Path

from common import digest, file_hash, read_json


def number(value, name, minimum=0, maximum=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite numeric data")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside the allowed range")
    return value


def validate_performance(path, prompts, output_tokens, profiling=False):
    lines = [line for line in Path(path).read_text().splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("Expected exactly one fresh performance record")
    import json

    data = json.loads(lines[0])
    if data.get("completed") != prompts:
        raise ValueError(
            f"Only {data.get('completed')} of {prompts} benchmark requests completed"
        )
    errors = data.get("errors")
    lengths = data.get("output_lens")
    if not isinstance(errors, list) or len(errors) != prompts or any(errors):
        raise ValueError("Missing request details or nonempty benchmark errors")
    if (
        not isinstance(lengths, list)
        or len(lengths) != prompts
        or any(n != output_tokens for n in lengths)
    ):
        raise ValueError(
            "Benchmark output lengths do not match the requested fixed workload"
        )
    for key in ["mean_ttft_ms", "mean_tpot_ms", "output_throughput"]:
        number(
            data.get(key),
            key,
            minimum=0 if profiling and key == "mean_tpot_ms" else 1e-12,
        )
    return data


def validate_accuracy(path, examples, selection):
    data = read_json(path)
    for key in ["requested", "scored", "completed"]:
        if data.get(key) != examples:
            raise ValueError(f"GSM8K {key} must equal {examples}")
    if data.get("errors") != 0:
        raise ValueError("GSM8K has terminal request errors")
    number(data.get("score"), "GSM8K score", maximum=1)
    if data.get("selection") != selection:
        raise ValueError("GSM8K selected different dataset/examples")
    rows_path = Path(path).with_name("gsm8k-examples.jsonl")
    if file_hash(rows_path) != data.get("records_sha256"):
        raise ValueError("GSM8K raw examples are missing or changed")
    import json

    rows = [
        json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()
    ]
    if any(
        r.get("status") != "completed" or not isinstance(r.get("content"), str)
        for r in rows
    ):
        raise ValueError("GSM8K raw examples contain failed or malformed requests")
    report = Path(path).with_name("gsm8k.html")
    if not report.is_file() or report.stat().st_size == 0:
        raise ValueError("GSM8K HTML report is missing")
    ids = [r["example_id"] for r in rows]
    if (
        len(ids) != examples
        or len(set(ids)) != examples
        or set(ids) != set(selection["example_ids"])
    ):
        raise ValueError("GSM8K raw examples do not cover the expected selection")
    correct = sum(r["correct"] is True for r in rows)
    if not math.isclose(correct / examples, data["score"], abs_tol=1e-12):
        raise ValueError("GSM8K summary and per-example score disagree")
    return data


def comparison_key(plan, case, assets, runtime):
    return digest(
        {
            "case": case,
            "protocol": plan["protocol"],
            "workload": {k: v for k, v in plan["workload"].items() if k != "repeats"},
            "defaults": {
                k: plan["defaults"][k]
                for k in [
                    "seed",
                    "concurrency",
                    "gsm8k_shots",
                    "gsm8k_max_tokens",
                    "gsm8k_threads",
                ]
            },
            "assets": assets["identity"],
            "base_image_digest": runtime.get("base_image_digest"),
            "client_sha": plan.get("sources", {}).get("client", {}).get("sha"),
            "hardware": runtime.get("hardware"),
            "stack": runtime.get("stack"),
        }
    )


def compare(metrics, baseline, key, mode):
    if not baseline or baseline.get("comparison_key") != key:
        return {
            "status": "uncalibrated",
            "failed": mode == "enforce",
            "deltas": {},
            "reasons": ["No compatible baseline"],
        }
    deltas, breaches = {}, []
    baseline_metrics = baseline["metrics"]
    for metric, value in metrics.items():
        if metric not in baseline_metrics:
            continue
        reference = number(baseline_metrics[metric], f"baseline {metric}")
        number(value, metric)
        if metric == "score":
            floor = number(baseline["accuracy_floor"], "accuracy_floor", maximum=1)
            drop = number(baseline["accuracy_drop"], "accuracy_drop", maximum=1)
            deltas[metric] = (value - reference) * 100
            if value < floor or value < reference - drop:
                breaches.append("accuracy_regression")
        else:
            if reference <= 0:
                raise ValueError("Performance baseline must be positive")
            tolerance = number(
                baseline["performance_tolerance"], "performance_tolerance", maximum=1
            )
            delta = value / reference - 1
            deltas[metric] = delta * 100
            if (metric == "output_throughput" and delta < -tolerance) or (
                metric != "output_throughput" and delta > tolerance
            ):
                breaches.append("performance_regression")
    if set(metrics) - baseline_metrics.keys():
        return {
            "status": "uncalibrated",
            "failed": mode == "enforce",
            "deltas": deltas,
            "reasons": ["Baseline is missing metrics"],
        }
    return {
        "status": "regression" if breaches else "within_baseline",
        "failed": bool(breaches) and mode == "enforce",
        "deltas": deltas,
        "reasons": sorted(set(breaches)),
    }


def summarize_case(plan, group, case, directory, assets, runtime, baseline=None):
    directory = Path(directory)
    state = read_json(directory / "execution.json")
    if (
        state.get("plan_hash") != plan["plan_hash"]
        or state.get("case_id") != case["id"]
    ):
        raise ValueError("Execution identity does not match plan")
    if runtime.get("backend") == "docker":
        if (
            not runtime.get("container_id")
            or state.get("container_id") != runtime["container_id"]
        ):
            raise ValueError("Case did not execute in its recorded target container")
        if runtime.get("target") != group["directory"]:
            raise ValueError("Container target identity changed")
    summary = {
        "schema_version": 1,
        "case_id": case["id"],
        "group_id": group["id"],
        "plan_hash": plan["plan_hash"],
        "container_id": runtime.get("container_id"),
        "execution": state["status"],
        "errors": list(state.get("errors", [])),
        "metrics": {},
        "timings": state.get("repeats", []),
        "comparison_key": comparison_key(plan, case, assets, runtime),
    }
    samples = []
    launches = []
    phase_ok = {"performance": 0, "accuracy": 0}
    counts = {"performance": [], "accuracy": []}
    for repeat in range(1, plan["workload"]["repeats"] + 1):
        path = directory / f"repeat-{repeat}"
        try:
            launch = read_json(path / "resolved-case.json")
            if (
                launch.get("case_id") != case["id"]
                or launch.get("group_id") != group["id"]
            ):
                raise ValueError("Launch artifact has a different case/group identity")
            launches.append(launch["argv"])
            for required in ["server.log", "server-info.json", "smoke.json"]:
                if not (path / required).is_file():
                    raise ValueError(f"Missing launch evidence: {required}")
        except Exception as exc:
            summary["errors"].append(f"repeat {repeat}: {exc}")
        metrics = {}
        for phase in ["performance", "accuracy"]:
            if not plan["workload"][phase]:
                continue
            try:
                if phase == "performance":
                    data = validate_performance(
                        path / "performance.jsonl",
                        plan["workload"]["prompts"],
                        plan["workload"].get("output_tokens", case["output_tokens"]),
                        profiling=plan["workload"].get("profiler", False),
                    )
                    metrics.update(
                        {
                            k: data[k]
                            for k in [
                                "mean_ttft_ms",
                                "mean_tpot_ms",
                                "output_throughput",
                            ]
                        }
                    )
                    counts[phase].append({"completed": data["completed"]})
                else:
                    data = validate_accuracy(
                        path / "gsm8k.json",
                        plan["workload"]["examples"],
                        assets["gsm8k"],
                    )
                    metrics["score"] = data["score"]
                    counts[phase].append(
                        {
                            key: data.get(key)
                            for key in [
                                "scored",
                                "completed",
                                "empty_outputs",
                                "truncated",
                            ]
                        }
                    )
                execution = state.get("repeats", [])[repeat - 1]
                if execution.get(phase) != 0:
                    raise ValueError("Client did not exit successfully")
                phase_ok[phase] += 1
            except Exception as exc:
                summary["errors"].append(f"repeat {repeat} {phase}: {exc}")
        samples.append(metrics)
    summary["samples"] = samples
    summary["launches"] = launches
    names = {key for sample in samples for key in sample}
    summary["metrics"] = {
        key: statistics.median(sample[key] for sample in samples if key in sample)
        for key in names
    }
    summary["phase_status"] = {
        name: "not_requested"
        if not plan["workload"][name]
        else "completed"
        if count == plan["workload"]["repeats"]
        else "eval_error"
        for name, count in phase_ok.items()
    }
    summary["counts"] = {}
    for phase, rows in counts.items():
        summary["counts"][phase] = {
            "expected": plan["workload"]["repeats"]
            * plan["workload"]["prompts" if phase == "performance" else "examples"]
        }
        for key in {key for row in rows for key in row}:
            values = [row[key] for row in rows if type(row.get(key)) is int]
            summary["counts"][phase][key] = sum(values) if values else None
    complete = state["status"] == "completed" and not summary["errors"]
    if not complete:
        summary["execution"] = (
            state["status"] if state["status"] != "completed" else "eval_error"
        )
        summary["comparison"] = {
            "status": "not_run",
            "failed": False,
            "deltas": {},
            "reasons": ["Evaluation incomplete; available measurements are diagnostic"],
        }
    elif plan["workload"].get("profiler"):
        summary["comparison"] = {
            "status": "diagnostic",
            "failed": False,
            "deltas": {},
            "reasons": ["Profiler timings are excluded from regression gates"],
        }
    else:
        summary["comparison"] = compare(
            summary["metrics"], baseline, summary["comparison_key"], plan["gate_mode"]
        )
    summary["failed"] = not complete or summary["comparison"]["failed"]
    return summary
