"""Reconcile artifacts against the plan and render the native Actions summary."""

import argparse
import html
import json
from pathlib import Path

from common import read_json, write_json
from config import validate_plan
from results import summarize_case


def cell(value):
    return html.escape(str(value)).replace("|", "\\|").replace("\n", " ")


def phase_status(result, phase, mode):
    state = result.get("phase_status", {}).get(phase, result["execution"])
    if state != "completed":
        return state
    comparison = result.get("comparison", {})
    reason = "accuracy_regression" if phase == "accuracy" else "performance_regression"
    if reason in comparison.get("reasons", []):
        return "regression (advisory)" if mode == "observe" else "regression"
    if comparison.get("status") == "not_run":
        return "completed (diagnostic)"
    if comparison.get("status") == "regression":
        return "within_baseline"  # The other evaluation dimension regressed.
    return comparison.get("status", "completed")


def render(plan, results, runtime, artifact_url=None):
    failed = sum(r["failed"] for r in results) or bool(runtime.get("failure"))
    lines = [
        "## Qwen4 MI308X CI",
        "",
        f"**Execution: {'❌ Failed' if failed else '✅ Complete'}** · {len(results)} cases · profile `{plan['profile']}`",
        f"**Accuracy/performance thresholds: {'advisory' if plan['gate_mode'] == 'observe' else 'enforced'}**",
        "",
        f"Base image: `{cell(plan['base_image'])}`",
        f"Resolved digest: `{cell(runtime.get('base_image_digest', 'existing container / not rebuilt'))}`",
        "",
    ]
    for name, source in plan.get("sources", {}).items():
        lines.append(f"- {cell(name)}: `{cell(source['repo'])}@{cell(source['sha'])}`")
    if runtime.get("note"):
        lines += ["", cell(runtime["note"])]
    if plan["workload"]["repeats"] > 1:
        lines += [
            "",
            "Metrics are medians across repetitions; completion counts sum verified repetitions.",
        ]
    if artifact_url:
        if not artifact_url.startswith("https://github.com/"):
            raise ValueError("Artifact link must be a GitHub HTTPS URL")
        lines.extend(["", f"[Download complete logs and results]({artifact_url})"])
    lines += [
        "",
        "### GSM8K Accuracy",
        "",
        "| Case | TP | Score | Verified / expected | Truncated | Delta (pp) | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    cases = {c["id"]: c for g in plan["groups"] for c in g["cases"]}
    for result in results:
        case = cases[result["case_id"]]
        score = result["metrics"].get("score")
        delta = result.get("comparison", {}).get("deltas", {}).get("score")
        counts = result.get("counts", {}).get("accuracy", {})
        scored = counts.get("scored")
        expected = plan["workload"]["examples"] * plan["workload"]["repeats"]
        truncated = counts.get("truncated")
        status = phase_status(result, "accuracy", plan["gate_mode"])
        lines.append(
            f"| {cell(case['id'])} | {case['tp']} | {f'{score:.4f}' if score is not None else '—'} | {scored if scored is not None else '—'} / {expected} | {truncated if truncated is not None else '—'} | {f'{delta:+.2f}' if delta is not None else '—'} | {cell(status)} |"
        )
    lines += [
        "",
        "### Serving Performance",
        "",
        "| Case | ISL / OSL | Verified / expected | Mean TTFT (ms; Δ%) | Mean TPOT (ms; Δ%) | Output tokens/s (Δ%) | Status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for result in results:
        case = cases[result["case_id"]]
        values = []
        completed = result.get("counts", {}).get("performance", {}).get("completed")
        expected = plan["workload"]["prompts"] * plan["workload"]["repeats"]
        for name in ["mean_ttft_ms", "mean_tpot_ms", "output_throughput"]:
            value = result["metrics"].get(name)
            delta = result.get("comparison", {}).get("deltas", {}).get(name)
            values.append(
                "—"
                if value is None
                else f"{value:.2f}" + (f" ({delta:+.2f}%)" if delta is not None else "")
            )
        lines.append(
            f"| {cell(case['id'])} | {case['input_tokens']} / {plan['workload'].get('output_tokens', case['output_tokens'])} | {completed if completed is not None else '—'} / {expected} | {' | '.join(values)} | {cell(phase_status(result, 'performance', plan['gate_mode']))} |"
        )
    for group in plan["groups"]:
        lines += [
            "",
            f"<details><summary>{cell(group['id'])}: container and case details</summary>",
            "",
            "Model mount: `/raid/models` → `/models` (read-only).",
            "",
        ]
        for result in results:
            if result["group_id"] != group["id"]:
                continue
            lines.append(
                f"**{cell(result['case_id'])}** — container `{cell(result.get('container_id', 'not started'))}`"
            )
            for reason in result.get("errors", []) + result.get("comparison", {}).get(
                "reasons", []
            ):
                lines.append(f"- {cell(reason)}")
            lines += ["", "```json", json.dumps(result, indent=2), "```", ""]
        lines.append("</details>")
    return "\n".join(lines) + "\n"


def aggregate(plan, output, baseline_path=None, artifact_url=None):
    validate_plan(plan, verify_environment=False)
    output = Path(output)
    issues = []
    job_failure = output / "github-job-failure.txt"
    if job_failure.exists():
        issues.append(
            f"GitHub evaluation job did not succeed: {job_failure.read_text().strip()}"
        )

    def load(path, fallback):
        try:
            data = read_json(path)
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object")
            return data
        except Exception as exc:
            issues.append(f"{Path(path).name}: {type(exc).__name__}: {exc}")
            return fallback

    baseline = load(baseline_path, {"cases": {}}) if baseline_path else {"cases": {}}
    runtime = load(output / "runtime.json", {})
    assets = load(output / "assets.json", {"identity": "missing"})
    results = []
    container_ids = set()
    for group in plan["groups"]:
        metadata = output / group["id"] / "container.json"
        group_runtime = {
            **runtime,
            **load(metadata, {}),
        }
        if runtime.get("backend") == "docker":
            cid = group_runtime.get("container_id")
            if not cid or cid in container_ids:
                issues.append(
                    f"{group['id']}: missing or reused target container identity"
                )
            container_ids.add(cid)
        for case in group["cases"]:
            directory = output / group["id"] / case["id"]
            try:
                result = summarize_case(
                    plan,
                    group,
                    case,
                    directory,
                    assets,
                    group_runtime,
                    baseline.get("cases", {}).get(case["id"]),
                )
            except Exception as exc:
                result = {
                    "case_id": case["id"],
                    "group_id": group["id"],
                    "execution": "not_run"
                    if not (directory / "execution.json").exists()
                    else "eval_error",
                    "failed": True,
                    "metrics": {},
                    "errors": [f"{type(exc).__name__}: {exc}"],
                    "container_id": group_runtime.get("container_id"),
                }
            write_json(directory / "result.json", result)
            results.append(result)
    summary = {
        "plan_hash": plan["plan_hash"],
        "profile": plan["profile"],
        "gate_mode": plan["gate_mode"],
        "failed": any(r["failed"] for r in results),
        "cases": results,
    }
    if issues:
        runtime["failure"] = "\n".join([runtime.get("failure", ""), *issues]).strip()
    if runtime.get("failure"):
        summary["failed"] = True
        summary["failure"] = runtime["failure"]
    write_json(output / "summary.json", summary)
    text = render(plan, results, runtime, artifact_url)
    if runtime.get("failure"):
        text += "\n**Infrastructure failure:** " + cell(runtime["failure"]) + "\n"
    (output / "summary.md").write_text(text)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--artifact-url")
    args = parser.parse_args()
    result = aggregate(
        read_json(args.plan), args.output, args.baseline, args.artifact_url
    )
    raise SystemExit(1 if result["failed"] else 0)
