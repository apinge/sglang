"""Regression coverage for the accuracy gate and owned-process cleanup."""

import argparse
import json
import subprocess
import sys

import pytest
import qwen4_ci as ci


def accuracy_result(tmp_path, score=0.97, examples=200):
    path = tmp_path / "gsm8k.json"
    path.write_text(
        json.dumps(
            dict(
                score=score,
                requested=examples,
                scored=examples,
                completed=examples,
                errors=0,
            )
        )
    )
    path.with_suffix(".html").write_text("<html>results</html>")
    return path


@pytest.mark.parametrize(
    "case,minimum_count",
    zip(ci.CASES, (191, 192, 191, 189, 189, 193, 192)),
    ids=[case[0] for case in ci.CASES],
)
def test_accuracy_gate_preserves_each_baseline_and_rejects_the_next_lower_count(
    case, minimum_count
):
    baseline = case[5]
    ci.check_accuracy_gate(minimum_count / 200, baseline)
    with pytest.raises(ValueError, match="regression"):
        ci.check_accuracy_gate((minimum_count - 1) / 200, baseline)
    ci.check_accuracy_gate(1, baseline)


@pytest.mark.parametrize(
    "key,value",
    [
        ("requested", 199),
        ("scored", 199),
        ("completed", 199),
        ("errors", 1),
        ("score", float("nan")),
        ("score", float("inf")),
        ("score", True),
        ("score", -0.01),
        ("score", 1.01),
        ("score", 0.971),
        ("errors", False),
    ],
)
def test_incomplete_or_invalid_accuracy_cannot_pass(tmp_path, key, value):
    path = accuracy_result(tmp_path)
    data = json.loads(path.read_text())
    data[key] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        ci.validate_accuracy(path)


def test_smoke_accuracy_is_diagnostic_but_must_be_complete(tmp_path):
    path = accuracy_result(tmp_path, score=0, examples=2)
    ci.validate_accuracy(path, smoke=True)
    ci.check_accuracy_gate(0, 0.97, smoke=True)
    with pytest.raises(ValueError, match="200"):
        ci.validate_accuracy(path)


def test_unlimited_benchmark_metadata_does_not_break_summary_json(tmp_path):
    data = dict(
        completed=2,
        errors=["", ""],
        output_lens=[500, 500],
        mean_ttft_ms=100,
        mean_tpot_ms=2,
        output_throughput=100,
        request_throughput=0.2,
        accept_length=4,
        request_rate=float("inf"),
    )
    path = tmp_path / "performance.jsonl"
    path.write_text(json.dumps(data) + "\n")
    metrics = ci.validate_performance(path, 2, 500, speculative=True)
    ci.write_json(tmp_path / "summary.json", metrics)
    assert metrics["output_throughput"] == 100 and "request_rate" not in metrics


def test_fresh_results_required_and_unknown_cases_rejected(tmp_path):
    for argv, message in [
        (["--case", "unknown"], "Unknown cases"),
        ([], "fresh output"),
    ]:
        (tmp_path / "old.json").write_text("{}")
        result = subprocess.run(
            [
                sys.executable,
                str(ci.ROOT / "scripts/ci/qwen4_ci.py"),
                "--output",
                str(tmp_path),
                *argv,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2 and message in result.stderr


def test_accuracy_regression_fails_the_run_and_reports_remaining_cases(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        ci,
        "command",
        lambda argv, **kwargs: ci.CLIENT_SHA if argv[-1] == "HEAD" else "",
    )
    monkeypatch.setattr(ci, "DATASETS", {})
    visited = []

    def case_run(case, args, directory, row):
        visited.append(case[0])
        row["accuracy"] = ci.validate_accuracy(accuracy_result(directory, score=0.9))
        ci.check_accuracy_gate(row["accuracy"]["score"], case[5])

    monkeypatch.setattr(ci, "run_case", case_run)
    args = argparse.Namespace(
        case=",".join(case[0] for case in ci.CASES[:2]),
        output=tmp_path / "results",
        client_root=tmp_path,
        models_root=tmp_path,
        datasets_root=None,
        accuracy_only=True,
        lock=tmp_path / "gpu.lock",
        smoke=False,
    )
    assert ci.run(args) == 1
    assert visited == [case[0] for case in ci.CASES[:2]]
    report = json.loads((args.output / "summary.json").read_text())
    assert report["failed"] and all(
        row["status"] == "failed" for row in report["cases"]
    )
    assert "90.00%" in (args.output / "summary.md").read_text()


@pytest.mark.parametrize(
    "execution_attempt,conclusion,has_report",
    [
        (2, "success", True),
        (6, "success", True),
        (6, "failure", True),
        (6, "failure", False),
    ],
)
def test_summary_uses_actual_execution_and_keeps_latest_conclusion(
    tmp_path, monkeypatch, execution_attempt, conclusion, has_report
):
    case, baseline = ci.CASES[0][0], ci.CASES[0][5]
    original = {
        "name": f"Qwen4 ({case})",
        "run_attempt": 2,
        "started_at": "2026-11-04T07:26:19Z",
        "conclusion": "success",
    }
    execution = dict(original, run_attempt=execution_attempt, conclusion=conclusion)
    if execution_attempt == 6:
        execution["started_at"] = "2026-11-05T11:57:29Z"
    copied = dict(execution, run_attempt=7)
    # Return history out of order and across pages, as the API may do.
    pages = [{"jobs": [copied]}, {"jobs": [execution, original]}]
    monkeypatch.setattr(ci, "command", lambda *args, **kwargs: json.dumps(pages))
    for key, value in {
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_REPOSITORY": "apinge/sglang",
        "GITHUB_RUN_ID": "41785211796",
        "GITHUB_STEP_SUMMARY": str(tmp_path / "step-summary.md"),
        "CASE_MATRIX": json.dumps({"case": [case]}),
        "VALIDATION_RESULT": "success",
        "CASES_RESULT": "success",
    }.items():
        monkeypatch.setenv(key, value)
    for attempt, score in [(2, 0.98)] + (
        [(6, 0.99)] if execution_attempt == 6 and has_report else []
    ):
        directory = (
            tmp_path
            / "qwen4-case-results"
            / f"qwen4-results-{case}-{attempt}"
            / "evaluation"
        )
        directory.mkdir(parents=True)
        row = dict(
            case=case,
            baseline=baseline,
            status="passed",
            errors=[],
            accuracy={"score": score},
        )
        ci.summary(directory, [row], argparse.Namespace(smoke=False), [])
    assert ci.summarize_ci() == (conclusion != "success")
    row = json.loads((tmp_path / "qwen4-summary/summary.json").read_text())["cases"][0]
    assert row["status"] == ("passed" if conclusion == "success" else "failed")
    if has_report:
        assert row["accuracy"]["score"] == (0.98 if execution_attempt == 2 else 0.99)
    else:
        assert "accuracy" not in row
        assert "qwen4-results-" + case + "-6" in row["errors"][0]


def test_gpu_selection_skips_busy_devices_and_maps_to_host_indices(monkeypatch):
    seen = {}

    def run(argv, env, **kwargs):
        seen["visible"] = env["HIP_VISIBLE_DEVICES"]
        return subprocess.CompletedProcess(argv, 0, stdout="[1, 3]\n", stderr="")

    monkeypatch.setattr(ci.subprocess, "run", run)
    assert ci.select_gpus({}, ["0", "2", "5", "7"], 2) == ["2", "7"]
    assert seen["visible"] == "0,2,5,7"
    assert ci.select_gpus({}, ["0", "2", "5", "7"], 1) == ["2"]
    with pytest.raises(RuntimeError, match="Need 4 idle GPUs.*idle: 2,7"):
        ci.select_gpus({}, ["0", "2", "5", "7"], 4)


def write_checksums(directory, rows):
    directory.mkdir()
    for rank, lines in rows.items():
        (directory / f"target_rank{rank}.tsv").write_text(
            "".join("\t".join(line) + "\n" for line in lines)
        )


def test_rfork_weight_checksums_must_match_on_every_rank(tmp_path):
    seed = [
        ("param", "w", "(2, 2)", "torch.bfloat16", "cuda", "8:10:30"),
        ("buffer", "b", "(2,)", "torch.float32", "cuda", "8:4:9"),
    ]
    write_checksums(tmp_path / "seed", {0: seed, 1: seed})
    # The device column is not compared.
    same = [line[:4] + ("cpu",) + line[5:] for line in seed]
    write_checksums(tmp_path / "same", {0: same, 1: same})
    assert ci.compare_weight_checksums(tmp_path / "seed", tmp_path / "same", 2) == 2
    changed = [seed[0][:5] + ("8:10:31",), seed[1]]
    write_checksums(tmp_path / "changed", {0: seed, 1: changed})
    with pytest.raises(ValueError, match="rank 1: 1/2 differ"):
        ci.compare_weight_checksums(tmp_path / "seed", tmp_path / "changed", 2)
    write_checksums(tmp_path / "missing", {0: seed, 1: seed[:1]})
    with pytest.raises(ValueError, match="rank 1: 1/2 differ"):
        ci.compare_weight_checksums(tmp_path / "seed", tmp_path / "missing", 2)


def test_rfork_case_needs_a_seed_and_a_fork_worth_of_gpus(tmp_path):
    case = next(case for case in ci.CASES if case[3] == "rfork")
    (tmp_path / case[1]).mkdir()
    (tmp_path / case[1] / "config.json").write_text("{}")
    args = argparse.Namespace(models_root=tmp_path, gpus="0,1,2", port=7080)
    with pytest.raises(ValueError, match=f"requires {2 * case[2]} GPUs"):
        ci.run_case(case, args, tmp_path, {"errors": []})
