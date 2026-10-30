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
    zip(ci.CASES, (191, 191, 195, 194, 193, 194)),
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
