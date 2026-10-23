import json

import pytest
from common import digest, file_hash, write_json
from config import make_plan, resolve_sources, server_command, validate_plan
from report import aggregate
from results import compare, validate_performance

from assets import gsm8k_selection


def test_six_cases_three_targets_and_off_on_recipe():
    plan = make_plan()
    assert [len(g["cases"]) for g in plan["groups"]] == [2, 2, 2]
    assert [c["tp"] for g in plan["groups"] for c in g["cases"]] == [4, 1, 8, 8, 2, 2]
    group = plan["groups"][-1]
    base, mtp = group["cases"]
    assert server_command(base, group) == server_command(mtp, group)
    assert base["launch_env"]["MTP"] == "0"
    assert mtp["launch_env"]["MTP"] == "1"
    assert plan["workload"]["examples"] == 200


@pytest.mark.parametrize(
    "selection", [[], "q35-35b-tp4-base", ["unknown"], ["q35-35b-tp4-base"] * 2]
)
def test_bad_selections_fail_before_execution(selection):
    with pytest.raises(ValueError):
        make_plan(selection=selection)


def test_plan_changes_cannot_silently_change_recipes():
    plan = make_plan()
    validate_plan(plan)
    plan["groups"][0]["cases"][0]["tp"] = 1
    with pytest.raises(ValueError):
        validate_plan(plan)
    plan["plan_hash"] = digest({k: v for k, v in plan.items() if k != "plan_hash"})
    with pytest.raises(ValueError):
        validate_plan(plan)


def test_latest_companion_resolves_once_and_keeps_aiter_pr(monkeypatch):
    calls = []

    def resolve(repo, ref):
        calls.append((repo, ref))
        return {"repo": repo, "ref": ref, "sha": "a" * 40}

    monkeypatch.setattr("config.resolve_commit", resolve)
    sources = resolve_sources("sglang", "apinge/sglang", "b" * 40)
    assert sources["aiter"]["ref"] == "qwen38-flash-next-dev"
    plan = make_plan(sources=sources)
    validate_plan(plan)
    assert len(calls) == 4  # Validation and six cases do not resolve again.
    sources = resolve_sources("aiter", "apinge/aiter", "c" * 40)
    assert sources["aiter"]["ref"] == "c" * 40
    assert sources["sglang"]["ref"] == "qwen4_v0.5.21"


def dataset(tmp_path, count=205):
    path = tmp_path / "gsm8k.jsonl"
    path.write_text(
        "".join(
            json.dumps({"question": f"Q{i}", "answer": f"#### {i}"}) + "\n"
            for i in range(count)
        )
    )
    return path


def test_200_examples_exclude_five_prompt_examples(tmp_path):
    selected = gsm8k_selection(dataset(tmp_path))
    assert len(selected["example_ids"]) == 200
    assert not set(selected["prompt_ids"]) & set(selected["example_ids"])
    with pytest.raises(ValueError, match="205"):
        gsm8k_selection(dataset(tmp_path, 200))


def test_incomplete_performance_cannot_pass(tmp_path):
    path = tmp_path / "perf.jsonl"
    record = {
        "completed": 2,
        "errors": ["", ""],
        "output_lens": [500, 500],
        "mean_ttft_ms": 10,
        "mean_tpot_ms": 1,
        "output_throughput": 100,
    }
    path.write_text(json.dumps(record) + "\n")
    validate_performance(path, 2, 500)
    for changes in [
        {"completed": 1},
        {"errors": ["", "HTTP 500"]},
        {"output_lens": [500, 499]},
        {"mean_ttft_ms": float("nan")},
    ]:
        path.write_text(json.dumps(record | changes) + "\n")
        with pytest.raises(ValueError):
            validate_performance(path, 2, 500)


def test_advisory_does_not_disable_completeness(tmp_path):
    plan = make_plan()
    output = tmp_path / "results"
    result = aggregate(plan, output)
    assert result["failed"]
    assert len(result["cases"]) == 6
    assert all(case["failed"] for case in result["cases"])
    assert "thresholds: advisory" in (output / "summary.md").read_text()


def test_numerical_regression_advisory_then_enforced():
    baseline = {
        "comparison_key": "key",
        "metrics": {"score": 0.9, "mean_tpot_ms": 10},
        "accuracy_floor": 0.8,
        "accuracy_drop": 0.01,
        "performance_tolerance": 0.05,
    }
    metrics = {"score": 0.5, "mean_tpot_ms": 20}
    assert not compare(metrics, baseline, "key", "observe")["failed"]
    assert compare(metrics, baseline, "key", "enforce")["failed"]
    assert not compare(metrics, None, "key", "observe")["failed"]
    assert compare(metrics, None, "key", "enforce")["failed"]


def test_corrupt_runtime_metadata_still_produces_complete_failure_report(tmp_path):
    (tmp_path / "runtime.json").write_text("not JSON")
    plan = make_plan()
    result = aggregate(plan, tmp_path)
    assert result["failed"] and len(result["cases"]) == 6
    assert "runtime.json" in result["failure"]
    assert "Execution: ❌ Failed" in (tmp_path / "summary.md").read_text()


def test_failed_performance_preserves_valid_accuracy_measurement(tmp_path):
    plan = make_plan(profile="smoke", selection=["q35-35b-tp4-base"])
    group, case = plan["groups"][0], plan["groups"][0]["cases"][0]
    selected = gsm8k_selection(dataset(tmp_path, 7), 2)
    write_json(tmp_path / "runtime.json", {"backend": "existing"})
    write_json(tmp_path / "assets.json", {"identity": "test", "gsm8k": selected})
    write_json(
        tmp_path / group["id"] / "container.json", {"container_id": "existing:test"}
    )
    directory = tmp_path / group["id"] / case["id"]
    write_json(
        directory / "execution.json",
        {
            "plan_hash": plan["plan_hash"],
            "case_id": case["id"],
            "status": "eval_error",
            "errors": ["performance exited 1"],
            "repeats": [{"performance": 1, "accuracy": 0}],
        },
    )
    repeat = directory / "repeat-1"
    write_json(
        repeat / "resolved-case.json",
        {"case_id": case["id"], "group_id": group["id"], "argv": ["bash", "launch.sh"]},
    )
    for name in ["server-info.json", "smoke.json"]:
        write_json(repeat / name, {})
    (repeat / "server.log").write_text("fixture")
    (repeat / "gsm8k.html").write_text("<html>fixture</html>")
    records = [
        {
            "example_id": identifier,
            "status": "completed",
            "content": "5",
            "correct": True,
        }
        for identifier in selected["example_ids"]
    ]
    raw = repeat / "gsm8k-examples.jsonl"
    raw.write_text("".join(json.dumps(record) + "\n" for record in records))
    write_json(
        repeat / "gsm8k.json",
        {
            "score": 1.0,
            "requested": 2,
            "scored": 2,
            "completed": 2,
            "errors": 0,
            "empty_outputs": 0,
            "selection": selected,
            "records_sha256": file_hash(raw),
        },
    )
    summary = aggregate(plan, tmp_path)
    assert summary["failed"]
    assert summary["cases"][0]["metrics"]["score"] == 1.0
    assert summary["cases"][0]["phase_status"]["accuracy"] == "completed"
    assert "1.0000" in (tmp_path / "summary.md").read_text()
    assert "completed (diagnostic)" in (tmp_path / "summary.md").read_text()
