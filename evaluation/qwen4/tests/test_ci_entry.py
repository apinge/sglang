import json

import pytest
from ci_entry import eligibility, main


def event():
    return {
        "pull_request": {
            "draft": False,
            "state": "open",
            "base": {"ref": "qwen4_v0.5.21"},
            "labels": [{"name": "run-qwen4-ci"}],
        }
    }


def test_only_maintainer_labeled_pr_can_use_gpu():
    assert eligibility(
        "pull_request_target", event(), "run-qwen4-ci", "apinge/sglang", "write"
    )[0]
    assert not eligibility(
        "pull_request_target", event(), "run-qwen4-ci", "apinge/sglang", "read"
    )[0]
    missing = event()
    missing["pull_request"]["labels"] = []
    assert not eligibility(
        "pull_request_target", missing, "run-qwen4-ci", "apinge/sglang", "write"
    )[0]
    draft = event()
    draft["pull_request"]["draft"] = True
    assert not eligibility(
        "pull_request_target", draft, "run-qwen4-ci", "apinge/sglang", "admin"
    )[0]


def test_schedule_and_manual_dispatch_are_supported():
    for trigger in ["schedule", "workflow_dispatch"]:
        assert eligibility(trigger, {}, "run-qwen4-ci", "apinge/sglang", None)[0]
    assert not eligibility(
        "pull_request", event(), "run-qwen4-ci", "apinge/sglang", "admin"
    )[0]


@pytest.mark.parametrize(
    "profile,expected", [("full", 200), ("smoke", 2), ("profile", 0)]
)
def test_dispatch_uses_profile_defaults_when_example_override_is_blank(
    tmp_path, monkeypatch, profile, expected
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "event.json").write_text("{}")
    environment = {
        "GITHUB_EVENT_PATH": str(tmp_path / "event.json"),
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REPOSITORY": "apinge/sglang",
        "GITHUB_SHA": "a" * 40,
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
        "GITHUB_RUN_ID": "1",
        "GITHUB_RUN_ATTEMPT": "1",
        "HARNESS_SHA": "f" * 40,
        "INPUT_PROFILE": profile,
        "INPUT_EXAMPLES": "",
        "INPUT_PR_NUMBER": "",
        "INPUT_CASES": "" if profile == "full" else '["q35-35b-tp4-base"]',
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    sources = {
        name: {"repo": "apinge/sglang", "ref": "test", "sha": "a" * 40}
        for name in ["sglang", "aiter", "pyhip", "client"]
    }
    monkeypatch.setattr("ci_entry.resolve_sources", lambda *args: sources)
    monkeypatch.setattr("ci_status.publish", lambda *args: None)
    main()
    plan = json.loads((tmp_path / "qwen4-plan/manifest.json").read_text())
    assert plan["workload"]["examples"] == expected
