from ci_status import status_request


def test_status_uses_tested_commit_and_marks_advisory_mode():
    plan = {
        "gate_mode": "observe",
        "ci_status_target": {"repo": "apinge/sglang", "sha": "a" * 40},
    }
    path, payload = status_request(
        plan, "success", "https://github.com/apinge/sglang/actions/runs/1"
    )
    assert path.endswith("/statuses/" + "a" * 40)
    assert payload["state"] == "success"
    assert "advisory" in payload["description"]
    assert payload["context"] == "Qwen4 CI"


def test_integration_run_does_not_post_a_pr_status():
    assert (
        status_request({"gate_mode": "observe"}, "success", "https://github.com")
        is None
    )
