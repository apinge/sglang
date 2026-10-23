from ci_artifacts import latest_result
from ci_comments import handle


class API:
    def __init__(self, permission="write", runs=None):
        self.permission = permission
        self.runs = runs or []
        self.posts = []

    def get(self, path):
        if path.endswith("/permission"):
            return {"permission": self.permission}
        if "/pulls/" in path:
            return {
                "state": "open",
                "draft": False,
                "base": {"ref": "qwen4_v0.5.21"},
                "head": {"sha": "a" * 40},
            }
        return {"workflow_runs": self.runs}

    def post(self, path, payload):
        self.posts.append((path, payload))


def comment(body):
    return {
        "comment": {"body": body, "user": {"login": "maintainer"}},
        "issue": {"number": 47, "pull_request": {"url": "unused"}},
    }


def failed_run(sha="a" * 40, status="completed", conclusion="failure"):
    return {
        "id": 123,
        "display_title": f"Qwen4 CI pr=47 sha={sha}",
        "status": status,
        "conclusion": conclusion,
        "html_url": "https://github.com/apinge/sglang/actions/runs/123",
    }


def test_maintainer_run_dispatches_explicitly_for_current_head():
    api = API()
    result = handle(comment("/ci run"), "apinge/sglang", api)
    assert result["action"] == "dispatched"
    assert len(api.posts) == 1
    assert api.posts[0][0].endswith("/qwen4-ci.yml/dispatches")
    assert api.posts[0][1]["inputs"]["candidate_sha"] == "a" * 40


def test_failed_retry_uses_github_job_retry_not_new_dispatch():
    api = API(runs=[failed_run()])
    result = handle(comment("/ci run_failed"), "apinge/sglang", api)
    assert result["action"] == "rerun_failed_jobs"
    assert api.posts == [("repos/apinge/sglang/actions/runs/123/rerun-failed-jobs", {})]


def test_unauthorized_and_old_head_commands_do_not_trigger_jobs():
    api = API(permission="read")
    assert handle(comment("/ci run"), "apinge/sglang", api)["action"] == "ignored"
    assert not api.posts
    api = API(runs=[failed_run("b" * 40)])
    assert (
        handle(comment("/ci run_failed"), "apinge/sglang", api)["action"] == "ignored"
    )
    assert not api.posts


def test_running_job_is_not_duplicated_and_extra_commands_are_rejected():
    api = API(runs=[failed_run(status="in_progress", conclusion=None)])
    assert (
        handle(comment("/ci run"), "apinge/sglang", api)["action"] == "already_running"
    )
    assert (
        handle(comment("/ci run; arbitrary command"), "apinge/sglang", api)["action"]
        == "ignored"
    )
    assert not api.posts


def test_report_retry_can_find_previous_successful_artifact():
    assert latest_result(["qwen4-plan", "qwen4-results-1"], 2) == "qwen4-results-1"
    assert (
        latest_result(["qwen4-results-1", "qwen4-results-2", "qwen4-results-3"], 2)
        == "qwen4-results-2"
    )
    assert latest_result(["qwen4-results-3", "other"], 2) == ""
