"""Attach the trusted report to the tested PR commit, not the base workflow SHA."""

import argparse
import json
import os
import subprocess

from common import REPO, SHA, read_json


def status_request(plan, state, run_url):
    target = plan.get("ci_status_target")
    if not target:
        return None
    if not REPO.fullmatch(target["repo"]) or not SHA.fullmatch(target["sha"]):
        raise ValueError("Invalid commit status target")
    if state not in {"pending", "success", "failure", "error"}:
        raise ValueError("Invalid commit status")
    description = "Build and evaluation in progress"
    if state == "success":
        description = (
            "Execution complete; numerical thresholds advisory"
            if plan["gate_mode"] == "observe"
            else "Execution and regression gates passed"
        )
    elif state in {"failure", "error"}:
        description = "Build, evaluation, or required result checks failed"
    return (
        f"repos/{target['repo']}/statuses/{target['sha']}",
        {
            "state": state,
            "context": "Qwen4 CI",
            "description": description,
            "target_url": run_url,
        },
    )


def publish(plan, state):
    url = f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    request = status_request(plan, state, url)
    if request:
        path, payload = request
        subprocess.run(
            ["gh", "api", "--method", "POST", path, "--input", "-"],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--state", required=True)
    args = parser.parse_args()
    publish(read_json(args.plan), args.state)
