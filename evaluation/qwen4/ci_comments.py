"""Maintainer-only /ci commands. This controller never executes PR code."""

import json
import os
import re
import subprocess
from urllib.parse import quote

from common import read_json


class GitHub:
    def get(self, path):
        return json.loads(subprocess.check_output(["gh", "api", path], text=True))

    def post(self, path, payload):
        result = subprocess.run(
            ["gh", "api", "--method", "POST", path, "--input", "-"],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=True,
        )
        return json.loads(result.stdout) if result.stdout.strip() else None


def handle(event, repository, api):
    body = event.get("comment", {}).get("body", "").strip()
    match = re.fullmatch(r"/ci\s+(run|run_failed)", body)
    issue = event.get("issue", {})
    if not match or "pull_request" not in issue:
        return {"action": "ignored", "reason": "Not a supported PR command"}
    actor = event["comment"]["user"]["login"]
    permission = api.get(
        f"repos/{repository}/collaborators/{quote(actor, safe='')}/permission"
    )["permission"]
    if permission not in {"write", "maintain", "admin"}:
        return {"action": "ignored", "reason": "Maintainer permission is required"}
    number = issue["number"]
    pr = api.get(f"repos/{repository}/pulls/{number}")
    branch = (
        "qwen38-flash-next-dev" if repository.endswith("/aiter") else "qwen4_v0.5.21"
    )
    workflow = (
        "qwen4-downstream.yml" if repository.endswith("/aiter") else "qwen4-ci.yml"
    )
    if pr["state"] != "open" or pr.get("draft") or pr["base"]["ref"] != branch:
        return {
            "action": "ignored",
            "reason": "PR must be open, ready, and target the migration branch",
        }
    sha = pr["head"]["sha"]
    runs = api.get(
        f"repos/{repository}/actions/workflows/{workflow}/runs?per_page=100"
    )["workflow_runs"]
    marker = f"pr={number} sha={sha}"
    matching = [run for run in runs if run.get("display_title") == f"Qwen4 CI {marker}"]
    latest = matching[0] if matching else None
    if latest and latest["status"] != "completed":
        return {"action": "already_running", "url": latest["html_url"]}
    if match[1] == "run_failed":
        if not latest or latest["conclusion"] not in {"failure", "timed_out"}:
            return {
                "action": "ignored",
                "reason": "No failed Qwen4 run for the current PR head; use /ci run",
            }
        api.post(
            f"repos/{repository}/actions/runs/{latest['id']}/rerun-failed-jobs", {}
        )
        return {"action": "rerun_failed_jobs", "url": latest["html_url"], "sha": sha}
    # This is a one-off dispatch for the current head. Persistent automatic
    # testing of later pushes is controlled separately by the maintainer label.
    api.post(
        f"repos/{repository}/actions/workflows/{workflow}/dispatches",
        {
            "ref": branch,
            "inputs": {
                "pr_number": str(number),
                "candidate_sha": sha,
                "profile": "full",
            },
        },
    )
    return {
        "action": "dispatched",
        "pr": number,
        "sha": sha,
        "url": f"https://github.com/{repository}/actions/workflows/{workflow}",
    }


if __name__ == "__main__":
    result = handle(
        read_json(os.environ["GITHUB_EVENT_PATH"]),
        os.environ["GITHUB_REPOSITORY"],
        GitHub(),
    )
    print(json.dumps(result, indent=2))
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
        stream.write(
            "### Qwen4 CI command\n\n```json\n"
            + json.dumps(result, indent=2)
            + "\n```\n"
        )
