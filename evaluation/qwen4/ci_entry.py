"""Trusted workflow planning/authorization, executed before any candidate code."""

import json
import os
from pathlib import Path

from common import SHA, command_output, digest, read_json, write_json
from config import make_plan, resolve_sources, validate_plan


def eligibility(event_name, event, label, repository, permission):
    if event_name in {"schedule", "workflow_dispatch"}:
        return True, "scheduled/manual run"
    if event_name != "pull_request_target":
        return False, "unsupported trigger"
    pr = event["pull_request"]
    if pr.get("draft") or pr.get("state") != "open":
        return False, "PR is draft or closed"
    expected = (
        "qwen38-flash-next-dev" if repository.endswith("/aiter") else "qwen4_v0.5.21"
    )
    if pr["base"]["ref"] != expected:
        return False, "different target branch"
    if label not in {item["name"] for item in pr.get("labels", [])}:
        return False, "maintainer label is absent"
    if permission not in {"admin", "maintain", "write"}:
        return False, "CI label must be authorized by a maintainer"
    return True, "maintainer-authorized labeled PR"


def main():
    if not SHA.fullmatch(os.environ["HARNESS_SHA"]):
        raise ValueError("HARNESS_SHA must be an immutable full commit SHA")
    event = read_json(os.environ["GITHUB_EVENT_PATH"])
    repository = os.environ["GITHUB_REPOSITORY"]
    event_name = os.environ["GITHUB_EVENT_NAME"]
    label = os.environ.get("QWEN4_CI_LABEL", "run-qwen4-ci")
    permission = None
    if event_name == "pull_request_target":
        # A maintainer's label remains valid for subsequent contributor pushes.
        # Check the person who granted the label, not the author of each push.
        if label in {item["name"] for item in event["pull_request"].get("labels", [])}:
            actor = os.environ["GITHUB_ACTOR"]
            if (
                event.get("action") != "labeled"
                or event.get("label", {}).get("name") != label
            ):
                authors = command_output(
                    [
                        "gh",
                        "api",
                        "--paginate",
                        f"repos/{repository}/issues/{event['number']}/events?per_page=100",
                        "--jq",
                        f'.[] | select(.event == "labeled" and .label.name == {json.dumps(label)}) | [.created_at, (.id|tostring), .actor.login] | @tsv',
                    ]
                ).splitlines()
                grants = [line.split("\t", 2) for line in authors]
                actor = (
                    max(grants, key=lambda row: (row[0], int(row[1])))[2]
                    if grants
                    else ""
                )
            if actor:
                permission = json.loads(
                    command_output(
                        [
                            "gh",
                            "api",
                            f"repos/{repository}/collaborators/{actor}/permission",
                        ]
                    )
                )["permission"]
    enabled, reason = eligibility(event_name, event, label, repository, permission)
    with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
        stream.write(f"enabled={'true' if enabled else 'false'}\n")
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
        stream.write(f"Qwen4 GPU evaluation: {reason}.\n")
    if not enabled:
        return
    existing_plan = Path("qwen4-plan/manifest.json")
    if existing_plan.is_file():
        saved_plan = read_json(existing_plan)
        validate_plan(saved_plan)
        from ci_status import publish

        publish(saved_plan, "pending")
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            stream.write("reused=true\n")
        print("Reusing the original run plan; source heads were not resolved again")
        return
    origin = "aiter" if repository.endswith("/aiter") else "sglang"
    candidate_repo, candidate_ref = repository, os.environ["GITHUB_SHA"]
    if event_name == "pull_request_target":
        candidate_repo = event["pull_request"]["head"]["repo"]["full_name"]
        candidate_ref = event["pull_request"]["head"]["sha"]
    elif event_name == "schedule":
        origin = "integration"
    elif os.environ.get("INPUT_PR_NUMBER"):
        number = os.environ["INPUT_PR_NUMBER"]
        if not number.isdigit() or int(number) <= 0:
            raise ValueError("Invalid PR number")
        pr = json.loads(
            command_output(["gh", "api", f"repos/{repository}/pulls/{number}"])
        )
        expected_branch = (
            "qwen38-flash-next-dev" if origin == "aiter" else "qwen4_v0.5.21"
        )
        if (
            pr["state"] != "open"
            or pr.get("draft")
            or pr["base"]["ref"] != expected_branch
        ):
            raise ValueError("Comment-dispatched PR is not eligible")
        expected_sha = os.environ.get("INPUT_CANDIDATE_SHA")
        if expected_sha and expected_sha != pr["head"]["sha"]:
            raise ValueError("PR head changed after dispatch; request a new /ci run")
        candidate_repo, candidate_ref = (
            pr["head"]["repo"]["full_name"],
            pr["head"]["sha"],
        )
    sources = resolve_sources(
        origin,
        candidate_repo,
        candidate_ref,
        {
            "sglang": os.environ.get("INPUT_SGLANG_REF"),
            "aiter": os.environ.get("INPUT_AITER_REF"),
            "pyhip": os.environ.get("INPUT_PYHIP_REF"),
        },
    )
    selection = os.environ.get("INPUT_CASES", "")
    plan = make_plan(
        profile=os.environ.get("INPUT_PROFILE") or "full",
        selection=json.loads(selection) if selection else None,
        sources=sources,
        run_id=f"{repository.replace('/', '-')}-{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}",
        examples=int(os.environ["INPUT_EXAMPLES"])
        if os.environ.get("INPUT_EXAMPLES")
        else None,
        base_image=os.environ.get("INPUT_BASE_IMAGE") or None,
        gate_mode=os.environ.get("INPUT_GATE_MODE") or "observe",
    )
    plan["harness"] = {"repo": "apinge/sglang", "sha": os.environ["HARNESS_SHA"]}
    if event_name == "pull_request_target" or os.environ.get("INPUT_PR_NUMBER"):
        plan["ci_status_target"] = {"repo": repository, "sha": sources[origin]["sha"]}
    plan["plan_hash"] = digest({k: v for k, v in plan.items() if k != "plan_hash"})
    write_json("qwen4-plan/manifest.json", plan)
    from ci_status import publish

    publish(plan, "pending")


if __name__ == "__main__":
    main()
