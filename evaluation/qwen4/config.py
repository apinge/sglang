"""Validate recipes and freeze one source/workload combination for an entire run."""

import copy
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import quote

from common import (
    REPO,
    ROOT,
    SHA,
    checked_name,
    command_output,
    digest,
    file_hash,
    read_json,
    utc_now,
)


def positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


EVALUATION_ROOT = Path(os.environ.get("QWEN4_EVALUATION_ROOT", ROOT.parent))


def load_cases():
    # JSON is a YAML subset; keeping these documents in that subset lets the
    # runner bootstrap without installing a YAML parser on the GPU host.
    data = read_json(ROOT / "cases.yaml")
    if data["schema_version"] != 1:
        raise ValueError("Unsupported case schema")
    ids = set()
    for group, spec in data["groups"].items():
        checked_name(group)
        checked_name(spec["directory"])
        if spec.get("draft_directory"):
            checked_name(spec["draft_directory"])
        positive_int(spec["gpus"], "group GPUs")
    for case in data["cases"]:
        case_id = checked_name(case["id"])
        if case_id in ids:
            raise ValueError(f"Duplicate case: {case_id}")
        ids.add(case_id)
        group = data["groups"][case["group"]]
        if case["mode"] not in {"base", "dflash", "mtp"}:
            raise ValueError("Unknown speculative mode")
        if not 1 <= positive_int(case["tp"], "TP") <= group["gpus"]:
            raise ValueError("TP exceeds the group's GPU allocation")
        if case["mode"] == "dflash" and not group.get("draft_directory"):
            raise ValueError("DFlash requires its corresponding draft")
        for key in ["launch_script", "benchmark_script"]:
            checked_name(case[key])
            if not (EVALUATION_ROOT / case[key]).is_file():
                raise ValueError(f"Missing evaluation script: {case[key]}")
        if not isinstance(case["launch_env"], dict):
            raise ValueError("launch_env must be a mapping")
        case["launch_sha256"] = file_hash(EVALUATION_ROOT / case["launch_script"])
        case["benchmark_sha256"] = file_hash(EVALUATION_ROOT / case["benchmark_script"])
    return data


def resolve_commit(repo, ref):
    if not REPO.fullmatch(repo) or not ref or len(ref) > 200:
        raise ValueError("Invalid source repository/ref")
    response = json.loads(
        command_output(["gh", "api", f"repos/{repo}/commits/{quote(ref, safe='')}"])
    )
    sha = response["sha"]
    if not SHA.fullmatch(sha):
        raise ValueError("GitHub did not return a full commit SHA")
    return {"repo": repo, "ref": ref, "sha": sha}


def resolve_sources(origin, candidate_repo, candidate_ref, overrides=None):
    pins = read_json(ROOT / "dependencies.lock.json")
    overrides = overrides or {}
    if origin not in {"sglang", "aiter", "integration"}:
        raise ValueError("Unknown run origin")
    sources = {}
    for name in ["sglang", "aiter", "pyhip", "client"]:
        repo = pins[f"{name}_repo"]
        ref = pins.get(f"{name}_sha", pins.get(f"{name}_branch"))
        if origin == name:
            repo, ref = candidate_repo or repo, candidate_ref or ref
        ref = overrides.get(name) or ref
        sources[name] = resolve_commit(repo, ref)
    return sources


def make_plan(
    profile="full",
    selection=None,
    sources=None,
    run_id=None,
    examples=None,
    base_image=None,
    gate_mode="observe",
    port=None,
):
    recipes = load_cases()
    config = read_json(ROOT / "profiles.yaml")
    if profile not in config["profiles"] or gate_mode not in {"observe", "enforce"}:
        raise ValueError("Unknown profile or gate mode")
    known = {x["id"]: x for x in recipes["cases"]}
    if selection is None:
        selection = list(known)
    if (
        not isinstance(selection, list)
        or not selection
        or not all(isinstance(x, str) for x in selection)
    ):
        raise ValueError("Case selection must be a nonempty list of case IDs")
    if len(set(selection)) != len(selection) or set(selection) - known.keys():
        raise ValueError("Duplicate or unknown case selection")
    if profile in {"full", "calibration"} and set(selection) != known.keys():
        raise ValueError("Full/calibration profiles require all six cases")
    if profile == "profile" and len(selection) != 1:
        raise ValueError("Profiling requires one explicitly selected case")
    workload = copy.deepcopy(config["profiles"][profile])
    if examples is not None:
        if not workload["accuracy"]:
            raise ValueError("Example override is only valid for accuracy workloads")
        workload["examples"] = positive_int(examples, "examples")
    defaults = config["defaults"]
    port = defaults["port"] if port is None else port
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError("Port must be in 1024..65535")
    image = base_image or defaults["base_image"]
    if not isinstance(image, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.:/@-]*", image
    ):
        raise ValueError("Invalid Docker image reference")
    groups = []
    for group_id, group in recipes["groups"].items():
        cases = [
            copy.deepcopy(c)
            for c in recipes["cases"]
            if c["group"] == group_id and c["id"] in selection
        ]
        if cases:
            groups.append({"id": group_id, **group, "cases": cases})
    for name, source in (sources or {}).items():
        if not SHA.fullmatch(source.get("sha", "")) or not REPO.fullmatch(
            source.get("repo", "")
        ):
            raise ValueError(f"Unresolved source: {name}")
    plan = {
        "schema_version": 1,
        "evaluation_scripts": {
            path.name: file_hash(path) for path in sorted(EVALUATION_ROOT.glob("*.sh"))
        },
        "harness_digest": digest(
            {
                str(p.relative_to(ROOT)): file_hash(p)
                for p in sorted(ROOT.iterdir())
                if p.is_file()
                and (p.suffix in {".py", ".json", ".yaml"} or p.name == "Dockerfile")
            }
        ),
        "created_at": utc_now(),
        "run_id": checked_name(run_id or f"local-{os.getpid()}"),
        "profile": profile,
        "gate_mode": gate_mode,
        "groups": groups,
        "workload": workload,
        "defaults": defaults,
        "port": port,
        "sources": sources or {},
        "base_image": image,
        "protocol": f"gsm8k-guide-5shot-{workload['examples']}-v1",
        "expected_cases": selection,
        "expected_results": [
            f"{c['group']}/{c['id']}/result.json"
            for c in recipes["cases"]
            if c["id"] in selection
        ],
    }
    plan["plan_hash"] = digest(plan)
    return plan


def validate_plan(plan, verify_environment=True):
    payload = {k: v for k, v in plan.items() if k != "plan_hash"}
    if digest(payload) != plan.get("plan_hash"):
        raise ValueError("Run plan was changed after preparation")
    expected_paths = []
    ids = []
    for group in plan["groups"]:
        checked_name(group["id"])
        for case in group["cases"]:
            checked_name(case["id"])
            if case["group"] != group["id"]:
                raise ValueError("Case group identity mismatch")
            expected_paths.append(f"{group['id']}/{case['id']}/result.json")
            ids.append(case["id"])
    if (
        len(ids) != len(set(ids))
        or set(ids) != set(plan["expected_cases"])
        or expected_paths != plan["expected_results"]
    ):
        raise ValueError("Expected-result manifest disagrees with its case groups")
    if not verify_environment:
        return  # Reporting may inspect a frozen plan produced by an older harness.
    fresh = make_plan(
        plan["profile"],
        plan["expected_cases"],
        plan["sources"],
        plan["run_id"],
        plan["workload"]["examples"] if plan["workload"]["accuracy"] else None,
        plan["base_image"],
        plan["gate_mode"],
        plan["port"],
    )
    for key in [
        "groups",
        "workload",
        "defaults",
        "protocol",
        "expected_results",
        "harness_digest",
        "evaluation_scripts",
    ]:
        if plan[key] != fresh[key]:
            raise ValueError(f"Run plan disagrees with trusted configuration: {key}")


def server_command(
    case, group, port=7080, models_root="/models", python=sys.executable
):
    return ["bash", str(EVALUATION_ROOT / case["launch_script"])]


def server_environment(
    case, group, port=7080, models_root="/models", python=sys.executable
):
    env = {
        **case["launch_env"],
        "SERVER_PYTHON": python,
        "MODEL_PATH": str(Path(models_root) / group["directory"]),
        "HOST": "127.0.0.1",
        "PORT": str(port),
        "SERVED_MODEL_NAME": case.get("served_model_name", case["id"]),
    }
    if case["mode"] == "dflash":
        env["DRAFT_MODEL_PATH"] = str(Path(models_root) / group["draft_directory"])
    return env
