#!/usr/bin/env python3
"""Qwen4 CI: prepare sources, build once, run three target pairs, and report.

Only the Python standard library is needed on the host. Model launch, serving
performance, and GSM8K scoring remain in the existing evaluation scripts.
"""

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve()
EVALUATION = (
    ROOT.parents[2] / "evaluation"
    if ROOT.parent.name == "ci" and ROOT.parent.parent.name == "scripts"
    else ROOT.parent / "evaluation"
)
BASE_IMAGE = "rocm/ali-private:ubuntu22.04_rocm7.14.0_cp310_torch2.12.0_sglang_d0a5b96_aiter_b7aed05_20261020"
CLIENT_SHA = "fc6ddbbf7aac80167ea04cc00bf200af8adf2b85"
PYHIP_SHA = "edde17b08ffe0dab0360fe3b1aa042b56ac72e1c"
BRANCHES = {"sglang": "qwen4_v0.5.21", "aiter": "qwen38-flash-next-dev"}
# User-approved references from the previous 200-example full-02 run.
# Update deliberately; never learn a baseline from the candidate being tested.
GSM8K_BASELINES = {
    "q35-35b-tp4-base": 0.970,
    "q35-35b-tp1-dflash": 0.970,
    "q35-397b-tp8-base": 0.990,
    "q35-397b-tp8-dflash": 0.985,
    "q38-flash-tp2-base": 0.980,
    "q38-flash-tp2-mtp": 0.985,
}
GSM8K_MAX_RELATIVE_DROP = 0.02
DATASETS = {
    "gsm8k/test.jsonl": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
    "ShareGPT_V3_unfiltered_cleaned_split.json": "35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4",
}
GROUPS = {
    "q35-35b": ("Qwen3.5-35B-A3B-PTPC-FP8", "Qwen3.5-35B-A3B-DFlash"),
    "q35-397b": ("Qwen3.5-397B-A17B-PTPC-FP8", "Qwen3.5-397B-A17B-DFlash"),
    "q38-flash": ("Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16", None),
}
# Case ID, target group, TP, speculative mode, existing launch script.
RECIPES = [
    (
        "q35-35b-tp4-base",
        "q35-35b",
        4,
        "base",
        "launch_qwen3.5-35B-fp8_tp4_disable_prefix_cache.sh",
    ),
    (
        "q35-35b-tp1-dflash",
        "q35-35b",
        1,
        "dflash",
        "launch_qwen3.5-35B-fp8_tp1_dflash_with_prefix_cache_mfs0.7_bs32.sh",
    ),
    (
        "q35-397b-tp8-base",
        "q35-397b",
        8,
        "base",
        "launch_qwen3.5-397B-fp8_tp8_disable_prefix_cache.sh",
    ),
    (
        "q35-397b-tp8-dflash",
        "q35-397b",
        8,
        "dflash",
        "launch_qwen3.5-397B-fp8_tp8_dflash_with_prefix_cache.sh",
    ),
    (
        "q38-flash-tp2-base",
        "q38-flash",
        2,
        "base",
        "launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh",
    ),
    (
        "q38-flash-tp2-mtp",
        "q38-flash",
        2,
        "mtp",
        "launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh",
    ),
]
CASES = [
    dict(
        id=i,
        group=g,
        tp=tp,
        mode=mode,
        launch_script=script,
        prefix_cache=mode == "dflash",
        output_tokens=350 if g == "q38-flash" else 500,
    )
    for i, g, tp, mode, script in RECIPES
]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def clean_environment(extra=None):
    """Build each child environment afresh; never carry a previous case's knobs."""
    allowed = {
        "PATH",
        "HOME",
        "USER",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "ROCM_PATH",
        "HIP_PATH",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
    }
    result = {key: value for key, value in os.environ.items() if key in allowed}
    result.update({"PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1"})
    result.update(extra or {})
    return result


@contextlib.contextmanager
def host_lock(path, timeout=3600):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Host is reserved: {path}")
                time.sleep(1)
        try:
            stream.seek(0)
            stream.truncate()
            stream.write(f"pid={os.getpid()}\n")
            stream.flush()
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def http_json(url, payload=None, timeout=5):
    body = json.dumps(payload).encode() if payload is not None else None
    request = Request(url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        raw = response.read()
        return json.loads(raw) if raw else None


def port_available(port):
    with socket.socket() as sock:
        # Uvicorn also uses SO_REUSEADDR. Closed connections in TIME_WAIT must
        # not be mistaken for a surviving listener after a clean shutdown.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def group_members(pgid):
    """Live members of an owned Linux process group, excluding unreaped zombies."""
    members = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(") ", 1)[1].split()
            if int(fields[2]) == pgid and fields[0] not in {"Z", "X"}:
                members.append(int(path.parent.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return members


def wait_port_released(port, timeout=10):
    deadline = time.monotonic() + timeout
    while not port_available(port):
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Server port {port} remained allocated after cleanup")
        time.sleep(0.1)


class ManagedProcess:
    def __init__(self, argv, log_path, env=None, cwd=None):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("w")
        self._stopped = False
        try:
            self.process = subprocess.Popen(
                argv,
                env=env,
                cwd=cwd,
                stdout=self.log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except BaseException:
            self.log.close()
            raise

    def alive(self):
        return self.process.poll() is None

    def stop(self, timeout=60):
        if self._stopped:
            return
        # Even an exited leader can have live worker descendants in its group.
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + timeout
        while group_members(self.process.pid) and time.monotonic() < deadline:
            self.process.poll()
            time.sleep(0.1)
        if group_members(self.process.pid):
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 10
            while group_members(self.process.pid) and time.monotonic() < deadline:
                self.process.poll()
                time.sleep(0.1)
        self.process.wait(timeout=10)
        self.log.close()
        if group_members(self.process.pid):
            raise RuntimeError("Owned server process group did not stop")
        self._stopped = True

    def wait(self, timeout, progress=False):
        try:
            if not progress:
                return self.process.wait(timeout=timeout)
            started = time.monotonic()
            deadline = started + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(self.process.args, timeout)
                try:
                    return self.process.wait(timeout=min(30, remaining))
                except subprocess.TimeoutExpired:
                    print(
                        f"Still running {self.log_path} ({time.monotonic() - started:.0f}s)",
                        flush=True,
                    )
        finally:
            self.stop()


def wait_ready(process, base_url, case, model_path, timeout=3600):
    deadline = time.monotonic() + timeout
    last_error = "no response"
    while time.monotonic() < deadline:
        if not process.alive():
            raise RuntimeError(
                f"Server exited before readiness: {process.process.returncode}"
            )
        try:
            models = http_json(base_url + "/v1/models")
            if case.get("served_model_name", case["id"]) not in [
                m["id"] for m in models["data"]
            ]:
                raise ValueError("Endpoint is serving a different model identity")
            info = http_json(base_url + "/server_info")
            args = info.get("server_args", info)
            if args.get("tp_size") != case["tp"]:
                raise ValueError(f"Server TP mismatch: {args.get('tp_size')}")
            if str(args.get("model_path", "")).rstrip("/") != str(model_path).rstrip(
                "/"
            ):
                raise ValueError("Server model path mismatch")
            expected = {"base": None, "dflash": "DFLASH", "mtp": "EAGLE"}[case["mode"]]
            if args.get("speculative_algorithm") != expected:
                raise ValueError("Server speculative mode mismatch")
            if bool(args.get("disable_radix_cache")) != (not case["prefix_cache"]):
                raise ValueError("Server prefix cache mode mismatch")
            if not process.alive():
                raise RuntimeError("Server exited during readiness")
            return info
        except ValueError:
            raise  # A responding but wrong endpoint must never count as ready.
        except Exception as exc:
            last_error = str(exc)
            time.sleep(min(2, max(0, deadline - time.monotonic())))
    raise TimeoutError(f"Server readiness timed out: {last_error}")


def smoke_generation(base_url, model, timeout=1200):
    result = http_json(
        base_url + "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "What is 1 + 1? Answer briefly."}],
            "temperature": 0,
            "max_tokens": 64,
        },
        timeout=timeout,
    )
    choice = result["choices"][0]
    message = choice["message"]
    if not (message.get("content") or message.get("reasoning_content")):
        raise ValueError("Smoke request returned no generated text")
    return result


def gsm8k_selection(path, count=200, shots=5):
    rows = [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]
    if len(rows) < shots + count:
        raise ValueError(f"GSM8K needs {shots + count} rows, found {len(rows)}")
    for row in rows:
        if not isinstance(row.get("question"), str) or not isinstance(
            row.get("answer"), str
        ):
            raise ValueError("Invalid GSM8K row")
    ids = [digest(row) for row in rows]
    prompt_ids, example_ids = ids[:shots], ids[shots : shots + count]
    if set(prompt_ids) & set(example_ids) or len(set(example_ids)) != count:
        raise ValueError("GSM8K prompt/test overlap or duplicate scored examples")
    return {
        "sha256": file_hash(path),
        "rows": len(rows),
        "shots": shots,
        "prompt_ids": prompt_ids,
        "example_ids": example_ids,
    }


def number(value, name, minimum=0, maximum=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite numeric data")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside the allowed range")
    return value


def validate_performance(path, prompts, output_tokens, profiling=False):
    lines = [line for line in Path(path).read_text().splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("Expected exactly one fresh performance record")
    import json

    data = json.loads(lines[0])
    if data.get("completed") != prompts:
        raise ValueError(
            f"Only {data.get('completed')} of {prompts} benchmark requests completed"
        )
    errors = data.get("errors")
    lengths = data.get("output_lens")
    if not isinstance(errors, list) or len(errors) != prompts or any(errors):
        raise ValueError("Missing request details or nonempty benchmark errors")
    if (
        not isinstance(lengths, list)
        or len(lengths) != prompts
        or any(n != output_tokens for n in lengths)
    ):
        raise ValueError(
            "Benchmark output lengths do not match the requested fixed workload"
        )
    for key in ["mean_ttft_ms", "mean_tpot_ms", "output_throughput"]:
        number(
            data.get(key),
            key,
            minimum=0 if profiling and key == "mean_tpot_ms" else 1e-12,
        )
    return data


def validate_accuracy(path, examples):
    data = read_json(path)
    for key in ["requested", "scored", "completed"]:
        if type(data.get(key)) is not int or data[key] != examples:
            raise ValueError(f"GSM8K {key} must equal {examples}")
    if type(data.get("errors")) is not int or data["errors"] != 0:
        raise ValueError("GSM8K has terminal request errors")
    number(data.get("score"), "GSM8K score", maximum=1)
    report = Path(path).with_name("gsm8k.html")
    if not report.is_file() or report.stat().st_size == 0:
        raise ValueError("GSM8K HTML report is missing")
    return data


def command(argv, **kwargs):
    kwargs.setdefault("timeout", 120)
    return subprocess.run(
        argv, check=True, text=True, stdout=subprocess.PIPE, **kwargs
    ).stdout.strip()


def github(path, payload=None):
    argv = ["gh", "api", path]
    if payload is not None:
        argv += ["--method", "POST", "--input", "-"]
    result = command(argv, input=json.dumps(payload) if payload is not None else None)
    return json.loads(result) if result else None


def prepare(args):
    """Runs on the trusted controller; no candidate code is checked out here."""
    repository = os.environ.get("GITHUB_REPOSITORY", "apinge/sglang")
    origin = "aiter" if repository.endswith("/aiter") else "sglang"
    event_name = os.environ.get("GITHUB_EVENT_NAME", "workflow_dispatch")
    event = (
        read_json(os.environ["GITHUB_EVENT_PATH"])
        if os.environ.get("GITHUB_EVENT_PATH")
        else {}
    )
    pr = event.get("pull_request")
    if event_name == "pull_request_target":
        label = os.environ.get("QWEN4_CI_LABEL") or "run-qwen4-ci"
        reason = ""
        if (
            not pr
            or pr.get("draft")
            or pr["state"] != "open"
            or pr["base"]["ref"] != BRANCHES[origin]
        ):
            reason = "PR must be open, ready, and target the migration branch"
        elif label not in [item["name"] for item in pr["labels"]]:
            reason = f"Maintainer label {label} is absent"
        else:
            # Check the last label grant, so contributor pushes remain eligible.
            rows = command(
                [
                    "gh",
                    "api",
                    "--paginate",
                    f"repos/{repository}/issues/{pr['number']}/events?per_page=100",
                    "--jq",
                    f'.[] | select(.event == "labeled" and .label.name == {json.dumps(label)}) | [(.id|tostring), .actor.login] | @tsv',
                ]
            )
            grants = [line.split("\t") for line in rows.splitlines()]
            actor = max(grants, key=lambda row: int(row[0]))[1] if grants else ""
            if not actor or github(
                f"repos/{repository}/collaborators/{quote(actor, safe='')}/permission"
            )["permission"] not in {"admin", "maintain", "write"}:
                reason = "CI label must be granted by a maintainer"
        if reason:
            actions_output("enabled", "false")
            actions_summary(f"Qwen4 GPU evaluation skipped: {reason}.\n")
            return 0
    elif event_name not in {"workflow_dispatch", "schedule"}:
        raise ValueError(f"Unsupported GPU trigger: {event_name}")
    elif os.environ.get("INPUT_PR_NUMBER"):
        number = os.environ["INPUT_PR_NUMBER"]
        if not number.isdigit() or int(number) < 1:
            raise ValueError("pr_number must be a positive integer")
        pr = github(f"repos/{repository}/pulls/{number}")
        # An explicit maintainer dispatch may pilot a draft PR.
        if pr["state"] != "open" or pr["base"]["ref"] != BRANCHES[origin]:
            raise ValueError("Manual PR must be open and target the migration branch")
    sources = {}
    for name in ["sglang", "aiter", "pyhip", "client"]:
        repo = (
            "tingqli/pyhip"
            if name == "pyhip"
            else "apinge/sglang"
            if name == "client"
            else f"apinge/{name}"
        )
        ref = (
            PYHIP_SHA
            if name == "pyhip"
            else CLIENT_SHA
            if name == "client"
            else os.environ.get(f"INPUT_{name.upper()}_REF") or BRANCHES[name]
        )
        if pr and name == origin:
            repo, ref = pr["head"]["repo"]["full_name"], pr["head"]["sha"]
        sha = github(f"repos/{repo}/commits/{quote(ref, safe='')}")["sha"]
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError("Expected a full source SHA")
        sources[name] = {"repo": repo, "sha": sha}
    plan = {
        "sources": sources,
        "base_image": os.environ.get("INPUT_BASE_IMAGE") or BASE_IMAGE,
        "smoke": args.smoke or os.environ.get("INPUT_SMOKE") == "true",
    }
    if pr:
        plan["status_target"] = {"repo": repository, "sha": pr["head"]["sha"]}
    write_json(Path(args.output) / "plan.json", plan)
    actions_output("plan", json.dumps(plan))
    actions_output("enabled", "true")
    actions_summary(
        "Qwen4 sources resolved once; full runs enforce the GSM8K accuracy threshold.\n"
    )
    publish_status(plan, "pending")
    return 0


def actions_output(name, value):
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            stream.write(f"{name}={value}\n")


def actions_summary(text):
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
            stream.write(text)


def publish_status(plan, state):
    if (
        plan.get("status_target")
        and os.environ.get("GH_TOKEN")
        and os.environ.get("GITHUB_ACTIONS") == "true"
    ):
        target = plan["status_target"]
        github(
            f"repos/{target['repo']}/statuses/{target['sha']}",
            {
                "state": state,
                "context": "Qwen4 CI",
                "description": (
                    "Smoke diagnostics; numerical thresholds advisory"
                    if plan["smoke"]
                    else "GSM8K relative drop <=2%; performance advisory"
                ),
                "target_url": f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}",
            },
        )


def install(plan):
    """Image build only: follow the evaluation guide without upgrading the base stack."""
    root = Path("/ci/src")
    for name, source in plan["sources"].items():
        dest = root / name
        for argv in [
            ["git", "init", str(dest)],
            [
                "git",
                "-C",
                str(dest),
                "remote",
                "add",
                "origin",
                f"https://github.com/{source['repo']}.git",
            ],
            ["git", "-C", str(dest), "fetch", "--depth=1", "origin", source["sha"]],
            ["git", "-C", str(dest), "checkout", "--detach", "FETCH_HEAD"],
        ]:
            subprocess.run(argv, check=True)
        if command(["git", "-C", str(dest), "rev-parse", "HEAD"]) != source["sha"]:
            raise RuntimeError(f"Wrong installed source: {name}")
        if name in {"aiter", "pyhip"}:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(dest),
                    "submodule",
                    "update",
                    "--init",
                    "--recursive",
                ],
                check=True,
            )
    pip = [sys.executable, "-m", "pip"]
    packages = json.loads(command(pip + ["list", "--format=json"]))
    managed = {"sglang", "sgl-kernel", "sglang-kernel", "aiter", "amd-aiter", "pyhip"}
    constraints = Path("/ci/constraints.txt")
    constraints.write_text(
        "\n".join(
            f"{p['name']}=={p['version']}"
            for p in packages
            if p["name"].lower().replace("_", "-") not in managed
        )
        + "\n"
    )
    env = {
        **os.environ,
        "PIP_CONSTRAINT": str(constraints),
        "SGLANG_BUILD_RUST_EXTS": "none",
    }
    subprocess.run(
        pip
        + [
            "uninstall",
            "-y",
            "sglang",
            "sgl-kernel",
            "sglang-kernel",
            "amd-aiter",
            "pyhip",
        ],
        check=True,
    )
    for name in ["aiter", "pyhip"]:
        subprocess.run(
            pip + ["install", "--no-build-isolation", "-e", str(root / name)],
            check=True,
            env=env,
        )
    package = root / "sglang/python"
    subprocess.run(
        [sys.executable, "setup_rocm.py", "install"],
        cwd=package / "sglang/kernels/aot",
        env=env,
        check=True,
    )
    if (package / "pyproject_other.toml").exists():
        shutil.copyfile(package / "pyproject_other.toml", package / "pyproject.toml")
    subprocess.run(
        pip + ["install", "--no-build-isolation", "-e", str(package) + "[all_hip]"],
        check=True,
        env=env,
    )
    # Module resolution works without importing AITER, whose import needs a GPU.
    origins = json.loads(
        command(
            [
                sys.executable,
                "-c",
                "import importlib.util,json; print(json.dumps({n:importlib.util.find_spec(n).origin for n in ['sglang','aiter','pyhip']}))",
            ]
        )
    )
    if any(
        not Path(path).resolve().is_relative_to(root / name)
        for name, path in origins.items()
    ):
        raise RuntimeError(
            "An installed module resolves outside the candidate checkout"
        )
    write_json(
        "/ci/build.json",
        {
            "sources": plan["sources"],
            "modules": origins,
            "packages": json.loads(command(pip + ["list", "--format=json"])),
        },
    )


def probe():
    """Executed in a short-lived child so the supervisor retains no GPU memory."""
    import torch

    if not torch.version.hip or torch.cuda.device_count() != 8:
        raise RuntimeError("This CI requires eight visible ROCm GPUs")
    devices = []
    for index in range(8):
        props = torch.cuda.get_device_properties(index)
        if "MI308" not in props.name or props.gcnArchName.split(":")[0] != "gfx942":
            raise RuntimeError(f"GPU {index} is not MI308X/gfx942")
        free, total = torch.cuda.mem_get_info(index)
        if total - free > 1024**3:
            raise RuntimeError(f"GPU {index} is busy")
        devices.append({"index": index, "name": props.name, "bytes": total})
    return {"devices": devices, "torch": torch.__version__, "rocm": torch.version.hip}


def gpu_idle():
    deadline = time.monotonic() + 30
    while True:
        result = subprocess.run(
            [sys.executable, str(ROOT), "probe"],
            env=clean_environment({"HIP_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}),
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
        if "is busy" not in result.stderr or time.monotonic() >= deadline:
            raise RuntimeError(result.stderr.strip())
        time.sleep(1)


def run_group(plan, group, output, models, datasets, client_root):
    output.mkdir(parents=True, exist_ok=True)
    client_env = clean_environment(
        {
            "PYTHONPATH": str(client_root / "python"),
            "HIP_VISIBLE_DEVICES": "0",
            "OPENAI_API_KEY": "EMPTY",
        }
    )
    if (
        command(["git", "-C", str(client_root), "rev-parse", "HEAD"])
        != plan["sources"]["client"]["sha"]
    ):
        raise ValueError("The evaluator checkout does not match the pinned SHA")
    if command(
        [
            "git",
            "-C",
            str(client_root),
            "status",
            "--porcelain",
            "--",
            "python/sglang/test",
            "python/sglang/benchmark",
        ]
    ):
        raise ValueError("The pinned evaluator has local source changes")
    module = command(
        [sys.executable, "-c", "import sglang; print(sglang.__file__)"], env=client_env
    )
    if not Path(module).resolve().is_relative_to(client_root / "python"):
        raise ValueError(
            "Client imported the candidate instead of the pinned evaluator"
        )
    for original in [c for c in CASES if c["group"] == group]:
        print(f"Starting {original['id']}", flush=True)
        case = {
            **original,
            "served_model_name": f"qwen4-{os.getpid()}-{original['id']}",
        }
        directory = output / case["id"]
        directory.mkdir()  # Existing case data must never count as a fresh result.
        state = {
            "case_id": case["id"],
            "container_id": os.environ.get("QWEN4_CONTAINER_ID"),
            "errors": [],
            "phases": {},
        }
        server = None
        started = time.monotonic()
        try:
            write_json(directory / "hardware.json", gpu_idle())
            if not port_available(7080):
                raise RuntimeError("Port 7080 already belongs to another server")
            env = clean_environment(
                {
                    "SERVER_PYTHON": sys.executable,
                    "MODEL_PATH": str(models / GROUPS[group][0]),
                    "HOST": "127.0.0.1",
                    "PORT": "7080",
                    "SERVED_MODEL_NAME": case["served_model_name"],
                    "HIP_VISIBLE_DEVICES": ",".join(map(str, range(case["tp"]))),
                    "TP_SIZE": str(case["tp"]),
                    "MTP": "1" if case["mode"] == "mtp" else "0",
                    "LOG_FILE": str(directory / "launch.log"),
                    "GPU_ARCHS": "gfx942",
                }
            )
            if case["mode"] == "dflash":
                env["DRAFT_MODEL_PATH"] = str(models / GROUPS[group][1])
            for key in [
                "TRITON_CACHE_DIR",
                "XDG_CACHE_HOME",
                "TORCH_EXTENSIONS_DIR",
                "AITER_JIT_DIR",
                "AITER_ROOT_DIR",
                "PYHIP_CACHE_DIR",
            ]:
                env[key] = str(directory / "cache" / key.lower())
                Path(env[key]).mkdir(parents=True)
            argv = ["bash", str(EVALUATION / case["launch_script"])]
            write_json(directory / "launch.json", {"argv": argv, "env": env})
            server = ManagedProcess(
                argv, directory / "server.log", env=env, cwd=directory
            )
            info = wait_ready(
                server, "http://127.0.0.1:7080", case, env["MODEL_PATH"], timeout=3600
            )
            write_json(directory / "server-info.json", info)
            write_json(
                directory / "smoke.json",
                smoke_generation("http://127.0.0.1:7080", case["served_model_name"]),
            )
            benchmark = (
                "run_qwen38_pure_text_bench.sh"
                if group == "q38-flash"
                else "run_qwen35_pure_text_bench.sh"
            )
            env = {
                **client_env,
                "CLIENT_PYTHON": sys.executable,
                "BENCH_MODEL": str(models / GROUPS[group][0]),
                "BENCH_SERVED_MODEL": case["served_model_name"],
                "BENCH_HOST": "127.0.0.1",
                "BENCH_PORT": "7080",
                "INPUT_TOKENS": "12000" if group == "q38-flash" else "8000",
                "OUTPUT_TOKENS": str(case["output_tokens"]),
                "NUM_PROMPTS": "2" if plan["smoke"] else "128",
                "WARMUP_REQUESTS": "1" if plan["smoke"] else "10",
                "MAX_CONCURRENCY": "1",
                "BENCH_SEED": "42",
                "DATASET_PATH": str(
                    datasets / "ShareGPT_V3_unfiltered_cleaned_split.json"
                ),
                "BENCH_OUTPUT_FILE": str(directory / "performance.jsonl"),
                "BENCH_LOG_FILE": str(directory / "benchmark.log"),
                "GSM8K_DATA_PATH": str(datasets / "gsm8k/test.jsonl"),
                "GSM8K_OUTPUT_DIR": str(directory),
                "GSM8K_NUM_EXAMPLES": "2" if plan["smoke"] else "200",
                "GSM8K_NUM_SHOTS": "5",
                "GSM8K_MAX_TOKENS": "16384",
                "GSM8K_NUM_THREADS": "512",
            }
            for phase, script, timeout in [
                ("performance", benchmark, 3600),
                ("accuracy", "run_gsm8k.sh", 7200),
            ]:
                try:
                    if not server.alive():
                        raise RuntimeError("Server exited before evaluation")
                    code = ManagedProcess(
                        ["bash", str(EVALUATION / script)],
                        directory / f"{phase}.log",
                        env=env,
                        cwd=directory,
                    ).wait(timeout, progress=True)
                    state["phases"][phase] = code
                    if code:
                        raise RuntimeError(f"{phase} exited {code}")
                except Exception as exc:
                    state["errors"].append(f"{phase}: {exc}")
        except Exception as exc:
            state["errors"].append(str(exc))
        except BaseException:
            state["errors"].append("Evaluation interrupted")
            raise
        finally:
            try:
                if server:
                    server.stop()
                    wait_port_released(7080)
                gpu_idle()
            except Exception as exc:
                state["errors"].append(f"Cleanup failed: {exc}")
                raise
            finally:
                state["elapsed_s"] = time.monotonic() - started
                write_json(directory / "execution.json", state)


def preflight(models, datasets, examples):
    for relative, checksum in DATASETS.items():
        if file_hash(datasets / relative) != checksum:
            raise ValueError(f"Dataset checksum mismatch: {datasets / relative}")
    model_info = {}
    for name in {name for pair in GROUPS.values() for name in pair if name}:
        path = models / name
        config = read_json(path / "config.json")
        index = path / "model.safetensors.index.json"
        names = (
            set(read_json(index)["weight_map"].values())
            if index.exists()
            else {p.name for p in path.glob("*.safetensors")}
        )
        if not names or any(
            Path(n).name != n
            or not (path / n).is_file()
            or not (path / n).stat().st_size
            for n in names
        ):
            raise ValueError(f"Missing model weights: {path}")
        model_info[name] = {
            "config_sha256": digest(config),
            "shards": {n: (path / n).stat().st_size for n in sorted(names)},
        }
    return {
        "models": model_info,
        "gsm8k": gsm8k_selection(datasets / "gsm8k/test.jsonl", examples),
    }


def run(plan, args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use a fresh output directory for each attempt")
    models = Path(args.models_root).resolve()
    datasets = Path(args.datasets_root or models / "benchmark_datasets").resolve()
    runtime = {"host": socket.gethostname(), "errors": [], "containers": {}}
    key = os.environ.get("QWEN4_RUN_KEY", f"{os.getpid()}-{time.time_ns()}")
    if not re.fullmatch(r"[a-zA-Z0-9-]+", key):
        raise ValueError("Invalid run identifier")
    image = f"qwen4-ci:{key}"
    runtime["image"] = image
    write_json(output / "manifest.json", plan)
    try:
        with host_lock(args.lock):
            relative = datasets.relative_to(models)
            write_json(
                output / "assets.json",
                preflight(models, datasets, 2 if plan["smoke"] else 200),
            )
            with tempfile.TemporaryDirectory(prefix="qwen4-build-") as temp:
                context = Path(temp)
                shutil.copyfile(ROOT, context / "qwen4_ci.py")
                write_json(context / "plan.json", plan)
                (context / "evaluation").mkdir()
                for name in {c["launch_script"] for c in CASES} | {
                    "run_gsm8k.sh",
                    "run_qwen35_pure_text_bench.sh",
                    "run_qwen38_pure_text_bench.sh",
                }:
                    shutil.copyfile(EVALUATION / name, context / "evaluation" / name)
                code = ManagedProcess(
                    ["docker", "pull", plan["base_image"]], output / "pull.log"
                ).wait(1800, progress=True)
                if code:
                    raise RuntimeError("Base image pull failed; see pull.log")
                info = json.loads(
                    command(["docker", "image", "inspect", plan["base_image"]])
                )[0]
                runtime["base_image_digest"] = next(
                    iter(info.get("RepoDigests", [])), info["Id"]
                )
                (context / "Dockerfile").write_text(
                    'ARG BASE_IMAGE\nFROM ${BASE_IMAGE}\nENV GPU_ARCHS=gfx942 PYTORCH_ROCM_ARCH=gfx942 SGLANG_BUILD_RUST_EXTS=none\nCOPY . /ci/\nRUN python3 /ci/qwen4_ci.py install --plan /ci/plan.json\nWORKDIR /ci\nENTRYPOINT []\nCMD ["sleep", "infinity"]\n'
                )
                code = ManagedProcess(
                    [
                        "docker",
                        "build",
                        "--network=host",
                        "--label",
                        f"qwen4.run={key}",
                        "--build-arg",
                        f"BASE_IMAGE={runtime['base_image_digest']}",
                        "-t",
                        image,
                        str(context),
                    ],
                    output / "build.log",
                ).wait(7200, progress=True)
                if code:
                    raise RuntimeError("Candidate build failed; see build.log")
                runtime["image_id"] = command(
                    ["docker", "image", "inspect", "--format={{.Id}}", image]
                )
            for group in GROUPS:
                directory = output / group
                directory.mkdir()
                name = f"{image.replace(':', '-')}-{group}"
                cid = command(
                    [
                        "docker",
                        "create",
                        "--init",
                        "--name",
                        name,
                        "--label",
                        f"qwen4.run={key}",
                        "--device=/dev/kfd",
                        "--device=/dev/dri",
                        "--group-add",
                        "video",
                        "--ipc=private",
                        "--shm-size=32g",
                        "--network=none",
                        "--security-opt",
                        "seccomp=unconfined",
                        "--ulimit",
                        "nofile=65536:65536",
                        "-e",
                        "HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7",
                        "-v",
                        f"{models}:/models:ro",
                        "-v",
                        f"{directory}:/results",
                        image,
                    ]
                )
                runtime["containers"][group] = cid
                write_json(output / "runtime.json", runtime)
                try:
                    command(["docker", "start", cid])
                    command(
                        [
                            "docker",
                            "cp",
                            f"{cid}:/ci/build.json",
                            str(output / "build.json"),
                        ]
                    )
                    code = ManagedProcess(
                        [
                            "docker",
                            "exec",
                            "-e",
                            f"QWEN4_CONTAINER_ID={cid}",
                            cid,
                            "python3",
                            "/ci/qwen4_ci.py",
                            "container",
                            "--plan",
                            "/ci/plan.json",
                            "--group",
                            group,
                            "--output",
                            "/results",
                            "--models-root",
                            "/models",
                            "--datasets-root",
                            str(Path("/models") / relative),
                        ],
                        directory / "container.log",
                    ).wait(30000, progress=True)
                    if code:
                        runtime["errors"].append(
                            f"Container {group} failed; see its container.log"
                        )
                finally:
                    # Change ownership while this same container still owns the mount.
                    # Failure here stops the run; later cases cannot mask cleanup errors.
                    try:
                        command(
                            [
                                "docker",
                                "exec",
                                cid,
                                "chown",
                                "-R",
                                f"{os.getuid()}:{os.getgid()}",
                                "/results",
                            ]
                        )
                    finally:
                        command(["docker", "rm", "-f", cid])
    except BaseException as exc:
        runtime["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        if shutil.which("docker"):
            subprocess.run(
                ["docker", "image", "rm", image],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        write_json(output / "runtime.json", runtime)
    return report(plan, output)


def run_local(args):
    """Run the full model suite in an existing installation, without a Docker build."""
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use a fresh output directory for each attempt")
    models = Path(args.models_root).resolve()
    datasets = Path(args.datasets_root or models / "benchmark_datasets").resolve()
    client = Path(args.client_root).resolve()
    roots = {
        "sglang": EVALUATION.parent,
        "aiter": Path("/opt/aiter"),
        "pyhip": Path("/opt/pyhip"),
        "client": client,
    }
    sources = {
        name: {
            "repo": "tingqli/pyhip"
            if name == "pyhip"
            else f"apinge/{'sglang' if name == 'client' else name}",
            "sha": command(["git", "-C", str(path), "rev-parse", "HEAD"]),
        }
        for name, path in roots.items()
    }
    if sources["client"]["sha"] != CLIENT_SHA:
        raise ValueError(f"Local evaluator must be at {CLIENT_SHA}")
    module = command(
        [sys.executable, "-c", "import sglang; print(sglang.__file__)"],
        env=clean_environment(),
    )
    if not Path(module).resolve().is_relative_to(roots["sglang"] / "python"):
        raise ValueError("Local server imports a different SGLang checkout")
    plan = {
        "sources": sources,
        "smoke": args.smoke,
        "mode": "existing-container",
        "base_image": "Existing environment; image build/digest not verified",
    }
    runtime = {"host": socket.gethostname(), "errors": [], "containers": {}}
    write_json(output / "manifest.json", plan)
    container = f"existing:{socket.gethostname()}"
    original_id = os.environ.get("QWEN4_CONTAINER_ID")
    os.environ["QWEN4_CONTAINER_ID"] = container
    try:
        with host_lock(args.lock):
            write_json(
                output / "assets.json",
                preflight(models, datasets, 2 if args.smoke else 200),
            )
            for group in GROUPS:
                runtime["containers"][group] = container
                write_json(output / "runtime.json", runtime)
                run_group(plan, group, output / group, models, datasets, client)
    except BaseException as exc:
        runtime["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        if original_id is None:
            os.environ.pop("QWEN4_CONTAINER_ID", None)
        else:
            os.environ["QWEN4_CONTAINER_ID"] = original_id
        write_json(output / "runtime.json", runtime)
    return report(plan, output)


def report(plan, output, job_result="success"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        runtime = read_json(output / "runtime.json")
        if not isinstance(runtime, dict) or not isinstance(runtime.get("errors"), list):
            raise ValueError("Malformed runtime.json")
    except (OSError, ValueError) as exc:
        runtime = {"errors": [f"Runtime evidence: {exc}"]}
    errors = list(runtime.get("errors", []))
    if job_result != "success":
        errors.append(f"GitHub GPU job: {job_result}")
    containers = runtime.get("containers", {})
    local = plan.get("mode") == "existing-container"
    if not isinstance(containers, dict) or not all(
        isinstance(value, str) and value for value in containers.values()
    ):
        containers = {}
    if set(containers) != set(GROUPS) or len(set(containers.values())) != (
        1 if local else 3
    ):
        errors.append("Missing target groups or incorrect container identities")
    try:
        read_json(output / "assets.json")["gsm8k"]
    except (OSError, ValueError, KeyError, TypeError):
        errors.append("Missing dataset selection evidence")
    examples, prompts = (2, 2) if plan["smoke"] else (200, 128)
    rows = []
    for case in CASES:
        directory = output / case["group"] / case["id"]
        reference = GSM8K_BASELINES[case["id"]]
        minimum = reference * (1 - GSM8K_MAX_RELATIVE_DROP)
        row = {
            "case": case["id"],
            "errors": [],
            "performance": None,
            "accuracy": None,
            "accuracy_reference": reference,
            "accuracy_minimum": None if plan["smoke"] else minimum,
        }
        try:
            state = read_json(directory / "execution.json")
            row["errors"].extend(state["errors"])
            if (
                state["case_id"] != case["id"]
                or not containers.get(case["group"])
                or state.get("container_id") != containers[case["group"]]
            ):
                raise ValueError("Case/container identity does not match")
            for phase in ["performance", "accuracy"]:
                if state["phases"].get(phase) != 0:
                    row["errors"].append(f"{phase} did not exit successfully")
            for name in ["launch.json", "server.log", "server-info.json", "smoke.json"]:
                if not (directory / name).is_file():
                    row["errors"].append(f"Missing {name}")
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            row["errors"].append(f"Execution: {exc}")
        for phase in ["performance", "accuracy"]:
            try:
                data = (
                    validate_performance(
                        directory / "performance.jsonl", prompts, case["output_tokens"]
                    )
                    if phase == "performance"
                    else validate_accuracy(directory / "gsm8k.json", examples)
                )
                keys = (
                    ["completed", "mean_ttft_ms", "mean_tpot_ms", "output_throughput"]
                    if phase == "performance"
                    else ["score", "scored", "completed"]
                )
                row[phase] = {key: data[key] for key in keys}
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                row["errors"].append(f"{phase}: {exc}")
        if not plan["smoke"] and row["accuracy"]:
            score = row["accuracy"]["score"]
            if score + 1e-12 < minimum:
                row["errors"].append(
                    f"GSM8K regression: {score:.2%} < {minimum:.2%}; "
                    f"relative drop {(reference - score) / reference:.2%} "
                    f"exceeds {GSM8K_MAX_RELATIVE_DROP:.0%} from reference {reference:.2%}"
                )
        rows.append(row)
    failed = bool(errors or any(r["errors"] for r in rows))
    title = "Qwen4 local model evaluation" if local else "Qwen4 MI308X CI"
    text = f"## {title} — {'Failed' if failed else 'Passed'}\n\n"
    if local:
        text += "Existing target-image container. **Docker build and three-container isolation: not verified.**\n\n"
    text += f"Mode: {'smoke (diagnostic)' if plan['smoke'] else 'full'}; {examples} scored GSM8K examples per case.\n\n"
    text += (
        "**Smoke accuracy is diagnostic; the 200-example threshold is not applied.**\n\n"
        if plan["smoke"]
        else f"**GSM8K gate: at most {GSM8K_MAX_RELATIVE_DROP:.0%} relative drop from each recorded reference; improvements pass.**\n\n"
    )
    text += "Performance thresholds: advisory.\n\n"
    text += f"Base image: `{plan['base_image']}`\n\nResolved image: `{runtime.get('base_image_digest', 'unavailable')}`\n\n"
    text += (
        " | ".join(f"{name}: `{s['sha']}`" for name, s in plan["sources"].items())
        + "\n\n"
    )
    text += "| Case | Status | GSM8K | Reference / minimum | Performance completed | TTFT (ms) | TPOT (ms) | Output tokens/s |\n|---|---|---:|---:|---:|---:|---:|---:|\n"
    for row in rows:
        a, p = row["accuracy"], row["performance"]
        metrics = [
            f"{p[k]:.3f}" if p else "—"
            for k in ["mean_ttft_ms", "mean_tpot_ms", "output_throughput"]
        ]
        accuracy = f"{a['score']:.3%} ({a['scored']}/{examples})" if a else "—"
        completed = f"{p['completed']}/{prompts}" if p else "—"
        threshold = (
            "Diagnostic"
            if plan["smoke"]
            else f"{row['accuracy_reference']:.2%} / {row['accuracy_minimum']:.2%} "
            f"(≥{math.ceil(row['accuracy_minimum'] * examples - 1e-12)}/{examples})"
        )
        text += (
            f"| {row['case']} | {'❌ Failed' if row['errors'] else '✅ Pass'} | {accuracy} | {threshold} | {completed} | "
            + " | ".join(metrics)
            + " |\n"
        )
    for message in errors + [f"{r['case']}: {e}" for r in rows for e in r["errors"]]:
        text += "\n- " + str(message).replace("\n", " ").replace("|", "\\|")
    text += "\n\nRaw logs, JSON, HTML, build provenance, and container IDs are in the run's result artifact.\n"
    write_json(
        output / "summary.json",
        {
            "failed": failed,
            "mode": plan.get("mode", "docker"),
            "errors": errors,
            "cases": rows,
        },
    )
    (output / "summary.md").write_text(text)
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["prepare", "run", "local", "container", "install", "report", "probe"],
    )
    parser.add_argument("--plan", help="Prepared JSON file; defaults to QWEN4_PLAN")
    parser.add_argument("--output", default="qwen4-results")
    parser.add_argument("--models-root", default="/raid/models")
    parser.add_argument("--datasets-root")
    parser.add_argument("--client-root", default="/ci/src/client")
    parser.add_argument("--group", choices=GROUPS)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Prepare a diagnostic 2-request/2-example run",
    )
    parser.add_argument("--lock", default="/var/lock/qwen4-mi308x.lock")
    args = parser.parse_args()
    signal.signal(
        signal.SIGTERM,
        lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("Terminated")),
    )
    if args.command == "probe":
        print(json.dumps(probe()))
        return 0
    if args.command == "prepare":
        return prepare(args)
    if args.command == "local":
        return run_local(args)
    plan = read_json(args.plan) if args.plan else json.loads(os.environ["QWEN4_PLAN"])
    if args.command == "install":
        install(plan)
        return 0
    if args.command == "container":
        if not args.group:
            parser.error("container requires --group")
        models = Path(args.models_root).resolve()
        run_group(
            plan,
            args.group,
            Path(args.output).resolve(),
            models,
            Path(args.datasets_root or models / "benchmark_datasets").resolve(),
            Path(args.client_root).resolve(),
        )
        return 0
    if args.command == "run":
        return run(plan, args)
    result = report(plan, args.output, os.environ.get("GPU_JOB_RESULT", "success"))
    actions_summary((Path(args.output) / "summary.md").read_text())
    publish_status(plan, "failure" if result else "success")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
