#!/usr/bin/env python3
"""Run the CI based on Qwen evaluation scripts and enforce the recorded GSM8K gate."""

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
EVALUATION = ROOT / "evaluation"
CLIENT_SHA = "fc6ddbbf7aac80167ea04cc00bf200af8adf2b85"
GSM8K_MAX_RELATIVE_DROP = 0.02
DATASETS = {
    "gsm8k/test.jsonl": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
    "ShareGPT_V3_unfiltered_cleaned_split.json": "35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4",
}
# Default order: Qwen3.8 first, then Qwen3.5.
# Case, model, TP, mode, launch script, approved 200-example full-02 accuracy.
CASES = [
    (
        "Qwen3.8-Flash-TP2-Base",
        "Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16",
        2,
        "base",
        "launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh",
        0.980,
    ),
    (
        "Qwen3.8-Flash-TP2-MTP",
        "Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16",
        2,
        "mtp",
        "launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh",
        0.985,
    ),
    (
        "Qwen3.5-35B-TP4-Base",
        "Qwen3.5-35B-A3B-PTPC-FP8",
        4,
        "base",
        "launch_qwen3.5-35B-fp8_tp4_disable_prefix_cache.sh",
        0.970,
    ),
    (
        "Qwen3.5-35B-TP1-DFlash",
        "Qwen3.5-35B-A3B-PTPC-FP8",
        1,
        "dflash",
        "launch_qwen3.5-35B-fp8_tp1_dflash_with_prefix_cache_mfs0.7_bs32.sh",
        0.970,
    ),
    (
        "Qwen3.5-397B-TP8-Base",
        "Qwen3.5-397B-A17B-PTPC-FP8",
        8,
        "base",
        "launch_qwen3.5-397B-fp8_tp8_disable_prefix_cache.sh",
        0.990,
    ),
    (
        "Qwen3.5-397B-TP8-DFlash",
        "Qwen3.5-397B-A17B-PTPC-FP8",
        8,
        "dflash",
        "launch_qwen3.5-397B-fp8_tp8_dflash_with_prefix_cache.sh",
        0.985,
    ),
]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def command(argv, **kwargs):
    return subprocess.check_output(argv, text=True, timeout=120, **kwargs).strip()


def clean_environment(**extra):
    allowed = (
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
    )
    return {
        **{k: os.environ[k] for k in allowed if k in os.environ},
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1",
        **extra,
    }


@contextlib.contextmanager
def managed_process(argv, log, env, cwd):
    """Stream full logs while owning the command's process group and exit status."""
    output = subprocess.Popen(
        ["tee", "--", str(log)], stdin=subprocess.PIPE, start_new_session=True
    )
    try:
        process = subprocess.Popen(
            argv,
            stdout=output.stdin,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=cwd,
            start_new_session=True,
        )
        output.stdin.close()
        try:
            yield process
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            except ProcessLookupError:
                pass
            # The leader may have exited while descendants still hold GPU memory.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=30)
    finally:
        # Close our writer even if starting the command failed. The relay drains
        # the final output after all owned workers have released their writers.
        output.stdin.close()
        try:
            output.wait(timeout=30)
        except subprocess.TimeoutExpired:
            output.kill()
            output.wait(timeout=30)
            raise RuntimeError(f"Timed out draining log output: {log}") from None
        if output.returncode:
            raise RuntimeError(
                f"Log streaming failed for {log}: exit {output.returncode}"
            )


def http_json(url):
    with urlopen(url, timeout=5) as response:
        return json.load(response)


def wait_ready(server, url, case, model_path, served_name, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"Server exited {server.returncode}; see server.log")
        try:
            with urlopen(url + "/health", timeout=5):
                pass
            models = http_json(url + "/v1/models")
            info = http_json(url + "/server_info")
        except OSError:
            time.sleep(2)
            continue
        args = info.get("server_args", info)
        if (
            served_name not in [m["id"] for m in models["data"]]
            or str(args.get("model_path", "")).rstrip("/") != str(model_path)
            or args.get("tp_size") != case[2]
            or args.get("speculative_algorithm")
            != {"base": None, "dflash": "DFLASH", "mtp": "EAGLE"}[case[3]]
            or bool(args.get("disable_radix_cache")) != (case[3] != "dflash")
        ):
            raise ValueError("Endpoint does not match the requested model, TP, or mode")
        return info
    raise TimeoutError(f"Server was not ready after {timeout}s; see server.log")


def validate_accuracy(path, smoke=False):
    data = json.loads(Path(path).read_text())
    examples = 2 if smoke else 200
    for key in ("requested", "scored", "completed"):
        if type(data.get(key)) is not int or data[key] != examples:
            raise ValueError(f"GSM8K {key} must equal {examples}")
    if type(data.get("errors")) is not int or data["errors"] != 0:
        raise ValueError("GSM8K has terminal request errors")
    score = data.get("score")
    if (
        type(score) not in (int, float)
        or not math.isfinite(score)
        or not 0 <= score <= 1
        or abs(score * examples - round(score * examples)) > 1e-6
    ):
        raise ValueError("GSM8K score must represent a finite count of correct answers")
    report = Path(path).with_suffix(".html")
    if not report.is_file() or not report.stat().st_size:
        raise ValueError("GSM8K HTML report is missing")
    return {
        key: data[key]
        for key in ("score", "requested", "scored", "completed", "errors")
    }


def check_accuracy_gate(score, baseline, smoke=False):
    minimum = baseline * (1 - GSM8K_MAX_RELATIVE_DROP)
    if not smoke and score + 1e-12 < minimum:
        raise ValueError(
            f"GSM8K regression: {score:.2%} < {minimum:.2%} "
            f"(reference {baseline:.2%}, maximum relative drop 2%)"
        )


def validate_performance(path, prompts, output_tokens, speculative=False):
    lines = [line for line in Path(path).read_text().splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("Expected one fresh performance record")
    data = json.loads(lines[0])
    if (
        data.get("completed") != prompts
        or data.get("errors") != [""] * prompts
        or data.get("output_lens") != [output_tokens] * prompts
    ):
        raise ValueError(
            "Benchmark requests failed or did not produce the fixed workload"
        )
    for key in (
        "mean_ttft_ms",
        "mean_tpot_ms",
        "output_throughput",
        "request_throughput",
    ):
        value = data.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid benchmark metric: {key}")
    if speculative or data.get("accept_length") is not None:
        value = data.get("accept_length")
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid benchmark metric: accept_length")
    # Native metadata can include an infinite request_rate for unlimited load.
    return {
        key: data[key]
        for key in (
            "completed",
            "mean_ttft_ms",
            "mean_tpot_ms",
            "output_throughput",
            "request_throughput",
            "accept_length",
        )
        if key in data
    }


def run_phase(script, directory, env, server, timeout):
    print(f"Running {script}; log: {directory / (script + '.log')}", flush=True)
    with managed_process(
        ["bash", str(EVALUATION / script)],
        directory / (script + ".log"),
        env,
        directory,
    ) as process:
        deadline = time.monotonic() + timeout
        while process.poll() is None:
            if server.poll() is not None:
                raise RuntimeError("Server exited during evaluation")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{script} exceeded {timeout}s")
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
        if process.returncode:
            raise RuntimeError(f"{script} exited {process.returncode}")
        if server.poll() is not None:
            raise RuntimeError("Server exited during evaluation")


def check_gpus(env, tp):
    # A child owns all GPU inspection so the supervisor never reserves VRAM.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import torch
assert torch.version.hip and torch.cuda.device_count() >= int(__import__('sys').argv[1]), 'Not enough ROCm GPUs'
for i in range(int(__import__('sys').argv[1])):
    assert torch.cuda.get_device_properties(i).gcnArchName.split(':')[0] == 'gfx942', 'MI308X/gfx942 required'
    free, total = torch.cuda.mem_get_info(i)
    assert total - free < 1024**3, f'Visible GPU {i} is busy; select free devices with --gpus'
""",
            str(tp),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip().splitlines()[-1])


def run_case(case, args, directory, row):
    name, model, tp, mode, script, baseline = case
    model_path = Path(args.models_root).resolve() / model
    if not (model_path / "config.json").is_file():
        raise ValueError(f"Missing model: {model_path}")
    devices = args.gpus.split(",")
    if len(devices) < tp:
        raise ValueError(f"{name} requires {tp} GPUs; --gpus supplies {len(devices)}")
    served_name = f"qwen4-{os.getpid()}-{name}"
    env = clean_environment(
        SERVER_PYTHON=sys.executable,
        HOST="127.0.0.1",
        PORT=str(args.port),
        MODEL_PATH=str(model_path),
        SERVED_MODEL_NAME=served_name,
        HIP_VISIBLE_DEVICES=",".join(devices[:tp]),
        GPU_ARCHS="gfx942",
        TP_SIZE=str(tp),
        MTP="1" if mode == "mtp" else "0",
        PLE_OFFLOAD_EMBEDDING="0",
        LOG_FILE=os.devnull,  # managed_process captures the complete server log.
    )
    if mode == "dflash":
        env["DRAFT_MODEL_PATH"] = str(
            model_path.with_name(model.replace("PTPC-FP8", "DFlash"))
        )
        if not (Path(env["DRAFT_MODEL_PATH"]) / "config.json").is_file():
            raise ValueError(f"Missing draft model: {env['DRAFT_MODEL_PATH']}")
    check_gpus(env, tp)
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", args.port))
    write_json(directory / "launch.json", {"script": script, "env": env})
    with managed_process(
        ["bash", str(EVALUATION / script)], directory / "server.log", env, directory
    ) as server:
        print(
            f"{name}: waiting for readiness; log: {directory / 'server.log'}",
            flush=True,
        )
        info = wait_ready(
            server,
            f"http://127.0.0.1:{args.port}",
            case,
            model_path,
            served_name,
            args.server_timeout,
        )
        write_json(directory / "server-info.json", info)
        q38 = model.startswith("Qwen3.8")
        output_tokens = 350 if q38 else 500
        datasets = Path(
            args.datasets_root or Path(args.models_root) / "benchmark_datasets"
        ).resolve()
        env = clean_environment(
            PYTHONPATH=str(Path(args.client_root).resolve() / "python"),
            BENCH_MODEL=str(model_path),
            BENCH_SERVED_MODEL=served_name,
            BENCH_HOST="127.0.0.1",
            BENCH_PORT=str(args.port),
            DATASET_PATH=str(datasets / "ShareGPT_V3_unfiltered_cleaned_split.json"),
            BENCH_OUTPUT_FILE=str(directory / "performance.jsonl"),
            BENCH_LOG_FILE=os.devnull,  # Avoid a second copy of the phase log.
            GSM8K_DATA_PATH=str(datasets / "gsm8k/test.jsonl"),
            GSM8K_OUTPUT_DIR=str(directory),
        )
        if args.smoke:
            env.update(NUM_PROMPTS="2", WARMUP_REQUESTS="1", GSM8K_NUM_EXAMPLES="2")
        phases = []
        if not args.accuracy_only:
            phases.append(
                (
                    "performance",
                    f"run_qwen{'38' if q38 else '35'}_pure_text_bench.sh",
                    3600,
                )
            )
        phases.append(("accuracy", "run_gsm8k.sh", 7200))
        for phase, script, timeout in phases:
            try:
                run_phase(script, directory, env, server, timeout)
                row[phase] = (
                    validate_accuracy(directory / "gsm8k.json", args.smoke)
                    if phase == "accuracy"
                    else validate_performance(
                        directory / "performance.jsonl",
                        2 if args.smoke else 128,
                        output_tokens,
                        speculative=mode != "base",
                    )
                )
                if phase == "accuracy":
                    check_accuracy_gate(row[phase]["score"], baseline, args.smoke)
            except Exception as exc:
                row["errors"].append(f"{phase}: {exc}")


def summary(output, rows, args, errors):
    failed = bool(errors or any(row["status"] != "passed" for row in rows))
    text = f"## Qwen4 evaluation — {'Failed' if failed else 'Passed'}\n\n"
    text += f"Coverage: {len(rows)}/6 selected cases. "
    text += (
        "Smoke is diagnostic: two examples; accuracy gate is disabled.\n\n"
        if args.smoke
        else "GSM8K: 200 examples; maximum 2% relative drop from recorded references.\n\n"
    )
    text += "Performance thresholds are advisory.\n\n"
    text += "| Case | Status | GSM8K | Reference / minimum | Output tokens/s | TTFT (ms) | TPOT (ms) | Accept length |\n"
    text += "|---|---|---:|---:|---:|---:|---:|---:|\n"
    for row in rows:
        accuracy, perf = row.get("accuracy", {}), row.get("performance", {})
        score = f"{accuracy['score']:.2%}" if accuracy else "—"
        minimum = row["baseline"] * (1 - GSM8K_MAX_RELATIVE_DROP)
        metrics = [
            f"{perf[key]:.3f}" if perf.get(key) is not None else "—"
            for key in (
                "output_throughput",
                "mean_ttft_ms",
                "mean_tpot_ms",
                "accept_length",
            )
        ]
        text += (
            f"| {row['case']} | {row['status']} | {score} | {row['baseline']:.2%} / {minimum:.2%} | "
            + " | ".join(metrics)
            + " |\n"
        )
    for error in errors + [
        f"{row['case']}: {e}" for row in rows for e in row["errors"]
    ]:
        text += "\n- " + str(error).replace("\n", " ").replace("|", "\\|")
    write_json(
        output / "summary.json",
        {"failed": failed, "smoke": args.smoke, "errors": errors, "cases": rows},
    )
    (output / "summary.md").write_text(text + "\n")
    return int(failed)


def run(args):
    selected = {name.strip() for name in args.case.split(",") if name.strip()}
    unknown = selected - {case[0] for case in CASES}
    if unknown:
        raise ValueError(f"Unknown cases: {', '.join(sorted(unknown))}")
    cases = [case for case in CASES if not selected or case[0] in selected]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use a fresh output directory for each attempt")
    rows = [
        {"case": case[0], "baseline": case[5], "status": "not run", "errors": []}
        for case in cases
    ]
    errors = []
    try:
        client = Path(args.client_root).resolve()
        if command(
            ["git", "-C", str(client), "rev-parse", "HEAD"]
        ) != CLIENT_SHA or command(
            ["git", "-C", str(client), "status", "--porcelain", "--", "python/sglang"]
        ):
            raise ValueError(f"Evaluator must be a clean checkout at {CLIENT_SHA}")
        datasets = Path(
            args.datasets_root or Path(args.models_root) / "benchmark_datasets"
        )
        for relative, expected in DATASETS.items():
            if args.accuracy_only and relative != "gsm8k/test.jsonl":
                continue
            with (datasets / relative).open("rb") as stream:
                checksum = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    checksum.update(block)
            if checksum.hexdigest() != expected:
                raise ValueError(f"Dataset checksum mismatch: {datasets / relative}")
        # Use the same bind-mounted inode for Actions and direct Docker runs.
        with Path(args.lock).open("a") as lock:
            print(f"Waiting for GPU lock: {args.lock}", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            for case, row in zip(cases, rows):
                directory = output / case[0]
                directory.mkdir()
                row["status"] = "failed"
                print(f"Starting {case[0]}", flush=True)
                try:
                    run_case(case, args, directory, row)
                    row["status"] = "failed" if row["errors"] else "passed"
                except Exception as exc:
                    row["errors"].append(str(exc))
                    print(f"{case[0]} failed: {exc}", file=sys.stderr, flush=True)
                finally:
                    write_json(directory / "result.json", row)
                    summary(output, rows, args, errors)
    except Exception as exc:
        errors.append(str(exc))
        print(str(exc), file=sys.stderr, flush=True)
    except BaseException:
        errors.append("Evaluation interrupted")
        raise
    finally:
        result = summary(output, rows, args, errors)
        print(f"Report: {output / 'summary.md'}", flush=True)
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as stream:
                stream.write((output / "summary.md").read_text())
    return result


def hip_requirements(requirements):
    from packaging.requirements import Requirement

    pending, seen, selected = ["", "srt_hip"], set(), {}
    while pending:
        extra = pending.pop()
        if extra in seen:
            continue
        seen.add(extra)
        for raw in requirements:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate({"extra": extra}):
                continue
            if requirement.name.lower() == "sglang":
                pending.extend(requirement.extras - seen)
            else:
                selected[str(requirement)] = requirement
    return list(selected.values())


def install():
    """Install candidate sources over the pinned image's runtime, during Docker build."""
    pip = [sys.executable, "-m", "pip"]
    env = {**os.environ, "AITER_USE_SYSTEM_TRITON": "1", "PYTHONPATH": ""}

    def packages():
        return {
            p["name"].lower().replace("_", "-"): p["version"]
            for p in json.loads(command(pip + ["list", "--format=json"], env=env))
        }

    before = packages()
    managed = {"sglang", "sgl-kernel", "sglang-kernel", "amd-aiter", "pyhip"}
    subprocess.run(pip + ["uninstall", "-y", *sorted(managed)], check=True, env=env)
    for path in (Path("/ci/aiter"), Path("/ci/pyhip")):
        subprocess.run(
            pip + ["install", "--no-deps", "--no-build-isolation", "-e", str(path)],
            check=True,
            env=env,
        )
    subprocess.run(
        [sys.executable, "setup_rocm.py", "install"],
        cwd=ROOT / "python/sglang/kernels/aot",
        check=True,
        env=env,
    )
    shutil.copyfile(
        ROOT / "python/pyproject_other.toml", ROOT / "python/pyproject.toml"
    )
    subprocess.run(
        pip
        + ["install", "--no-deps", "--no-build-isolation", "-e", str(ROOT / "python")],
        check=True,
        env=env,
    )
    requirements = hip_requirements(
        json.loads(
            command(
                [
                    sys.executable,
                    "-c",
                    "import importlib.metadata as m, json; print(json.dumps(m.requires('sglang')))",
                ],
                env=env,
            )
        )
    )
    installed = packages()
    gaps = [
        r
        for r in requirements
        if r.name.lower().replace("_", "-") not in installed
        or not r.specifier.contains(
            installed[r.name.lower().replace("_", "-")], prereleases=True
        )
    ]
    replaced = {r.name.lower().replace("_", "-") for r in gaps}
    if replaced & {"numpy", "torch", "triton"}:
        raise RuntimeError(
            "Candidate requires a different GPU runtime; update the base image"
        )
    if gaps:
        constraints = Path("/ci/constraints.txt")
        constraints.write_text(
            "\n".join(f"{n}=={v}" for n, v in installed.items() if n not in replaced)
            + "\n"
        )
        subprocess.run(
            pip
            + [
                "install",
                "-c",
                str(constraints),
                *[f"{r.name}{r.specifier}" for r in gaps],
            ],
            check=True,
            env=env,
        )
    after = packages()
    if any(after.get(n) != v for n, v in before.items() if n not in managed | replaced):
        raise RuntimeError("Candidate installation changed the base runtime")
    for requirement in requirements:
        name = requirement.name.lower().replace("_", "-")
        if name not in after or not requirement.specifier.contains(
            after[name], prereleases=True
        ):
            raise RuntimeError(f"Unsatisfied HIP requirement: {requirement}")
    for name, path in (
        ("sglang", ROOT),
        ("aiter", Path("/ci/aiter")),
        ("pyhip", Path("/ci/pyhip")),
    ):
        origin = command(
            [
                sys.executable,
                "-c",
                f"import importlib.util; print(importlib.util.find_spec('{name}').origin)",
            ],
            env=env,
        )
        if not Path(origin).resolve().is_relative_to(path):
            raise RuntimeError(f"{name} imports outside the candidate checkout")
    subprocess.run(
        [sys.executable, "-c", "import sglang.launch_server"], check=True, env=env
    )
    subprocess.run(
        [sys.executable, "-c", "import sglang.test.run_eval"],
        check=True,
        env={**env, "PYTHONPATH": "/ci/client/python"},
    )
    write_json(
        "/ci/build.json",
        {
            "packages": after,
            "sources": {
                name: command(["git", "-C", str(path), "rev-parse", "HEAD"])
                for name, path in (
                    ("sglang", ROOT),
                    ("aiter", Path("/ci/aiter")),
                    ("pyhip", Path("/ci/pyhip")),
                    ("client", Path("/ci/client")),
                )
            },
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("run", "install"), default="run")
    parser.add_argument("--output", default="qwen4-results")
    parser.add_argument(
        "--case", default="", help="Comma-separated case IDs; empty runs all six"
    )
    parser.add_argument("--models-root", default="/models")
    parser.add_argument("--datasets-root")
    parser.add_argument("--client-root", default="/ci/client")
    parser.add_argument("--port", type=int, default=7080)
    parser.add_argument(
        "--gpus", default=os.environ.get("HIP_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    )
    parser.add_argument("--lock", default="/tmp/qwen4-mi308x.lock")
    parser.add_argument("--server-timeout", type=int, default=3600)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Diagnostic two-request/two-example run; no accuracy gate",
    )
    parser.add_argument(
        "--accuracy-only",
        action="store_true",
        help="Run the GSM8K gate without the performance benchmark",
    )
    args = parser.parse_args()
    devices = args.gpus.split(",")
    if any(not device.isdigit() for device in devices) or len(set(devices)) != len(
        devices
    ):
        parser.error("--gpus must contain distinct, comma-separated GPU indices")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit("Terminated"))
    try:
        if args.command == "install":
            install()
            return 0
        return run(args)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    sys.exit(main())
