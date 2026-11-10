#!/usr/bin/env python3
"""Run the CI based on Qwen evaluation scripts and enforce the recorded GSM8K gate."""

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
import time
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
EVALUATION = ROOT / "evaluation"
CLIENT_SHA = "fc6ddbbf7aac80167ea04cc00bf200af8adf2b85"
PRESSURE_BENCHMARK = "benchmark_example-20261013-363695ea/concurrency"
GSM8K_MAX_RELATIVE_DROP = 0.03
# R-Fork transfer groups listen on the seed; keep them outside the default
# ip_local_port_range (32768-60999) so an ephemeral socket cannot take one.
RFORK_GROUP_PORT_BASE = 62000
DATASETS = {
    "gsm8k/test.jsonl": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
    "ShareGPT_V3_unfiltered_cleaned_split.json": "35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4",
}
# Default order: Qwen3.8 first, then Qwen3.5.
# Case, model, TP, mode, launch script, approved 200-example full-02 accuracy.
# "rfork" runs a seed and an R-Fork instance in the same container: 2 x TP GPUs.
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
        "Qwen3.8-Flash-TP2-RFork",
        "Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16",
        2,
        "rfork",
        "launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh",
        0.980,
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


# Off the daily schedule, CI runs these only when selected explicitly: they take
# four GPUs and check startup paths rather than serving accuracy or speed.
DAILY_ONLY_CASES = {"Qwen3.8-Flash-TP2-RFork"}
# R-Fork validates startup and transferred weights; serving cases also test images.
MULTIMODAL_CASES = {case[0] for case in CASES if case[3] != "rfork"}


def matrix_cases(selected, event_name):
    """Case IDs for the Actions matrix, in CASES order."""
    ids = [case[0] for case in CASES]
    unknown = selected - set(ids)
    if unknown:
        raise ValueError(f"Unknown cases: {', '.join(sorted(unknown))}")
    if selected:
        return [name for name in ids if name in selected]
    return [
        name for name in ids if event_name == "schedule" or name not in DAILY_ONLY_CASES
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
            != {"base": None, "dflash": "DFLASH", "mtp": "EAGLE", "rfork": None}[
                case[3]
            ]
            or bool(args.get("disable_radix_cache")) != (case[3] != "dflash")
            or args.get("mm_feature_transport") != "cuda_ipc"
        ):
            raise ValueError(
                "Endpoint does not match the requested model, TP, mode, or CUDA IPC transport"
            )
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
            f"(reference {baseline:.2%}, maximum relative drop "
            f"{GSM8K_MAX_RELATIVE_DROP:.0%})"
        )


def validate_multimodal(path, model):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Multimodal response must be a JSON object")
    choices = data.get("choices")
    if (
        data.get("error")
        or data.get("model") != model
        or not isinstance(choices, list)
        or len(choices) != 1
    ):
        raise ValueError(
            "Multimodal response must contain one completion from the requested model"
        )
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise ValueError("Multimodal response must contain an assistant message")
    message = choice.get("message", {})
    answer = message.get("content")
    tokens = data.get("usage", {}).get("completion_tokens")
    image_tokens = (
        data.get("usage", {}).get("prompt_tokens_details", {}).get("image_tokens")
    )
    if (
        message.get("role") != "assistant"
        or not isinstance(answer, str)
        or not answer.strip()
        or message.get("refusal")
        or choice.get("finish_reason") not in ("stop", "length")
        or type(tokens) is not int
        or tokens <= 0
        or type(image_tokens) is not int
        or image_tokens <= 0
    ):
        raise ValueError(
            "Multimodal response has no valid assistant text or token usage"
        )
    return {
        "status": "passed",
        "completion_tokens": tokens,
        "image_tokens": image_tokens,
        "answer": answer,
    }


def check_rocjpeg_log(path):
    text = Path(path).read_text(errors="replace")
    if not re.search(r"rocJPEG GPU JPEG decoding: on\b", text):
        raise ValueError(
            "rocJPEG GPU decoding was not confirmed: expected "
            "'rocJPEG GPU JPEG decoding: on' in server.log"
        )
    if "rocJPEG batch decode failed" in text:
        raise ValueError("rocJPEG request batch fell back to PIL; see server.log")
    if not re.search(r"rocJPEG request batch decoded: [1-9]\d* image\(s\)", text):
        raise ValueError(
            "rocJPEG request batch decoding was not confirmed in server.log"
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


def check_server_log(path):
    ignored_imports = (
        "ignore import error when loading sglang.srt.models.",
        "ignore import error when loading sglang.srt.multimodal.processors.",
    )
    fault = re.compile(
        r"\b(?:[a-z]*error|traceback|"
        r"core[ _-]dump(?:ed)?|segmentation[ _-]fault|segfault|sigsegv|sigabrt|aborted|"
        r"illegal[ _-]memory[ _-]access|memory[ _-]access[ _-]fault|"
        r"misaligned[ _-]address|device[ _-]side[ _-]assert|gpu[ _-]fault|"
        r"hipError\w*|HSA_STATUS_ERROR\w*)\b",
        re.IGNORECASE,
    )
    errors = []
    for line in Path(path).read_text(errors="replace").splitlines():
        if any(prefix in line.lower() for prefix in ignored_imports):
            continue
        checked = line
        if "rocJPEG GPU JPEG decoding: on," in line:
            # The successful self-check reports a pixel-error metric. Ignore
            # that phrase only; any actual error/fault on the line still fails.
            checked = line.replace(
                "mean abs error vs PIL", "mean abs difference vs PIL"
            )
        if fault.search(checked):
            errors.append(line)
    if errors:
        raise RuntimeError(
            "Server log contains errors or GPU faults:\n" + "\n".join(errors[:10])
        )


def run_phase(script, directory, env, server, timeout):
    script = Path(script)
    log = directory / (script.name + ".log")
    print(f"Running {script}; log: {log}", flush=True)
    with managed_process(
        ["bash", str(EVALUATION / script)],
        log,
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


def select_gpus(env, devices, tp):
    """Return the first `tp` idle gfx942 GPUs among `devices`."""
    # A child owns all GPU inspection so the supervisor never reserves VRAM.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json, torch
assert torch.version.hip, 'ROCm PyTorch required'
idle = []
for i in range(torch.cuda.device_count()):
    assert torch.cuda.get_device_properties(i).gcnArchName.split(':')[0] == 'gfx942', 'MI308X/gfx942 required'
    free, total = torch.cuda.mem_get_info(i)
    if total - free < 1024**3:
        idle.append(i)
print(json.dumps(idle))
""",
        ],
        env={**env, "HIP_VISIBLE_DEVICES": ",".join(devices)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip().splitlines()[-1])
    # Child ordinals index into the HIP_VISIBLE_DEVICES list it was given.
    idle = [devices[i] for i in json.loads(result.stdout.strip().splitlines()[-1])]
    if len(idle) < tp:
        raise RuntimeError(
            f"Need {tp} idle GPUs among --gpus {','.join(devices)}; "
            f"idle: {','.join(idle) or 'none'}"
        )
    return idle[:tp]


def client_environment(args, model_path, served_name, port, directory):
    datasets = Path(
        args.datasets_root or Path(args.models_root) / "benchmark_datasets"
    ).resolve()
    env = clean_environment(
        PYTHONPATH=str(Path(args.client_root).resolve() / "python"),
        BENCH_MODEL=str(model_path),
        BENCH_SERVED_MODEL=served_name,
        BENCH_HOST="127.0.0.1",
        BENCH_PORT=str(port),
        DATASET_PATH=str(datasets / "ShareGPT_V3_unfiltered_cleaned_split.json"),
        BENCH_OUTPUT_FILE=str(directory / "performance.jsonl"),
        BENCH_LOG_FILE=os.devnull,  # Avoid a second copy of the phase log.
        GSM8K_DATA_PATH=str(datasets / "gsm8k/test.jsonl"),
        GSM8K_OUTPUT_DIR=str(directory),
        MM_OUTPUT_FILE=str(directory / "multimodal.json"),
        MM_IMAGE_URL=os.environ.get("MM_IMAGE_URL", ""),
    )
    if args.smoke:
        env.update(NUM_PROMPTS="2", WARMUP_REQUESTS="1", GSM8K_NUM_EXAMPLES="2")
    return env


def read_weight_checksums(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        kind, name, shape, dtype, _device, checksum = line.split("\t")
        rows[(kind, name)] = (shape, dtype, checksum)
    return rows


def compare_weight_checksums(seed_dir, fork_dir, tp):
    """Every rank's params and buffers must match the seed in shape, dtype and bytes."""
    mismatched = []
    for rank in range(tp):
        seed = read_weight_checksums(Path(seed_dir) / f"target_rank{rank}.tsv")
        fork = read_weight_checksums(Path(fork_dir) / f"target_rank{rank}.tsv")
        if not seed:
            raise ValueError(f"Seed rank {rank} wrote no weight checksums")
        differ = sorted(
            k for k in seed.keys() | fork.keys() if seed.get(k) != fork.get(k)
        )
        if differ:
            mismatched.append(
                f"rank {rank}: {len(differ)}/{len(seed)} differ (e.g. {differ[0][1]})"
            )
    if mismatched:
        raise ValueError(
            "R-Fork weights differ from the seed: " + "; ".join(mismatched)
        )
    return len(seed)


def run_rfork_case(case, args, directory, row):
    """Start a seed, then an R-Fork instance on other GPUs that pulls the seed's weights."""
    name, model, tp, _mode, script, baseline = case
    model_path = Path(args.models_root).resolve() / model
    if not (model_path / "config.json").is_file():
        raise ValueError(f"Missing model: {model_path}")
    devices = args.gpus.split(",")
    if len(devices) < 2 * tp:
        raise ValueError(
            f"{name} requires {2 * tp} GPUs; --gpus supplies {len(devices)}"
        )
    served_name = f"qwen4-{os.getpid()}-{name}"
    seed_port, fork_port = args.port, args.port + 1
    group_ports = list(range(RFORK_GROUP_PORT_BASE, RFORK_GROUP_PORT_BASE + tp))
    env = clean_environment(
        SERVER_PYTHON=sys.executable,
        HOST="127.0.0.1",
        MODEL_PATH=str(model_path),
        SERVED_MODEL_NAME=served_name,
        GPU_ARCHS="gfx942",
        TP_SIZE=str(tp),
        MTP="0",
        PLE_OFFLOAD_EMBEDDING="0",
        LOG_FILE=os.devnull,  # managed_process captures the complete server log.
    )
    gpus = select_gpus(env, devices, 2 * tp)
    seed_env = {
        **env,
        "PORT": str(seed_port),
        "HIP_VISIBLE_DEVICES": ",".join(gpus[:tp]),
        "SGLANG_DEBUG_WEIGHT_CHECKSUM_DIR": str(directory / "checksums-seed"),
    }
    fork_env = {
        **env,
        "PORT": str(fork_port),
        "HIP_VISIBLE_DEVICES": ",".join(gpus[tp:]),
        "SGLANG_DEBUG_WEIGHT_CHECKSUM_DIR": str(directory / "checksums-fork"),
    }
    fork_args = [
        "--load-format",
        "remote_instance",
        "--remote-instance-weight-loader-backend",
        "nccl",
        "--remote-instance-weight-loader-seed-instance-ip",
        "127.0.0.1",
        "--remote-instance-weight-loader-seed-instance-service-port",
        str(seed_port),
        "--remote-instance-weight-loader-send-weights-group-ports",
        json.dumps(group_ports),
    ]
    print(
        f"{name}: seed on GPUs {seed_env['HIP_VISIBLE_DEVICES']}, "
        f"R-Fork instance on GPUs {fork_env['HIP_VISIBLE_DEVICES']}",
        flush=True,
    )
    for port in (seed_port, fork_port, *group_ports):
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
    write_json(
        directory / "launch.json",
        {
            "script": script,
            "seed_env": seed_env,
            "fork_env": fork_env,
            "fork_args": fork_args,
        },
    )
    argv = ["bash", str(EVALUATION / script)]
    with managed_process(argv, directory / "seed.log", seed_env, directory) as seed:
        print(
            f"{name}: waiting for the seed; log: {directory / 'seed.log'}", flush=True
        )
        wait_ready(
            seed,
            f"http://127.0.0.1:{seed_port}",
            case,
            model_path,
            served_name,
            args.server_timeout,
        )
        started = time.monotonic()
        with managed_process(
            argv + fork_args, directory / "server.log", fork_env, directory
        ) as fork:
            print(
                f"{name}: waiting for the R-Fork instance; log: {directory / 'server.log'}",
                flush=True,
            )
            info = wait_ready(
                fork,
                f"http://127.0.0.1:{fork_port}",
                case,
                model_path,
                served_name,
                args.server_timeout,
            )
            startup = time.monotonic() - started
            write_json(directory / "server-info.json", info)
            log = (directory / "server.log").read_text(errors="replace")
            if "Loading weights from remote instance" not in log:
                raise ValueError("The new instance did not load its weights via R-Fork")
            load_weight = re.findall(r"load_weight=([0-9.]+)", log)
            row["rfork"] = {
                "startup_s": round(startup, 1),
                "load_weight_s": float(load_weight[-1]) if load_weight else None,
                "tensors_per_rank": compare_weight_checksums(
                    directory / "checksums-seed", directory / "checksums-fork", tp
                ),
            }
            if seed.poll() is not None:
                raise RuntimeError("Seed exited during the weight transfer")
            if not getattr(args, "pressure", False):
                try:
                    run_phase(
                        "run_gsm8k.sh",
                        directory,
                        client_environment(
                            args, model_path, served_name, fork_port, directory
                        ),
                        fork,
                        7200,
                    )
                    row["accuracy"] = validate_accuracy(
                        directory / "gsm8k.json", args.smoke
                    )
                    check_accuracy_gate(row["accuracy"]["score"], baseline, args.smoke)
                except Exception as exc:
                    row["errors"].append(f"accuracy: {exc}")
    if getattr(args, "pressure", False):
        # Inspect both complete logs after shutdown, like the other pressure cases.
        check_server_log(directory / "seed.log")
        check_server_log(directory / "server.log")


def run_case(case, args, directory, row):
    name, model, tp, mode, script, baseline = case
    if mode == "rfork":
        return run_rfork_case(case, args, directory, row)
    model_path = Path(args.models_root).resolve() / model
    if not (model_path / "config.json").is_file():
        raise ValueError(f"Missing model: {model_path}")
    devices = args.gpus.split(",")
    if len(devices) < tp:
        raise ValueError(f"{name} requires {tp} GPUs; --gpus supplies {len(devices)}")
    pressure = getattr(args, "pressure", False)
    pressure_dir = directory / "pressure"
    pressure_source = pressure_dir / ".benchmark"
    if pressure:
        shutil.copytree(
            Path(args.models_root).resolve() / PRESSURE_BENCHMARK,
            pressure_source,
            ignore=shutil.ignore_patterns("__pycache__", "results", "*.log"),
        )
    served_name = f"qwen4-{os.getpid()}-{name}"
    env = clean_environment(
        SERVER_PYTHON=sys.executable,
        HOST="127.0.0.1",
        PORT=str(args.port),
        MODEL_PATH=str(model_path),
        SERVED_MODEL_NAME=served_name,
        GPU_ARCHS="gfx942",
        TP_SIZE=str(tp),
        MTP="1" if mode == "mtp" else "0",
        PLE_OFFLOAD_EMBEDDING="0",
        LOG_FILE=os.devnull,  # managed_process captures the complete server log.
        SGLANG_ROCM_GPU_JPEG_DECODE="1",
        LIBVA_DRIVERS_PATH=os.environ.get("LIBVA_DRIVERS_PATH")
        or f"{os.environ.get('ROCM_PATH') or '/opt/rocm'}/lib/rocm_sysdeps/lib",
        LIBVA_DRIVER_NAME=os.environ.get("LIBVA_DRIVER_NAME") or "radeonsi",
    )
    if mode == "dflash":
        env["DRAFT_MODEL_PATH"] = str(
            model_path.with_name(model.replace("PTPC-FP8", "DFlash"))
        )
        if not (Path(env["DRAFT_MODEL_PATH"]) / "config.json").is_file():
            raise ValueError(f"Missing draft model: {env['DRAFT_MODEL_PATH']}")
    env["HIP_VISIBLE_DEVICES"] = ",".join(select_gpus(env, devices, tp))
    print(f"{name}: using GPUs {env['HIP_VISIBLE_DEVICES']}", flush=True)
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
        env = client_environment(args, model_path, served_name, args.port, directory)
        if pressure:
            concurrents = [1] if args.smoke else [1, 2, 4, 8, 12, 16, 32]
            dataset = f"tk_in:{12000 if q38 else 8000},tk_out:{output_tokens}"
            dataset += ",conc:" + "-".join(map(str, concurrents))
            if args.smoke:
                dataset += ",prm:2"
            env.update(
                LLM_API_MODEL=served_name,
                LLM_LOCAL_TOKENIZER_PATH=str(model_path),
                LLM_INFER_FRAMEWORK="sglang",
                LLM_BENCH_BACKEND="sglang",
                LLM_API_URL=f"http://127.0.0.1:{args.port}/generate",
                LLM_AUTO_SEQ_SIGMA="4",
                LLM_AUTO_DATASET_CONF=dataset,
                LLM_AUTO_PROMPT_FACTOR="2",
                LLM_REPOST_PROMPT_CNT_PER_THREAD="0",
                LLM_LOOP_FOREVER="False",
                LLM_WAIT_IF_TIMEOUT="False",
                LLM_TIMEOUT="120",
                LLM_DISABLE_REPORT="True",
                LLM_ENABLE_REPORT="False",
                LLM_ENABLE_UPLOAD_PROMPT="False",
                LLM_AVG_FIRST_TOKEN_TIMEOUT="20000",
                LLM_AVG_STEP_LATENCY_TIMEOUT="100",
            )
        if pressure:
            phases = [("pressure", str(pressure_source / "run.sh"), 3300)]
        else:
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
        # Reuse the same model process, after the existing text workloads.
        phases.append(("multimodal", "run_multimodal_test.sh", 240))
        for phase, script, timeout in phases:
            if phase == "multimodal":
                row[phase] = {"status": "failed"}
            try:
                run_phase(script, directory, env, server, timeout)
                if phase == "accuracy":
                    row[phase] = validate_accuracy(directory / "gsm8k.json", args.smoke)
                    check_accuracy_gate(row[phase]["score"], baseline, args.smoke)
                elif phase == "multimodal":
                    row[phase] = validate_multimodal(
                        directory / "multimodal.json", served_name
                    )
                elif phase == "pressure":
                    log = (directory / "run.sh.log").read_text(errors="replace")
                    failures = [
                        line
                        for line in log.splitlines()
                        if re.search(r"\bError:|Traceback", line)
                    ]
                    if failures:
                        raise ValueError(
                            "Concurrency request errors:\n" + "\n".join(failures[:10])
                        )
                    completed = [
                        int(value)
                        for value in re.findall(r"Concurrent Nums: (\d+)", log)
                    ]
                    error_rates = re.findall(r"Error Requests: ([0-9.]+)%", log)
                    if (
                        completed != concurrents
                        or len(error_rates) != len(concurrents)
                        or any(float(rate) != 0 for rate in error_rates)
                    ):
                        raise ValueError(
                            f"Incomplete concurrency pressure run: completed levels {completed}"
                        )
                else:
                    row[phase] = validate_performance(
                        directory / "performance.jsonl",
                        2 if args.smoke else 128,
                        output_tokens,
                        speculative=mode != "base",
                    )
            except Exception as exc:
                row["errors"].append(f"{phase}: {exc}")
    # managed_process has stopped the server and drained its log relay here.
    # A successful image response alone could have used the PIL fallback.
    if row.get("multimodal", {}).get("status") == "passed":
        try:
            check_rocjpeg_log(directory / "server.log")
            row["multimodal"]["rocjpeg"] = "on"
        except (OSError, ValueError) as exc:
            row["multimodal"]["status"] = "failed"
            row["errors"].append(f"multimodal: {exc}")
    if pressure:
        # Inspect the complete log after shutdown, including teardown failures.
        check_server_log(directory / "server.log")


def multimodal_status(row):
    if row["case"] not in MULTIMODAL_CASES:
        return "not applicable"
    return row.get("multimodal", {}).get("status", "not run")


def summary(output, rows, args, errors):
    pressure_only = getattr(args, "pressure", False)
    failed = bool(errors or any(row["status"] != "passed" for row in rows))
    title = "Qwen4 daily pressure" if pressure_only else "Qwen4 evaluation"
    text = f"## {title} — {'Failed' if failed else 'Passed'}\n\n"
    text += f"Cases passed: **{sum(row['status'] == 'passed' for row in rows)}/{len(rows)}**. "
    text += f"Coverage: {len(rows)}/{len(CASES)} selected cases. "
    text += (
        "Multimodal: serving cases require assistant text, rocJPEG activation and "
        "successful request-batch decoding; R-Fork checks startup and transferred weights.\n\n"
    )
    if pressure_only:
        text += "Pressure: serving cases require request completion, a multimodal response, and final server-log checks.\n\n"
        if args.smoke:
            text += "Smoke is diagnostic: one two-request pressure round.\n\n"
        text += "| Case | Status | Multimodal |\n|---|---|---|\n"
        for row in rows:
            status = "passed" if row["status"] == "passed" else "failed"
            mm_status = multimodal_status(row)
            text += f"| {row['case']} | {status} | {mm_status} |\n"
    else:
        text += (
            "Smoke is diagnostic: two examples; accuracy gate is disabled.\n\n"
            if args.smoke
            else (
                "GSM8K: 200 examples; maximum "
                f"{GSM8K_MAX_RELATIVE_DROP:.0%} relative drop from recorded references.\n\n"
            )
        )
    if not pressure_only:
        text += "Performance thresholds are advisory.\n\n"
        text += "| Case | Status | GSM8K | Reference / minimum | Output tokens/s | QPS | TTFT (ms) | TPOT (ms) | Accept length | Multimodal |\n"
        text += "|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n"
        for row in rows:
            accuracy, perf = row.get("accuracy", {}), row.get("performance", {})
            score = f"{accuracy['score']:.2%}" if accuracy else "—"
            minimum = row["baseline"] * (1 - GSM8K_MAX_RELATIVE_DROP)
            metrics = [
                f"{perf[key]:.3f}" if perf.get(key) is not None else "—"
                for key in (
                    "output_throughput",
                    "request_throughput",
                    "mean_ttft_ms",
                    "mean_tpot_ms",
                    "accept_length",
                )
            ]
            text += (
                f"| {row['case']} | {row['status']} | {score} | {row['baseline']:.2%} / {minimum:.2%} | "
                + " | ".join(metrics)
                + f" | {multimodal_status(row)} |\n"
            )
    for row in rows:
        rfork = row.get("rfork")
        if rfork:
            text += (
                f"\n{row['case']}: R-Fork instance ready in {rfork['startup_s']:.1f} s "
                f"(load_weight {rfork['load_weight_s']} s); "
                f"{rfork['tensors_per_rank']} tensors per rank match the seed.\n"
            )
    for error in errors + [
        f"{row['case']}: {e}" for row in rows for e in row["errors"]
    ]:
        text += "\n- " + str(error).replace("\n", " ").replace("|", "\\|")
    write_json(
        output / "summary.json",
        {
            "failed": failed,
            "smoke": args.smoke,
            "pressure_only": pressure_only,
            "errors": errors,
            "cases": rows,
        },
    )
    (output / "summary.md").write_text(text + "\n")
    return int(failed)


def summarize_ci():
    """Combine the latest GitHub case attempts and enforce the final CI gate."""
    selected = json.loads(os.environ.get("CASE_MATRIX") or "{}").get("case")
    if selected is None:
        selected = [
            name.strip()
            for name in os.environ.get("CASE_FILTER", "").split(",")
            if name.strip()
        ]
    root = Path(os.environ["RUNNER_TEMP"]) / "qwen4-case-results"
    pages = json.loads(
        command(
            [
                "gh",
                "api",
                "--paginate",
                "--slurp",
                f"repos/{os.environ['GITHUB_REPOSITORY']}/actions/runs/"
                f"{os.environ['GITHUB_RUN_ID']}/jobs?filter=all&per_page=100",
            ]
        )
    )
    jobs, attempts = {}, {}
    for job in sorted(
        (job for page in pages for job in page["jobs"]),
        key=lambda job: job["run_attempt"],
    ):
        name = job["name"].rsplit(" / ", 1)[-1]
        jobs[name] = job
        # GitHub copies completed jobs into reruns with a new run_attempt.
        # Their original start time identifies the attempt that uploaded results.
        if job.get("started_at"):
            attempts.setdefault((name, job["started_at"]), job["run_attempt"])
    rows = []
    for case in CASES:
        name, baseline = case[0], case[5]
        if selected and name not in selected:
            continue
        row = {"case": name, "baseline": baseline, "status": "not run", "errors": []}
        job_name = f"Qwen4 ({name})"
        job = jobs.get(job_name, {})
        attempt = attempts.get(
            (job_name, job.get("started_at")), job.get("run_attempt", 0)
        )
        artifact = root / f"qwen4-results-{name}-{attempt}"
        try:
            report = json.loads((artifact / "evaluation/summary.json").read_text())
            result = next(row for row in report["cases"] if row["case"] == name)
            result["baseline"] = baseline
            result["errors"].extend(report["errors"])
            if report["failed"]:
                result["status"] = "failed"
            if name in MULTIMODAL_CASES:
                if result.get("multimodal", {}).get("status") != "passed":
                    result["status"] = "failed"
                    result["errors"].append("Missing or failed multimodal result")
                elif result["multimodal"].get("rocjpeg") != "on":
                    result["status"] = "failed"
                    result["errors"].append("Missing rocJPEG activation confirmation")
            row = result
        except Exception as exc:
            row["status"] = "failed"
            row["errors"].append(f"{artifact.name}: {exc}")
        conclusion = job.get("conclusion")
        if conclusion != "success":
            row["status"] = (
                "failed" if conclusion == "failure" else conclusion or "not run"
            )
            row["errors"].append(f"GitHub case job: {conclusion or 'not run'}")
        rows.append(row)
    errors = [
        f"{name}: {os.environ[key]}"
        for name, key in (
            ("CPU validation", "VALIDATION_RESULT"),
            ("GPU jobs", "CASES_RESULT"),
        )
        if os.environ[key] != "success"
    ]
    if os.environ["CASES_RESULT"] == "skipped":
        errors.append(
            "Qwen4 GPU cases did not run. Fork PRs require the run-qwen4-ci label."
        )
    output = root.parent / "qwen4-summary"
    output.mkdir(exist_ok=True)
    result = summary(
        output,
        rows,
        argparse.Namespace(
            smoke=os.environ.get("SMOKE") == "true",
            pressure=os.environ.get("DAILY_PRESSURE") == "true",
        ),
        errors,
    )
    text = (output / "summary.md").read_text()
    print(text, flush=True)
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as stream:
        stream.write(text)
    return result


def run(args):
    pressure = getattr(args, "pressure", False)
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
        if not pressure and (
            command(["git", "-C", str(client), "rev-parse", "HEAD"]) != CLIENT_SHA
            or command(
                [
                    "git",
                    "-C",
                    str(client),
                    "status",
                    "--porcelain",
                    "--",
                    "python/sglang",
                ]
            )
        ):
            raise ValueError(f"Evaluator must be a clean checkout at {CLIENT_SHA}")
        datasets = Path(
            args.datasets_root or Path(args.models_root) / "benchmark_datasets"
        )
        for relative, expected in DATASETS.items():
            if pressure or (args.accuracy_only and relative != "gsm8k/test.jsonl"):
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
    parser.add_argument(
        "command", nargs="?", choices=("run", "install", "summarize"), default="run"
    )
    parser.add_argument("--output", default="qwen4-results")
    parser.add_argument(
        "--case", default="", help="Comma-separated case IDs; empty runs all cases"
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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--accuracy-only",
        action="store_true",
        help="Run GSM8K and multimodal checks without the performance benchmark",
    )
    mode.add_argument(
        "--pressure",
        action="store_true",
        help="Run daily pressure and multimodal checks, then check the final server log",
    )
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit("Terminated"))
    try:
        if args.command == "summarize":
            return summarize_ci()
        devices = args.gpus.split(",")
        if any(not device.isdigit() for device in devices) or len(set(devices)) != len(
            devices
        ):
            parser.error("--gpus must contain distinct, comma-separated GPU indices")
        if args.command == "install":
            install()
            return 0
        return run(args)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    sys.exit(main())
