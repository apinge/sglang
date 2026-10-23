"""Evaluation inside an owned container (or explicit existing-container local mode)."""

import os
import socket
import sys
import time
from pathlib import Path

from common import clean_environment, command_output, utc_now, write_json
from config import EVALUATION_ROOT, server_command, server_environment
from process import (
    ManagedProcess,
    port_available,
    smoke_generation,
    wait_port_released,
    wait_ready,
)


def client_env(client_root):
    env = clean_environment({"HIP_VISIBLE_DEVICES": "0", "OPENAI_API_KEY": "EMPTY"})
    if client_root:
        env["PYTHONPATH"] = str(Path(client_root) / "python")
    return env


def verify_client(plan, client_root, client_python):
    if not client_root:
        raise ValueError("A pinned client checkout is required")
    expected = plan["sources"].get("client", {}).get("sha")
    actual = command_output(["git", "-C", str(client_root), "rev-parse", "HEAD"])
    if actual != expected:
        raise ValueError(f"Client SHA {actual} differs from planned {expected}")
    # A dirty scorer defeats the SHA pin even if editable install metadata agrees.
    changes = command_output(
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
    )
    if changes:
        raise ValueError("Pinned client benchmark/scorer sources have local changes")
    module = command_output(
        [client_python, "-c", "import sglang; print(sglang.__file__)"],
        env=client_env(client_root),
    )
    if (
        not Path(module)
        .resolve()
        .is_relative_to((Path(client_root) / "python").resolve())
    ):
        raise ValueError(f"Client imports unintended SGLang module: {module}")
    return {"sha": actual, "module": module, "python": client_python}


def perf_command(plan, case, group, output, models_root, sharegpt, python):
    name = case["benchmark_script"]
    if plan["workload"].get("profiler"):
        name = name.replace("_bench.sh", "_profile.sh")
    return ["bash", str(EVALUATION_ROOT / name)]


def accuracy_command(plan, case, output, data, python):
    return ["bash", str(EVALUATION_ROOT / "run_gsm8k.sh")]


def evaluation_environment(
    plan, case, group, output, models_root, gsm8k, sharegpt, client_root, python
):
    w, d = plan["workload"], plan["defaults"]
    return {
        **client_env(client_root),
        "CLIENT_PYTHON": python,
        "BENCH_MODEL": str(Path(models_root) / group["directory"]),
        "BENCH_SERVED_MODEL": case.get("served_model_name", case["id"]),
        "BENCH_HOST": "127.0.0.1",
        "BENCH_PORT": str(plan["port"]),
        "INPUT_TOKENS": str(case["input_tokens"]),
        "OUTPUT_TOKENS": str(w.get("output_tokens", case["output_tokens"])),
        "NUM_PROMPTS": str(w["prompts"]),
        "WARMUP_REQUESTS": str(w["warmups"]),
        "MAX_CONCURRENCY": str(d["concurrency"]),
        "BENCH_SEED": str(d["seed"]),
        "DATASET_PATH": str(sharegpt),
        "BENCH_OUTPUT_FILE": str(output / "performance.jsonl"),
        "BENCH_LOG_FILE": str(output / "benchmark-script.log"),
        "PROFILE_LOG_FILE": str(output / "profile-script.log"),
        "GSM8K_DATA_PATH": str(gsm8k),
        "GSM8K_OUTPUT_DIR": str(output),
        "GSM8K_NUM_EXAMPLES": str(w["examples"]),
        "GSM8K_NUM_SHOTS": str(d["gsm8k_shots"]),
        "GSM8K_MAX_TOKENS": str(d["gsm8k_max_tokens"]),
        "GSM8K_NUM_THREADS": str(d["gsm8k_threads"]),
    }


def run_case(
    plan,
    group,
    case,
    output,
    models_root,
    gsm8k,
    sharegpt,
    server_python=sys.executable,
    client_python=sys.executable,
    client_root=None,
    process_factory=ManagedProcess,
    readiness=wait_ready,
    smoke=smoke_generation,
):
    output = Path(output)
    directory = output / group["id"] / case["id"]
    if directory.exists():
        raise ValueError(f"Refusing to reuse case output directory: {directory}")
    directory.mkdir(parents=True)
    state = {
        "plan_hash": plan["plan_hash"],
        "case_id": case["id"],
        "status": "running",
        "started_at": utc_now(),
        "errors": [],
        "repeats": [],
        "container_id": os.environ.get(
            "QWEN4_CONTAINER_ID", f"existing:{socket.gethostname()}"
        ),
    }
    state_path = directory / "execution.json"
    write_json(state_path, state)
    for repeat in range(1, plan["workload"]["repeats"] + 1):
        case = {**case, "served_model_name": f"{plan['run_id']}-{case['id']}-r{repeat}"}
        run_dir = directory / f"repeat-{repeat}"
        run_dir.mkdir()
        process = None
        phase = "startup"
        detail = {"repeat": repeat}
        started = time.monotonic()
        try:
            if not port_available(plan["port"]):
                raise RuntimeError(
                    f"Port {plan['port']} already belongs to another process"
                )
            env = clean_environment(
                server_environment(
                    case, group, plan["port"], models_root, server_python
                )
            )
            env["LOG_FILE"] = str(run_dir / "launch-script.log")
            env["HIP_VISIBLE_DEVICES"] = ",".join(str(x) for x in range(case["tp"]))
            env.update(
                {
                    "TRITON_CACHE_DIR": str(run_dir / "cache/triton"),
                    "XDG_CACHE_HOME": str(run_dir / "cache/xdg"),
                    "TORCH_EXTENSIONS_DIR": str(run_dir / "cache/torch"),
                    "AITER_JIT_DIR": str(run_dir / "cache/aiter"),
                    "AITER_ROOT_DIR": str(run_dir / "cache/aiter-templates"),
                    "PYHIP_CACHE_DIR": str(run_dir / "cache/pyhip"),
                    "GPU_ARCHS": "gfx942",
                }
            )
            for key in [
                "TRITON_CACHE_DIR",
                "XDG_CACHE_HOME",
                "TORCH_EXTENSIONS_DIR",
                "AITER_JIT_DIR",
                "AITER_ROOT_DIR",
                "PYHIP_CACHE_DIR",
            ]:
                Path(env[key]).mkdir(parents=True, exist_ok=True)
            if plan["workload"].get("profiler"):
                env.update(
                    {
                        "SGLANG_TORCH_PROFILER_DIR": str(run_dir / "profile"),
                        "SGLANG_PROFILE_WITH_STACK": "1",
                        "SGLANG_PROFILE_RECORD_SHAPES": "1",
                    }
                )
            argv = server_command(case, group, plan["port"], models_root, server_python)
            write_json(
                run_dir / "resolved-case.json",
                {
                    "argv": argv,
                    "env": env,
                    "case_id": case["id"],
                    "group_id": group["id"],
                },
            )
            process = process_factory(
                argv, run_dir / "server.log", env=env, cwd=run_dir
            )
            detail["server_pid"] = process.process.pid
            info = readiness(
                process,
                f"http://127.0.0.1:{plan['port']}",
                case,
                str(Path(models_root) / group["directory"]),
                plan["defaults"]["timeouts"]["startup"],
            )
            write_json(run_dir / "server-info.json", info)
            write_json(
                run_dir / "smoke.json",
                smoke(f"http://127.0.0.1:{plan['port']}", case["served_model_name"]),
            )
            detail["startup_s"] = time.monotonic() - started
            for phase in ["performance", "accuracy"]:
                if not plan["workload"][phase]:
                    continue
                if not process.alive():
                    state["errors"].append(
                        f"repeat {repeat}: server exited before {phase}"
                    )
                    detail[phase] = "not_run"
                    continue
                command = (
                    perf_command(
                        plan, case, group, run_dir, models_root, sharegpt, client_python
                    )
                    if phase == "performance"
                    else accuracy_command(plan, case, run_dir, gsm8k, client_python)
                )
                environment = evaluation_environment(
                    plan,
                    case,
                    group,
                    run_dir,
                    models_root,
                    gsm8k,
                    sharegpt,
                    client_root,
                    client_python,
                )
                write_json(
                    run_dir / f"{phase}-command.json",
                    {"argv": command, "env": environment},
                )
                state["current_phase"] = phase
                write_json(state_path, state)
                phase_started = time.monotonic()
                child = process_factory(
                    command,
                    run_dir / f"{phase}.log",
                    env=environment,
                    cwd=run_dir,
                )
                try:
                    code = child.wait(plan["defaults"]["timeouts"][phase])
                    detail[phase] = code
                    if code:
                        state["errors"].append(
                            f"repeat {repeat}: {phase} exited with {code}"
                        )
                except Exception as exc:
                    state["errors"].append(f"repeat {repeat}: {phase}: {exc}")
                    detail[phase] = "error"
                finally:
                    child.stop()
                    detail[f"{phase}_s"] = time.monotonic() - phase_started
        except Exception as exc:
            state["errors"].append(
                f"repeat {repeat}: {phase}: {type(exc).__name__}: {exc}"
            )
        except BaseException:
            state["status"] = "cancelled"
            raise
        finally:
            cleanup_started = time.monotonic()
            cleanup_error = None
            try:
                if process:
                    process.stop(plan["defaults"]["timeouts"]["stop"])
                    wait_port_released(plan["port"])
            except Exception as exc:
                state["errors"].append(f"repeat {repeat}: cleanup: {exc}")
                cleanup_error = exc
            detail["cleanup_s"] = time.monotonic() - cleanup_started
            detail["total_s"] = time.monotonic() - started
            state["repeats"].append(detail)
            if state["status"] != "cancelled":
                state["status"] = "eval_error" if state["errors"] else "completed"
            state["finished_at"] = utc_now()
            state["current_phase"] = "finished"
            write_json(state_path, state)
            if cleanup_error:
                raise cleanup_error
    return state


def run_group(
    plan,
    group_id,
    output,
    models_root,
    gsm8k,
    sharegpt,
    server_python=sys.executable,
    client_python=sys.executable,
    client_root=None,
    check_hardware=True,
):
    from hardware import inspect_hardware, wait_gpu_idle

    group = next(g for g in plan["groups"] if g["id"] == group_id)
    verify_client(plan, client_root, client_python)
    for case in group["cases"]:
        if check_hardware:
            inspect_hardware(group["gpus"])
        run_case(
            plan,
            group,
            case,
            output,
            models_root,
            gsm8k,
            sharegpt,
            server_python,
            client_python,
            client_root,
        )
        if check_hardware:
            wait_gpu_idle(group["gpus"])
    if check_hardware:
        # One final probe before letting another target container start.
        inspect_hardware(group["gpus"])
