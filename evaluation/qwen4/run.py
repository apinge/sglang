"""Qwen4 CI and local entrypoint. Use --help for the reproducible run interface."""

import argparse
import json
import os
import signal
import socket
import sys
from pathlib import Path

from common import ROOT, command_output, read_json, write_json
from config import make_plan, resolve_sources, validate_plan
from process import host_lock

from assets import inventory


def interrupt(signum, frame):
    raise KeyboardInterrupt(f"Signal {signum}")


def local_sources():
    pins = read_json(ROOT / "dependencies.lock.json")
    result = {}
    for name, path in {
        "sglang": ROOT.parents[1],
        "aiter": Path("/opt/aiter"),
        "pyhip": Path("/opt/pyhip"),
    }.items():
        sha = command_output(["git", "-C", str(path), "rev-parse", "HEAD"])
        result[name] = {
            "repo": pins[f"{name}_repo"],
            "ref": "local-checkout",
            "sha": sha,
        }
    result["client"] = {
        "repo": pins["client_repo"],
        "ref": pins["client_sha"],
        "sha": pins["client_sha"],
    }
    return result


def execute(args, plan):
    from report import aggregate

    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "Execution needs a fresh output directory; use report for existing results"
        )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", plan)
    runtime = {
        "backend": args.backend,
        "host": socket.gethostname(),
        "base_image_digest": None,
        "attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "local"),
    }
    write_json(output / "runtime.json", runtime)
    backend = None
    try:
        with host_lock(args.lock, plan["defaults"]["timeouts"]["lock"]):
            try:
                inventory(
                    plan,
                    args.models_root,
                    args.gsm8k,
                    args.sharegpt,
                    output / "assets.json",
                )
                if args.backend == "docker":
                    from docker_backend import DockerBackend

                    if set(plan["sources"]) != {"sglang", "aiter", "pyhip", "client"}:
                        raise ValueError(
                            "Docker execution requires resolved source SHAs"
                        )
                    try:
                        gsm8k_relative = (
                            Path(args.gsm8k)
                            .resolve()
                            .relative_to(Path(args.models_root).resolve())
                        )
                        sharegpt_relative = (
                            Path(args.sharegpt)
                            .resolve()
                            .relative_to(Path(args.models_root).resolve())
                        )
                    except ValueError as exc:
                        raise ValueError(
                            "Docker datasets must be staged within --models-root"
                        ) from exc
                    backend = DockerBackend(
                        plan, output, args.models_root, image_lock=args.image_lock
                    )
                    backend.build()
                    for group in plan["groups"]:
                        # Failures of an individual case are recorded by the group
                        # runner. A group infrastructure/cleanup failure stops reuse.
                        backend.evaluate_group(
                            group,
                            gsm8k_relative,
                            sharegpt_relative,
                        )
                else:
                    from hardware import inspect_hardware
                    from runner import run_group

                    runtime.update(
                        inspect_hardware(max(g["gpus"] for g in plan["groups"]))
                    )
                    runtime["container_isolation_verified"] = False
                    runtime["note"] = (
                        "Local validation in the existing target-image container; no image was built and no distinct target containers were created."
                    )
                    write_json(output / "runtime.json", runtime)
                    for group in plan["groups"]:
                        write_json(
                            output / group["id"] / "container.json",
                            {
                                "container_id": f"existing:{socket.gethostname()}",
                                "target": group["directory"],
                                "backend": "existing",
                                "container_isolation_verified": False,
                            },
                        )
                        run_group(
                            plan,
                            group["id"],
                            output,
                            args.models_root,
                            args.gsm8k,
                            args.sharegpt,
                            client_python=args.client_python,
                            client_root=args.client_root,
                        )
            finally:
                if backend:
                    backend.cleanup()
    except BaseException as exc:
        runtime = read_json(output / "runtime.json")
        runtime["failure"] = f"{type(exc).__name__}: {exc}"
        write_json(output / "runtime.json", runtime)
        print(runtime["failure"], file=sys.stderr)
    finally:
        summary = aggregate(plan, output, args.baseline)
    print(f"Results: {output / 'summary.md'}")
    return int(summary["failed"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan")
    plan_parser.add_argument(
        "--profile", default="full", choices=["full", "smoke", "calibration", "profile"]
    )
    plan_parser.add_argument(
        "--cases", help="JSON array of case IDs; full/calibration require all six"
    )
    plan_parser.add_argument(
        "--origin", default="integration", choices=["integration", "sglang", "aiter"]
    )
    plan_parser.add_argument("--candidate-repo")
    plan_parser.add_argument("--candidate-ref")
    plan_parser.add_argument("--sglang-ref")
    plan_parser.add_argument("--aiter-ref")
    plan_parser.add_argument("--pyhip-ref")
    plan_parser.add_argument("--base-image")
    plan_parser.add_argument(
        "--gate-mode", default="observe", choices=["observe", "enforce"]
    )
    plan_parser.add_argument("--examples", type=int)
    plan_parser.add_argument("--run-id")
    plan_parser.add_argument("--port", type=int, default=7080)
    plan_parser.add_argument(
        "--sources", choices=["remote", "local", "none"], default="remote"
    )
    plan_parser.add_argument("--output", required=True)
    for name in ["execute", "group"]:
        sub = commands.add_parser(name)
        sub.add_argument("--plan", required=True)
        sub.add_argument("--output", required=True)
        sub.add_argument(
            "--models-root", default="/models" if name == "group" else "/raid/models"
        )
        sub.add_argument("--gsm8k", required=True)
        sub.add_argument("--sharegpt", required=True)
        sub.add_argument("--client-root", default="/opt/sglang")
        sub.add_argument("--client-python", default=sys.executable)
        if name == "execute":
            sub.add_argument(
                "--backend", default="docker", choices=["docker", "existing"]
            )
            sub.add_argument("--lock", default="/var/lock/qwen4-mi308x.lock")
            sub.add_argument("--baseline")
            sub.add_argument(
                "--image-lock",
                help="Immutable base-image lock from the original attempt",
            )
        else:
            sub.add_argument("--group", required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupt)
    if args.command == "plan":
        sources = (
            local_sources()
            if args.sources == "local"
            else resolve_sources(
                args.origin,
                args.candidate_repo,
                args.candidate_ref,
                {
                    "sglang": args.sglang_ref,
                    "aiter": args.aiter_ref,
                    "pyhip": args.pyhip_ref,
                },
            )
            if args.sources == "remote"
            else {}
        )
        plan = make_plan(
            args.profile,
            json.loads(args.cases) if args.cases else None,
            sources,
            args.run_id,
            args.examples,
            args.base_image,
            args.gate_mode,
            args.port,
        )
        write_json(args.output, plan)
        print(f"Prepared {len(plan['expected_cases'])} cases: {plan['plan_hash']}")
        return 0
    plan = read_json(args.plan)
    validate_plan(plan)
    if args.command == "execute":
        return execute(args, plan)
    from hardware import inspect_hardware
    from runner import run_group

    metadata_path = Path(args.output) / args.group / "container.json"
    runtime = read_json(metadata_path)
    runtime.update(
        inspect_hardware(
            next(g["gpus"] for g in plan["groups"] if g["id"] == args.group)
        )
    )
    write_json(metadata_path, runtime)
    run_group(
        plan,
        args.group,
        args.output,
        args.models_root,
        args.gsm8k,
        args.sharegpt,
        client_python=args.client_python,
        client_root=args.client_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
