"""Host-side Docker lifecycle: one image and one container per target pair."""

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from common import ROOT, command_output, read_json, write_json
from config import EVALUATION_ROOT
from process import ManagedProcess


class DockerBackend:
    def __init__(
        self, plan, output, models_root="/raid/models", docker="docker", image_lock=None
    ):
        self.plan = plan
        self.output = Path(output).resolve()
        self.models_root = Path(models_root).resolve()
        self.docker = docker
        self.image = f"qwen4-ci:{plan['run_id']}"
        self.containers = []
        self.image_lock = read_json(image_lock) if image_lock else None
        if self.image_lock is not None and not isinstance(self.image_lock, dict):
            raise ValueError("Image lock must be a JSON object")
        self.built = False

    def run(self, argv):
        return command_output([self.docker, *argv])

    def build(self):
        reference = self.plan["base_image"]
        if self.image_lock is not None:
            if self.image_lock.get("plan_hash") != self.plan["plan_hash"]:
                raise ValueError("Retry image lock belongs to another run plan")
            reference = self.image_lock["base_image_digest"]
            if not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.:/-]*@sha256:[0-9a-f]{64}", reference
            ):
                raise ValueError("Retry image lock is not immutable")
        # Under the host lock, recover only containers from this same logical
        # run. Other runs or manual services remain untouched.
        orphans = self.run(
            ["ps", "-aq", "--filter", f"label=qwen4.run={self.plan['run_id']}"]
        ).splitlines()
        if orphans:
            if not all(re.fullmatch(r"[0-9a-f]{12,64}", cid) for cid in orphans):
                raise ValueError("Unexpected Docker container ID during recovery")
            write_json(
                self.output / "recovery.json",
                {"containers": orphans, "status": "removing"},
            )
            self.run(["rm", "-f", *orphans])
            write_json(
                self.output / "recovery.json",
                {"containers": orphans, "status": "removed"},
            )
        self.run(["pull", reference])
        info = json.loads(self.run(["image", "inspect", reference]))[0]
        digests = info.get("RepoDigests", [])
        if not digests:
            raise RuntimeError(
                "Selected registry image has no immutable repository digest"
            )
        base = reference if "@sha256:" in reference else digests[0]
        write_json(
            self.output / "image-lock.json",
            {
                **(self.image_lock or {}),
                "plan_hash": self.plan["plan_hash"],
                "base_image_digest": base,
            },
        )
        runtime_path = self.output / "runtime.json"
        runtime = read_json(runtime_path) if runtime_path.exists() else {}
        runtime.update(backend="docker", base_image_digest=base)
        write_json(runtime_path, runtime)
        with tempfile.TemporaryDirectory(prefix="qwen4-build-") as directory:
            context = Path(directory)
            shutil.copytree(
                ROOT,
                context / "harness",
                ignore=shutil.ignore_patterns("__pycache__", "tests"),
            )
            (context / "scripts").mkdir()
            for script in EVALUATION_ROOT.glob("*.sh"):
                shutil.copy2(script, context / "scripts" / script.name)
            shutil.copy2(ROOT / "Dockerfile", context / "Dockerfile")
            shutil.copy2(ROOT / "build_install.py", context / "build_install.py")
            # Workload/run IDs must not invalidate compiled dependency layers.
            write_json(
                context / "plan.json",
                {
                    "sources": {
                        name: {"repo": source["repo"], "sha": source["sha"]}
                        for name, source in sorted(self.plan["sources"].items())
                    }
                },
            )
            (context / "retry-constraints.txt").write_text(
                "\n".join((self.image_lock or {}).get("pip_constraints", []))
            )
            argv = [
                self.docker,
                "build",
                "--build-arg",
                f"BASE_IMAGE={base}",
                "--label",
                f"qwen4.run={self.plan['run_id']}",
                "-t",
                self.image,
                str(context),
            ]
            process = ManagedProcess(argv, self.output / "build.log")
            if process.wait(self.plan["defaults"]["timeouts"]["build"], progress=True):
                raise RuntimeError("Candidate image build failed; see build.log")
        runtime = read_json(self.output / "runtime.json")
        runtime["candidate_image_id"] = self.run(
            ["image", "inspect", "--format", "{{.Id}}", self.image]
        )
        self.built = True
        for name in ["build-provenance.json", "installed-packages.json"]:
            (self.output / name).write_text(
                self.run(
                    ["run", "--rm", "--network=none", self.image, "cat", f"/ci/{name}"]
                )
                + "\n"
            )
        provenance = read_json(self.output / "build-provenance.json")
        lock = read_json(self.output / "image-lock.json")
        lock["pip_constraints"] = provenance["pip_constraints"]
        write_json(self.output / "image-lock.json", lock)
        # Probe the entire eight-GPU lease before any target container starts,
        # including cards unused by a TP1/TP2/TP4 case.
        runtime.update(
            json.loads(
                self.run(
                    [
                        "run",
                        "--rm",
                        "--network=none",
                        "--device=/dev/kfd",
                        "--device=/dev/dri",
                        "-e",
                        "HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7",
                        self.image,
                        "python",
                        "/ci/harness/hardware.py",
                        "--gpus",
                        "8",
                    ]
                )
            )
        )
        write_json(self.output / "runtime.json", runtime)

    def evaluate_group(self, group, gsm8k_relative, sharegpt_relative):
        name = f"qwen4-{self.plan['run_id']}-{group['id']}"
        group_path = self.output / group["id"]
        group_path.mkdir(parents=True, exist_ok=True)
        argv = [
            "create",
            "--init",
            "--name",
            name,
            "--label",
            f"qwen4.run={self.plan['run_id']}",
            "--label",
            f"qwen4.target={group['id']}",
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
            f"HIP_VISIBLE_DEVICES={','.join(map(str, range(group['gpus'])))}",
            "--mount",
            f"type=bind,source={self.models_root},target=/models,readonly",
            "--mount",
            f"type=bind,source={self.output / 'manifest.json'},target=/ci/output/manifest.json,readonly",
            "--mount",
            f"type=bind,source={group_path},target=/ci/output/{group['id']}",
            self.image,
        ]
        if self.plan["workload"].get("profiler"):
            argv[-1:-1] = ["--cap-add=SYS_PTRACE"]
        container_id = self.run(argv)
        self.containers.append(container_id)
        write_json(
            group_path / "container.json",
            {
                "container_id": container_id,
                "target": group["directory"],
                "candidate_image": self.image,
                "models_mount": f"{self.models_root}:/models:ro",
            },
        )
        try:
            self.run(["start", container_id])
            command = [
                self.docker,
                "exec",
                "-e",
                f"QWEN4_CONTAINER_ID={container_id}",
                name,
                "python",
                "/ci/harness/run.py",
                "group",
                "--plan",
                "/ci/output/manifest.json",
                "--output",
                "/ci/output",
                "--group",
                group["id"],
                "--models-root",
                "/models",
                "--gsm8k",
                f"/models/{gsm8k_relative}",
                "--sharegpt",
                f"/models/{sharegpt_relative}",
                "--client-root",
                "/ci/src/client",
                "--client-python",
                "/ci/client-venv/bin/python",
            ]
            process = ManagedProcess(command, group_path / "container.log")
            timeout = sum(
                self.plan["defaults"]["timeouts"][x]
                for x in ["startup", "performance", "accuracy", "stop"]
            )
            code = process.wait(
                timeout * len(group["cases"]) * self.plan["workload"]["repeats"] + 300,
                progress=True,
            )
            if code:
                raise RuntimeError(
                    f"Target group {group['id']} exited {code}; see container.log"
                )
        finally:
            self.run(["rm", "-f", container_id])
            self.containers.remove(container_id)

    def cleanup(self):
        for name in self.containers[:]:
            self.run(["rm", "-f", name])
            self.containers.remove(name)
        if self.built:
            # Candidate containers run as root. Return only this run's bind
            # mount to the runner user after every writer has stopped, including
            # cancellation; otherwise the next checkout cannot clean the tree.
            self.run(
                [
                    "run",
                    "--rm",
                    "--network=none",
                    "--user",
                    "0",
                    "--mount",
                    f"type=bind,source={self.output},target=/ci/output",
                    self.image,
                    "chown",
                    "-R",
                    f"{os.getuid()}:{os.getgid()}",
                    "/ci/output",
                ]
            )
            self.built = False
        # Only this run's tag, not a shared base image or global Docker cache.
        if shutil.which(self.docker):
            subprocess.run([self.docker, "image", "rm", self.image], check=False)
