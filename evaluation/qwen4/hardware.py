"""Run in the ROCm environment, never on the CPU planning runner."""

import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


def _inspect_hardware(required=8, require_idle=True):
    import torch

    if not torch.version.hip or torch.cuda.device_count() < required:
        raise RuntimeError(f"Need {required} visible ROCm GPUs")
    devices = []
    for index in range(required):
        p = torch.cuda.get_device_properties(index)
        arch = p.gcnArchName.split(":")[0]
        if arch != "gfx942" or "MI308" not in p.name:
            raise RuntimeError(
                f"GPU {index} is {p.name}/{arch}, expected MI308X/gfx942"
            )
        free, total = torch.cuda.mem_get_info(index)
        if require_idle and total - free > 1024**3:
            raise RuntimeError(
                f"GPU {index} is busy ({(total - free) / 1024**3:.2f} GiB allocated); no process was killed"
            )
        devices.append(
            {
                "index": index,
                "name": p.name,
                "arch": arch,
                "total_memory": p.total_memory,
                "uuid": str(getattr(p, "uuid", "unavailable")),
            }
        )
    stack = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "kernel": platform.release(),
    }
    for name in ["triton", "transformers"]:
        stack[name] = importlib.metadata.version(name)
    driver = Path("/sys/module/amdgpu/version")
    stack["amdgpu_driver"] = (
        driver.read_text().strip() if driver.exists() else "unavailable"
    )
    modules = {}
    for name in ["sglang", "aiter", "pyhip"]:
        spec = importlib.util.find_spec(name)
        modules[name] = spec.origin if spec else None
    return {"hardware": devices, "stack": stack, "module_origins": modules}


def inspect_hardware(required=8, require_idle=True):
    """Probe in a short-lived process; the supervisor must not retain GPU RAM."""
    command = [sys.executable, str(Path(__file__).resolve()), "--gpus", str(required)]
    if not require_idle:
        command.append("--allow-busy")
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "GPU preflight failed")
    return json.loads(result.stdout)


def wait_gpu_idle(required, timeout=30):
    deadline = time.monotonic() + timeout
    while True:
        try:
            return inspect_hardware(required)
        except RuntimeError as exc:
            if "is busy" not in str(exc) or time.monotonic() >= deadline:
                raise
            time.sleep(0.5)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--allow-busy", action="store_true")
    args = parser.parse_args()
    if args.gpus <= 0:
        parser.error("--gpus must be positive")
    print(json.dumps(_inspect_hardware(args.gpus, not args.allow_busy)))
