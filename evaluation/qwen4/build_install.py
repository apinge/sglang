"""Image-only installation of resolved fork sources; never run on a shared host."""

import json
import os
import subprocess
import sys
from pathlib import Path


def run(argv, **kwargs):
    subprocess.run(argv, check=True, **kwargs)


def dependency_constraints(packages):
    managed = {"sglang", "sgl-kernel", "sglang-kernel", "amd-aiter", "aiter", "pyhip"}
    return sorted(
        f"{p['name']}=={p['version']}"
        for p in packages
        if p["name"].lower().replace("_", "-") not in managed
    )


def install(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    root = Path("/ci/src")
    root.mkdir(parents=True, exist_ok=False)
    sources = plan["sources"]
    for name in ["sglang", "aiter", "pyhip", "client"]:
        source = sources[name]
        destination = root / name
        run(["git", "init", str(destination)])
        run(
            [
                "git",
                "-C",
                str(destination),
                "remote",
                "add",
                "origin",
                f"https://github.com/{source['repo']}.git",
            ]
        )
        run(
            [
                "git",
                "-C",
                str(destination),
                "fetch",
                "--depth=1",
                "origin",
                source["sha"],
            ]
        )
        run(["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"])
        actual = subprocess.check_output(
            ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != source["sha"]:
            raise RuntimeError(f"Unexpected checkout for {name}")
        if name in {"aiter", "pyhip"}:
            run(["git", "-C", str(destination), "submodule", "sync", "--recursive"])
            run(
                [
                    "git",
                    "-C",
                    str(destination),
                    "submodule",
                    "update",
                    "--init",
                    "--recursive",
                ]
            )
    # Keep the base image's dependency versions. Deliberate dependency changes
    # belong in a reviewed image/constraints update, not a surprise pip upgrade.
    packages = json.loads(
        subprocess.check_output(
            [sys.executable, "-m", "pip", "list", "--format=json"], text=True
        )
    )
    constraints = dependency_constraints(packages)
    retry_constraints = Path("/ci/retry-constraints.txt")
    if retry_constraints.is_file() and retry_constraints.read_text().strip():
        constraints = retry_constraints.read_text().splitlines()
    constraints_path = Path("/ci/base-constraints.txt")
    constraints_path.write_text("\n".join(constraints) + "\n")
    env = {
        **os.environ,
        "PIP_CONSTRAINT": str(constraints_path),
        "SGLANG_BUILD_RUST_EXTS": "none",
    }
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "uninstall",
            "-y",
            "sglang",
            "sgl-kernel",
            "sglang-kernel",
            "amd-aiter",
            "pyhip",
        ]
    )
    for name in ["aiter", "pyhip"]:
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-build-isolation",
                "-e",
                str(root / name),
            ],
            env=env,
        )
    kernel = root / "sglang/python/sglang/kernels/aot"
    run([sys.executable, "setup_rocm.py", "install"], cwd=kernel, env=env)
    package = root / "sglang/python"
    alternate = package / "pyproject_other.toml"
    if alternate.exists():
        (package / "pyproject.toml").write_bytes(alternate.read_bytes())
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "-e",
            str(package) + "[all_hip]",
        ],
        env=env,
    )
    # Do not let ensurepip seed older pip/setuptools copies that shadow the
    # constrained packages already present in the base image.
    run(
        [
            sys.executable,
            "-m",
            "venv",
            "--without-pip",
            "--system-site-packages",
            "/ci/client-venv",
        ]
    )
    client_python = "/ci/client-venv/bin/python"
    site = subprocess.check_output(
        [client_python, "-c", "import site; print(site.getsitepackages()[0])"],
        text=True,
    ).strip()
    # Client source is imported ahead of the candidate editable installation.
    Path(site, "qwen4_client.pth").write_text(
        "import sys; sys.path.insert(0, '/ci/src/client/python')\n"
    )
    installed = json.loads(
        subprocess.check_output(
            [sys.executable, "-m", "pip", "list", "--format=json"], text=True
        )
    )
    Path("/ci/installed-packages.json").write_text(json.dumps(installed, indent=2))
    submodules = {}
    for name in ["aiter", "pyhip"]:
        submodules[name] = subprocess.check_output(
            ["git", "-C", str(root / name), "submodule", "status", "--recursive"],
            text=True,
        )
    # AITER import initializes GPU-dependent Triton modules. Image builds have
    # no GPU, so verify module resolution without importing those modules.
    origins = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import importlib.util,json; print(json.dumps({n:importlib.util.find_spec(n).origin for n in ['sglang','aiter','pyhip']}))",
            ],
            text=True,
        )
    )
    for name, origin in origins.items():
        if not Path(origin).resolve().is_relative_to(root / name):
            raise RuntimeError(
                f"{name} resolves outside the intended source checkout: {origin}"
            )
    Path("/ci/build-provenance.json").write_text(
        json.dumps(
            {
                "sources": sources,
                "submodules": submodules,
                "module_origins": origins,
                "pip_constraints": dependency_constraints(installed),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    install(sys.argv[1])
