"""CI authorization, script recipes, lifecycle failures, and native result contracts."""

import argparse
import http.server
import io
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time

import pytest
import qwen4_ci as ci
from qwen4_ci import (
    ManagedProcess,
    host_lock,
    port_available,
    read_json,
    validate_accuracy,
    wait_ready,
)


@pytest.mark.parametrize("status", [401, 403, 404])
def test_prepare_api_failure_is_actionable_and_never_writes_a_plan(tmp_path, status):
    fake_gh = tmp_path / "gh"
    reason = (
        "Bad credentials"
        if status == 401
        else "Forbidden"
        if status == 403
        else "Not Found"
    )
    fake_gh.write_text(
        f"#!/bin/sh\nprintf '%s\\n' 'gh: {reason} (HTTP {status})' >&2\nexit 1\n"
    )
    fake_gh.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REPOSITORY": "apinge/sglang",
        "INPUT_PR_NUMBER": "65",
    }
    env.pop("GITHUB_EVENT_PATH", None)
    output = tmp_path / "plan"
    result = subprocess.run(
        [sys.executable, str(ci.ROOT), "prepare", "--output", str(output)],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert f"HTTP {status}" in result.stderr and "Traceback" not in result.stderr
    assert not (output / "plan.json").exists()
    if status == 401:
        assert (
            "GH_TOKEN/GITHUB_TOKEN" in result.stderr
            and "gh auth login" in result.stderr
        )
    else:
        assert (
            "repository permissions" in result.stderr
            and "gh auth login" not in result.stderr
        )


@pytest.mark.parametrize("explicit_path", [True, False])
def test_missing_plan_exits_before_docker_with_prepare_instructions(
    tmp_path, explicit_path
):
    env = dict(os.environ)
    env.pop("QWEN4_PLAN", None)
    argv = [sys.executable, str(ci.ROOT), "run", "--output", str(tmp_path / "results")]
    if explicit_path:
        argv += ["--plan", str(tmp_path / "missing/plan.json")]
    result = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert "prepare" in result.stderr and "Traceback" not in result.stderr
    assert not (tmp_path / "results").exists()


def test_non_github_command_failure_keeps_its_original_exit_code():
    with pytest.raises(subprocess.CalledProcessError) as error:
        ci.command([sys.executable, "-c", "raise SystemExit(7)"])
    assert error.value.returncode == 7


@pytest.fixture
def runtime_packages():
    return [
        {"name": name, "version": version}
        for name, version in [
            ("sglang", "0.0.0.dev1"),
            ("torch", "2.12.0+rocm7.14"),
            ("triton", "3.8.0+rocm7.14"),
            ("numpy", "2.2.6"),
            ("transformers", "5.12.1"),
        ]
    ]


def test_runtime_checks_follow_hip_extras_without_resolving_diffusion(runtime_packages):
    after = [dict(p) for p in runtime_packages]
    after[0]["version"] = "0.0.0.dev2"
    requirements = [
        "numpy>=2",
        "sglang[runtime_common]; extra == 'srt_hip'",
        "sglang[runtime_base]; extra == 'runtime_common'",
        "sglang[srt_hip]; extra == 'runtime_base'",  # Self-extra cycles terminate.
        "transformers==5.12.1; extra == 'runtime_base'",
        "torch; extra == 'srt_hip'",
        "diffusers==99; extra == 'diffusion_hip'",
        "missing-package; python_version < '3.0'",
    ]
    checked = ci.check_base_runtime(runtime_packages, after, requirements, {"sglang"})
    assert any("transformers" in r for r in checked)
    assert any("torch" in r for r in checked)
    assert all("diffusers" not in r and "missing-package" not in r for r in checked)


@pytest.mark.parametrize("change", ["torch", "triton", "removed"])
def test_runtime_checks_reject_replaced_or_removed_base_packages(
    runtime_packages, change
):
    after = [dict(p) for p in runtime_packages]
    if change == "removed":
        after = [p for p in after if p["name"] != "triton"]
    else:
        next(p for p in after if p["name"] == change)["version"] = "9.0.0"
    with pytest.raises(RuntimeError, match="Base package"):
        ci.check_base_runtime(runtime_packages, after, [], {"sglang"})


@pytest.mark.parametrize(
    "requirement",
    ["missing-package; extra == 'srt_hip'", "transformers>=6; extra == 'srt_hip'"],
)
def test_runtime_checks_reject_missing_or_incompatible_candidate_requirements(
    runtime_packages, requirement
):
    with pytest.raises(RuntimeError, match="HIP runtime requires"):
        ci.check_base_runtime(
            runtime_packages, runtime_packages, [requirement], {"sglang"}
        )


@pytest.mark.parametrize("replace_triton", [False, True])
def test_image_install_reuses_base_dependencies_and_checks_fresh_imports(
    tmp_path, monkeypatch, plan, runtime_packages, replace_triton
):
    from pathlib import Path

    root = tmp_path / "ci"
    root.mkdir()
    package = root / "src/sglang/python"
    package.mkdir(parents=True)
    (package / "pyproject_other.toml").write_text("# alternate HIP metadata\n")
    monkeypatch.setattr(
        ci,
        "Path",
        lambda path: (
            root / str(path).removeprefix("/ci/")
            if str(path).startswith("/ci/")
            else Path(path)
        ),
    )
    monkeypatch.setenv("PYTHONPATH", "/opt/aiter")
    monkeypatch.setenv("AITER_USE_SYSTEM_TRITON", "0")
    calls, snapshots = [], []
    after = [dict(p) for p in runtime_packages]
    after[0]["version"] = "0.0.0.dev2"
    if replace_triton:
        after[2]["version"] = "3.7.0"

    def command(argv, **kwargs):
        if argv[:2] == ["git", "-C"]:
            return plan["sources"][Path(argv[2]).name]["sha"]
        if argv[-2:] == ["list", "--format=json"]:
            snapshots.append(True)
            return json.dumps(runtime_packages if len(snapshots) == 1 else after)
        assert "PYTHONPATH" not in kwargs["env"]
        return json.dumps(
            {
                "origins": {
                    n: str(root / "src" / n / n / "__init__.py")
                    for n in ["sglang", "aiter", "pyhip"]
                },
                "requirements": ["numpy>=2", "torch; extra == 'srt_hip'"],
            }
        )

    monkeypatch.setattr(ci, "command", command)
    monkeypatch.setattr(
        ci.subprocess, "run", lambda argv, **kwargs: calls.append((argv, kwargs))
    )
    if replace_triton:
        with pytest.raises(RuntimeError, match="Base package triton"):
            ci.install(plan)
        assert not (root / "build.json").exists()
    else:
        ci.install(plan)
        info = ci.read_json(root / "build.json")
        assert info["packages"] == after and info["checked_hip_requirements"]
        imports = [(a, k) for a, k in calls if "-c" in a]
        assert any("import sglang.launch_server" in a for a, _ in imports)
        assert any(
            k["env"].get("PYTHONPATH") == str(root / "src/client/python")
            for _, k in imports
        )
    install_args, options = next(
        (a, k) for a, k in calls if "-e" in a and str(package) in a
    )
    assert "--no-deps" in install_args and "--no-build-isolation" in install_args
    assert all("[all_hip]" not in arg for arg in install_args)
    assert "PYTHONPATH" not in options["env"]
    assert options["env"]["AITER_USE_SYSTEM_TRITON"] == "1"
    aiter_args, aiter_options = next(
        (a, k) for a, k in calls if "-e" in a and str(root / "src/aiter") in a
    )
    assert aiter_options["env"]["AITER_USE_SYSTEM_TRITON"] == "1"
    assert "PYTHONPATH" not in aiter_options["env"]


def test_image_install_adds_only_missing_hip_pins(
    tmp_path, monkeypatch, plan, runtime_packages
):
    from pathlib import Path

    root = tmp_path / "ci"
    root.mkdir()
    package = root / "src/sglang/python"
    package.mkdir(parents=True)
    (package / "pyproject_other.toml").write_text("# alternate HIP metadata\n")
    monkeypatch.setattr(
        ci,
        "Path",
        lambda path: (
            root / str(path).removeprefix("/ci/")
            if str(path).startswith("/ci/")
            else Path(path)
        ),
    )
    before = [dict(item) for item in runtime_packages] + [
        {"name": "sentencepiece", "version": "0.2.2"},
        {"name": "xgrammar", "version": "0.2.1"},
    ]
    satisfied = [dict(item) for item in before]
    next(item for item in satisfied if item["name"] == "sglang")["version"] = (
        "0.0.0.dev2"
    )
    pinned = [dict(item) for item in satisfied]
    next(item for item in pinned if item["name"] == "sentencepiece")["version"] = (
        "0.2.1"
    )
    next(item for item in pinned if item["name"] == "xgrammar")["version"] = "0.2.7"
    pinned.extend(
        [
            {"name": "outlines", "version": "0.1.11"},
            {"name": "petit-kernel", "version": "0.0.2"},
        ]
    )
    lists = []
    calls = []

    def command(argv, **kwargs):
        if argv[:2] == ["git", "-C"]:
            return plan["sources"][Path(argv[2]).name]["sha"]
        if argv[-2:] == ["list", "--format=json"]:
            lists.append(True)
            return json.dumps((before, satisfied, pinned)[len(lists) - 1])
        return json.dumps(
            {
                "origins": {
                    n: str(root / "src" / n / n / "__init__.py")
                    for n in ["sglang", "aiter", "pyhip"]
                },
                "requirements": [
                    "sglang[runtime_common]; extra == 'srt_hip'",
                    "sglang[runtime_base]; extra == 'runtime_common'",
                    "sentencepiece==0.2.1; extra == 'runtime_base'",
                    "outlines==0.1.11; extra == 'runtime_common'",
                    "xgrammar==0.2.7; extra == 'runtime_common'",
                    "petit_kernel==0.0.2; extra == 'srt_hip'",
                    "torch; extra == 'srt_hip'",
                    "diffusers==99; extra == 'diffusion_hip'",
                ],
            }
        )

    monkeypatch.setattr(ci, "command", command)
    monkeypatch.setattr(
        ci.subprocess, "run", lambda argv, **kwargs: calls.append((argv, kwargs))
    )
    ci.install(plan)
    pin_args, pin_options = next(
        (argv, options)
        for argv, options in calls
        if "sentencepiece==0.2.1" in argv
    )
    assert "outlines==0.1.11" in pin_args
    assert "xgrammar==0.2.7" in pin_args
    assert "petit_kernel==0.0.2" in pin_args
    assert "--no-deps" not in pin_args and all("[all_hip]" not in arg for arg in pin_args)
    frozen = Path(pin_options["env"]["PIP_CONSTRAINT"]).read_text()
    assert "torch==2.12.0+rocm7.14" in frozen
    assert "sentencepiece" not in frozen and "xgrammar" not in frozen
    assert "outlines" not in frozen and "petit" not in frozen
    info = ci.read_json(root / "build.json")
    assert info["packages"] == pinned
    assert "sentencepiece==0.2.1" in info["dependency_policy"]
    assert all("diffusers" not in item for item in info["checked_hip_requirements"])


@pytest.mark.parametrize(
    "mode,requested,strict",
    [
        ("ok", 2, True),
        ("error", 2, True),
        ("mixed", 2, True),
        ("empty", 2, True),
        ("ok", 3, True),
        ("error", 2, False),
    ],
)
def test_native_counts_and_strict_failures_preserve_original_scoring(
    tmp_path, mode, requested, strict
):
    pytest.importorskip("sglang", reason="Installed-runtime client integration")
    from sglang.test.run_eval import run_eval

    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            index = int(
                re.findall(r"Question: Q(\d+)", body["messages"][0]["content"])[-1]
            )
            seen.append(index)
            if mode == "error" or (mode == "mixed" and index == 6):
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"test failure"}}')
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "id": "fixture",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "test-model",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "" if mode == "empty" else str(index),
                                },
                                "finish_reason": "length"
                                if mode == "empty"
                                else "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 1,
                            "total_tokens": 101,
                        },
                    }
                ).encode()
            )

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    data = tmp_path / "test.jsonl"
    data.write_text(
        "".join(
            json.dumps({"question": f"Q{i}", "answer": f"#### {i}"}) + "\n"
            for i in range(7)
        )
    )
    args = argparse.Namespace(
        eval_name="gsm8k",
        gsm8k_data_path=str(data),
        num_examples=requested,
        num_shots=5,
        num_threads=2,
        output_dir=str(tmp_path / "results"),
        base_url=None,
        host="127.0.0.1",
        port=server.server_port,
        model="test-model",
        max_tokens=32,
        repeat=1,
        api="chat",
        strict=strict,
        temperature=0.0,
        top_p=1.0,
    )
    errors = {"ok": 0, "empty": 0, "error": 2, "mixed": 1}[mode]
    incomplete = strict and (errors > 0 or requested > 2)
    try:
        if incomplete:
            with pytest.raises(RuntimeError, match="Incomplete GSM8K"):
                run_eval(args)
        else:
            run_eval(args)
        path = tmp_path / "results/gsm8k.json"
        result = read_json(path)
        assert sorted(seen) == [
            5,
            6,
        ]  # Prompt examples remain excluded by the original scorer.
        assert result["score"] == {"ok": 1, "empty": 0, "error": 0, "mixed": 0.5}[mode]
        assert path.with_suffix(".html").stat().st_size > 0
        assert not path.with_name("gsm8k-examples.jsonl").exists()
        if strict:
            assert {
                k: result[k] for k in ["requested", "scored", "completed", "errors"]
            } == {
                "requested": requested,
                "scored": 2,
                "completed": 2 - errors,
                "errors": errors,
            }
            if incomplete:
                with pytest.raises(ValueError):
                    validate_accuracy(path, requested)
            else:
                validate_accuracy(path, requested)
        else:
            assert not {"requested", "scored", "completed", "errors"} & result.keys()
    finally:
        server.shutdown()
        server.server_close()


def test_sampler_counts_logical_requests_across_retries_and_threads(monkeypatch):
    pytest.importorskip("sglang", reason="Installed-runtime client integration")
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    from sglang.test import simple_eval_common as common

    attempts, lock = {}, threading.Lock()

    def create(**kwargs):
        index = int(kwargs["messages"][-1]["content"])
        with lock:
            attempts[index] = attempts.get(index, 0) + 1
            attempt = attempts[index]
        if index % 4 == 0 or (index % 4 == 1 and attempt <= 2):
            raise RuntimeError("fixture transport error")
        return SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None if index % 4 == 2 else "answer"
                    )
                )
            ],
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(common, "OpenAI", lambda **kwargs: client)
    monkeypatch.setattr(common, "LargerHttpxClient", lambda: None)
    monkeypatch.setattr(common.time, "sleep", lambda _: None)
    sampler = common.ChatCompletionSampler(model="fixture")
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(
            pool.map(
                lambda i: sampler([{"role": "user", "content": str(i)}]), range(200)
            )
        )
    assert sampler.request_counts == {"attempted": 200, "completed": 150}
    assert (
        sum(attempts.values()) == 550
    )  # Retries do not inflate the logical request count.
    assert (
        results[0] == results[2] == ""
    )  # A failed request and a completed empty answer differ only in accounting.


def test_dead_server_is_detected_without_waiting_for_timeout(tmp_path):
    process = ManagedProcess(
        [sys.executable, "-c", "raise SystemExit(7)"], tmp_path / "server.log"
    )
    process.process.wait()
    case = ci.CASES[0]
    start = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="exited"):
            wait_ready(process, "http://127.0.0.1:1", case, "/models/model", 30)
        assert time.monotonic() - start < 2
    finally:
        process.stop()


def test_wrong_server_identity_is_not_ready(tmp_path):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"data": [{"id": "wrong-model"}]}).encode())

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    process = ManagedProcess(
        [sys.executable, "-c", "import time; time.sleep(30)"], tmp_path / "server.log"
    )
    try:
        case = ci.CASES[0]
        with pytest.raises(ValueError, match="different model"):
            wait_ready(
                process,
                f"http://127.0.0.1:{server.server_port}",
                case,
                "/models/model",
                30,
            )
    finally:
        process.stop()
        server.shutdown()
        server.server_close()


def test_host_lock_excludes_other_jobs(tmp_path):
    with host_lock(tmp_path / "host.lock", timeout=0):
        with pytest.raises(TimeoutError):
            with host_lock(tmp_path / "host.lock", timeout=0):
                pass
    with host_lock(tmp_path / "host.lock", timeout=0):
        pass


def test_stop_reaps_the_owned_leader(tmp_path):
    process = ManagedProcess(
        [sys.executable, "-c", "import time; time.sleep(60)"], tmp_path / "log"
    )
    process.stop(timeout=1)
    assert not process.alive()


def test_progress_streams_before_exit_and_preserves_the_exit_code(
    tmp_path, monkeypatch
):
    release = tmp_path / "release"
    script = (
        "import time\nfrom pathlib import Path\n"
        "print('LIVE OUTPUT', flush=True)\n"
        f"while not Path({str(release)!r}).exists(): time.sleep(0.01)\n"
        "print('FINAL OUTPUT', flush=True)\nraise SystemExit(7)\n"
    )
    process = ManagedProcess([sys.executable, "-c", script], tmp_path / "process.log")
    visible = threading.Event()

    class Console(io.StringIO):
        def write(self, text):
            result = super().write(text)
            if "LIVE OUTPUT" in text:
                visible.set()
            return result

    console = Console()
    monkeypatch.setattr(sys, "stdout", console)
    outcomes = []
    worker = threading.Thread(
        target=lambda: outcomes.append(process.wait(10, progress=True))
    )
    worker.start()
    try:
        assert visible.wait(5), (
            "Output must be visible while the child is still running"
        )
        assert process.alive()
    finally:
        release.touch()
        worker.join(timeout=10)
        process.stop(timeout=1)
    assert outcomes == [7]
    assert "FINAL OUTPUT" in console.getvalue()
    assert (tmp_path / "process.log").read_text() == "LIVE OUTPUT\nFINAL OUTPUT\n"


def test_progress_timeout_retains_output_and_stops_the_child(tmp_path, capsys):
    ready = tmp_path / "ready"
    script = (
        "import time\nfrom pathlib import Path\nprint('BEFORE TIMEOUT', flush=True)\n"
        f"Path({str(ready)!r}).touch()\ntime.sleep(60)\n"
    )
    process = ManagedProcess([sys.executable, "-c", script], tmp_path / "process.log")
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        with pytest.raises(subprocess.TimeoutExpired):
            process.wait(0.1, progress=True)
        assert not process.alive()
        assert "BEFORE TIMEOUT" in capsys.readouterr().out
        assert "BEFORE TIMEOUT" in (tmp_path / "process.log").read_text()
    finally:
        process.stop(timeout=1)


@pytest.mark.parametrize("failure", ["lock", "dataset"])
def test_pre_docker_failures_are_printed_with_the_report(
    tmp_path, monkeypatch, capsys, plan, failure
):
    def lock_error(*args):
        raise PermissionError(13, "Permission denied", "/var/lock/qwen4-mi308x.lock")

    def dataset_error(*args):
        raise FileNotFoundError(
            2,
            "No such file or directory",
            "/raid/models/benchmark_datasets/gsm8k/test.jsonl",
        )

    if failure == "lock":
        monkeypatch.setattr(ci, "host_lock", lock_error)
    monkeypatch.setattr(ci, "preflight", dataset_error)
    monkeypatch.setattr(
        ci,
        "command",
        lambda *a, **k: pytest.fail("Docker must not start after failed preflight"),
    )
    monkeypatch.setattr(ci.shutil, "which", lambda _: None)
    args = argparse.Namespace(
        output=str(tmp_path / "results"),
        models_root="/raid/models",
        datasets_root=None,
        lock=str(tmp_path / "lock"),
    )
    assert ci.run(plan, args) == 1
    captured = capsys.readouterr()
    assert "Starting Docker CI" in captured.out and "FAILED" in captured.out
    assert (
        "Permission denied" if failure == "lock" else "gsm8k/test.jsonl"
    ) in captured.err
    assert not (tmp_path / "results/pull.log").exists()
    assert (tmp_path / "results/summary.md").is_file()


def test_port_probe_detects_listener_and_allows_clean_reuse():
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    server.listen()
    assert not port_available(port)
    client = socket.create_connection(("127.0.0.1", port))
    peer, _ = server.accept()
    peer.close()
    client.close()
    server.close()
    assert port_available(port)


def test_stop_waits_for_shell_child_to_release_listener(tmp_path):
    # Shell exits promptly on SIGTERM, but its child needs time to drain. The
    # supervisor must not report completion merely because the shell exited.
    ready = tmp_path / "ready"
    script = tmp_path / "server.py"
    script.write_text(
        "import signal, socket, time\n"
        "from pathlib import Path\n"
        "s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)\n"
        "s.bind(('127.0.0.1',0)); s.listen()\n"
        f"Path({str(ready)!r}).write_text(str(s.getsockname()[1]))\n"
        "def stop(*args):\n    time.sleep(0.3)\n    s.close()\n    raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM,stop)\n"
        "while True: time.sleep(0.05)\n"
    )
    process = ManagedProcess(
        ["bash", "-c", '"$1" "$2" & wait', "fixture", sys.executable, str(script)],
        tmp_path / "group.log",
    )
    unrelated = ManagedProcess(
        [sys.executable, "-c", "import time; time.sleep(30)"], tmp_path / "other.log"
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        port = int(ready.read_text())
        assert not port_available(port)
        process.stop(timeout=2)
        assert port_available(port)
        assert unrelated.alive()
        process.stop(timeout=2)  # Idempotent; no stale PID/group reuse.
    finally:
        process.stop(timeout=1)
        unrelated.stop(timeout=1)


@pytest.fixture
def plan(request):
    return {
        "sources": {
            name: {"repo": f"apinge/{name}", "sha": str(i) * 40}
            for i, name in enumerate(["sglang", "aiter", "pyhip", "client"], 1)
        },
        "base_image": ci.BASE_IMAGE,
        "smoke": getattr(request, "param", True),
    }


@pytest.fixture
def complete_results(tmp_path, plan):
    examples, prompts = (2, 2) if plan["smoke"] else (200, 128)
    data = tmp_path / "gsm8k-source.jsonl"
    data.write_text(
        "".join(
            json.dumps({"question": f"Q{i}", "answer": f"#### {i}"}) + "\n"
            for i in range(5 + examples)
        )
    )
    selection = ci.gsm8k_selection(data, examples)
    ci.write_json(tmp_path / "assets.json", {"gsm8k": selection})
    ci.write_json(
        tmp_path / "runtime.json",
        {"errors": [], "containers": {g: g + "-container" for g in ci.GROUPS}},
    )
    for case in ci.CASES:
        path = tmp_path / case["group"] / case["id"]
        ci.write_json(
            path / "execution.json",
            {
                "case_id": case["id"],
                "container_id": case["group"] + "-container",
                "errors": [],
                "phases": {"accuracy": 0, "performance": 0},
            },
        )
        for name in [
            "launch.json",
            "server-info.json",
            "smoke.json",
            "server.log",
            "gsm8k.html",
        ]:
            (path / name).write_text("{}")
        ci.write_json(
            path / "gsm8k.json",
            {
                "requested": examples,
                "scored": examples,
                "completed": examples,
                "errors": 0,
                "score": 0 if plan["smoke"] else ci.GSM8K_BASELINES[case["id"]],
            },
        )
        (path / "performance.jsonl").write_text(
            json.dumps(
                {
                    "completed": prompts,
                    "errors": [""] * prompts,
                    "output_lens": [case["output_tokens"]] * prompts,
                    "mean_ttft_ms": 10,
                    "mean_tpot_ms": 2,
                    "output_throughput": 50,
                    "request_throughput": 0.5,
                    "accept_length": 3.5 if case["mode"] in {"dflash", "mtp"} else None,
                }
            )
            + "\n"
        )
    return tmp_path


def test_smoke_accuracy_is_diagnostic_when_all_six_cases_complete(
    complete_results, plan
):
    assert ci.report(plan, complete_results) == 0
    summary = ci.read_json(complete_results / "summary.json")
    assert len(summary["cases"]) == 6
    assert all(row["accuracy"]["score"] == 0 for row in summary["cases"])
    text = (complete_results / "summary.md").read_text()
    assert "| QPS |" in text and "| Accept length |" in text
    by_case = {row["case"]: row for row in summary["cases"]}
    assert by_case["Qwen3.5-35B-TP4-Base"]["performance"]["request_throughput"] == 0.5
    assert by_case["Qwen3.5-35B-TP4-Base"]["performance"]["accept_length"] is None
    assert by_case["Qwen3.5-35B-TP1-DFlash"]["performance"]["accept_length"] == 3.5
    assert by_case["Qwen3.8-Flash-TP2-MTP"]["performance"]["accept_length"] == 3.5


def test_existing_container_report_does_not_claim_docker_isolation(
    complete_results, plan
):
    runtime = ci.read_json(complete_results / "runtime.json")
    runtime["containers"] = {group: "existing:fixture" for group in ci.GROUPS}
    ci.write_json(complete_results / "runtime.json", runtime)
    for case in ci.CASES:
        path = complete_results / case["group"] / case["id"] / "execution.json"
        state = ci.read_json(path)
        state["container_id"] = "existing:fixture"
        ci.write_json(path, state)
    assert (
        ci.report(plan, complete_results) == 1
    )  # Docker mode requires three containers.
    plan["mode"] = "existing-container"
    assert ci.report(plan, complete_results) == 0
    text = (complete_results / "summary.md").read_text()
    assert "Qwen4 local model evaluation" in text and "not verified" in text


@pytest.mark.parametrize("group_failure", [False, True])
def test_local_runner_visits_groups_and_preserves_failure_evidence(
    tmp_path, monkeypatch, group_failure
):
    client = tmp_path / "client"

    def command(argv, **kwargs):
        if argv[0] == "git":
            return ci.CLIENT_SHA if str(client) in argv else "a" * 40
        return str(ci.EVALUATION.parent / "python/sglang/__init__.py")

    groups = []

    def group(*args, **kwargs):
        groups.append(args[1])
        if group_failure:
            raise RuntimeError("fixture group failure")

    monkeypatch.setattr(ci, "command", command)
    monkeypatch.setattr(ci, "preflight", lambda *args: {"gsm8k": {}})
    monkeypatch.setattr(ci, "run_group", group)
    monkeypatch.setenv("QWEN4_CONTAINER_ID", "original")
    args = argparse.Namespace(
        output=str(tmp_path / "results"),
        models_root=str(tmp_path / "models"),
        datasets_root=None,
        client_root=str(client),
        smoke=False,
        lock=str(tmp_path / "lock"),
    )
    assert ci.run_local(args) == 1  # Missing measurements cannot become a pass.
    assert groups == (list(ci.GROUPS)[:1] if group_failure else list(ci.GROUPS))
    assert ci.os.environ["QWEN4_CONTAINER_ID"] == "original"
    assert (
        ci.read_json(tmp_path / "results/summary.json")["mode"] == "existing-container"
    )


@pytest.mark.parametrize("plan", [False], indirect=True)
@pytest.mark.parametrize("case", ci.CASES, ids=lambda case: case["id"])
@pytest.mark.parametrize("lost_answers,failed", [(-1, False), (3, False), (4, True)])
def test_full_accuracy_gate_uses_relative_drop_for_each_case(
    complete_results, plan, case, lost_answers, failed
):
    path = complete_results / case["group"] / case["id"] / "gsm8k.json"
    data = ci.read_json(path)
    reference = ci.GSM8K_BASELINES[case["id"]]
    data["score"] = (round(reference * 200) - lost_answers) / 200
    ci.write_json(path, data)
    assert ci.report(plan, complete_results) == int(failed)
    summary = ci.read_json(complete_results / "summary.json")
    row = next(row for row in summary["cases"] if row["case"] == case["id"])
    assert any("GSM8K regression" in error for error in row["errors"]) == failed
    assert row["accuracy_reference"] == reference
    assert row["performance"]["completed"] == 128
    assert "Reference / minimum" in (complete_results / "summary.md").read_text()


@pytest.mark.parametrize(
    "failure",
    [
        "missing_accuracy",
        "short_performance",
        "request_error",
        "wrong_container",
        "failed_process",
        "lost_runner",
    ],
)
def test_incomplete_results_fail_and_keep_valid_other_measurements(
    complete_results, plan, failure
):
    case = ci.CASES[0]
    path = complete_results / case["group"] / case["id"]
    if failure == "missing_accuracy":
        (path / "gsm8k.json").unlink()
    elif failure in {"short_performance", "request_error"}:
        perf = ci.read_json(path / "performance.jsonl")
        perf["completed"] = 1 if failure == "short_performance" else 2
        perf["errors"] = (
            ["transport failure", ""] if failure == "request_error" else ["", ""]
        )
        (path / "performance.jsonl").write_text(json.dumps(perf) + "\n")
    elif failure == "wrong_container":
        runtime = ci.read_json(complete_results / "runtime.json")
        runtime["containers"] = {g: "shared-container" for g in ci.GROUPS}
        ci.write_json(complete_results / "runtime.json", runtime)
    elif failure == "failed_process":
        state = ci.read_json(path / "execution.json")
        state["phases"]["accuracy"] = 1
        ci.write_json(path / "execution.json", state)
    assert (
        ci.report(
            plan, complete_results, "failure" if failure == "lost_runner" else "success"
        )
        == 1
    )
    summary = ci.read_json(complete_results / "summary.json")
    assert len(summary["cases"]) == 6
    assert (
        summary["cases"][0][
            "performance" if failure == "missing_accuracy" else "accuracy"
        ]
        is not None
    )


def test_missing_or_malformed_artifacts_still_produce_six_row_report(tmp_path, plan):
    (tmp_path / "runtime.json").write_text("not JSON")
    assert ci.report(plan, tmp_path) == 1
    assert len(ci.read_json(tmp_path / "summary.json")["cases"]) == 6
    assert "Missing" in (tmp_path / "summary.md").read_text()


@pytest.mark.parametrize(
    "malformed", ["containers", "assets", "accuracy", "performance"]
)
def test_wrong_json_types_fail_without_losing_the_report(
    complete_results, plan, malformed
):
    path = complete_results / ci.CASES[0]["group"] / ci.CASES[0]["id"]
    if malformed == "containers":
        ci.write_json(
            complete_results / "runtime.json", {"errors": [], "containers": None}
        )
    elif malformed == "assets":
        ci.write_json(complete_results / "assets.json", None)
    elif malformed == "accuracy":
        ci.write_json(path / "gsm8k.json", [])
    else:
        (path / "performance.jsonl").write_text("null\n")
    assert ci.report(plan, complete_results) == 1
    assert len(ci.read_json(complete_results / "summary.json")["cases"]) == 6


@pytest.mark.parametrize(
    "origin,event_name,draft,label,permission,enabled",
    [
        ("sglang", "pull_request_target", False, True, "write", True),
        ("sglang", "pull_request_target", False, True, "read", False),
        ("sglang", "pull_request_target", True, True, "write", False),
        ("sglang", "pull_request_target", False, False, "write", False),
        ("aiter", "pull_request_target", False, True, "write", True),
        ("sglang", "workflow_dispatch", True, False, "write", True),
    ],
)
def test_authorization_and_candidate_source_selection(
    tmp_path, monkeypatch, origin, event_name, draft, label, permission, enabled
):
    pr = {
        "number": 65,
        "draft": draft,
        "state": "open",
        "labels": [{"name": "run-qwen4-ci"}] if label else [],
        "base": {"ref": ci.BRANCHES[origin]},
        "head": {"repo": {"full_name": f"contributor/{origin}"}, "sha": "a" * 40},
    }
    event_path = tmp_path / "event.json"
    ci.write_json(event_path, {"pull_request": pr})
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setenv("GITHUB_EVENT_NAME", event_name)
    monkeypatch.setenv("GITHUB_REPOSITORY", f"apinge/{origin}")
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    monkeypatch.setenv(
        "INPUT_PR_NUMBER", "65" if event_name == "workflow_dispatch" else ""
    )
    monkeypatch.delenv("GH_TOKEN", raising=False)
    calls = []

    def api(path, payload=None):
        assert payload is None
        calls.append(path)
        if "/permission" in path:
            return {"permission": permission}
        if "/pulls/" in path:
            return pr
        return {"sha": "a" * 40 if "contributor" in path else "b" * 40}

    monkeypatch.setattr(ci, "github", api)
    monkeypatch.setattr(
        ci, "command", lambda *a, **k: "1\told-maintainer\n20\tcurrent-maintainer"
    )
    assert ci.prepare(argparse.Namespace(output=str(tmp_path), smoke=False)) == 0
    assert f"enabled={str(enabled).lower()}" in (tmp_path / "output").read_text()
    if enabled:
        saved = ci.read_json(tmp_path / "plan.json")
        assert saved["sources"][origin] == {
            "repo": f"contributor/{origin}",
            "sha": "a" * 40,
        }
        companion = "aiter" if origin == "sglang" else "sglang"
        assert any(ci.BRANCHES[companion] in path for path in calls)
        assert saved["status_target"]["sha"] == "a" * 40
    else:
        assert not any("/commits/" in path for path in calls)


@pytest.mark.parametrize("failure", [None, "build", "group", "create"])
def test_docker_one_build_three_owned_containers_and_failure_cleanup(
    tmp_path, monkeypatch, plan, failure
):
    calls = []
    build_dirs = []
    work_root = tmp_path / "work"
    monkeypatch.setenv("QWEN4_WORK_ROOT", str(work_root))
    monkeypatch.setenv("DOCKER_CONFIG", "/home/ci-fixture/.docker")
    monkeypatch.setenv("BUILDX_CONFIG", "/home/ci-fixture/.docker/buildx")

    class Process:
        def __init__(self, argv, log_path, **kwargs):
            self.argv = argv
            calls.append(argv)
            if argv[:2] == ["docker", "build"]:
                context = ci.Path(argv[-1])
                buildx = ci.Path(kwargs["env"]["BUILDX_CONFIG"])
                assert buildx.is_dir() and buildx.is_relative_to(work_root)
                assert context.is_dir() and context.is_relative_to(work_root)
                assert not buildx.is_relative_to(context)
                assert kwargs["env"]["DOCKER_CONFIG"] == "/home/ci-fixture/.docker"
                assert kwargs["env"]["BUILDKIT_PROGRESS"] == "plain"
                assert not (context / "buildx").exists()
                build_dirs.extend([context, buildx])

        def wait(self, *args, **kwargs):
            return int(
                (failure == "build" and self.argv[:2] == ["docker", "build"])
                or (failure == "group" and "container" in self.argv)
            )

    def command(argv, **kwargs):
        calls.append(argv)
        if argv[:2] == ["docker", "info"]:
            return "/raid/docker"
        if argv[:3] == ["docker", "image", "inspect"]:
            return (
                "sha256:built"
                if any(x.startswith("--format") for x in argv)
                else json.dumps(
                    [{"Id": "sha256:base", "RepoDigests": ["image@sha256:" + "a" * 64]}]
                )
            )
        if argv[:2] == ["docker", "create"]:
            if failure == "create":
                raise RuntimeError("name already in use")
            return argv[argv.index("--name") + 1]
        return ""

    monkeypatch.setattr(ci, "ManagedProcess", Process)
    monkeypatch.setattr(ci, "command", command)
    monkeypatch.setattr(ci, "preflight", lambda *args: {"gsm8k": {}})
    monkeypatch.setattr(ci.shutil, "which", lambda name: None)
    args = argparse.Namespace(
        output=str(tmp_path / "results"),
        models_root=str(tmp_path / "models"),
        datasets_root=None,
        lock=str(tmp_path / "host.lock"),
    )
    # Even a zero container exit cannot pass without actual complete results.
    assert ci.run(plan, args) == 1
    builds = [c for c in calls if c[:2] == ["docker", "build"]]
    creates = [c for c in calls if c[:2] == ["docker", "create"]]
    removes = [c for c in calls if c[:3] == ["docker", "rm", "-f"]]
    assert len(builds) == 1
    assert len(build_dirs) == 2 and all(not path.exists() for path in build_dirs)
    assert (
        ci.read_json(tmp_path / "results/runtime.json")["docker_root_dir"]
        == "/raid/docker"
    )
    assert len(removes) == (0 if failure in {"build", "create"} else 3)
    if failure not in {"build", "create"}:
        names = [c[c.index("--name") + 1] for c in creates]
        assert len(set(names)) == 3
        assert [c[-1] for c in removes] == names
        assert all(
            "--network=none" in c and any(v.endswith(":/models:ro") for v in c)
            for c in creates
        )


def test_partial_run_requires_previous_results_before_docker(
    tmp_path, monkeypatch, plan
):
    monkeypatch.setattr(
        ci,
        "ManagedProcess",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("docker started")),
    )
    args = argparse.Namespace(
        output=str(tmp_path / "results"),
        models_root=str(tmp_path / "models"),
        datasets_root=None,
        lock=str(tmp_path / "host.lock"),
        cases=["Qwen3.5-397B-TP8-DFlash"],
        reuse=None,
    )
    assert ci.run(plan, args) == 1
    text = (tmp_path / "results" / "summary.md").read_text()
    assert "--reuse" in text and "Qwen3.5-397B-TP8-DFlash" in text


def test_reuse_copies_passed_cases_and_rejects_a_failed_one(tmp_path, complete_results):
    rerun = ci.CASES[0]["id"]
    kept = ci.CASES[1]
    cache = complete_results / kept["group"] / kept["id"] / "cache"
    cache.mkdir()
    (cache / "blob").write_text("skip")
    output = tmp_path / "next"
    output.mkdir()
    adopted = ci.adopt_previous_results(complete_results, output, [rerun])
    assert rerun not in adopted and kept["id"] in adopted
    assert (output / kept["group"] / kept["id"] / "gsm8k.json").is_file()
    assert not (output / kept["group"] / kept["id"] / "cache").exists()
    failed = ci.CASES[2]
    state = ci.read_json(complete_results / failed["group"] / failed["id"] / "execution.json")
    state["errors"] = ["boom"]
    ci.write_json(complete_results / failed["group"] / failed["id"] / "execution.json", state)
    with pytest.raises(ValueError, match=failed["id"]):
        ci.adopt_previous_results(complete_results, tmp_path / "again", [rerun])


def test_report_accepts_one_rerun_beside_reused_cases(complete_results, plan):
    rerun = ci.CASES[0]
    runtime = ci.read_json(complete_results / "runtime.json")
    runtime["containers"] = {rerun["group"]: "rerun-container"}
    runtime["reused_containers"] = {
        case["id"]: case["group"] + "-container"
        for case in ci.CASES
        if case["id"] != rerun["id"]
    }
    ci.write_json(complete_results / "runtime.json", runtime)
    state = ci.read_json(complete_results / rerun["group"] / rerun["id"] / "execution.json")
    state["container_id"] = "rerun-container"
    ci.write_json(complete_results / rerun["group"] / rerun["id"] / "execution.json", state)
    assert ci.report(plan, complete_results) == 0
    assert "Reused passed cases" in (complete_results / "summary.md").read_text()
    assert "`Qwen3.5-35B-TP1-DFlash`" in (complete_results / "summary.md").read_text()


def test_docker_rerun_starts_only_the_selected_group(tmp_path, monkeypatch, plan):
    calls = []

    class Process:
        def __init__(self, argv, log_path, **kwargs):
            calls.append(argv)

        def wait(self, *args, **kwargs):
            return 0

    def command(argv, **kwargs):
        calls.append(argv)
        if argv[:2] == ["docker", "info"]:
            return "/raid/docker"
        if argv[:3] == ["docker", "image", "inspect"]:
            return (
                "sha256:built"
                if any(item.startswith("--format") for item in argv)
                else json.dumps(
                    [{"Id": "sha256:base", "RepoDigests": ["image@sha256:" + "a" * 64]}]
                )
            )
        if argv[:2] == ["docker", "create"]:
            return "cid"
        return ""

    monkeypatch.setattr(ci, "ManagedProcess", Process)
    monkeypatch.setattr(ci, "command", command)
    monkeypatch.setattr(ci, "preflight", lambda *args: {"gsm8k": {}})
    monkeypatch.setattr(ci.shutil, "which", lambda name: None)
    monkeypatch.setenv("QWEN4_WORK_ROOT", str(tmp_path / "work"))
    reused = {
        case["id"]: "old-" + case["id"]
        for case in ci.CASES
        if case["id"] != "Qwen3.8-Flash-TP2-MTP"
    }
    monkeypatch.setattr(ci, "adopt_previous_results", lambda *args: reused)
    args = argparse.Namespace(
        output=str(tmp_path / "results"),
        models_root=str(tmp_path / "models"),
        datasets_root=None,
        lock=str(tmp_path / "host.lock"),
        cases=["Qwen3.8-Flash-TP2-MTP"],
        reuse=str(tmp_path / "previous"),
    )
    assert ci.run(plan, args) == 1
    creates = [argv for argv in calls if argv[:2] == ["docker", "create"]]
    execs = [argv for argv in calls if "container" in argv]
    assert len(creates) == 1 and len(execs) == 1
    assert "q38-flash" in execs[0]
    assert execs[0].count("--case") == 1 and "Qwen3.8-Flash-TP2-MTP" in execs[0]


def test_build_failure_marks_cases_not_run_without_missing_file_noise(tmp_path, plan):
    ci.write_json(
        tmp_path / "runtime.json",
        {"errors": ["Candidate build failed; see build.log"], "containers": {}},
    )
    ci.write_json(tmp_path / "assets.json", {"gsm8k": {}})
    assert ci.report(plan, tmp_path) == 1
    rows = ci.read_json(tmp_path / "summary.json")["cases"]
    assert len(rows) == 6 and all(row["status"] == "not_run" for row in rows)
    text = (tmp_path / "summary.md").read_text()
    assert "Candidate build failed" in text
    assert text.count("⚪ Not run") == 6
    assert "No such file" not in text


def test_missing_execution_record_preserves_existing_measurements(
    complete_results, plan
):
    case = ci.CASES[0]
    (complete_results / case["group"] / case["id"] / "execution.json").unlink()
    assert ci.report(plan, complete_results) == 1
    row = ci.read_json(complete_results / "summary.json")["cases"][0]
    assert row["status"] == "failed"
    assert row["performance"] is not None and row["accuracy"] is not None


@pytest.mark.parametrize("in_actions", [False, True])
def test_status_targets_the_candidate_only_in_actions(monkeypatch, plan, in_actions):
    plan["status_target"] = {"repo": "apinge/aiter", "sha": "c" * 40}
    for name, value in {
        "GH_TOKEN": "test-token",
        "GITHUB_ACTIONS": str(in_actions).lower(),
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_REPOSITORY": "apinge/aiter",
        "GITHUB_RUN_ID": "123",
    }.items():
        monkeypatch.setenv(name, value)
    calls = []
    monkeypatch.setattr(
        ci, "github", lambda path, payload: calls.append((path, payload))
    )
    ci.publish_status(plan, "failure")
    assert len(calls) == int(in_actions)
    if in_actions:
        assert calls[0][0] == "repos/apinge/aiter/statuses/" + "c" * 40
        assert calls[0][1]["state"] == "failure"
        assert (
            calls[0][1]["target_url"]
            == "https://github.com/apinge/aiter/actions/runs/123"
        )


@pytest.mark.parametrize("case", ci.CASES, ids=lambda c: c["id"])
def test_existing_launch_recipes_preserve_tp_and_speculative_mode(tmp_path, case):
    import os
    import subprocess

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    capture = tmp_path / "args.json"
    fake = tmp_path / "python"
    fake.write_text(
        "#!/usr/bin/env python3\nimport sys,json\nfrom pathlib import Path\n"
        f"if '-m' in sys.argv: Path({str(capture)!r}).write_text(json.dumps(sys.argv[1:]))\n"
    )
    fake.chmod(0o755)
    env = {
        **os.environ,
        "MODEL_PATH": str(model),
        "DRAFT_MODEL_PATH": str(tmp_path / "draft"),
        "SERVER_PYTHON": str(fake),
        "LOG_FILE": str(tmp_path / "launch.log"),
        "SERVED_MODEL_NAME": "fixture",
        "TP_SIZE": str(case["tp"]),
        "MTP": "1" if case["mode"] == "mtp" else "0",
    }
    subprocess.run(
        ["bash", str(ci.EVALUATION / case["launch_script"])],
        env=env,
        cwd=tmp_path,
        check=True,
        capture_output=True,
        timeout=15,
    )
    args = ci.read_json(capture)
    assert args[args.index("--tp-size") + 1] == str(case["tp"])
    assert args[args.index("--served-model-name") + 1] == "fixture"
    if case["mode"] == "base":
        assert "--speculative-algorithm" not in args
    else:
        assert args[args.index("--speculative-algorithm") + 1] == (
            "DFLASH" if case["mode"] == "dflash" else "EAGLE"
        )
    assert ("--disable-radix-cache" in args) == (not case["prefix_cache"])
