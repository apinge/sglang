"""CI authorization, script recipes, lifecycle failures, and native result contracts."""

import argparse
import http.server
import json
import re
import socket
import sys
import threading
import time

import pytest
import qwen4_ci as ci
from qwen4_ci import (
    ManagedProcess,
    gsm8k_selection,
    host_lock,
    port_available,
    read_json,
    validate_accuracy,
    wait_ready,
)


@pytest.mark.parametrize(
    "http_error,truncate", [(False, False), (True, False), (False, True)]
)
def test_audited_sampler_counts_errors_without_changing_scorer(
    tmp_path, http_error, truncate
):
    pytest.importorskip("sglang", reason="Installed-runtime client integration")
    from sglang.test.run_eval import run_eval

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if http_error:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"test failure"}}')
                return
            index = int(
                re.findall(r"Question: Q(\d+)", body["messages"][0]["content"])[-1]
            )
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
                                    "content": "" if truncate else str(index),
                                },
                                "finish_reason": "length" if truncate else "stop",
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
        num_examples=2,
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
        strict=True,
        temperature=0.0,
        top_p=1.0,
    )
    try:
        if http_error:
            with pytest.raises(RuntimeError, match="Incomplete GSM8K"):
                run_eval(args)
        else:
            run_eval(args)
        result = read_json(tmp_path / "results/gsm8k.json")
        assert result["scored"] == 2
        if http_error:
            assert result["errors"] == 2 and result["completed"] == 0
            with pytest.raises(ValueError):
                validate_accuracy(
                    tmp_path / "results/gsm8k.json", 2, gsm8k_selection(data, 2)
                )
        else:
            assert result["score"] == (0 if truncate else 1)
            if truncate:
                assert result["empty_outputs"] == 2 and result["truncated"] == 2
            validate_accuracy(
                tmp_path / "results/gsm8k.json", 2, gsm8k_selection(data, 2)
            )
    finally:
        server.shutdown()
        server.server_close()


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
def plan():
    return {
        "sources": {
            name: {"repo": f"apinge/{name}", "sha": str(i) * 40}
            for i, name in enumerate(["sglang", "aiter", "pyhip", "client"], 1)
        },
        "base_image": ci.BASE_IMAGE,
        "smoke": True,
    }


@pytest.fixture
def complete_results(tmp_path, plan):
    data = tmp_path / "gsm8k-source.jsonl"
    data.write_text(
        "".join(
            json.dumps({"question": f"Q{i}", "answer": f"#### {i}"}) + "\n"
            for i in range(7)
        )
    )
    selection = ci.gsm8k_selection(data, 2)
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
        records = [
            {"example_id": i, "status": "completed", "content": "", "correct": False}
            for i in selection["example_ids"]
        ]
        (path / "gsm8k-examples.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in records)
        )
        ci.write_json(
            path / "gsm8k.json",
            {
                "requested": 2,
                "scored": 2,
                "completed": 2,
                "errors": 0,
                "score": 0,
                "selection": selection,
                "records_sha256": ci.file_hash(path / "gsm8k-examples.jsonl"),
            },
        )
        (path / "performance.jsonl").write_text(
            json.dumps(
                {
                    "completed": 2,
                    "errors": ["", ""],
                    "output_lens": [case["output_tokens"]] * 2,
                    "mean_ttft_ms": 10,
                    "mean_tpot_ms": 2,
                    "output_throughput": 50,
                }
            )
            + "\n"
        )
    return tmp_path


def test_low_accuracy_is_advisory_when_all_six_cases_complete(complete_results, plan):
    assert ci.report(plan, complete_results) == 0
    summary = ci.read_json(complete_results / "summary.json")
    assert len(summary["cases"]) == 6
    assert all(row["accuracy"]["score"] == 0 for row in summary["cases"])


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

    class Process:
        def __init__(self, argv, log_path, **kwargs):
            self.argv = argv
            calls.append(argv)

        def wait(self, *args, **kwargs):
            return int(
                (failure == "build" and self.argv[:2] == ["docker", "build"])
                or (failure == "group" and "container" in self.argv)
            )

    def command(argv, **kwargs):
        calls.append(argv)
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
    assert len(removes) == (0 if failure in {"build", "create"} else 3)
    if failure not in {"build", "create"}:
        names = [c[c.index("--name") + 1] for c in creates]
        assert len(set(names)) == 3
        assert [c[-1] for c in removes] == names
        assert all(
            "--network=none" in c and any(v.endswith(":/models:ro") for v in c)
            for c in creates
        )


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
