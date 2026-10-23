import http.server
import json
import socket
import sys
import threading
import time

import pytest
from config import make_plan
from process import (
    ManagedProcess,
    host_lock,
    port_available,
    wait_ready,
)


def test_dead_server_is_detected_without_waiting_for_timeout(tmp_path):
    process = ManagedProcess(
        [sys.executable, "-c", "raise SystemExit(7)"], tmp_path / "server.log"
    )
    process.process.wait()
    case = make_plan()["groups"][0]["cases"][0]
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
        case = make_plan()["groups"][0]["cases"][0]
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
