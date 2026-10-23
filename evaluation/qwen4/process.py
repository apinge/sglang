"""Owned process groups, bounded HTTP checks, and a cross-repository host lock."""

import contextlib
import fcntl
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from urllib.request import Request, urlopen


@contextlib.contextmanager
def host_lock(path, timeout=3600):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Host is reserved: {path}")
                time.sleep(1)
        try:
            stream.seek(0)
            stream.truncate()
            stream.write(f"pid={os.getpid()}\n")
            stream.flush()
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def http_json(url, payload=None, timeout=5):
    body = json.dumps(payload).encode() if payload is not None else None
    request = Request(url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        raw = response.read()
        return json.loads(raw) if raw else None


def port_available(port):
    with socket.socket() as sock:
        # Uvicorn also uses SO_REUSEADDR. Closed connections in TIME_WAIT must
        # not be mistaken for a surviving listener after a clean shutdown.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def group_members(pgid):
    """Live members of an owned Linux process group, excluding unreaped zombies."""
    members = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(") ", 1)[1].split()
            if int(fields[2]) == pgid and fields[0] not in {"Z", "X"}:
                members.append(int(path.parent.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return members


def wait_port_released(port, timeout=10):
    deadline = time.monotonic() + timeout
    while not port_available(port):
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Server port {port} remained allocated after cleanup")
        time.sleep(0.1)


class ManagedProcess:
    def __init__(self, argv, log_path, env=None, cwd=None):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("w")
        self._stopped = False
        try:
            self.process = subprocess.Popen(
                argv,
                env=env,
                cwd=cwd,
                stdout=self.log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except BaseException:
            self.log.close()
            raise

    def alive(self):
        return self.process.poll() is None

    def stop(self, timeout=60):
        if self._stopped:
            return
        # Even an exited leader can have live worker descendants in its group.
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + timeout
        while group_members(self.process.pid) and time.monotonic() < deadline:
            self.process.poll()
            time.sleep(0.1)
        if group_members(self.process.pid):
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 10
            while group_members(self.process.pid) and time.monotonic() < deadline:
                self.process.poll()
                time.sleep(0.1)
        self.process.wait(timeout=10)
        self.log.close()
        if group_members(self.process.pid):
            raise RuntimeError("Owned server process group did not stop")
        self._stopped = True

    def wait(self, timeout, progress=False):
        try:
            if not progress:
                return self.process.wait(timeout=timeout)
            started = time.monotonic()
            deadline = started + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(self.process.args, timeout)
                try:
                    return self.process.wait(timeout=min(30, remaining))
                except subprocess.TimeoutExpired:
                    print(
                        f"Still running {self.log_path} ({time.monotonic() - started:.0f}s)",
                        flush=True,
                    )
        finally:
            self.stop()


def wait_ready(process, base_url, case, model_path, timeout=3600):
    deadline = time.monotonic() + timeout
    last_error = "no response"
    while time.monotonic() < deadline:
        if not process.alive():
            raise RuntimeError(
                f"Server exited before readiness: {process.process.returncode}"
            )
        try:
            models = http_json(base_url + "/v1/models")
            if case.get("served_model_name", case["id"]) not in [
                m["id"] for m in models["data"]
            ]:
                raise ValueError("Endpoint is serving a different model identity")
            info = http_json(base_url + "/server_info")
            args = info.get("server_args", info)
            if args.get("tp_size") != case["tp"]:
                raise ValueError(f"Server TP mismatch: {args.get('tp_size')}")
            if str(args.get("model_path", "")).rstrip("/") != str(model_path).rstrip(
                "/"
            ):
                raise ValueError("Server model path mismatch")
            expected = {"base": None, "dflash": "DFLASH", "mtp": "EAGLE"}[case["mode"]]
            if args.get("speculative_algorithm") != expected:
                raise ValueError("Server speculative mode mismatch")
            if bool(args.get("disable_radix_cache")) != (not case["prefix_cache"]):
                raise ValueError("Server prefix cache mode mismatch")
            if not process.alive():
                raise RuntimeError("Server exited during readiness")
            return info
        except ValueError:
            raise  # A responding but wrong endpoint must never count as ready.
        except Exception as exc:
            last_error = str(exc)
            time.sleep(min(2, max(0, deadline - time.monotonic())))
    raise TimeoutError(f"Server readiness timed out: {last_error}")


def smoke_generation(base_url, model, timeout=1200):
    result = http_json(
        base_url + "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "What is 1 + 1? Answer briefly."}],
            "temperature": 0,
            "max_tokens": 64,
        },
        timeout=timeout,
    )
    choice = result["choices"][0]
    message = choice["message"]
    if not (message.get("content") or message.get("reasoning_content")):
        raise ValueError("Smoke request returned no generated text")
    return result
