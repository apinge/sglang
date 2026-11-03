"""Fail closed when a stream-ordered ROCm IPC readiness wait stops progressing.

A wait must not return on timeout: copies and acknowledgements are already
queued behind it. Terminating the consumer lets the engine's subprocess
watchdog tear down the worker group without exposing unfinished feature data.
"""

import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class _PendingWait:
    device_id: int
    address: int
    generation: int
    transport_name: str
    timeout: float
    deadline: float
    event: Optional[Any] = None


def _terminate_consumer(wait: _PendingWait, reason: str) -> None:
    try:
        logger.error(
            "ROCm IPC readiness wait failed: %s; transport=%s device=cuda:%d "
            "address=%#x generation=%d timeout=%.3fs. Terminating consumer "
            "for worker-group cleanup; the feature must not be copied or acknowledged.",
            reason,
            wait.transport_name,
            wait.device_id,
            wait.address,
            wait.generation,
            wait.timeout,
        )
    finally:
        # GPU synchronization or Python cleanup can block behind the stalled
        # wait. A nonzero exit triggers the engine's SubprocessWatchdog.
        os._exit(1)


class IpcWaitWatchdog:
    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        on_failure: Callable[[_PendingWait, str], None] = _terminate_consumer,
    ) -> None:
        self._clock = clock
        self._on_failure = on_failure
        self._pending: set[_PendingWait] = set()
        self._lock = threading.Lock()
        self._wake = threading.Event()

    def start(self) -> None:
        threading.Thread(
            target=self._run, name="ROCmIpcWaitWatchdog", daemon=True
        ).start()

    def arm(
        self,
        *,
        device_id: int,
        address: int,
        generation: int,
        transport_name: str,
        timeout: float,
    ) -> _PendingWait:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(
                "--watchdog-timeout must be finite and positive for ROCm IPC waits"
            )
        wait = _PendingWait(
            device_id,
            address,
            generation,
            transport_name,
            timeout,
            self._clock() + timeout,
        )
        with self._lock:
            self._pending.add(wait)
        self._wake.set()
        return wait

    def launched(self, wait: _PendingWait, event: Any) -> None:
        # An unrecorded CUDA event reports completion, so publish it only after
        # record() has enqueued it behind the readiness kernel.
        with self._lock:
            wait.event = event
        self._wake.set()

    def cancel(self, wait: _PendingWait) -> bool:
        with self._lock:
            if wait not in self._pending:
                return False
            self._pending.remove(wait)
            return True

    def _poll_once(self) -> None:
        with self._lock:
            pending = tuple(self._pending)
        for wait in pending:
            try:
                # CUDA/HIP Event.query() is nonblocking and selects its event's
                # device internally. Never synchronize the consumer stream here.
                if wait.event is not None and wait.event.query():
                    self.cancel(wait)
                    continue
            except Exception as exc:
                if self.cancel(wait):
                    self._on_failure(wait, f"GPU completion query failed: {exc}")
                continue
            if self._clock() >= wait.deadline and self.cancel(wait):
                self._on_failure(wait, "readiness deadline exceeded")

    def _run(self) -> None:
        while True:
            with self._lock:
                interval = 0.05 if self._pending else None
            self._wake.wait(interval)
            self._wake.clear()
            self._poll_once()


_watchdog: Optional[IpcWaitWatchdog] = None
_watchdog_lock = threading.Lock()


def get_ipc_wait_watchdog() -> IpcWaitWatchdog:
    global _watchdog
    with _watchdog_lock:
        if _watchdog is None:
            watchdog = IpcWaitWatchdog()
            watchdog.start()
            _watchdog = watchdog
        return _watchdog
