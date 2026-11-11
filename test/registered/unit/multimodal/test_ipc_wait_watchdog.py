"""Host-side deadline and failure handling for ROCm IPC readiness waits."""

import unittest
from unittest.mock import Mock, patch

from sglang.srt.multimodal.transport import ipc_wait_watchdog
from sglang.srt.multimodal.transport.ipc_wait_watchdog import IpcWaitWatchdog
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestIpcWaitWatchdog(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.now = 0.0
        self.failure = Mock()
        self.watchdog = IpcWaitWatchdog(clock=lambda: self.now, on_failure=self.failure)

    def arm(self, timeout=5.0):
        return self.watchdog.arm(
            device_id=1,
            address=0x1000,
            generation=17,
            transport_name="test IPC",
            timeout=timeout,
        )

    def test_completed_wait_does_not_fail_after_deadline(self):
        wait = self.arm()
        event = Mock()
        event.query.return_value = True
        self.watchdog.launched(wait, event)
        self.now = 6.0
        self.watchdog._poll_once()
        self.watchdog._poll_once()
        event.query.assert_called_once_with()
        self.failure.assert_not_called()

    def test_pending_wait_expires_once_with_diagnostics(self):
        wait = self.arm()
        event = Mock()
        event.query.return_value = False
        self.watchdog.launched(wait, event)
        self.now = 4.0
        self.watchdog._poll_once()
        self.failure.assert_not_called()
        self.now = 5.0
        self.watchdog._poll_once()
        self.watchdog._poll_once()
        self.failure.assert_called_once_with(wait, "readiness deadline exceeded")
        self.assertEqual(
            (wait.device_id, wait.address, wait.generation, wait.transport_name),
            (1, 0x1000, 17, "test IPC"),
        )

    def test_wait_is_armed_before_completion_event_is_recorded(self):
        wait = self.arm()
        self.now = 5.0
        self.watchdog._poll_once()
        self.failure.assert_called_once_with(wait, "readiness deadline exceeded")

    def test_cancelled_launch_does_not_fail(self):
        wait = self.arm()
        self.watchdog.cancel(wait)
        self.now = 10.0
        self.watchdog._poll_once()
        self.failure.assert_not_called()

    def test_later_wait_does_not_extend_earlier_deadline(self):
        first = self.arm()
        self.now = 4.0
        second = self.arm()
        self.now = 5.0
        self.watchdog._poll_once()
        self.failure.assert_called_once_with(first, "readiness deadline exceeded")
        self.now = 9.0
        self.watchdog._poll_once()
        self.assertEqual(self.failure.call_count, 2)
        self.failure.assert_called_with(second, "readiness deadline exceeded")

    def test_gpu_query_failure_terminates_consumer(self):
        wait = self.arm()
        event = Mock()
        event.query.side_effect = RuntimeError("GPU context lost")
        self.watchdog.launched(wait, event)
        self.watchdog._poll_once()
        self.failure.assert_called_once_with(
            wait, "GPU completion query failed: GPU context lost"
        )

    def test_cancel_during_query_prevents_stale_failure(self):
        wait = self.arm()
        event = Mock()

        def query():
            self.watchdog.cancel(wait)
            raise RuntimeError("cancelled")

        event.query.side_effect = query
        self.watchdog.launched(wait, event)
        self.watchdog._poll_once()
        self.failure.assert_not_called()

    def test_timeout_cannot_disable_supervision(self):
        for timeout in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(timeout=timeout):
                with self.assertRaisesRegex(
                    ValueError, "--watchdog-timeout.*finite and positive"
                ):
                    self.arm(timeout)

    def test_failed_thread_start_is_not_cached(self):
        failed, healthy = Mock(), Mock()
        failed.start.side_effect = RuntimeError("cannot start thread")
        with patch.object(ipc_wait_watchdog, "_watchdog", None), patch.object(
            ipc_wait_watchdog, "IpcWaitWatchdog", side_effect=[failed, healthy]
        ):
            with self.assertRaisesRegex(RuntimeError, "cannot start thread"):
                ipc_wait_watchdog.get_ipc_wait_watchdog()
            self.assertIsNone(ipc_wait_watchdog._watchdog)
            self.assertIs(ipc_wait_watchdog.get_ipc_wait_watchdog(), healthy)
            self.assertIs(ipc_wait_watchdog.get_ipc_wait_watchdog(), healthy)
        healthy.start.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
