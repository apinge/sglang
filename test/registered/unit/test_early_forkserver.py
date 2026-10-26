"""Unit test for sglang._early_forkserver: children forked from the early
forkserver see the environment the launcher has when it starts them."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

import os
import subprocess
import sys
import tempfile
import unittest

from sglang.test.test_utils import CustomTestCase

_TARGET = """
import os

AT_IMPORT = os.environ.get("EARLY_FORKSERVER_TEST_VAR")


def child(queue):
    queue.put((AT_IMPORT, os.environ.get("EARLY_FORKSERVER_TEST_VAR")))
"""

_LAUNCHER = """
import multiprocessing as mp
import os

import sglang  # starts the early forkserver
import target

if __name__ == "__main__":
    os.environ["EARLY_FORKSERVER_TEST_VAR"] = "set-after-import"
    mp.set_start_method("forkserver", force=True)
    queue = mp.Queue()
    process = mp.Process(target=target.child, args=(queue,))
    process.start()
    print(queue.get(timeout=240))
    process.join()
"""


class TestEarlyForkserver(CustomTestCase):
    def test_child_sees_environment_set_after_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, source in (("target.py", _TARGET), ("launcher.py", _LAUNCHER)):
                with open(os.path.join(tmp, name), "w") as f:
                    f.write(source)
            result = subprocess.run(
                [sys.executable, "launcher.py"],
                cwd=tmp,
                env={**os.environ, "SGLANG_EARLY_FORKSERVER": "1"},
                capture_output=True,
                text=True,
                timeout=300,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.strip().splitlines()[-1],
            str(("set-after-import", "set-after-import")),
        )


if __name__ == "__main__":
    unittest.main()
