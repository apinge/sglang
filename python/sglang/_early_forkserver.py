"""Opt-in (SGLANG_EARLY_FORKSERVER=1): start the multiprocessing forkserver while
the launching process is still importing, so scheduler and detokenizer processes
are forked from a server that already imported torch and other heavy modules
instead of being spawned and importing everything themselves.

The forkserver does not import sglang: on some platforms importing it initializes
CUDA, and a process forked after that cannot use the GPU. Children inherit the
forkserver's environment, so the launcher sends its current environment along
with each child's target. The child applies it and imports the main module, as
prepare() does for a spawned child, before anything from sglang is imported.

Only the standard library may be imported at module level: the child runs this
file by path, because importing it as a module would import sglang first.
"""

import gc
import multiprocessing
import multiprocessing.context
import multiprocessing.forkserver
import multiprocessing.spawn
import operator
import os
import runpy
import threading
from multiprocessing.reduction import ForkingPickler

_STARTED_ENV = "_SGLANG_EARLY_FORKSERVER_STARTED"

# None of these may initialize CUDA/HIP or query it in a way that poisons fork.
_PRELOAD = ["torch", "torch._inductor.compile_fx", "transformers", "triton", "fastapi"]

# Preparation data that makes a child import the launcher's main module.
_MAIN_KEYS = ("init_main_from_name", "init_main_from_path")

_launching = threading.local()


def _load_target(environ, main, target):
    # Keep the collector off the objects inherited from the forkserver: walking
    # them slows the child's imports and copies their pages.
    gc.freeze()
    try:
        import setproctitle

        # setproctitle sets itself up lazily from the original environ
        # block; it silently stops working once that block is replaced.
        setproctitle.getproctitle()
    except ImportError:
        pass
    os.environ.clear()
    os.environ.update(environ)
    multiprocessing.spawn.prepare(main)
    return ForkingPickler.loads(target)


class _FromThisFile:
    """Unpickles as this file's globals, or one of them, by running the file."""

    def __init__(self, name=None):
        self.name = name

    # pickle requires the first item returned by __reduce__ to be callable.
    def __call__(self, *args):
        return globals()[self.name](*args)

    def __reduce__(self):
        if self.name is None:
            return runpy.run_path, (__file__,)
        return operator.getitem, (_FromThisFile(), self.name)


class _Target:
    """Stands in for a child's target while the launcher pickles the child.

    The child unpickles the process object right after prepare(); its target
    comes before its arguments, so both are loaded after _load_target() ran.
    """

    def __init__(self, target, main):
        self.target = target
        self.main = main

    def __reduce__(self):
        target = bytes(ForkingPickler.dumps(self.target))
        return _FromThisFile("_load_target"), (dict(os.environ), self.main, target)


def _get_preparation_data(name):
    data = _orig_get_preparation_data(name)
    main = getattr(_launching, "main", None)
    if main is not None:
        for key in _MAIN_KEYS:
            if key in data:
                main[key] = data.pop(key)
    return data


def _forkserver_popen(process_obj):
    target = process_obj._target
    _launching.main = {}
    process_obj._target = _Target(target, _launching.main)
    try:
        return _orig_forkserver_popen(process_obj)
    finally:
        _launching.main = None
        process_obj._target = target


_orig_get_preparation_data = multiprocessing.spawn.get_preparation_data
_orig_forkserver_popen = multiprocessing.context.ForkServerProcess._Popen


def maybe_start() -> None:
    # Children import sglang too; only the launcher starts the server.
    if os.environ.get(_STARTED_ENV) == "1":
        return
    os.environ[_STARTED_ENV] = "1"
    multiprocessing.spawn.get_preparation_data = _get_preparation_data
    multiprocessing.context.ForkServerProcess._Popen = staticmethod(_forkserver_popen)
    # transformers calls torch.cuda.is_available() at import, which poisons
    # fork unless it uses the management-library check.
    had_check = os.environ.get("PYTORCH_NVML_BASED_CUDA_CHECK")
    os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
    try:
        multiprocessing.set_forkserver_preload(_PRELOAD)
        multiprocessing.forkserver.ensure_running()
    finally:
        if had_check is None:
            del os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"]
        else:
            os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = had_check
