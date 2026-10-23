"""Small standard-library helpers shared by the host and container entrypoints."""

import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SHA = re.compile(r"^[0-9a-f]{40}$")
NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
REPO = re.compile(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def command_output(argv, **kwargs):
    return subprocess.run(
        argv, check=True, text=True, stdout=subprocess.PIPE, **kwargs
    ).stdout.strip()


def checked_name(value):
    if not isinstance(value, str) or not NAME.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"Invalid identifier: {value!r}")
    return value


def clean_environment(extra=None):
    """Build each child environment afresh; never carry a previous case's knobs."""
    allowed = {
        "PATH",
        "HOME",
        "USER",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "ROCM_PATH",
        "HIP_PATH",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
    }
    result = {key: value for key, value in os.environ.items() if key in allowed}
    result.update({"PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1"})
    result.update(extra or {})
    return result
