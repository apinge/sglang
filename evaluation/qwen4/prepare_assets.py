"""Stage the pinned GSM8K file and verify the existing ShareGPT asset."""

import argparse
from pathlib import Path
from urllib.request import urlopen

from common import ROOT, file_hash, read_json


def prepare(root):
    root = Path(root)
    lock = read_json(ROOT / "datasets.lock.json")
    for name in ["gsm8k", "sharegpt"]:
        entry = lock[name]
        path = root / entry["path"]
        if not path.exists():
            if "url" not in entry:
                raise ValueError(
                    f"Stage the approved {name} file at {path} before running CI"
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".download")
            try:
                with urlopen(entry["url"], timeout=60) as response:
                    temporary.write_bytes(response.read())
                if file_hash(temporary) != entry["sha256"]:
                    raise ValueError(f"Downloaded {name} checksum mismatch")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        if file_hash(path) != entry["sha256"]:
            raise ValueError(
                f"{name} differs from datasets.lock.json; no file was overwritten"
            )
        print(f"Verified {name}: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/raid/models/benchmark_datasets")
    prepare(parser.parse_args().root)
