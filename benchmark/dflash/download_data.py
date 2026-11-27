"""Download the five public workloads into distinct, validated JSONL files."""

import argparse
import gzip
import hashlib
import json
import urllib.request
from pathlib import Path

DATASETS = {
    "gsm8k": (
        "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
        {"question", "answer"},
    ),
    "math500": (
        "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl",
        {"problem", "answer"},
    ),
    "humaneval": (
        "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz",
        {"prompt", "test", "entry_point"},
    ),
    "mbpp": (
        "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl",
        {"text", "test_list"},
    ),
    "mtbench": (
        "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",
        {"turns"},
    ),
}


def validate(name, data):
    fields = DATASETS[name][1]
    rows = [
        json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()
    ]
    if not rows or any(
        not isinstance(row, dict) or not fields <= row.keys() for row in rows
    ):
        raise RuntimeError(
            f"Invalid {name} dataset: expected {sorted(fields)} in every row"
        )
    return len(rows)


def ensure_dataset(name, path):
    path = Path(path)
    url, _ = DATASETS[name]
    if path.is_file():
        data = path.read_bytes()
    else:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        if url.endswith(".gz"):
            data = gzip.decompress(data)
        validate(name, data)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    count = validate(name, data)
    print(f"{name}: {count} rows; SHA256 {hashlib.sha256(data).hexdigest()}; {path}")
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent / "data"
    )
    args = parser.parse_args()
    for name in DATASETS:
        ensure_dataset(name, args.output_dir / f"{name}.jsonl")


if __name__ == "__main__":
    main()
