"""Inventory local, read-only assets before loading any model."""

import json
from pathlib import Path

from common import ROOT, digest, file_hash, read_json, write_json


def gsm8k_selection(path, count=200, shots=5):
    rows = [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]
    if len(rows) < shots + count:
        raise ValueError(f"GSM8K needs {shots + count} rows, found {len(rows)}")
    for row in rows:
        if not isinstance(row.get("question"), str) or not isinstance(
            row.get("answer"), str
        ):
            raise ValueError("Invalid GSM8K row")
    ids = [digest(row) for row in rows]
    prompt_ids, example_ids = ids[:shots], ids[shots : shots + count]
    if set(prompt_ids) & set(example_ids) or len(set(example_ids)) != count:
        raise ValueError("GSM8K prompt/test overlap or duplicate scored examples")
    return {
        "sha256": file_hash(path),
        "rows": len(rows),
        "shots": shots,
        "prompt_ids": prompt_ids,
        "example_ids": example_ids,
    }


def model_manifest(directory, hash_weights=False):
    directory = Path(directory)
    config = directory / "config.json"
    if not config.is_file():
        raise ValueError(f"Missing model config: {config}")
    json.loads(config.read_text())
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        shards = [p.name for p in sorted(directory.glob("*.safetensors"))]
    if not shards:
        raise ValueError(f"No safetensors weights found: {directory}")
    files = {}
    for name in shards:
        path = directory / name
        if Path(name).name != name or not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or invalid weight shard: {path}")
        stat = path.stat()
        entry = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if hash_weights:
            entry["sha256"] = file_hash(path)
        metadata = directory / ".cache/huggingface/download" / f"{name}.metadata"
        if metadata.is_file():
            entry["huggingface_metadata"] = metadata.read_text().splitlines()[:2]
        files[name] = entry
    small_files = {}
    for pattern in ["*.json", "*.jinja", "*.model"]:
        for path in sorted(directory.glob(pattern)):
            small_files[path.name] = file_hash(path)
    return {
        "directory": directory.name,
        "files": files,
        "metadata_sha256": small_files,
        "weight_identity": "sha256" if hash_weights else "size-mtime-hf-metadata",
        "identity": digest({"files": files, "metadata": small_files}),
    }


def inventory(plan, models_root, gsm8k_path, sharegpt_path, output, hash_weights=False):
    models = {}
    for group in plan["groups"]:
        names = [group["directory"]]
        if any(case["mode"] == "dflash" for case in group["cases"]):
            names.append(group["draft_directory"])
        for name in names:
            models[name] = model_manifest(Path(models_root) / name, hash_weights)
    result = {"models": models}
    if plan["workload"]["accuracy"]:
        result["gsm8k"] = gsm8k_selection(
            gsm8k_path, plan["workload"]["examples"], plan["defaults"]["gsm8k_shots"]
        )
    if plan["workload"]["performance"]:
        dataset = json.loads(Path(sharegpt_path).read_text())
        if not isinstance(dataset, list) or len(dataset) < plan["workload"]["prompts"]:
            raise ValueError("ShareGPT is not a sufficiently large conversation list")
        result["sharegpt"] = {"sha256": file_hash(sharegpt_path), "rows": len(dataset)}
    approved = read_json(ROOT / "datasets.lock.json")
    for name in ["gsm8k", "sharegpt"]:
        if name in result and result[name]["sha256"] != approved[name]["sha256"]:
            raise ValueError(f"{name} does not match the approved dataset checksum")
    result["identity"] = digest(result)
    write_json(output, result)
    return result
