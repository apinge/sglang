"""Merge the validated gfx942 overrides into supplied AITER tables without replacing inputs."""

import argparse
import csv
import hashlib
import io
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AITER_REVISION = "514c166e9abad59acde74309923c2d923a89eee0"
KEYS = {
    "ck": ("gfx", "cu_num", "M", "N", "K", "q_dtype_w"),
    "tgemm": (
        "gfx",
        "cu_num",
        "M",
        "N",
        "K",
        "bias",
        "dtype",
        "outdtype",
        "scaleAB",
        "bpreshuffle",
    ),
}
REQUIRED = {
    "ck": set(KEYS["ck"]) | {"libtype", "kernelId", "splitK", "kernelName"},
    "tgemm": set(KEYS["tgemm"]) | {"libtype", "solidx", "splitK", "kernelName"},
}
OVERLAYS = {
    "ck": ROOT / "configs/ck_accepted_rows.csv",
    "tgemm": ROOT / "configs/tgemm_accepted_rows.csv",
}


def shape_key(row, family):
    """The original pandas merge used numeric coordinates and boolean layout keys."""
    result = []
    for name in KEYS[family]:
        value = str(row[name]).strip()
        if name in ("cu_num", "M", "N", "K"):
            try:
                number = Decimal(value)
            except InvalidOperation as exc:
                raise ValueError(f"Invalid integer coordinate {name}={value}") from exc
            if not number.is_finite() or number != number.to_integral_value():
                raise ValueError(f"Invalid integer coordinate {name}={value}")
            value = int(number)
        elif name in ("bias", "scaleAB", "bpreshuffle"):
            if value.lower() not in ("true", "false"):
                raise ValueError(f"Invalid boolean shape key {name}={value}")
            value = value.lower() == "true"
        result.append(value)
    return tuple(result)


def read_table(path, family):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames or []
        if len(set(names)) != len(names) or not REQUIRED[family].issubset(names):
            raise ValueError(f"Invalid {family} CSV schema: {path}")
        rows = list(reader)
    if not rows or any(
        None in row or any(value is None for value in row.values()) for row in rows
    ):
        raise ValueError(f"Empty or malformed {family} table: {path}")
    seen = set()
    for row in rows:
        key = shape_key(row, family)
        if key in seen:
            raise ValueError(f"Duplicate {family} shape key: {key}")
        seen.add(key)
    return names, rows


def merge_tables(base_names, base_rows, overlay_names, overlay_rows, family):
    replacements = {shape_key(row, family) for row in overlay_rows}
    names = base_names + [name for name in overlay_names if name not in base_names]
    # Same keep-last operation as concat(base, validated_rows).drop_duplicates(keys).
    rows = [row for row in base_rows if shape_key(row, family) not in replacements]
    rows += overlay_rows
    return names, rows


def encode_table(names, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=names, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def prepare_configs(ck_base, tgemm_base, output_dir):
    bases = {"ck": Path(ck_base), "tgemm": Path(tgemm_base)}
    output_dir = Path(output_dir)
    input_paths = {path.resolve() for path in [*bases.values(), *OVERLAYS.values()]}
    outputs = {
        family + "_" + variant + ".csv": output_dir / (family + "_" + variant + ".csv")
        for family in bases
        for variant in ("base", "accepted")
    }
    outputs["MANIFEST.json"] = output_dir / "MANIFEST.json"
    if any(path.resolve() in input_paths for path in outputs.values()):
        raise ValueError("Output would overwrite an input table")
    if any(path.exists() for path in outputs.values()):
        raise ValueError("Output files already exist; use a fresh output directory")
    data, manifest = {}, {"validated_aiter_revision": AITER_REVISION, "tables": {}}
    for family, base_path in bases.items():
        base_names, base_rows = read_table(base_path, family)
        overlay_path = OVERLAYS[family]
        overlay_names, overlay_rows = read_table(overlay_path, family)
        if len(overlay_rows) != (1 if family == "ck" else 4) or any(
            row["gfx"] != "gfx942" or Decimal(row["cu_num"]) != 80
            for row in overlay_rows
        ):
            raise ValueError(
                "Validated overlay must retain the recorded gfx942/80-CU rows"
            )
        names, merged = merge_tables(
            base_names, base_rows, overlay_names, overlay_rows, family
        )
        base_bytes = base_path.read_bytes()
        data[family + "_base.csv"] = base_bytes
        data[family + "_accepted.csv"] = encode_table(names, merged)
        manifest["tables"][family] = {
            "base_path": str(base_path.resolve()),
            "base_sha256": sha256(base_bytes),
            "overlay_sha256": sha256(overlay_path.read_bytes()),
            "merged_sha256": sha256(data[family + "_accepted.csv"]),
            "base_rows": len(base_rows),
            "overlay_rows": len(overlay_rows),
            "merged_rows": len(merged),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in data.items():
        outputs[name].write_bytes(value)
    outputs["MANIFEST.json"].write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ck-base", type=Path, required=True)
    parser.add_argument("--tgemm-base", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_configs(args.ck_base, args.tgemm_base, args.output_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
