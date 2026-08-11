#!/bin/bash

set -euo pipefail

DATASETS_ROOT=${1:-/models/benchmark_datasets}
SHAREGPT_REPO_ID=${SHAREGPT_REPO_ID:-anon8231489123/ShareGPT_Vicuna_unfiltered}
SHAREGPT_FILENAME=${SHAREGPT_FILENAME:-ShareGPT_V3_unfiltered_cleaned_split.json}
TARGET_PATH="${DATASETS_ROOT}/${SHAREGPT_FILENAME}"

mkdir -p "${DATASETS_ROOT}"

if python3 - <<'PY' "${TARGET_PATH}"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
try:
    json.loads(path.read_text(encoding="utf-8"))
except json.JSONDecodeError:
    raise SystemExit(1)
print(f"ShareGPT dataset already present: {path}")
PY
then
  exit 0
fi

echo "Downloading ShareGPT dataset to ${TARGET_PATH}"
python3 - <<'PY' "${SHAREGPT_REPO_ID}" "${SHAREGPT_FILENAME}" "${TARGET_PATH}"
import shutil
import sys

from huggingface_hub import hf_hub_download

repo_id, filename, target_path = sys.argv[1:4]
downloaded = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    repo_type="dataset",
)
shutil.copyfile(downloaded, target_path)
print(f"Saved ShareGPT dataset to {target_path}")
PY
