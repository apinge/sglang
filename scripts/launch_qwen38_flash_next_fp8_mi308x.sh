#!/usr/bin/env bash
# Launch Qwen3.8-Flash-Next-FP8 on two MI308X (gfx942) GPUs.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/Qwen3.8-Flash-Next-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.8-Flash-Next-FP8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-2}"
EP_SIZE="${EP_SIZE:-2}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.75}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-4}"
CUDA_GRAPH_MAX_BS_DECODE="${CUDA_GRAPH_MAX_BS_DECODE:-4}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Model config not found: ${MODEL_PATH}/config.json" >&2
  exit 1
fi

if (( TP_SIZE <= 0 || EP_SIZE <= 0 || TP_SIZE % EP_SIZE != 0 )); then
  echo "TP_SIZE must be positive and divisible by EP_SIZE (got TP=${TP_SIZE}, EP=${EP_SIZE})." >&2
  exit 1
fi

python - "${TP_SIZE}" <<'PY'
import sys

import torch

tp_size = int(sys.argv[1])
if torch.version.hip is None:
    raise SystemExit("ROCm PyTorch is required: torch.version.hip is empty.")
if not torch.cuda.is_available():
    raise SystemExit("No ROCm GPU is available to PyTorch.")
if torch.cuda.device_count() < tp_size:
    raise SystemExit(
        f"TP_SIZE={tp_size} requires at least {tp_size} visible GPUs; "
        f"only {torch.cuda.device_count()} are visible."
    )
for index in range(tp_size):
    arch = torch.cuda.get_device_properties(index).gcnArchName.split(":", 1)[0]
    if arch != "gfx942":
        raise SystemExit(
            f"GPU {index} has architecture {arch}; this MI308X script requires gfx942."
        )
PY

# Match the AMD nightly correctness configuration. Explicit AITER backends are
# selected below while this disables the unreleased global paged-QSA path.
export SGLANG_USE_AITER=0

command=(
  sglang serve
  --model-path "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  # ROCm PyTorch exposes GPU devices through the torch.cuda API.
  --device cuda
  --tp-size "${TP_SIZE}"
  --ep-size "${EP_SIZE}"
  --attention-backend aiter
  --moe-runner-backend aiter
  --kv-cache-dtype auto
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
  --watchdog-timeout 1200
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  --cuda-graph-max-bs-decode "${CUDA_GRAPH_MAX_BS_DECODE}"
  --speculative-algorithm EAGLE
  --speculative-num-steps 3
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens 4
)

if (( $# > 0 )); then
  command+=("$@")
fi

printf 'Launching: '
printf '%q ' "${command[@]}"
printf '\n'
exec "${command[@]}"
