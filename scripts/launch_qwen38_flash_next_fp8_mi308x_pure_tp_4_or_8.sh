#!/usr/bin/env bash
# Launch Qwen3.8-Flash-Next-FP8 with pure TP4 or TP8 on MI308X (gfx942).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/logs/qwen3.8_flash_next_fp8_mi308x_pure_tp_4_or_8_$(date -u +%Y%m%dT%H%M%SZ).log}"
mkdir -p "$(dirname -- "${LOG_FILE}")"
# Capture both this script's preflight checks and all sglang serve output.
# Using a process substitution (instead of `command 2>&1 | tee`) preserves the
# server process's exit status when it terminates.
exec > >(tee "${LOG_FILE}") 2>&1
printf 'Logging to: %s\n' "${LOG_FILE}"

MODEL_PATH="${MODEL_PATH:-/models/Qwen3.8-Flash-Next-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.8-Flash-Next-FP8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7080}"
TP_SIZE="${TP_SIZE:-8}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
#CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
CUDA_GRAPH_MAX_BS_DECODE="${CUDA_GRAPH_MAX_BS_DECODE:-32}"
AITER_MOE_PADDING_SIZE="${AITER_MOE_PADDING_SIZE:-128}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Model config not found: ${MODEL_PATH}/config.json" >&2
  exit 1
fi

if (( TP_SIZE != 4 && TP_SIZE != 8 )); then
  echo "This pure-TP script supports TP_SIZE=4 or TP_SIZE=8 (got ${TP_SIZE})." >&2
  exit 1
fi

# Qwen3.8's native FP8 MoE uses 128-wide checkpoint blocks. This makes the
# local MoE buffers 160 -> 256 for TP4 and 80 -> 128 for TP8.
if (( AITER_MOE_PADDING_SIZE != 128 )); then
  echo "AITER_MOE_PADDING_SIZE must be 128 for pure TP4/TP8 (got ${AITER_MOE_PADDING_SIZE})." >&2
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
#unset SGLANG_USE_AITER
export SGLANG_USE_AITER=0
export AITER_MOE_PADDING_SIZE

command=(
  sglang serve
  --model-path "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --tp-size "${TP_SIZE}"
  --attention-backend aiter
  --moe-runner-backend aiter
  --kv-cache-dtype auto
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
  --watchdog-timeout 1200
  --disable-radix-cache
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  --cuda-graph-max-bs-decode "${CUDA_GRAPH_MAX_BS_DECODE}"
  #--moe-runner-backend triton 
  # --speculative-algorithm EAGLE
  # --speculative-num-steps 3
  # --speculative-eagle-topk 1
  # --speculative-num-draft-tokens 4
)

if (( $# > 0 )); then
  command+=("$@")
fi

printf 'Launching: '
printf '%q ' "${command[@]}"
printf '\n'
exec "${command[@]}"
