#!/usr/bin/env bash
# Launch Qwen3.8-Flash-Next-FP8 with pure TP2/TP4/TP8 on MI308X (gfx942).
# Examples (other settings stay the same):
#   MAMBA_SSM_DTYPE=float32 bash scripts/launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh
#   MAMBA_SSM_DTYPE=bfloat16 bash scripts/launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PLE_OFFLOAD_EMBEDDING="${PLE_OFFLOAD_EMBEDDING:-0}"
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/logs/qwen3.8_flash_next_fp8_mi308x_pure_tp_4_or_8_pleoffload${PLE_OFFLOAD_EMBEDDING}_$(date -u +%Y%m%dT%H%M%SZ).log}"
mkdir -p "$(dirname -- "${LOG_FILE}")"
# Capture both this script's preflight checks and all sglang serve output.
# Using a process substitution (instead of `command 2>&1 | tee`) preserves the
# server process's exit status when it terminates.
exec > >(tee "${LOG_FILE}") 2>&1
printf 'Logging to: %s\n' "${LOG_FILE}"

MODEL_PATH="${MODEL_PATH:-/models/Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7080}"
TP_SIZE="${TP_SIZE:-2}"
# TP2 needs room for HIPBLAS initialization and graph capture after KV sizing;
# TP4/TP8 have previously been validated with the more conservative 0.85.
if [[ -z "${MEM_FRACTION_STATIC:-}" ]]; then
  if (( TP_SIZE == 2 )); then
    MEM_FRACTION_STATIC=0.95
  else
    MEM_FRACTION_STATIC=0.85
  fi
fi
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
# Persistent SSM state pool; the AITER per-chunk h snapshots remain BF16.
# Default to float32 for the FP32-state control run; set MAMBA_SSM_DTYPE=bfloat16 to use BF16.
MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-float32}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
CUDA_GRAPH_MAX_BS_DECODE="${CUDA_GRAPH_MAX_BS_DECODE:-32}"
AITER_MOE_PADDING_SIZE="${AITER_MOE_PADDING_SIZE:-64}"
# Select the public prefill backend; use triton for the control run.
LINEAR_ATTN_PREFILL_BACKEND="${LINEAR_ATTN_PREFILL_BACKEND:-aiter}"
case "${LINEAR_ATTN_PREFILL_BACKEND}" in
  triton|aiter)
    ;;
  *)
    echo "LINEAR_ATTN_PREFILL_BACKEND must be 'triton' or 'aiter' (got ${LINEAR_ATTN_PREFILL_BACKEND})." >&2
    exit 1
    ;;
esac
DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-1}"
# Report cache hits by default when radix caching is enabled.
ENABLE_CACHE_REPORT="${ENABLE_CACHE_REPORT:-1}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Model config not found: ${MODEL_PATH}/config.json" >&2
  exit 1
fi

if (( TP_SIZE != 4 && TP_SIZE != 8 && TP_SIZE != 2)); then
  echo "This pure-TP script supports TP_SIZE=4 or TP_SIZE=8 or TP 2(got ${TP_SIZE})." >&2
  exit 1
fi

if [[ "${PLE_OFFLOAD_EMBEDDING}" != "0" && "${PLE_OFFLOAD_EMBEDDING}" != "1" ]]; then
  echo "PLE_OFFLOAD_EMBEDDING must be 0 or 1 (got ${PLE_OFFLOAD_EMBEDDING})." >&2
  exit 1
fi

case "${MAMBA_SSM_DTYPE}" in
  float32|bfloat16) ;;
  *)
    echo "MAMBA_SSM_DTYPE must be float32 or bfloat16 for this AITER GDN setup (got ${MAMBA_SSM_DTYPE})." >&2
    exit 1
    ;;
esac

if (( DISABLE_RADIX_CACHE )); then
  ENABLE_CACHE_REPORT=0
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
export SGLANG_USE_AITER=1
export SGLANG_GR_READ_FLYDSL="${SGLANG_GR_READ_FLYDSL:-1}"
export SGLANG_USE_PYHIP_QSA="${SGLANG_USE_PYHIP_QSA:-1}"
export AITER_MOE_PADDING_SIZE
printf 'GR read: installed PyHIP enabled=%s; TP=%s; AITER_MOE_PADDING_SIZE=%s; LINEAR_ATTN_PREFILL_BACKEND=%s\n' \
  "${SGLANG_GR_READ_FLYDSL}" "${TP_SIZE}" "${AITER_MOE_PADDING_SIZE}" \
  "${LINEAR_ATTN_PREFILL_BACKEND}"
printf 'SSM state dtype: %s (extra CLI arguments may override script defaults)\n' "${MAMBA_SSM_DTYPE}"
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# export HIP_VISIBLE_DEVICES=4,5,6,7
command=(
  sglang serve
  --model-path "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --tp-size "${TP_SIZE}"
  --attention-backend aiter
  --moe-runner-backend aiter
  --linear-attn-backend triton
  --linear-attn-decode-backend triton
  --linear-attn-prefill-backend "${LINEAR_ATTN_PREFILL_BACKEND}"
  --linear-attn-verify-backend triton
  --kv-cache-dtype auto
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
  --watchdog-timeout 1200
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  --cuda-graph-max-bs-decode "${CUDA_GRAPH_MAX_BS_DECODE}"
  --mamba-ssm-dtype "${MAMBA_SSM_DTYPE}"
  # --speculative-algorithm NEXTN
  # --speculative-num-steps 3
  # --speculative-eagle-topk 1
  # --speculative-num-draft-tokens 4
)

if (( DISABLE_RADIX_CACHE )); then
  command+=(--disable-radix-cache)
fi

if (( ENABLE_CACHE_REPORT )); then
  command+=(--enable-cache-report)
fi

# Use an explicit negative flag for the control run so the A/B result cannot be
# affected by an upstream model-specific default.
if (( PLE_OFFLOAD_EMBEDDING )); then
  command+=(--ple-offload-embedding)
else
  command+=(--no-ple-offload-embedding)
fi

if (( $# > 0 )); then
  command+=("$@")
fi

printf 'Launching: '
printf '%q ' "${command[@]}"
printf '\n'
exec "${command[@]}"
