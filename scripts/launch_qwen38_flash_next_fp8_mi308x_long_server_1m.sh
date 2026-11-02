#!/usr/bin/env bash
# Customer server_args from /opt/sglang/customer_server.md.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/models/Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7080}"
GPU_DEVICES="${GPU_DEVICES:-6,7}"
CONTEXT_LEN="${CONTEXT_LEN:-1048576}"
CUDA_GRAPH_MAX_BS_DECODE="${CUDA_GRAPH_MAX_BS_DECODE:-8}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-8}"
for name in CONTEXT_LEN CUDA_GRAPH_MAX_BS_DECODE MAX_RUNNING_REQUESTS; do
  if [[ ! "${!name}" =~ ^[1-9][0-9]*$ ]]; then
    echo "$name must be a positive integer (got ${!name})" >&2
    exit 2
  fi
done
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/logs/qwen38_customer_tp2_ctx${CONTEXT_LEN}_c${MAX_RUNNING_REQUESTS}_gb${CUDA_GRAPH_MAX_BS_DECODE}_ple1_$(date -u +%Y%m%dT%H%M%SZ).log}"
if [[ "$LOG_FILE" != /* ]]; then
  LOG_FILE="$PWD/$LOG_FILE"
fi
if [[ "$MODEL_PATH" != /* ]]; then
  MODEL_PATH="$PWD/$MODEL_PATH"
fi
mkdir -p "$(dirname -- "$LOG_FILE")"
exec > >(tee "$LOG_FILE") 2>&1
printf 'Logging to: %s\n' "$LOG_FILE"

if (( $# > 0 )); then
  echo "Set CONTEXT_LEN/CUDA_GRAPH_MAX_BS_DECODE/MAX_RUNNING_REQUESTS or PORT/HOST/MODEL_PATH/GPU_DEVICES/LOG_FILE as environment variables." >&2
  exit 2
fi

# Context/request/graph limits and deployment options above are configurable.
# Other inference settings below remain fixed.
export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
export HIP_VISIBLE_DEVICES="$GPU_DEVICES"
export SGLANG_USE_AITER=1
export SGLANG_GR_READ_FLYDSL=1
export SGLANG_USE_PYHIP_QSA=1
export SGLANG_OPT_PYHIP_QSA_SPLIT_INDEXER=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export AITER_MOE_PADDING_SIZE=64
# Avoid the /opt/sglang repository directory shadowing the editable package.
cd /tmp

printf 'Customer config: TP=2; GPUs=%s; context=%s; Mamba cache=160; PLE offload=1\n' "$GPU_DEVICES" "$CONTEXT_LEN"
printf 'mem-fraction argument=1.0; max running=%s; graph max batch=%s; warmups=voice_chat\n' "$MAX_RUNNING_REQUESTS" "$CUDA_GRAPH_MAX_BS_DECODE"
printf 'GR_READ=1; PYHIP_QSA=1; dtype=bfloat16; reasoning parser=qwen3; tool parser=qwen3_coder\n'

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Model config not found: ${MODEL_PATH}/config.json" >&2
  exit 1
fi
python3 - <<'PY'
import torch

if torch.version.hip is None or not torch.cuda.is_available():
    raise SystemExit("ROCm PyTorch and visible GPUs are required.")
if torch.cuda.device_count() < 2:
    raise SystemExit(f"TP2 requires 2 visible GPUs; got {torch.cuda.device_count()}.")
for index in range(2):
    arch = torch.cuda.get_device_properties(index).gcnArchName.split(":", 1)[0]
    if arch != "gfx942":
        raise SystemExit(f"GPU {index}: expected gfx942, got {arch}.")
PY
# Defaults: 1M context (1048576), max running 8, graph batch 8.
command=(
  python3 -m sglang.launch_server
  --model-path "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --tp-size 2
  --context-length "$CONTEXT_LEN"
  --max-mamba-cache-size 160
  --chunked-prefill-size 16384
  --ple-offload-embedding
  --moe-runner-backend aiter
  --attention-backend aiter
  --tokenizer-mode auto
  --trust-remote-code
  --reasoning-parser qwen3
  --tool-call-parser qwen3_coder
  --watchdog-timeout 1200
  --mm-process-config '{"image":{"max_pixels":1048576,"min_pixels":784}}'
  --warmups voice_chat
  --speculative-eagle-topk 1
  --speculative-algorithm NEXTN
  --mem-fraction-static 1.0
  --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS_DECODE"
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --speculative-num-draft-tokens 4
  --kv-cache-dtype auto
  --dtype bfloat16
  --enable-metrics
  --speculative-num-steps 3
  --log-requests-level 0
)
# disable_radix_cache=false: retain the default enabled radix cache.
printf 'Launching: '
printf '%q ' "${command[@]}"
printf '\n'
exec "${command[@]}"
