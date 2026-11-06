#!/usr/bin/env bash
# Customer server_args from /opt/sglang/customer_server.md.
# ROPE_TYPE=default keeps checkpoint RoPE; ROPE_TYPE=yarn enables HF YaRN.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/models/Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7080}"
GPU_DEVICES="${GPU_DEVICES:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-2}"
CONTEXT_LEN="${CONTEXT_LEN:-1048576}"
ROPE_TYPE="${ROPE_TYPE:-default}"
CUDA_GRAPH_MAX_BS_DECODE="${CUDA_GRAPH_MAX_BS_DECODE:-8}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-8}"
for name in TP_SIZE CONTEXT_LEN CUDA_GRAPH_MAX_BS_DECODE MAX_RUNNING_REQUESTS; do
  if [[ ! "${!name}" =~ ^[1-9][0-9]*$ ]]; then
    echo "$name must be a positive integer (got ${!name})" >&2
    exit 2
  fi
done
case "$ROPE_TYPE" in
  default|yarn) ;;
  *) echo "ROPE_TYPE must be default or yarn (got $ROPE_TYPE)." >&2; exit 2 ;;
esac
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/logs/qwen38_customer_tp${TP_SIZE}_ctx${CONTEXT_LEN}_c${MAX_RUNNING_REQUESTS}_gb${CUDA_GRAPH_MAX_BS_DECODE}_rope${ROPE_TYPE}_ple1_$(date -u +%Y%m%dT%H%M%SZ).log}"
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
  echo "Set TP_SIZE/CONTEXT_LEN/ROPE_TYPE/CUDA_GRAPH_MAX_BS_DECODE/MAX_RUNNING_REQUESTS or PORT/HOST/MODEL_PATH/GPU_DEVICES/LOG_FILE as environment variables." >&2
  exit 2
fi

# TP size, RoPE, context/request/graph limits and deployment options are configurable.
# Other inference settings below remain fixed.
export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
export HIP_VISIBLE_DEVICES="$GPU_DEVICES"
export SGLANG_USE_AITER=1
export SGLANG_GR_READ_FLYDSL=1
export SGLANG_USE_PYHIP_QSA=1
export SGLANG_OPT_PYHIP_QSA_SPLIT_INDEXER=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export AITER_MOE_PADDING_SIZE=64
export SGLANG_LOG_QSA_INDEXER_BACKEND="${SGLANG_LOG_QSA_INDEXER_BACKEND:-1}"
# Avoid the /opt/sglang repository directory shadowing the editable package.
cd /tmp

printf 'Customer config: TP=%s; GPUs=%s; context=%s; Mamba cache=160; PLE offload=1\n' "$TP_SIZE" "$GPU_DEVICES" "$CONTEXT_LEN"
printf 'mem-fraction argument=1.0; max running=%s; graph max batch=%s; warmups=voice_chat\n' "$MAX_RUNNING_REQUESTS" "$CUDA_GRAPH_MAX_BS_DECODE"
printf 'GR_READ=1; PYHIP_QSA=1; dtype=bfloat16; reasoning parser=qwen3; tool parser=qwen3_coder\n'

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Model config not found: ${MODEL_PATH}/config.json" >&2
  exit 1
fi
MODEL_OVERRIDE_JSON=""
if [[ "$ROPE_TYPE" == yarn ]]; then
  if (( 10#$CONTEXT_LEN > 1048576 )); then
    echo "CONTEXT_LEN=$CONTEXT_LEN exceeds the fixed YaRN limit 1048576 (262144 * 4.0)." >&2
    exit 2
  fi
  # Fixed HF recipe from huggingface.md; do not modify the checkpoint file.
  MODEL_OVERRIDE_JSON='{"text_config":{"rope_parameters":{
    "mrope_interleaved":true,
    "mrope_section":[11,11,10],
    "rope_type":"yarn",
    "rope_theta":10000000,
    "partial_rotary_factor":0.25,
    "factor":4.0,
    "original_max_position_embeddings":262144
  }}}'
  printf 'RoPE: yarn; factor=4.0; original context=262144; model override=%s\n' "$MODEL_OVERRIDE_JSON"
else
  printf 'RoPE: checkpoint configuration (ROPE_TYPE=default)\n'
fi
python3 - "$TP_SIZE" <<'PY'
import sys

import torch

tp_size = int(sys.argv[1])
if torch.version.hip is None or not torch.cuda.is_available():
    raise SystemExit("ROCm PyTorch and visible GPUs are required.")
if torch.cuda.device_count() < tp_size:
    raise SystemExit(f"TP{tp_size} requires {tp_size} visible GPUs; got {torch.cuda.device_count()}.")
for index in range(tp_size):
    arch = torch.cuda.get_device_properties(index).gcnArchName.split(":", 1)[0]
    if arch != "gfx942":
        raise SystemExit(f"GPU {index}: expected gfx942, got {arch}.")
PY
# Defaults: TP2, default checkpoint RoPE, context 1048576, running/graph batch 8.
command=(
  python3 -m sglang.launch_server
  --model-path "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --tp-size "$TP_SIZE"
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
if [[ "$ROPE_TYPE" == yarn ]]; then
  command+=(--json-model-override-args "$MODEL_OVERRIDE_JSON")
fi
# disable_radix_cache=false: retain the default enabled radix cache.
printf 'Launching: '
printf '%q ' "${command[@]}"
printf '\n'
exec "${command[@]}"
