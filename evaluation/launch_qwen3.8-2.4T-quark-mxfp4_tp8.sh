#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/models/Qwen3.8-2.4T-A95B-Quark-MXFP4/}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-Qwen3.8-2.4T-A95B-Quark-MXFP4}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-30000}

export SGLANG_USE_AITER=1
export USE_AITER_COMM=1
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTHONPATH=/opt/sglang/python${PYTHONPATH:+:${PYTHONPATH}}

if [[ -n "${SGLANG_TORCH_PROFILER_DIR:-}" ]]; then
  mkdir -p "${SGLANG_TORCH_PROFILER_DIR}"
fi

cd /opt/sglang
exec python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tp-size 8 \
  --moe-runner-backend aiter \
  --attention-backend aiter \
  --linear-attn-backend aiter \
  --linear-attn-decode-backend aiter \
  --linear-attn-prefill-backend aiter \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --chunked-prefill-size 8192 \
  --max-prefill-tokens 8192 \
  --max-total-tokens 273536 \
  --max-running-requests 9 \
  --max-mamba-cache-size 9 \
  --mamba-ssm-dtype bfloat16 \
  --mem-fraction-static 1.05 \
  --disable-radix-cache \
  --disable-custom-all-reduce \
  --cuda-graph-max-bs-decode 8 \
  --cuda-graph-bs-decode 1 2 4 8 \
  --cuda-graph-backend-prefill disabled \
  --watchdog-timeout 1200 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --trust-remote-code \
  --host "${HOST}" \
  --port "${PORT}" \
  "$@"
