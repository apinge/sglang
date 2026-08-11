#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:9080/v1}"
MODEL="${MODEL:-/models/Qwen3.6-27B/}"
NUM_EXAMPLES="${NUM_EXAMPLES:-200}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1}"
SEED="${SEED:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${SCRIPT_DIR}/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}"

sgl-eval ping \
  --base-url "${BASE_URL}" \
  --model "${MODEL}" \
  --max-tokens 64 \
  --temperature 0

sgl-eval run gsm8k \
  --base-url "${BASE_URL}" \
  --model "${MODEL}" \
  --num-examples "${NUM_EXAMPLES}" \
  --max-tokens "${MAX_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --seed "${SEED}" \
  --no-thinking \
  --chat-template-kwarg enable_thinking=false \
  --out-dir "${RUN_DIR}" \
  2>&1 | tee "${RUN_DIR}/sgl_eval_gsm8k_${NUM_EXAMPLES}.log"
