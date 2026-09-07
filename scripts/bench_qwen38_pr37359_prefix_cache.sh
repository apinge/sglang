#!/usr/bin/env bash
# One-prefix-group smoke/point/matrix runs through the PR's validated runner.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MODE="${1:-smoke}"
if (( $# > 0 )); then shift; fi

case "${MODE}" in
  smoke)
    DEFAULT_INPUT_LENS=4096
    DEFAULT_OUTPUT_LENS=32
    DEFAULT_HIT_PERCENTAGES=50
    DEFAULT_CONCURRENCIES=1
    DEFAULT_NUM_PROMPTS=8
    DEFAULT_WARMUP_REQUESTS=1
    ;;
  point)
    DEFAULT_INPUT_LENS=32768
    DEFAULT_OUTPUT_LENS=512
    DEFAULT_HIT_PERCENTAGES=50
    DEFAULT_CONCURRENCIES=8
    DEFAULT_NUM_PROMPTS=50
    DEFAULT_WARMUP_REQUESTS=5
    ;;
  matrix)
    DEFAULT_INPUT_LENS=32768
    DEFAULT_OUTPUT_LENS=64
    DEFAULT_HIT_PERCENTAGES="50 90"
    DEFAULT_CONCURRENCIES="1 8"
    DEFAULT_NUM_PROMPTS=8
    DEFAULT_WARMUP_REQUESTS=1
    ;;
  *)
    printf 'Usage: bash %s {smoke|point|matrix} [runner arguments]\n' "$0" >&2
    exit 2
    ;;
esac

MODEL_PATH="${MODEL_PATH:-/models/Qwen3.8-Flash-Next-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.8-Flash-Next-FP8}"
BASE_URL="${BASE_URL:-http://127.0.0.1:7080}"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/logs/pr37359_${MODE}_$(date -u +%Y%m%dT%H%M%SZ)}"
NUM_PROMPTS="${NUM_PROMPTS:-${DEFAULT_NUM_PROMPTS}}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-${DEFAULT_WARMUP_REQUESTS}}"
read -r -a input_lens <<< "${INPUT_LENS:-${DEFAULT_INPUT_LENS}}"
read -r -a output_lens <<< "${OUTPUT_LENS:-${DEFAULT_OUTPUT_LENS}}"
read -r -a hit_percentages <<< "${HIT_PERCENTAGES:-${DEFAULT_HIT_PERCENTAGES}}"
read -r -a concurrencies <<< "${CONCURRENCIES:-${DEFAULT_CONCURRENCIES}}"
export PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

# For nonzero hit targets the PR translates --num-groups 1 directly into
# --gsp-num-groups 1. Its special 0%-hit mode uses one unique group per request;
# the default experiments here use nonzero targets to preserve the user's group count.
command=(
  python "${REPO_ROOT}/benchmark/prefix_cache/bench_prefix_cache.py"
  --base-url "${BASE_URL}"
  --model "${SERVED_MODEL_NAME}"
  --tokenizer "${MODEL_PATH}"
  --input-lens "${input_lens[@]}"
  --output-lens "${output_lens[@]}"
  --cache-hit-percentages "${hit_percentages[@]}"
  --concurrencies "${concurrencies[@]}"
  --num-prompts "${NUM_PROMPTS}"
  --num-groups 1
  --warmup-requests "${WARMUP_REQUESTS}"
  --prewarm-concurrency 1
  --request-rate inf
  --repetitions "${REPETITIONS:-1}"
  --seed "${SEED:-42}"
  --tag-prefix "qwen38-pr37359-${MODE}"
  --result-dir "${RESULT_DIR}"
  --output-details
  --quiet
)
command+=("$@")
printf 'Result directory: %s\n' "${RESULT_DIR}"
printf 'Launching: '
printf '%q ' "${command[@]}"
printf '\n'
exec "${command[@]}"
