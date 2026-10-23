#!/usr/bin/env bash
set -euo pipefail

model="${BENCH_MODEL:-/models/Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16}"
input_tokens="${INPUT_TOKENS:-12000}"
output_tokens="${OUTPUT_TOKENS:-350}"
num_prompts="${NUM_PROMPTS:-128}"
max_concurrency="${MAX_CONCURRENCY:-1}"
warmup_requests="${WARMUP_REQUESTS:-10}"
dataset_name="${DATASET_NAME:-random}"
host="${BENCH_HOST:-localhost}"
port="${BENCH_PORT:-7080}"
log_file="${BENCH_LOG_FILE:-pure_text_perf.log}"

echo "bench model: ${model}"
echo "input tokens: ${input_tokens}"
echo "output tokens: ${output_tokens}"
echo "max concurrency: ${max_concurrency}"
echo "num prompts: ${num_prompts}"
echo "dataset-name: ${dataset_name}"

"${CLIENT_PYTHON:-python3}" -m sglang.bench_serving \
    --backend sglang \
    --model "${model}" \
    --served-model-name "${BENCH_SERVED_MODEL:-$model}" \
    --dataset-path "${DATASET_PATH:-/models/benchmark_datasets/ShareGPT_V3_unfiltered_cleaned_split.json}" \
    --seed "${BENCH_SEED:-42}" \
    --flush-cache --output-details \
    --output-file "${BENCH_OUTPUT_FILE:-performance.jsonl}" \
    --dataset-name "${dataset_name}" \
    --host "${host}" \
    --port "${port}" \
    --num-prompts "${num_prompts}" \
    --random-input-len "${input_tokens}" \
    --random-output-len "${output_tokens}" \
    --random-range-ratio 1.0 \
    --warmup-requests "${warmup_requests}" \
    --max-concurrency "${max_concurrency}" 2>&1 | tee "${log_file}"
