#!/usr/bin/env bash
set -o pipefail

model="${BENCH_MODEL:-/models/Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16}"
input_tokens="${INPUT_TOKENS:-12000}"
output_tokens="${OUTPUT_TOKENS:-5}"
num_prompts="${NUM_PROMPTS:-4}"
max_concurrency="${MAX_CONCURRENCY:-1}"
dataset_name="${DATASET_NAME:-random}"
warmup_requests="${WARMUP_REQUESTS:-10}"
host="${BENCH_HOST:-localhost}"
port="${BENCH_PORT:-7080}"
profile_log_file="${PROFILE_LOG_FILE:-pure_text_profile.log}"

echo "bench model: ${model}"
echo "input tokens: ${input_tokens}"
echo "output tokens: ${output_tokens}"
echo "max concurrency: ${max_concurrency}"
echo "num prompts: ${num_prompts}"
echo "dataset-name: ${dataset_name}"

export SGLANG_VLM_CACHE_SIZE_MB=0
export SGLANG_TORCH_PROFILER_DIR="${SGLANG_TORCH_PROFILER_DIR:-./sglang_profile_res}"
export SGLANG_PROFILE_WITH_STACK=1
export SGLANG_PROFILE_RECORD_SHAPES=1

python3 -m sglang.bench_serving \
    --backend sglang \
    --model ${model} \
    --dataset-name ${dataset_name} \
    --host ${host} \
    --port ${port} \
    --num-prompts ${num_prompts} \
    --random-input ${input_tokens} \
    --random-output ${output_tokens} \
    --random-range-ratio 1.0 \
    --warmup-requests ${warmup_requests} \
    --max-concurrency ${max_concurrency} \
    --profile \
    2>&1 | tee "${profile_log_file}"
