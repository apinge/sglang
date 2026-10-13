#!/usr/bin/env bash
set -o pipefail

model="${BENCH_MODEL:-${MODEL_PATH:-/models/Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16}}"
input_tokens="${INPUT_TOKENS:-12000}"
output_tokens="${OUTPUT_TOKENS:-350}"
num_prompts="${NUM_PROMPTS:-128}"
max_concurrency="${MAX_CONCURRENCY:-8}"
warmup_requests="${WARMUP_REQUESTS:-10}"
dataset_name="${DATASET_NAME:-generated-shared-prefix}"
host="${BENCH_HOST:-localhost}"
port="${BENCH_PORT:-7080}"
log_file="${BENCH_LOG_FILE:-pure_text_perf_prefix_cache_hit.log}"
tokenizer="${BENCH_TOKENIZER:-${model}}"
base_url="${BENCH_BASE_URL:-http://${host}:${port}}"
output_file="${BENCH_OUTPUT_FILE:-prefix-cache-single-point.jsonl}"
cache_hit_percent="${CACHE_HIT_PERCENT:-55}"
num_groups="${GSP_NUM_GROUPS:-1}"
gsp_range_ratio="${GSP_RANGE_RATIO:-1.0}"
system_prompt_tokens=$((input_tokens * cache_hit_percent / 100))
question_tokens=$((input_tokens - system_prompt_tokens))
prompts_per_group=$((num_prompts / num_groups))
tag="prefix-cache-in${input_tokens}-out${output_tokens}-hit${cache_hit_percent}-c${max_concurrency}"

echo "bench model: ${model}"
echo "input tokens: ${input_tokens}"
echo "output tokens: ${output_tokens}"
echo "max concurrency: ${max_concurrency}"
echo "num prompts: ${num_prompts}"
echo "dataset-name: ${dataset_name}"
echo "cache hit percent: ${cache_hit_percent}"
echo "system prompt tokens: ${system_prompt_tokens}"
echo "question tokens: ${question_tokens}"
echo "prefix groups: ${num_groups}"
echo "gsp range ratio: ${gsp_range_ratio}"
echo "base-url: ${base_url}"
echo "output file: ${output_file}"

# Requires a server with radix caching enabled and --enable-cache-report.
# The flush clears its prefix cache before prefix priming and measurement.
python3 -m sglang.benchmark.serving \
    --backend sglang \
    --base-url "${base_url}" \
    --model "${model}" \
    --tokenizer "${tokenizer}" \
    --dataset-name "${dataset_name}" \
    --num-prompts "${num_prompts}" \
    --gsp-num-groups "${num_groups}" \
    --gsp-prompts-per-group "${prompts_per_group}" \
    --gsp-system-prompt-len "${system_prompt_tokens}" \
    --gsp-question-len "${question_tokens}" \
    --gsp-output-len "${output_tokens}" \
    --gsp-range-ratio "${gsp_range_ratio}" \
    --gsp-group-distribution uniform \
    --max-concurrency "${max_concurrency}" \
    --request-rate inf \
    --warmup-requests "${warmup_requests}" \
    --flush-cache \
    --gsp-prewarm-prefixes \
    --gsp-prewarm-concurrency 1 \
    --seed 42 \
    --cache-report \
    --tag "${tag}" \
    --output-file "${output_file}" 2>&1 | tee "${log_file}"
