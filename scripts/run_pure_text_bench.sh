model="${BENCH_MODEL:-/models/Qwen3.8-Flash-Next-PTPC-FP8}"
input_tokens="${INPUT_TOKENS:-12000}"
output_tokens="${OUTPUT_TOKENS:-350}"
num_prompts="${NUM_PROMPTS:-32}"
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
    --max-concurrency ${max_concurrency} 2>&1 | tee "${log_file}"
