model=/models/Qwen3.5-397B-A17B/
input_tokens=8000
output_tokens=10
num_prompts=4
max_concurrency=1
dataset_name="random"

echo "bench model: ${model}"
echo "input tokens: ${input_tokens}"
echo "output tokens: ${output_tokens}"
echo "max concurrency: ${max_concurrency}"
echo "num prompts: ${num_prompts}"
echo "dataset-name: ${dataset_name}"

export SGLANG_VLM_CACHE_SIZE_MB=0
export SGLANG_TORCH_PROFILER_DIR="./sglang_profile_res"
export SGLANG_PROFILE_WITH_STACK=1
export SGLANG_PROFILE_RECORD_SHAPES=1

python3 -m sglang.bench_serving \
    --backend sglang \
    --model ${model} \
    --dataset-name ${dataset_name} \
    --host localhost \
    --port 9000 \
    --num-prompts ${num_prompts} \
    --random-input ${input_tokens} \
    --random-output ${output_tokens} \
    --random-range-ratio 1.0 \
    --max-concurrency ${max_concurrency} \
    --profile \
    2>&1 | tee pure_text_profile.log
