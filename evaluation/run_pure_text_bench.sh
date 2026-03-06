model=/models/Qwen3.5-397B-A17B/
input_tokens=8000
output_tokens=500
num_prompts=32
max_concurrency=1
dataset_name="random"

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
    --host localhost \
    --port 9000 \
    --num-prompts ${num_prompts} \
    --random-input ${input_tokens} \
    --random-output ${output_tokens} \
    --random-range-ratio 1.0 \
    --max-concurrency ${max_concurrency}
    2>&1 | tee pure_text_perf.log
