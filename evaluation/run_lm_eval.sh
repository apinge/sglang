pip install -y lm-eval[api]

lm_eval --model local-completions \
    --model_args '{"base_url": "http://localhost:9000/v1/completions", "model": "/models/Qwen3.5-397B-A17B", "num_concurrent": 64, "max_retries": 10, "max_gen_toks": 2048}' \
    --tasks gsm8k \
    --batch_size auto \
    --num_fewshot 5 \
    --trust_remote_code
