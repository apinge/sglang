#!/usr/bin/env bash
set -euo pipefail

python3 -m sglang.benchmark.serving \
  --backend sglang \
  --base-url http://0.0.0.0:7080 \
  --model /models/Qwen3.8-Flash-Next-FP8 \
  --tokenizer /models/Qwen3.8-Flash-Next-FP8 \
  --dataset-name generated-shared-prefix \
  --num-prompts 128 \
  --gsp-num-groups 1 \
  --gsp-prompts-per-group 128 \
  --gsp-system-prompt-len 6600 \
  --gsp-question-len 5400 \
  --gsp-output-len 350 \
  --gsp-range-ratio 1.0 \
  --gsp-group-distribution uniform \
  --max-concurrency 64 \
  --request-rate inf \
  --warmup-requests 8 \
  --flush-cache \
  --gsp-prewarm-prefixes \
  --gsp-prewarm-concurrency 1 \
  --seed 42 \
  --cache-report \
  --tag prefix-cache-in12000-out350-hit55-c64 \
  --output-file prefix-cache-in12000-out350-hit55-c64.jsonl
