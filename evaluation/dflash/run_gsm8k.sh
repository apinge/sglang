#!/bin/bash
set -euo pipefail
# Locate the repository so paths work from any working directory.
cd -- "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p "logs/dflash/gsm8k_accuracy" "results/dflash/gsm8k_accuracy"
cd "results/dflash/gsm8k_accuracy"

export TVM_FFI_DISABLE_TORCH_C_DLPACK=1

# --data-path is passed explicitly: the default (test.jsonl) is not found locally,
# so bench_sglang.py falls back to download_and_cache_file(), which caches to
# /tmp/<url-basename> = /tmp/test.jsonl. That file already holds MATH-500 here, so
# the cache check short-circuits and gsm8k dies on KeyError: 'question'.
python3 "../../../benchmark/gsm8k/bench_sglang.py" \
  --host 127.0.0.1 \
  --port 9080 \
  --backend srt \
  --parallel 64 \
  --num-shots 5 \
  --num-questions 200 \
  --temperature 0.0 \
  --top-p 1.0 \
  --data-path "../../../benchmark/dflash/data/gsm8k.jsonl" \
  --tokenizer-path /models/Qwen3.5-397B-A17B-PTPC-FP8 \
  --max-new-tokens 4096 \
  2>&1 | tee "../../../logs/dflash/gsm8k_accuracy/gsm8k.log"
