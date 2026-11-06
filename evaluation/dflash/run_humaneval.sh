#!/bin/bash
set -euo pipefail
cd -- "$(dirname "${BASH_SOURCE[0]}")/../.."
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export ACCEPTANCE_DIR="results/dflash/humaneval/acceptance"
mkdir -p "logs/dflash/humaneval" "results/dflash/humaneval"

python3 -u benchmark/dflash/record_acceptance.py \
  --benchmark benchmark/dflash/bench_dflash_humaneval_sweep.py \
  --base-url http://127.0.0.1:9080 \
  --external-mode dflash \
  --target-model /models/Qwen3.5-397B-A17B-PTPC-FP8 \
  --draft-model /models/Qwen3.5-397B-A17B-DFlash \
  --attention-backends aiter \
  --tp-sizes 8 \
  --speculative-num-draft-tokens 16 \
  --data-path benchmark/dflash/data/humaneval.jsonl \
  --result-json results/dflash/humaneval/humaneval.json \
  --concurrencies 1 \
  --temperature 0.0 --top-p 1.0 --top-k 1 \
  --timeout-s 3600 \
  --questions-per-concurrency-base 164 \
  --max-questions-per-config 164 \
  --max-new-tokens 1024 \
  2>&1 | tee logs/dflash/humaneval/humaneval.log
