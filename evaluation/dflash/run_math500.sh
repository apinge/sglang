#!/bin/bash
set -euo pipefail
cd -- "$(dirname "${BASH_SOURCE[0]}")/../.."
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export ACCEPTANCE_DIR="results/dflash/math500/acceptance"
mkdir -p "logs/dflash/math500" "results/dflash/math500"

python3 -u benchmark/dflash/record_acceptance.py \
  --benchmark benchmark/dflash/bench_dflash_math500_sweep.py \
  --base-url http://127.0.0.1:9080 \
  --external-mode dflash \
  --target-model /models/Qwen3.5-397B-A17B-PTPC-FP8 \
  --draft-model /models/Qwen3.5-397B-A17B-DFlash \
  --attention-backends aiter \
  --tp-sizes 8 \
  --speculative-num-draft-tokens 16 \
  --data-path benchmark/dflash/data/math500.jsonl \
  --result-json results/dflash/math500/math500.json \
  --concurrencies 1 \
  --temperature 0.0 --top-p 1.0 --top-k 1 \
  --timeout-s 3600 \
  --questions-per-concurrency-base 128 \
  --max-questions-per-config 500 \
  --max-new-tokens 2048 \
  2>&1 | tee logs/dflash/math500/math500.log
