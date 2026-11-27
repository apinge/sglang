#!/bin/bash
set -euo pipefail
cd -- "$(dirname "${BASH_SOURCE[0]}")/../.."
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export ACCEPTANCE_DIR="results/dflash/mtbench/acceptance"
mkdir -p "logs/dflash/mtbench" "results/dflash/mtbench"

python3 -u benchmark/dflash/record_acceptance.py \
  --benchmark benchmark/dflash/bench_dflash_mtbench_sweep.py \
  --base-url http://127.0.0.1:9080 \
  --external-mode dflash \
  --target-model /models/Qwen3.5-397B-A17B-PTPC-FP8 \
  --draft-model /models/Qwen3.5-397B-A17B-DFlash \
  --attention-backends aiter \
  --tp-sizes 8 \
  --speculative-num-draft-tokens 16 \
  --data-path benchmark/dflash/data/mtbench.jsonl \
  --result-json results/dflash/mtbench/mtbench.json \
  --concurrencies 1 \
  --temperature 0.0 --top-p 1.0 --top-k 1 \
  --timeout-s 3600 \
  --questions-per-concurrency-base 80 \
  --max-questions-per-config 80 \
  --max-new-tokens 1024 \
  2>&1 | tee logs/dflash/mtbench/mtbench.log
