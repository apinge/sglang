#!/usr/bin/env bash
set -euo pipefail
args=(--eval-name gsm8k
      --num-examples "${GSM8K_NUM_EXAMPLES:-200}"
      --num-shots "${GSM8K_NUM_SHOTS:-5}"
      --num-threads "${GSM8K_NUM_THREADS:-512}"
      --max-tokens "${GSM8K_MAX_TOKENS:-16384}"
      --host "${BENCH_HOST:-127.0.0.1}" --port "${BENCH_PORT:-7080}"
      --api chat --temperature 0 --top-p 1
      --gsm8k-data-path "${GSM8K_DATA_PATH:-/models/benchmark_datasets/gsm8k/test.jsonl}"
      --output-dir "${GSM8K_OUTPUT_DIR:-gsm8k-results}" --strict)
if [[ -n "${BENCH_SERVED_MODEL:-}" ]]; then args+=(--model "$BENCH_SERVED_MODEL"); fi
exec "${CLIENT_PYTHON:-python}" -m sglang.test.run_eval "${args[@]}" "$@"
