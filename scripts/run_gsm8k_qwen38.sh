port="${BENCH_PORT:-7080}"
python -m sglang.test.run_eval --eval-name gsm8k --num-examples 1319 --max-tokens 16384 --port ${port}
