#!/usr/bin/env bash
# PR #37359 server experiment; prefix cache and custom all-reduce stay enabled.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_PATH="${MODEL_PATH:-/models/Qwen3.8-Flash-Next-FP8}"
export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-7080}"
# TP4 passed startup with prefix caching; TP2 exhausted its static cache budget.
export TP_SIZE="${TP_SIZE:-4}"
export LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/logs/pr37359_qwen38_server_tp${TP_SIZE}_$(date -u +%Y%m%dT%H%M%SZ).log}"

printf 'PR37359 server launcher PID: %s\n' "$$"
exec bash "${SCRIPT_DIR}/launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh" "$@"
