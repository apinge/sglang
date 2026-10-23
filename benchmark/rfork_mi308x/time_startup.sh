#!/usr/bin/env bash
# Launch one Qwen3.8 instance, wait for "fired up", print startup timings, optionally
# keep it running. Usage:
#   time_startup.sh <log_file> [--keep] [-- extra sglang args...]
# Environment is passed through to the launch script (HIP_VISIBLE_DEVICES, TP_SIZE,
# PORT, SGLANG_CACHE_DIR, LAUNCH_SCRIPT, ...). EVICT_MODEL_CACHE=1 drops the model's
# page cache first (evict_model_cache.py).
set -euo pipefail

BENCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${BENCH_DIR}/../.." && pwd)"
LOG_FILE="$1"
shift
KEEP=0
if [[ "${1:-}" == "--keep" ]]; then
  KEEP=1
  shift
fi
[[ "${1:-}" == "--" ]] && shift
MODEL_PATH="${MODEL_PATH:-/models/Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16}"
LAUNCH_SCRIPT="${LAUNCH_SCRIPT:-${REPO_DIR}/scripts/launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh}"
mkdir -p "$(dirname -- "${LOG_FILE}")"

if [[ "${EVICT_MODEL_CACHE:-0}" == "1" ]]; then
  python3 -I "${BENCH_DIR}/evict_model_cache.py" "${MODEL_PATH}" --evict
fi

start_ns=$(date +%s%N)
LOG_FILE="${LOG_FILE}" MODEL_PATH="${MODEL_PATH}" nohup bash "${LAUNCH_SCRIPT}" "$@" >/dev/null 2>&1 &
launcher_pid=$!
until grep -q -E "fired up|Traceback" "${LOG_FILE}" 2>/dev/null; do
  if ! kill -0 "${launcher_pid}" 2>/dev/null; then
    break
  fi
  sleep 1
done
end_ns=$(date +%s%N)

grep -E "Load weight end|startup timings|fired up|Traceback|Error" "${LOG_FILE}" | grep -E "TP0|startup|fired|Traceback|Error" | cut -c1-240 || true
python3 -c "import sys; print(f'launch->ready wall: {(int(sys.argv[2]) - int(sys.argv[1])) / 1e9:.1f} s')" "${start_ns}" "${end_ns}"

if (( ! KEEP )); then
  pids=$(pgrep -f sglang || true)
  for p in ${pids}; do
    grep -qa "LOG_FILE=${LOG_FILE}" "/proc/${p}/environ" 2>/dev/null && kill "${p}" || true
  done
  sleep 25
  for p in ${pids}; do
    grep -qa "LOG_FILE=${LOG_FILE}" "/proc/${p}/environ" 2>/dev/null && kill -9 "${p}" || true
  done
  sleep 5
fi
