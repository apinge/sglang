#!/bin/bash

set -euo pipefail

TYPE=${1:-launch}
MODEL_NAME=${2:-offical_qwen3p5_397B_ptpc}
MODEL_PATH=${3:-/models/offical_qwen3p5_397B_ptpc}
TP=${4:-8}
EP=${5:-1}
TIMEOUT=${6:-45}
PORT=${SGLANG_BENCHMARK_PORT:-8080}
GSM8K_NUM_QUESTIONS=${SGLANG_BENCHMARK_GSM8K_NUM_QUESTIONS:-200}
GSM8K_PARALLEL=${SGLANG_BENCHMARK_GSM8K_PARALLEL:-128}
GSM8K_MAX_NEW_TOKENS=${SGLANG_BENCHMARK_GSM8K_MAX_NEW_TOKENS:-4096}
ACCURACY_RESULTS_DIR=${SGLANG_BENCHMARK_ACCURACY_RESULTS_DIR:-accuracy_test_results}
SERVER_LOG=${SGLANG_BENCHMARK_SERVER_LOG:-/tmp/sglang_qwen35_server.log}
BENCHMARK_RESULTS_DIR=${SGLANG_BENCHMARK_RESULTS_DIR:-benchmark_test_results}
BENCHMARK_EXAMPLE_ROOT=${SGLANG_BENCHMARK_EXAMPLE_ROOT:-/models/benchamark_example}

export SGLANG_DISABLE_CUDNN_CHECK=1
export SGLANG_USE_CUDA_IPC_TRANSPORT=1
# Override to 0 for Ali-specific runs when required.
export SGLANG_VLM_CACHE_SIZE_MB="${SGLANG_VLM_CACHE_SIZE_MB:-8192}"
export SGLANG_USE_AITER=1
export SGLANG_ROCM_USE_AITER_LINEAR_SHUFFLE=1
export SGLANG_ROCM_USE_AITER_LINEAR_FP8HIPB=1
export SGLANG_USE_AITER_NEW_CA=false

if [[ "${MODEL_NAME}" == "offical_qwen3p5_397B_ptpc" ]]; then
  export AITER_MOE_PADDING_SIZE="${AITER_MOE_PADDING_SIZE:-256}"
fi

echo "Detect TYPE: ${TYPE}"
echo "Detect model_name: ${MODEL_NAME}"
echo "Detect model_path: ${MODEL_PATH}"
echo "Detect TP: ${TP}"
echo "Detect EP: ${EP}"
echo "Detect TIMEOUT: ${TIMEOUT}"
echo "Detect PORT: ${PORT}"
echo "Detect benchmark_example_root: ${BENCHMARK_EXAMPLE_ROOT}"
echo "Detect benchmark_results_dir: ${BENCHMARK_RESULTS_DIR}"

wait_for_server() {
  local max_retries=${1}
  local retry_interval=60

  echo
  echo "========== WAITING FOR SERVER TO BE READY =========="
  for ((i=1; i<=max_retries; i++)); do
    if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null; then
      echo "SGLang server is up."
      return 0
    fi

    echo "Waiting for SGLang server to be ready... (${i}/${max_retries})"

    if [[ -f "${SERVER_LOG}" ]]; then
      tail -n 20 "${SERVER_LOG}" || true
    fi

    sleep "${retry_interval}"
  done

  echo "SGLang server did not start after $((max_retries * retry_interval)) seconds."
  if [[ -f "${SERVER_LOG}" ]]; then
    echo "========== SERVER LOG =========="
    tail -n 200 "${SERVER_LOG}" || true
  fi
  return 1
}

print_launch_recipe() {
  local model=${1}
  local attention_backend=${2}

  echo
  echo "========== LAUNCH ENVIRONMENT =========="
  echo "export SGLANG_DISABLE_CUDNN_CHECK=${SGLANG_DISABLE_CUDNN_CHECK}"
  echo "export SGLANG_USE_CUDA_IPC_TRANSPORT=${SGLANG_USE_CUDA_IPC_TRANSPORT}"
  echo "export SGLANG_VLM_CACHE_SIZE_MB=${SGLANG_VLM_CACHE_SIZE_MB}"
  echo "export SGLANG_USE_AITER=${SGLANG_USE_AITER}"
  echo "export SGLANG_ROCM_USE_AITER_LINEAR_SHUFFLE=${SGLANG_ROCM_USE_AITER_LINEAR_SHUFFLE}"
  echo "export SGLANG_ROCM_USE_AITER_LINEAR_FP8HIPB=${SGLANG_ROCM_USE_AITER_LINEAR_FP8HIPB}"
  echo "export SGLANG_USE_AITER_NEW_CA=${SGLANG_USE_AITER_NEW_CA}"
  if [[ -n "${AITER_MOE_PADDING_SIZE:-}" ]]; then
    echo "export AITER_MOE_PADDING_SIZE=${AITER_MOE_PADDING_SIZE}"
  fi

  echo
  echo "setup server for TP${TP}:"
  cat <<EOF
nohup python3 -m sglang.launch_server \\
  --port ${PORT} \\
  --model-path ${model} \\
  --tp-size ${TP} \\
  --attention-backend ${attention_backend} \\
  --reasoning-parser qwen3 \\
  --tool-call-parser qwen3_coder \\
  --enable-multimodal \\
  --trust-remote-code \\
  --chunked-prefill-size 32768 \\
  --mem-fraction-static 0.9 \\
  --max-prefill-tokens 32768 \\
  --max-running-requests 128 \\
EOF
  if [[ "${MODEL_NAME}" == "offical_qwen3p5_397B_ptpc" ]]; then
    echo "  --disable-custom-all-reduce \\"
  fi
  cat <<EOF
  --disable-radix-cache \\
  --mm-attention-backend aiter_attn \\
  > ${SERVER_LOG} 2>&1 &
EOF
}

smoke_test_server() {
  local max_retries=3
  local retry_interval=10

  echo
  echo "========== TESTING SERVER =========="
  for ((i=1; i<=max_retries; i++)); do
    if curl --fail --silent --show-error --max-time 120 --request POST \
      --url "http://localhost:${PORT}/v1/chat/completions" \
      --header "Content-Type: application/json" \
      --data "$(cat <<EOF
{
  "model": "${MODEL_PATH}",
  "messages": [
    {
      "role": "user",
      "content": "Reply with exactly one word: ready"
    }
  ],
  "temperature": 0.0,
  "top_p": 0.0001,
  "top_k": 1,
  "max_tokens": 8
}
EOF
)" >/dev/null; then
      echo "Smoke test passed."
      return 0
    fi

    echo "Smoke test attempt ${i}/${max_retries} failed."
    if ! curl -sf "http://localhost:${PORT}/v1/models" >/dev/null; then
      echo "SGLang server stopped responding during smoke test."
      return 1
    fi

    if (( i < max_retries )); then
      sleep "${retry_interval}"
    fi
  done

  echo "Smoke test did not succeed after ${max_retries} attempts."
  return 1
}

# Prepare compatibility aliases instead of rewriting external benchmark sources.
prepare_external_benchmark_environment() {
  local alias_name
  local alias_path
  local alias_root
  local model_basename

  model_basename=$(basename "${MODEL_PATH}")

  for alias_name in "${MODEL_NAME}" "${model_basename}"; do
    [[ -n "${alias_name}" ]] || continue
    for alias_root in \
      "/data/models" \
      "/data/models/Qwen/uc" \
      "/mnt/raid0" \
      "/mnt/raid0/models" \
      "/mnt/raid0/Qwen/uc" \
      "/mnt/raid0/models/Qwen/uc"; do
      mkdir -p "${alias_root}"
      alias_path="${alias_root}/${alias_name}"
      if [[ -e "${alias_path}" && ! -L "${alias_path}" ]]; then
        echo "Keeping existing benchmark alias path: ${alias_path}"
        continue
      fi
      ln -sfn "${MODEL_PATH}" "${alias_path}"
    done
  done
}

rewrite_external_benchmark_runtime_config() {
  local benchmark_type=${1}
  local work_dir=${2}

  python3 - "${benchmark_type}" "${work_dir}" "${PORT}" <<'PY'
from pathlib import Path
import sys

benchmark_type = sys.argv[1]
work_dir = Path(sys.argv[2])
port = sys.argv[3]

if benchmark_type != "request_rate" or port == "8080":
    raise SystemExit(0)

patterns = [
    ("--host 127.0.0.1 --port 8080", f"--host 127.0.0.1 --port {port}"),
    ("http://127.0.0.1:8080", f"http://127.0.0.1:{port}"),
    ("https://127.0.0.1:8080", f"https://127.0.0.1:{port}"),
    ("127.0.0.1:8080", f"127.0.0.1:{port}"),
    ("localhost:8080", f"localhost:{port}"),
    ("--port 8080", f"--port {port}"),
    ("port=8080", f"port={port}"),
    ("port = 8080", f"port = {port}"),
    ('"port": 8080', f'"port": {port}'),
]

modified_files = []
candidate_suffixes = {
    ".py",
    ".sh",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".txt",
}

for path in work_dir.rglob("*"):
    if not path.is_file():
        continue
    if path.suffix and path.suffix not in candidate_suffixes:
        continue

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        continue

    new_text = text
    for old, new in patterns:
        new_text = new_text.replace(old, new)

    if new_text != text:
        path.write_text(new_text, encoding="utf-8")
        modified_files.append(str(path.relative_to(work_dir)))

print(
    f"Rewrote request_rate benchmark runtime config to use port {port} "
    f"in {len(modified_files)} file(s)."
)
for file_name in modified_files:
    print(f"  - {file_name}")
PY
}

verify_external_benchmark_log() {
  local log_path=${1}
  python3 - "${log_path}" <<'PY'
import pathlib
import re
import sys

log_path = pathlib.Path(sys.argv[1])
text = log_path.read_text(encoding="utf-8", errors="replace")

failure_patterns = [
    ("python traceback", r"Traceback \(most recent call last\):"),
    ("python syntax error", r"SyntaxError:"),
    ("wrapper failure", r"Benchmark iteration \d+ failed"),
    ("nonzero return code", r"Return code:\s*[1-9]\d*"),
    ("all requests failed", r"Error Requests:\s*100(?:\.0+)?%"),
    ("broken request threshold", r"too many broken_error=\d+"),
]

for label, pattern in failure_patterns:
    if re.search(pattern, text):
        print(f"Detected benchmark failure marker ({label}) in {log_path}")
        raise SystemExit(1)
PY
}

run_external_benchmark() {
  local benchmark_type=${1}
  local source_dir="${BENCHMARK_EXAMPLE_ROOT}/${benchmark_type}"
  local source_script="${source_dir}/run.sh"
  local work_dir
  local log_path="${BENCHMARK_RESULTS_DIR}/${benchmark_type}_benchmark_${MODEL_NAME}_TP${TP}_EP${EP}.log"

  echo
  echo "========== STARTING ${benchmark_type^^} BENCHMARK =========="

  if [[ ! -f "${source_script}" ]]; then
    echo "Benchmark script not found: ${source_script}"
    exit 1
  fi

  mkdir -p "${BENCHMARK_RESULTS_DIR}"
  work_dir=$(mktemp -d "/tmp/sglang_external_${benchmark_type}.XXXXXX")
  cp -a "${source_dir}/." "${work_dir}/"
  chmod -R u+w "${work_dir}"
  prepare_external_benchmark_environment
  rewrite_external_benchmark_runtime_config "${benchmark_type}" "${work_dir}"

  (
    cd "${work_dir}"
    export MODEL="${MODEL_PATH}"
    export MODEL_PATH="${MODEL_PATH}"
    export TOKENIZER="${MODEL_PATH}"
    export TOKENIZER_PATH="${MODEL_PATH}"
    export HOST="127.0.0.1"
    export PORT="${PORT}"
    export BASE_URL="http://127.0.0.1:${PORT}"
    export SGLANG_BENCHMARK_MODEL_PATH="${MODEL_PATH}"
    export SGLANG_BENCHMARK_PORT="${PORT}"
    bash -euo pipefail "./run.sh"
  ) | tee "${log_path}"

  verify_external_benchmark_log "${log_path}"
}

if [[ "${TYPE}" == "launch" ]]; then
  echo
  echo "========== LAUNCHING SERVER =========="

  if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Model path does not exist: ${MODEL_PATH}"
    exit 1
  fi

  pkill -f "python3 -m sglang.launch_server" || true
  rm -f "${SERVER_LOG}"
  model="${MODEL_PATH}"
  attention_backend="aiter"
  if [[ "${MODEL_NAME}" == "Qwen3.5-27B-PTPC-compressor" ]]; then
    attention_backend="triton"
  fi

  launch_args=(
    --port "${PORT}"
    --model-path "${model}"
    --tp-size "${TP}"
    --attention-backend "${attention_backend}"
    --reasoning-parser qwen3
    --tool-call-parser qwen3_coder
    --enable-multimodal
    --trust-remote-code
    --chunked-prefill-size 32768
    --mem-fraction-static 0.9
    --max-prefill-tokens 32768
    --max-running-requests 128
    --disable-radix-cache
    --mm-attention-backend aiter_attn
  )

  if [[ "${MODEL_NAME}" == "offical_qwen3p5_397B_ptpc" ]]; then
    launch_args+=(--disable-custom-all-reduce)
  fi

  print_launch_recipe "${model}" "${attention_backend}"

  nohup python3 -m sglang.launch_server "${launch_args[@]}" > "${SERVER_LOG}" 2>&1 &

  if ! wait_for_server "${TIMEOUT}"; then
    pkill -f "python3 -m sglang.launch_server" || true
    exit 1
  fi

  smoke_test_server

elif [[ "${TYPE}" == "evaluation" ]]; then
  echo
  echo "========== STARTING GSM8K ACCURACY EVALUATION =========="
  mkdir -p "${ACCURACY_RESULTS_DIR}"
  result_jsonl="${ACCURACY_RESULTS_DIR}/${MODEL_NAME}_gsm8k_result.jsonl"
  raw_result_file="${ACCURACY_RESULTS_DIR}/${MODEL_NAME}_gsm8k_raw_results.jsonl"
  rm -f "${result_jsonl}" "${raw_result_file}"

  benchmark_args=(
    --host "http://127.0.0.1"
    --port "${PORT}"
    --backend srt
    --parallel "${GSM8K_PARALLEL}"
    --num-questions "${GSM8K_NUM_QUESTIONS}"
    --result-file "${result_jsonl}"
    --raw-result-file "${raw_result_file}"
  )

  if [[ "${MODEL_NAME}" != "offical_qwen3p5_397B_ptpc" ]]; then
    benchmark_args+=(
      --max-new-tokens "${GSM8K_MAX_NEW_TOKENS}"
    )
  fi

  echo "Running benchmark/gsm8k/bench_sglang.py."
  python3 benchmark/gsm8k/bench_sglang.py "${benchmark_args[@]}" \
    | tee "gsm8k_accuracy_${MODEL_NAME}_TP${TP}_EP${EP}.log"

  MODEL_NAME="${MODEL_NAME}" \
  RESULT_JSONL="${result_jsonl}" \
  ACCURACY_RESULTS_DIR="${ACCURACY_RESULTS_DIR}" \
  python3 - <<'PY'
import json
import os
from pathlib import Path

result_jsonl = Path(os.environ["RESULT_JSONL"])
result_lines = [
    line.strip()
    for line in result_jsonl.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if not result_lines:
    raise SystemExit(f"No GSM8K benchmark results found in {result_jsonl}")

metrics = json.loads(result_lines[-1])

result_path = os.path.join(
    os.environ["ACCURACY_RESULTS_DIR"], f'{os.environ["MODEL_NAME"]}_gsm8k_results.json'
)
payload = {
    "results": {"gsm8k": {"exact_match,flexible-extract": metrics["accuracy"]}},
    "metrics": metrics,
}

with open(result_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

print(f"Wrote GSM8K results to {result_path}")
print(json.dumps(payload, indent=2))
PY

elif [[ "${TYPE}" == "concurrency" || "${TYPE}" == "request_rate" ]]; then
  run_external_benchmark "${TYPE}"

elif [[ "${TYPE}" == "performance" ]]; then
  echo
  echo "========== STARTING PERFORMANCE BENCHMARK =========="
  mkdir -p "${BENCHMARK_RESULTS_DIR}"
  python3 -m sglang.bench_serving \
    --backend sglang-oai-chat \
    --port "${PORT}" \
    --model "${MODEL_PATH}" \
    --dataset-name image \
    --num-prompts 10 \
    --image-count 5 \
    --image-resolution 960x1280 \
    --random-input-len 8000 \
    --random-output-len 650 \
    --max-concurrency 1 \
    --random-range-ratio 1.0 \
    | tee "${BENCHMARK_RESULTS_DIR}/performance_benchmark_${MODEL_NAME}_TP${TP}_EP${EP}.log"

else
  echo "Unknown TYPE: ${TYPE}"
  echo "Usage: $0 {launch|evaluation|concurrency|request_rate|performance} [model_name] [model_path] [TP] [EP] [timeout]"
  exit 1
fi

exit_code=$?
if [[ ${exit_code} -eq 0 ]]; then
  echo
  echo "========== SGLANG BENCHMARK ${TYPE} COMPLETED SUCCESSFULLY =========="
else
  echo
  echo "========== SGLANG BENCHMARK ${TYPE} FAILED WITH EXIT CODE ${exit_code} =========="
  exit "${exit_code}"
fi
