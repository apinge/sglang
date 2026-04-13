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

# Patch the copied benchmark files so they target the active model and port.
rewrite_external_benchmark_files() {
  local benchmark_root=${1}
  python3 - "${benchmark_root}" "${MODEL_PATH}" "${PORT}" <<'PY'
import pathlib
import py_compile
import re
import sys

benchmark_root = pathlib.Path(sys.argv[1])
model_path = sys.argv[2]
port = sys.argv[3]


def preserve_quote_replacement(match, new_value):
    prefix = match.group(1)
    original_value = match.group(0)[len(prefix):]
    if original_value.startswith('"') and original_value.endswith('"'):
        return f'{prefix}"{new_value}"'
    if original_value.startswith("'") and original_value.endswith("'"):
        return f"{prefix}'{new_value}'"
    return f"{prefix}{new_value}"

model_patterns = [
    (
        re.compile(
            r'(^\s*(?:MODEL_PATH|MODEL|model_path|model)\s*=\s*)(?:"[^"]*"|\'[^\']*\'|[^\s#]+)',
            re.MULTILINE,
        ),
        lambda m: preserve_quote_replacement(m, model_path),
    ),
    (
        re.compile(r'(--model-path(?:=|\s+))(?:\"[^\"]*\"|\'[^\']*\'|[^\s\\\"\')]+)'),
        lambda m: f'{m.group(1)}{model_path}',
    ),
    (
        re.compile(r'(--model(?!-path)(?:=|\s+))(?:\"[^\"]*\"|\'[^\']*\'|[^\s\\\"\')]+)'),
        lambda m: f'{m.group(1)}{model_path}',
    ),
    (
        re.compile(
            r'/(?:data/|mnt/raid0/)?models/(?!benchamark_example\b|benchmark_example\b)[^\s"\'\\]+'
        ),
        lambda _: model_path,
    ),
]

port_patterns = [
    (
        re.compile(
            r'(^\s*(?:PORT|port)\s*=\s*)(?:"[^"]*"|\'[^\']*\'|[^\s#]+)',
            re.MULTILINE,
        ),
        lambda m: preserve_quote_replacement(m, port),
    ),
    (
        re.compile(r'(--port(?:=|\s+))(?:\"[^\"]*\"|\'[^\']*\'|[^\s\\\"\')]+)'),
        lambda m: f'{m.group(1)}{port}',
    ),
    (
        re.compile(r'(https?://localhost:)\d+'),
        lambda m: f'{m.group(1)}{port}',
    ),
    (
        re.compile(r'(https?://127\.0\.0\.1:)\d+'),
        lambda m: f'{m.group(1)}{port}',
    ),
]

updated_files = []
for path in sorted(benchmark_root.rglob("*")):
    if not path.is_file():
        continue

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        continue

    updated = text
    model_updates = 0
    for pattern, repl in model_patterns:
        updated, count = pattern.subn(repl, updated)
        model_updates += count

    port_updates = 0
    for pattern, repl in port_patterns:
        updated, count = pattern.subn(repl, updated)
        port_updates += count

    if updated != text:
        path.write_text(updated, encoding="utf-8")
        updated_files.append((path.relative_to(benchmark_root), model_updates, port_updates))

    if path.suffix == ".py":
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError:
            repaired = path.read_text(encoding="utf-8")
            repaired = re.sub(
                r'(--model-path(?:=|\s+))"([^"\n]+)"',
                r"\1\2",
                repaired,
            )
            repaired = re.sub(
                r'(--model(?!-path)(?:=|\s+))"([^"\n]+)"',
                r"\1\2",
                repaired,
            )
            repaired = re.sub(
                r'(--port(?:=|\s+))"([^"\n]+)"',
                r"\1\2",
                repaired,
            )
            if repaired != path.read_text(encoding="utf-8"):
                path.write_text(repaired, encoding="utf-8")
                updated_files.append((path.relative_to(benchmark_root), 0, 0))

if not updated_files:
    print(f"WARNING: No benchmark files required rewriting under {benchmark_root}")
else:
    for rel_path, model_updates, port_updates in updated_files:
        print(f"Prepared benchmark file: {benchmark_root / rel_path}")
        print(f"Applied model replacements: {model_updates}")
        print(f"Applied port replacements: {port_updates}")
PY
}

patch_request_rate_wrapper_invocation() {
  local work_dir=${1}
  local wrapper_path="${work_dir}/run_benchmark_serving_wrapper.py"

  if [[ ! -f "${wrapper_path}" ]]; then
    return 0
  fi

  if python3 -m py_compile "${wrapper_path}" >/dev/null 2>&1; then
    return 0
  fi

  echo "Request rate wrapper still fails Python compile, sanitizing rewritten CLI quoting."
  python3 - "${wrapper_path}" <<'PY'
import pathlib
import re
import sys

wrapper_path = pathlib.Path(sys.argv[1])
text = wrapper_path.read_text(encoding="utf-8")
updated = re.sub(
    r'(--model-path(?:=|\s+))"([^"\n]+)"',
    r"\1\2",
    text,
)
updated = re.sub(
    r'(--model(?!-path)(?:=|\s+))"([^"\n]+)"',
    r"\1\2",
    updated,
)
updated = re.sub(
    r'(--port(?:=|\s+))"([^"\n]+)"',
    r"\1\2",
    updated,
)
wrapper_path.write_text(updated, encoding="utf-8")
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
  rewrite_external_benchmark_files "${work_dir}"
  if [[ "${benchmark_type}" == "request_rate" ]]; then
    patch_request_rate_wrapper_invocation "${work_dir}"
  fi

  (
    cd "${work_dir}"
    export MODEL="${MODEL_PATH}"
    export MODEL_PATH="${MODEL_PATH}"
    export PORT="${PORT}"
    export SGLANG_BENCHMARK_MODEL_PATH="${MODEL_PATH}"
    export SGLANG_BENCHMARK_PORT="${PORT}"
    bash -euo pipefail "./run.sh"
  ) | tee "${log_path}"
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

  nohup python3 -m sglang.launch_server \
    --port "${PORT}" \
    --model-path "${model}" \
    --tp-size "${TP}" \
    --attention-backend triton \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --enable-multimodal \
    --trust-remote-code \
    --chunked-prefill-size 32768 \
    --mem-fraction-static 0.9 \
    --max-prefill-tokens 32768 \
    --max-running-requests 128 \
    --disable-custom-all-reduce \
    --disable-radix-cache \
    --mm-attention-backend aiter_attn \
    > "${SERVER_LOG}" 2>&1 &

  if ! wait_for_server "${TIMEOUT}"; then
    pkill -f "python3 -m sglang.launch_server" || true
    exit 1
  fi

  smoke_test_server

elif [[ "${TYPE}" == "evaluation" ]]; then
  echo
  echo "========== STARTING GSM8K ACCURACY EVALUATION =========="
  mkdir -p "${ACCURACY_RESULTS_DIR}"
  MODEL_NAME="${MODEL_NAME}" \
  PORT="${PORT}" \
  GSM8K_NUM_QUESTIONS="${GSM8K_NUM_QUESTIONS}" \
  GSM8K_PARALLEL="${GSM8K_PARALLEL}" \
  ACCURACY_RESULTS_DIR="${ACCURACY_RESULTS_DIR}" \
  python3 - <<'PY' | tee "gsm8k_accuracy_${MODEL_NAME}_TP${TP}_EP${EP}.log"
import json
import os
from types import SimpleNamespace

from sglang.test.few_shot_gsm8k import run_eval

metrics = run_eval(
    SimpleNamespace(
        num_shots=5,
        data_path=None,
        num_questions=int(os.environ["GSM8K_NUM_QUESTIONS"]),
        max_new_tokens=512,
        parallel=int(os.environ["GSM8K_PARALLEL"]),
        host="http://127.0.0.1",
        port=int(os.environ["PORT"]),
        temperature=0.0,
    )
)

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
