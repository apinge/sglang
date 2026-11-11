#!/usr/bin/env bash
# One image/text request against an already running OpenAI-compatible server.
set -euo pipefail

host="${BENCH_HOST:-localhost}"
port="${BENCH_PORT:-9000}"
model="${BENCH_SERVED_MODEL:-${BENCH_MODEL:-/models/Qwen3.5-397B-A17B}}"
image_url="${MM_IMAGE_URL:-https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/CI_Demo/mathv-1327.jpg}"
output="${MM_OUTPUT_FILE:-multimodal.json}"
mkdir -p "$(dirname -- "$output")"

payload=$("${CLIENT_PYTHON:-python3}" - "$model" "$image_url" <<'PY'
import json
import sys

print(json.dumps({
    "model": sys.argv[1],
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": sys.argv[2]}},
        {"type": "text", "text": "Describe this image in detail."},
    ]}],
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 100,
    "chat_template_kwargs": {"enable_thinking": False},
}))
PY
)

status=0
curl --fail-with-body --silent --show-error --connect-timeout 15 --max-time 180 \
  --request POST --url "http://${host}:${port}/v1/chat/completions" \
  --header "Content-Type: application/json" --data "$payload" \
  --output "$output" || status=$?
if [[ -f "$output" ]]; then cat "$output"; printf '\n'; fi
exit "$status"
