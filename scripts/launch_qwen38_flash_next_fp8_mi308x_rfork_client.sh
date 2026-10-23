#!/usr/bin/env bash
# Start a Qwen3.8-Flash-Next-FP8 instance that pulls its weights from a running
# seed instance over RCCL (R-Fork, --load-format remote_instance), instead of disk.
# The seed is a normal launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh
# instance with the same TP_SIZE and model.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SEED_HOST="${SEED_HOST:-127.0.0.1}"
SEED_PORT="${SEED_PORT:-7080}"
TP_SIZE="${TP_SIZE:-2}"
# One port per TP rank, opened on the seed. Keep them outside
# /proc/sys/net/ipv4/ip_local_port_range (32768-60999 here), otherwise an
# ephemeral socket can take one and the seed rank fails with EADDRINUSE.
GROUP_PORT_BASE="${GROUP_PORT_BASE:-62000}"

group_ports="$(python3 -c "import json,sys; b,n=int(sys.argv[1]),int(sys.argv[2]); print(json.dumps(list(range(b,b+n))))" "${GROUP_PORT_BASE}" "${TP_SIZE}")"

export TP_SIZE
exec bash "${SCRIPT_DIR}/launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh" \
  --load-format remote_instance \
  --remote-instance-weight-loader-backend nccl \
  --remote-instance-weight-loader-seed-instance-ip "${SEED_HOST}" \
  --remote-instance-weight-loader-seed-instance-service-port "${SEED_PORT}" \
  --remote-instance-weight-loader-send-weights-group-ports "${group_ports}" \
  "$@"
