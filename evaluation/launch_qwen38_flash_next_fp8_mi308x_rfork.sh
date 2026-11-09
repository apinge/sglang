#!/usr/bin/env bash
# R-Fork scale-out for Qwen3.8-Flash-Next-FP8 on MI308X. Wraps
# launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh (same TP_SIZE, PORT,
# LOG_FILE, ... variables):
#
#   MODE=seed   a normal instance (loads from disk) that new instances pull weights from
#   MODE=rfork  a new instance that pulls the seed's processed weights over RCCL instead
#               of reading the checkpoint; TP_SIZE and model must match the seed
#
# Examples (TP4, seed on GPU0-3, new instance on GPU4-7):
#   HIP_VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 PORT=7090 MODE=seed bash evaluation/launch_qwen38_flash_next_fp8_mi308x_rfork.sh
#   HIP_VISIBLE_DEVICES=4,5,6,7 TP_SIZE=4 PORT=7091 MODE=rfork SEED_PORT=7090 bash evaluation/launch_qwen38_flash_next_fp8_mi308x_rfork.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${MODE:-seed}"
# R-Fork is validated without the MTP draft model; the base launch script defaults MTP=1.
export MTP="${MTP:-0}"
TP_SIZE="${TP_SIZE:-2}"
SEED_HOST="${SEED_HOST:-127.0.0.1}"
SEED_PORT="${SEED_PORT:-7080}"
# One port per TP rank, opened on the seed. Keep them outside
# /proc/sys/net/ipv4/ip_local_port_range, otherwise an ephemeral socket can take one.
GROUP_PORT_BASE="${GROUP_PORT_BASE:-62000}"

extra_args=()
case "${MODE}" in
  seed)
    ;;
  rfork)
    group_ports="$(python3 -c "import json,sys; b,n=int(sys.argv[1]),int(sys.argv[2]); print(json.dumps(list(range(b,b+n))))" "${GROUP_PORT_BASE}" "${TP_SIZE}")"
    extra_args+=(
      --load-format remote_instance
      --remote-instance-weight-loader-backend nccl
      --remote-instance-weight-loader-seed-instance-ip "${SEED_HOST}"
      --remote-instance-weight-loader-seed-instance-service-port "${SEED_PORT}"
      --remote-instance-weight-loader-send-weights-group-ports "${group_ports}"
    )
    ;;
  *)
    echo "Unknown MODE=${MODE}; use seed or rfork." >&2
    exit 1
    ;;
esac

export TP_SIZE
exec bash "${SCRIPT_DIR}/launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh" \
  "${extra_args[@]}" "$@"
