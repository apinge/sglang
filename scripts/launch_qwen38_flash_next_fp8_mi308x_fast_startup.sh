#!/usr/bin/env bash
# Reference launcher for the fast-startup options of Qwen3.8-Flash-Next-FP8 on MI308X.
# Wraps launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh (same TP_SIZE, PORT,
# LOG_FILE, ... variables) and adds the arguments for one weight-loading MODE:
#
#   MODE=disk             default checkpoint loading (baseline)
#   MODE=seed             a normal instance that other instances can pull weights from
#   MODE=rfork            pull processed weights from a running seed (R-Fork), no disk read
#   MODE=fastsafetensors  fastsafetensors reader (GDS off); TP4/TP8 only for this checkpoint
#   MODE=runai            runai model streamer (libaio_direct)
#
# RFORK_BACKEND=nccl (default) or mooncake selects the R-Fork transport; seed and new
# instance must use the same backend, TP_SIZE and model. With mooncake, HIP IPC needs the
# peer GPUs visible, so HIP_VISIBLE_DEVICES is ignored and BASE_GPU_ID places the instance.
#
# Examples (TP4, seed on GPU0-3, new instance on GPU4-7):
#   HIP_VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 PORT=7090 MODE=seed bash <this script>
#   HIP_VISIBLE_DEVICES=4,5,6,7 TP_SIZE=4 PORT=7091 MODE=rfork SEED_PORT=7090 bash <this script>
#   TP_SIZE=4 PORT=7090 MODE=seed RFORK_BACKEND=mooncake BASE_GPU_ID=0 bash <this script>
#   TP_SIZE=4 PORT=7091 MODE=rfork RFORK_BACKEND=mooncake BASE_GPU_ID=4 SEED_PORT=7090 bash <this script>
#
# Compiled-kernel caches are reused across launches from $HOME (~/.flydsl matters most);
# to prepare a new node, copy ~/.flydsl, ~/.cache/sglang and ~/.cache/comgr into its $HOME.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${MODE:-disk}"
RFORK_BACKEND="${RFORK_BACKEND:-nccl}"
TP_SIZE="${TP_SIZE:-2}"
SEED_HOST="${SEED_HOST:-127.0.0.1}"
SEED_PORT="${SEED_PORT:-7080}"
# One port per TP rank, opened on the seed. Keep them outside
# /proc/sys/net/ipv4/ip_local_port_range, otherwise an ephemeral socket can take one.
GROUP_PORT_BASE="${GROUP_PORT_BASE:-62000}"
BASE_GPU_ID="${BASE_GPU_ID:-}"

if [[ "${RFORK_BACKEND}" != "nccl" && "${RFORK_BACKEND}" != "mooncake" ]]; then
  echo "RFORK_BACKEND must be nccl or mooncake (got ${RFORK_BACKEND})." >&2
  exit 1
fi

extra_args=()
if [[ ( "${MODE}" == "seed" || "${MODE}" == "rfork" ) && "${RFORK_BACKEND}" == "mooncake" ]]; then
  if [[ -n "${HIP_VISIBLE_DEVICES:-}" || -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "RFORK_BACKEND=mooncake: ignoring HIP_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES; placing by BASE_GPU_ID." >&2
    unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
  fi
  if [[ -z "${BASE_GPU_ID}" ]]; then
    echo "RFORK_BACKEND=mooncake needs BASE_GPU_ID (first GPU of this instance)." >&2
    exit 1
  fi
  # Same-node transfer over HIP IPC; SGLang's default protocol is rdma.
  export MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-hip}"
fi
if [[ -n "${BASE_GPU_ID}" ]]; then
  extra_args+=(--base-gpu-id "${BASE_GPU_ID}")
fi

case "${MODE}" in
  disk)
    ;;
  seed)
    if [[ "${RFORK_BACKEND}" == "mooncake" ]]; then
      extra_args+=(--remote-instance-weight-loader-start-seed-via-transfer-engine)
    fi
    ;;
  rfork)
    extra_args+=(
      --load-format remote_instance
      --remote-instance-weight-loader-seed-instance-ip "${SEED_HOST}"
      --remote-instance-weight-loader-seed-instance-service-port "${SEED_PORT}"
    )
    if [[ "${RFORK_BACKEND}" == "mooncake" ]]; then
      extra_args+=(--remote-instance-weight-loader-backend transfer_engine)
    else
      group_ports="$(python3 -c "import json,sys; b,n=int(sys.argv[1]),int(sys.argv[2]); print(json.dumps(list(range(b,b+n))))" "${GROUP_PORT_BASE}" "${TP_SIZE}")"
      extra_args+=(
        --remote-instance-weight-loader-backend nccl
        --remote-instance-weight-loader-send-weights-group-ports "${group_ports}"
      )
    fi
    ;;
  fastsafetensors)
    if (( TP_SIZE == 2 )); then
      # It copies each file whole to one GPU; the 95 GiB first shard does not fit
      # next to the TP2 weights.
      echo "MODE=fastsafetensors runs out of memory at TP_SIZE=2 with this checkpoint." >&2
      exit 1
    fi
    extra_args+=(--load-format fastsafetensors --model-loader-extra-config '{"enable_gds": false}')
    ;;
  runai)
    # The default io_uring strategy fails on older kernels (e.g. 5.10).
    export RUNAI_STREAMER_FS_STRATEGY="${RUNAI_STREAMER_FS_STRATEGY:-libaio_direct}"
    extra_args+=(--load-format runai_streamer)
    ;;
  *)
    echo "Unknown MODE=${MODE}; use disk, seed, rfork, fastsafetensors or runai." >&2
    exit 1
    ;;
esac

export TP_SIZE
exec bash "${SCRIPT_DIR}/launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh" \
  "${extra_args[@]}" "$@"
