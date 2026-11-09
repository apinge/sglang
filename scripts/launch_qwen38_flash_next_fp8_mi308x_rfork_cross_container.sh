#!/usr/bin/env bash
# Run on the host: start a new container on the same node that pulls weights from a
# seed instance running in another container (R-Fork), then serves.
#
# The seed is a normal instance started in its own container, e.g.
#   HIP_VISIBLE_DEVICES=0,1 TP_SIZE=2 PORT=7090 MODE=seed \
#     bash scripts/launch_qwen38_flash_next_fp8_mi308x_rfork.sh
# Both containers need --network=host --ipc=host and the GPU devices; with those, RCCL
# moves the weights over P2P/XGMI as it does within one container.
#
# Example (TP2, seed on GPU0-1 at port 7090, new instance on GPU2-3 at port 7091):
#   IMAGE=<same image as the seed container> \
#   HIP_VISIBLE_DEVICES=2,3 TP_SIZE=2 PORT=7091 SEED_PORT=7090 \
#     bash scripts/launch_qwen38_flash_next_fp8_mi308x_rfork_cross_container.sh
set -euo pipefail

IMAGE="${IMAGE:?set IMAGE to the seed container image}"
HOST_MODELS="${HOST_MODELS:-/models}"
CONTAINER_NAME="${CONTAINER_NAME:-rfork-fork}"
# SGLang checkout inside the new container; defaults to the image's own install.
REPO_DIR="${REPO_DIR:-/opt/sglang}"
# Packages the image lacks but the seed container has (e.g. a PyHIP source tree and
# its dist-info dir); appended to PYTHONPATH.
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-}"
# Optional one-time setup inside the new container, run before the server starts
# (e.g. restoring kernel caches or prebuilt aiter JIT modules the image does not ship).
SETUP_CMD="${SETUP_CMD:-true}"
# Extra docker run arguments, e.g. "-v <host checkout>:/src" together with REPO_DIR=/src.
read -r -a docker_args <<< "${DOCKER_ARGS:-}"

TP_SIZE="${TP_SIZE:-2}"
PORT="${PORT:-7091}"
SEED_HOST="${SEED_HOST:-127.0.0.1}"
SEED_PORT="${SEED_PORT:-7090}"
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:?set HIP_VISIBLE_DEVICES to the new instance GPUs}"
LOG_FILE="${LOG_FILE:-${REPO_DIR}/scripts/logs/rfork_cross_container_$(date -u +%Y%m%dT%H%M%SZ).log}"

pythonpath="${REPO_DIR}/python${EXTRA_PYTHONPATH:+:${EXTRA_PYTHONPATH}}"
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
docker run -d --name="${CONTAINER_NAME}" \
  --privileged --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --network=host --ipc=host --device /dev/kfd --device /dev/dri \
  -v "${HOST_MODELS}:/models" "${docker_args[@]}" \
  -e PYTHONPATH="${pythonpath}" \
  -e MODE=rfork -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}" \
  -e TP_SIZE="${TP_SIZE}" -e PORT="${PORT}" \
  -e SEED_HOST="${SEED_HOST}" -e SEED_PORT="${SEED_PORT}" \
  -e LOG_FILE="${LOG_FILE}" \
  --entrypoint /bin/bash "${IMAGE}" \
  -c "set -e; ${SETUP_CMD}; cd ${REPO_DIR}; exec bash scripts/launch_qwen38_flash_next_fp8_mi308x_rfork.sh"

echo "Started ${CONTAINER_NAME}; server log: ${LOG_FILE} (inside the container)"
echo "Follow with: docker logs -f ${CONTAINER_NAME}"
