#!/usr/bin/env bash
# Build mori for the Qwen4-Exp SP SDMA overlap (MI308X / gfx942).
#
#   bash 3rdparty/amd/mori/install_mori.sh [TARGET_DIR]
#
# Clones the pinned mori branch and builds it with SDMA enabled. With TARGET_DIR the
# package goes there (add it to PYTHONPATH); without it, it is pip-installed into the
# current environment. MORI_REF also takes a commit or a pull ref.
set -euo pipefail

MORI_REPO="${MORI_REPO:-https://github.com/apinge/mori.git}"
MORI_REF="${MORI_REF:-qwen3.8-flash-sdma}"
MORI_SRC="${MORI_SRC:-${TMPDIR:-/tmp}/mori-src}"
TARGET_DIR="${1:-}"

# mori's setup.py refuses to build without the ibverbs headers.
if ! ls /usr/include/infiniband/verbs.h >/dev/null 2>&1; then
  echo "Missing libibverbs headers: apt-get install -y libibverbs-dev (or dnf install -y rdma-core-devel)" >&2
  exit 1
fi
python3 -c "import flydsl" 2>/dev/null || {
  echo "FlyDSL is not importable; mori's device API needs it (tested with flydsl 0.3.2)" >&2
  exit 1
}

if [[ ! -d "${MORI_SRC}/.git" ]]; then
  git clone --filter=blob:none "${MORI_REPO}" "${MORI_SRC}"
fi
cd "${MORI_SRC}"
git fetch --quiet "${MORI_REPO}" "${MORI_REF}"
git checkout --quiet --force FETCH_HEAD
echo "mori ${MORI_REF} at $(git rev-parse --short HEAD)"
git submodule update --init --depth 1 3rdparty/msgpack-c 3rdparty/spdlog

# BUILD_CCO_SDMA=ON is required: a library built without it has no SDMA queues and
# every put silently moves no bytes. UMBP is not needed.
export MORI_GPU_ARCHS="${MORI_GPU_ARCHS:-gfx942}" BUILD_UMBP=OFF BUILD_CCO_SDMA=ON
if [[ -n "${TARGET_DIR}" ]]; then
  pip install --no-deps --target "${TARGET_DIR}" .
  echo "mori installed in ${TARGET_DIR}; run with PYTHONPATH=${TARGET_DIR}:\$PYTHONPATH"
else
  pip install --no-deps .
fi
