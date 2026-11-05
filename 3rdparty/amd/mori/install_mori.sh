#!/usr/bin/env bash
# Build mori for the Qwen4-Exp SP SDMA overlap (MI308X / gfx942).
#
#   bash 3rdparty/amd/mori/install_mori.sh [TARGET_DIR]
#
# Clones ROCm/mori at the tested commit, applies the MI308X engine-spread patch and
# builds it with SDMA enabled. With TARGET_DIR the package goes there (add it to
# PYTHONPATH); without it, it is pip-installed into the current environment.
set -euo pipefail

MORI_REPO="${MORI_REPO:-https://github.com/ROCm/mori.git}"
MORI_COMMIT="${MORI_COMMIT:-752efd3703136a09ed5cda73563d98e42f16ac69}"
MORI_SRC="${MORI_SRC:-${TMPDIR:-/tmp}/mori-${MORI_COMMIT:0:7}}"
TARGET_DIR="${1:-}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

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
git fetch --quiet origin "${MORI_COMMIT}" 2>/dev/null || true
git checkout --quiet --force "${MORI_COMMIT}"
git submodule update --init --depth 1 3rdparty/msgpack-c 3rdparty/spdlog
if git apply --check "${HERE}/mori_mi308x_engine_spread.patch" 2>/dev/null; then
  git apply "${HERE}/mori_mi308x_engine_spread.patch"
elif ! git apply --reverse --check "${HERE}/mori_mi308x_engine_spread.patch" 2>/dev/null; then
  echo "mori_mi308x_engine_spread.patch does not apply to ${MORI_COMMIT}" >&2
  exit 1
fi

# BUILD_CCO_SDMA=ON is required: a library built without it has no SDMA queues and
# every put silently moves no bytes. UMBP is not needed.
export MORI_GPU_ARCHS="${MORI_GPU_ARCHS:-gfx942}" BUILD_UMBP=OFF BUILD_CCO_SDMA=ON
if [[ -n "${TARGET_DIR}" ]]; then
  pip install --no-deps --target "${TARGET_DIR}" .
  echo "mori installed in ${TARGET_DIR}; run with PYTHONPATH=${TARGET_DIR}:\$PYTHONPATH"
else
  pip install --no-deps .
fi
