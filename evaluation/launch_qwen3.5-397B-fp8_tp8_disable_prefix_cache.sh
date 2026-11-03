#!/usr/bin/env bash
set -euo pipefail
export SGLANG_DISABLE_CUDNN_CHECK=1
export SGLANG_VLM_CACHE_SIZE_MB=8192 #阿里用0

export SGLANG_USE_AITER=1
export SGLANG_ROCM_USE_AITER_LINEAR_SHUFFLE=1
export SGLANG_ROCM_USE_AITER_LINEAR_FP8HIPB=1
export AITER_QUICK_REDUCE_QUANTIZATION=INT6
export USE_AITER_COMM=1

export SGLANG_USE_AITER_NEW_CA=false
export SGLANG_USE_IPC_POOL_HANDLE_CACHE=1
export HIP_GDN_SORT_IDX_BS=32768


export TVM_FFI_DISABLE_TORCH_C_DLPACK=1 # pip uninstall torch-c-dlpack-ext

# --watchdog-timeout 1200 for profile
model="${MODEL_PATH:-/models/Qwen3.5-397B-A17B-PTPC-FP8}"
"${SERVER_PYTHON:-python3}" -m sglang.launch_server \
        --host "${HOST:-0.0.0.0}" --port "${PORT:-7080}" --served-model-name "${SERVED_MODEL_NAME:-$model}" \
        --model-path "${model}" \
        --tp-size 8 \
        --reasoning-parser qwen3 \
        --tool-call-parser qwen3_coder \
        --enable-multimodal \
        --mm-feature-transport cuda_ipc \
        --trust-remote-code \
        --chunked-prefill-size 32768 \
        --mem-fraction-static 0.9 \
        --max-prefill-tokens 32768 \
        --max-running-requests 128 \
        --attention-backend aiter \
        --mm-attention-backend aiter_attn \
        --kv-cache-dtype fp8_e4m3 \
        --cuda-graph-max-bs-decode 128 \
        --watchdog-timeout 3600 \
        --disable-radix-cache  2>&1 | tee "${LOG_FILE:-launch_qwen3.5-397B-fp8_tp8_disable_prefix_cache.sh.log}"
        # --linear-attn-backend aiter \
        # --linear-attn-decode-backend aiter \
        # --linear-attn-prefill-backend aiter \
        # --mamba-radix-cache-strategy extra_buffer \
        # --page-size 64  2>&1 | tee "${LOG_FILE:-launch_qwen3.5-397B-fp8_tp8_disable_prefix_cache.sh.log}"

