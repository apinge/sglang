#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${ROOT}/logs/server"

export SGLANG_DISABLE_CUDNN_CHECK=1
export SGLANG_USE_CUDA_IPC_TRANSPORT=1
export SGLANG_VLM_CACHE_SIZE_MB=8192
export SGLANG_USE_AITER=1
export SGLANG_ROCM_USE_AITER_LINEAR_SHUFFLE=1
export SGLANG_ROCM_USE_AITER_LINEAR_FP8HIPB=1
export SGLANG_USE_AITER_NEW_CA=false
export SGLANG_USE_IPC_POOL_HANDLE_CACHE=1
export HIP_GDN_SORT_IDX_BS=32768

export TVM_FFI_DISABLE_TORCH_C_DLPACK=1

model=/model/Qwen3.5-27B-PTPC-compressor

python3 -m sglang.launch_server \
  --port 9080 \
  --model-path ${model} \
  --tp-size 2 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-multimodal \
  --trust-remote-code \
  --chunked-prefill-size 32768 \
  --mem-fraction-static 0.9 \
  --max-prefill-tokens 32768 \
  --max-running-requests 128 \
  --cuda-graph-max-bs 128 \
  --attention-backend aiter \
  --mm-attention-backend aiter_attn \
  --disable-radix-cache \
  --kv-cache-dtype fp8_e4m3 \
  --linear-attn-backend aiter \
  --linear-attn-decode-backend aiter \
  --linear-attn-prefill-backend aiter \
  --watchdog-timeout 1200 \
  --disable-custom-all-reduce 2>&1 | tee "${ROOT}/logs/server"/qwen3.5-27B-fp8_tp2_no_prefix_cache_origin.log
