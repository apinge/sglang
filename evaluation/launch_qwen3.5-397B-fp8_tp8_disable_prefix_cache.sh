export SGLANG_DISABLE_CUDNN_CHECK=1
export SGLANG_USE_CUDA_IPC_TRANSPORT=1
export SGLANG_VLM_CACHE_SIZE_MB=8192 #阿里用0

export SGLANG_USE_AITER=1
export SGLANG_ROCM_USE_AITER_LINEAR_SHUFFLE=1
export SGLANG_ROCM_USE_AITER_LINEAR_FP8HIPB=1
export AITER_QUICK_REDUCE_QUANTIZATION=INT6

export USE_AITER_COMM=1
#export USE_HIP_LINEAR_ATTN=1
export SGLANG_USE_AITER_NEW_CA=false
export SGLANG_USE_IPC_POOL_HANDLE_CACHE=1
# Keep PR277 fused old-CA allreduce + RMSNorm enabled.
unset SGLANG_DISABLE_AITER_FUSED_AR_RMSNORM
unset SGLANG_DISABLE_AITER_FUSED_MLP_QUANT
export HIP_GDN_SORT_IDX_BS=32768

# export SGLANG_DISABLE_AITER_FUSED_AR_RMSNORM=1

export TVM_FFI_DISABLE_TORCH_C_DLPACK=1 # pip uninstall torch-c-dlpack-ext

# --watchdog-timeout 1200 for profile
model=/model/offical_qwen3p5_397B_ptpc
python3 -m sglang.launch_server \
        --port 7080 \
        --model-path ${model} \
        --tp-size 8 \
        --reasoning-parser qwen3 \
        --tool-call-parser qwen3_coder \
        --enable-multimodal \
        --trust-remote-code \
        --chunked-prefill-size 32768 \
        --mem-fraction-static 0.9 \
        --max-prefill-tokens 32768 \
        --max-running-requests 128 \
        --attention-backend aiter \
        --mm-attention-backend aiter_attn \
        --kv-cache-dtype fp8_e4m3 \
        --cuda-graph-max-bs 128 \
        --linear-attn-backend aiter \
        --linear-attn-decode-backend aiter \
        --linear-attn-prefill-backend aiter \
        --watchdog-timeout 1200 \
        --disable-radix-cache  2>&1 | tee launch_qwen3.5-397B-fp8_tp8_disable_prefix_cache.sh.log
        # --mamba-scheduler-strategy extra_buffer \
        # --page-size 64  2>&1 | tee launch_qwen3.5-397B-fp8_tp8_prefix_cache_origin_ali2.sh.log
 
