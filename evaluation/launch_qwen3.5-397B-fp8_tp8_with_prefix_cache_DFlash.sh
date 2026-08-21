export SGLANG_DISABLE_CUDNN_CHECK=1
export SGLANG_USE_CUDA_IPC_TRANSPORT=1
export SGLANG_VLM_CACHE_SIZE_MB=8192 #阿里用0


export SGLANG_USE_AITER=1
export SGLANG_ROCM_USE_AITER_LINEAR_SHUFFLE=1
export SGLANG_ROCM_USE_AITER_LINEAR_FP8HIPB=1
export AITER_QUICK_REDUCE_QUANTIZATION=INT6
export USE_AITER_COMM=1
export USE_HIP_LINEAR_ATTN=1
export SGLANG_USE_AITER_NEW_CA=false
export SGLANG_USE_IPC_POOL_HANDLE_CACHE=1

#export AITER_MOE_PADDING_SIZE=192
export HIP_GDN_SORT_IDX_BS=32768
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1 # pip uninstall torch-c-dlpack-ext

model=/models/Qwen/Qwen3.5-397B-A17B-PTPC-FP8
model1=/models/Qwen/Qwen3.5-397B-A17B-Dflash

# --disable-radix-cache
python3 -m sglang.launch_server \
 --port 7080 \
 --model-path ${model} \
 --tp-size 8 \
 --reasoning-parser qwen3 \
 --tool-call-parser qwen3_coder \
 --enable-multimodal \
 --trust-remote-code \
 --speculative-algorithm DFLASH \
 --speculative-draft-model-path ${model1} \
 --speculative-num-draft-tokens 16 \
 --speculative-draft-attention-backend triton \
 --chunked-prefill-size 32768 \
 --mem-fraction-static 0.9 \
 --max-prefill-tokens 32768 \
 --max-running-requests 128 \
 --linear-attn-backend aiter \
 --linear-attn-decode-backend aiter \
 --linear-attn-prefill-backend aiter \
 --attention-backend aiter \
 --mm-attention-backend aiter_attn \
 --mamba-scheduler-strategy extra_buffer \
 --kv-cache-dtype fp8_e4m3 \
 --page-size 64 2>&1 | tee qwen3.5-397B-fp8_tp8_dflash.log
