export SGLANG_USE_AITER=1
export SGLANG_USE_CUDA_IPC_TRANSPORT=1
export SGLANG_USE_IPC_POOL_HANDLE_CACHE=1

model=/models/Qwen3.5-397B-A17B-FP8
model1=/models/Qwen3.5-397B-A17B-DFlash

python3 -m sglang.launch_server \
 --port 9080 \
 --model-path ${model} \
 --tp-size 4 \
 --reasoning-parser qwen3 \
 --tool-call-parser qwen3_coder \
 --enable-multimodal \
 --trust-remote-code \
 --speculative-algorithm DFLASH \
 --speculative-draft-model-path ${model1} \
 --speculative-dflash-block-size 16 \
 --speculative-draft-attention-backend triton \
 --chunked-prefill-size 32768 \
 --mem-fraction-static 0.9 \
 --max-prefill-tokens 32768 \
 --max-running-requests 32 \
 --attention-backend aiter \
 --mm-attention-backend aiter_attn \
 --disable-custom-all-reduce \
 --kv-cache-dtype fp8_e4m3 \
 --page-size 64 \
 --disable-radix-cache 2>&1 | tee qwen3.5-397B-fp8_tp4_dflash.log
