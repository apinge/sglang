export SGLANG_USE_AITER=1
export SGLANG_USE_CUDA_IPC_TRANSPORT=1
export SGLANG_USE_IPC_POOL_HANDLE_CACHE=1

model=/models/Qwen3.6-27B/
model1=/models/Qwen3.6-27B-Dflash/

python3 -m sglang.launch_server \
 --port 9080 \
 --model-path ${model} \
 --tp-size 1 \
 --reasoning-parser qwen3 \
 --tool-call-parser qwen3_coder \
 --enable-multimodal \
 --trust-remote-code \
 --speculative-algorithm DFLASH \
 --speculative-draft-model-path ${model1} \
 --speculative-dflash-block-size 16 \
 --speculative-draft-attention-backend triton \
 --attention-backend aiter \
 --mm-attention-backend aiter_attn \
 --kv-cache-dtype fp8_e4m3 \
 --page-size 64 \
 --chunked-prefill-size 32768 \
 --mem-fraction-static 0.9 \
 --max-prefill-tokens 32768 \
 --max-running-requests 32 \
 --disable-custom-all-reduce \
 --disable-radix-cache  2>&1 | tee qwen3.6-27B_tp1_dflash.log
