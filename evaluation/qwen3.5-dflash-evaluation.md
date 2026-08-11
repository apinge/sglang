## Qwen3.5 sglang dflash evaluation guide

### 1. Prepare docker image
```bash
docker pull lmsysorg/sglang-rocm:v0.5.15-rocm720-mi35x-20260713
docker run -it --name qwen3.5-dflash-xisun --device=/dev/dri --device=/dev/kfd --device=/dev/infiniband --device=/dev/infiniband/rdma_cm --privileged --network=host --ipc=host --cap-add=SYS_ADMIN --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --group-add video --ipc=host -v $HOME:/root/workspace   -v /data2/models:/models lmsysorg/sglang-rocm:v0.5.15-rocm720-mi35x-20260713  bash
```

### 2. Setup environment
```bash
pip uninstall sglang-kernel sgl-kernel sglang -y
rm -rf /sgl-workspace/sglang
git clone https://github.com/apinge/sglang -b qwen3_5_v0.5.15_dflash_aiter
cd sglang
# Compile sgl-kernel
pip install --upgrade pip
cd sgl-kernel
python3 setup_rocm.py install

# Install sglang python package
cd ..
rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
pip install -e "python[all_hip]"
pip install git+https://github.com/sgl-project/sgl-eval
```

### 3. Run dflash accpetance legnth test with offical harness benchmark
```bash
cd evaluation
python3 run_dflash_official_harness_rocm_aiter.py \
    --target-model /models/Qwen3.5-397B-A17B-FP8 \
    --dflash-draft-model /models/Qwen3.5-397B-A17B-DFlash \
    --spec-modes dflash \
    --dflash-block-sizes 16 \
    --workloads gsm8k \
    --num-samples 200 \
    --concurrencies 1
```

### 4. Run gsm8k accuracy check with sgl-eval
```bash
cd evaluation
./launch_qwen3.6-27B-bf16_tp1_Dflash.sh
```

```bash
./run_gsm8k.sh
```
