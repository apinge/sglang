# Qwen3.5 & Qwen3.8 PTPC Model Evaluation Guide

## Target Model
Qwen3.5
- [sammysun0711/Qwen3.5-397B-A17B-PTPC-FP8](https://huggingface.co/sammysun0711/Qwen3.5-397B-A17B-PTPC-FP8)
- [sammysun0711/Qwen3.5-35B-A3B-PTPC-FP8](https://huggingface.co/sammysun0711/Qwen3.5-35B-A3B-PTPC-FP8)

Qwen3.8
- [sammysun0711/Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16](https://huggingface.co/sammysun0711/Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16)

## Set Environment

1. Docker image:
   ```bash
   rocm/ali-private:ubuntu22.04_rocm7.14.0_cp310_torch2.12.0_sglang_d0a5b96_aiter_b7aed05_20261020
   ```

2. Install aiter:
   ```bash
   cd /opt
   pip uninstall -y amd-aiter
   rm -rf /opt/aiter ~/.aiter
   git clone https://github.com/apinge/aiter.git -b qwen38-flash-next-dev
   cd aiter
   # git checkout <target_commit>
   git submodule sync && git submodule update --init --recursive
   pip install -e .
   ```

3. Setup pyhip
    ```bash
    cd /opt
    pip uninstall -y pyhip
    rm -rf pyhip
    git clone https://github.com/tingqli/pyhip.git -b dev
    pip install -e .
    cd ..
    ```

4. Install sglang:
   ```bash
   cd /opt
   pip uninstall -y sglang sglang-kernel
   rm -rf /opt/sglang
   git clone https://github.com/apinge/sglang.git -b qwen4_v0.5.21
   cd sglang
   # git checkout <target_commit>
   # Compile sgl-kernel
   pip install --upgrade pip
   cd python/sglang/kernels/aot
   python setup_rocm.py install

   # Install sglang python package
   cd ../../../..
   rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
   SGLANG_BUILD_RUST_EXTS=none pip install -e "python[all_hip]"
   ```

## Qwen3.5 Evaluation
### Launch server
Target model: [sammysun0711/Qwen3.5-397B-A17B-PTPC-FP8](https://huggingface.co/sammysun0711/Qwen3.5-397B-A17B-PTPC-FP8)
```bash
./launch_qwen3.5-397B-fp8_tp8_disable_prefix_cache.sh
```

Target model: [sammysun0711/Qwen3.5-35B-A3B-PTPC-FP8](https://huggingface.co/sammysun0711/Qwen3.5-35B-A3B-PTPC-FP8)
```bash
./launch_qwen3.5-35B-fp8_tp4_disable_prefix_cache.sh
```

### Run benchmark
```bash
./run_qwen35_pure_text_bench.sh
```

### Run GSM8K Accuracy verification
```bash
./run_gsm8k.sh
```

### Run Profile
```bash
./run_qwen35_pure_text_profile.sh
```

## Qwen3.8 Evaluation

### Launch server
Target model: [sammysun0711/Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16](https://huggingface.co/sammysun0711/Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16)
```bash
./launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh
```

### Run benchmark
```bash
./run_qwen38_pure_text_bench.sh
```

### Run GSM8K Accuracy verification
```bash
./run_gsm8k.sh
```

### Run Profile
```bash
./run_qwen38_pure_text_profile.sh
```
