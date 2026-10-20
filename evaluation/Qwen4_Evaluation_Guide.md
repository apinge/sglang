# Set Environment

1. Docker image:
   ```bash
   rocm/ali-private:ubuntu22.04_rocm7.14.0_cp310_torch2.12.0_sglang_c9ef753_aiter_4752a40_20260923_v2
   ```

2. Install aiter:
   ```bash
   pip uninstall -y aiter
   git clone -b qwen38-flash-next-dev https://github.com/apinge/aiter.git
   cd aiter
   git checkout <target_commit>
   git submodule sync && git submodule update --init --recursive
   python3 setup.py develop
   ```

3. Setup pyhip
    ```bash
    pip uninstall -y pyhip
    git clone https://github.com/tingqli/pyhip.git -b dev
    pip install -e .
    cd ..
    ```

4. Install sglang:
   ```bash
   pip uninstall -y sglang
   git clone https://github.com/sammysun0711/sglang.git -b qwen4_v0.5.21 
   cd sglang
   git checkout <target_commit>
   # Compile sgl-kernel
   pip install --upgrade pip
   cd python/sglang/kernels/aot
   python setup_rocm.py install

   # Install sglang python package
   cd ../../../..
   rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
   SGLANG_BUILD_RUST_EXTS=none pip install -e "python[all_hip]"
   ```

# Qwen3.5 Evaluation
## Launch server
```bash
./launch_qwen3.5-397B-fp8_tp8_disable_prefix_cache.sh
```

## Run benchmark
```bash
./run_qwen35_pure_text_bench.sh
```

## GSM8K Accuracy verification
```bash
./run_gsm8k.sh
```

# Qwen3.5 Evaluation
## Launch server
```bash
./launch_qwen3.5-397B-fp8_tp8_disable_prefix_cache.sh
```

## Run benchmark
```bash
./run_qwen35_pure_text_bench.sh
```

## GSM8K Accuracy verification
```bash
./run_gsm8k.sh
```


# Qwen3.8 Evaluation
## Launch server
```bash
./launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh
```

## Run benchmark
```bash
./run_qwen38_pure_text_bench.sh
```

## GSM8K Accuracy verification
```bash
./run_gsm8k.sh
```

# Profile
./run_sgl_profile.sh