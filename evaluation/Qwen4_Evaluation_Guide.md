# Qwen3.5 & Qwen3.8 PTPC Model Evaluation Guide


**Unified CI and local evaluation.** The existing launch, benchmark, and GSM8K scripts are also the CI execution interfaces. CI runs six configurations: 35B TP4 base / TP1 DFlash, 397B TP8 base / DFlash, and Qwen3.8 TP2 MTP off / on. It builds one image, creates a separate container per target model, and restarts the server between the pair's two cases. See [the CI runner guide](qwen4/README.md) for workflow setup, source pinning, and complete local-run commands.

The default GSM8K test now scores **200 examples**, with five separate few-shot examples excluded from scoring. It writes JSON, HTML, and per-example JSONL and fails on incomplete requests. Set `GSM8K_DATA_PATH` to the pinned local test file and choose a fresh `GSM8K_OUTPUT_DIR` for each evaluation. The dataset setup command is `python qwen4/prepare_assets.py --root /models/benchmark_datasets` inside a writable provisioning container; CI mounts the staged datasets read-only.

Benchmark scripts accept `BENCH_MODEL`, `BENCH_SERVED_MODEL`, `BENCH_HOST`, `BENCH_PORT`, `DATASET_PATH`, `BENCH_OUTPUT_FILE`, and `BENCH_LOG_FILE`. Always set `BENCH_MODEL` for the selected checkpoint, especially when testing 35B. CI supplies these values automatically and keeps every case's results separate. For example, after launching the 35B target-only server:

```bash
BENCH_MODEL=/models/Qwen3.5-35B-A3B-PTPC-FP8 \
  BENCH_OUTPUT_FILE=35b-base-performance.jsonl \
  ./run_qwen35_pure_text_bench.sh
GSM8K_OUTPUT_DIR=35b-base-gsm8k ./run_gsm8k.sh
```

For Qwen3.8, explicitly select the two modes:

```bash
MTP=0 TP_SIZE=2 ./launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh
# Stop that server before the second launch in the same target container.
MTP=1 TP_SIZE=2 ./launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh
```

The model mount is `/raid/models:/models:ro` on the bare-metal runner. The initial CI result gates check build/startup and evaluation completeness; numerical thresholds remain advisory during calibration. Maintainers can use the `run-qwen4-ci` label, manual dispatch, or the optional `/ci run` and `/ci run_failed` comment controls after the workflows are installed.

## Target Model
Qwen3.5 + DFlash
Target Model
- [sammysun0711/Qwen3.5-397B-A17B-PTPC-FP8](https://huggingface.co/sammysun0711/Qwen3.5-397B-A17B-PTPC-FP8)
- [sammysun0711/Qwen3.5-35B-A3B-PTPC-FP8](https://huggingface.co/sammysun0711/Qwen3.5-35B-A3B-PTPC-FP8)

Draft Model：
- [z-lab/Qwen3.5-397B-A17B-DFlash](https://huggingface.co/z-lab/Qwen3.5-397B-A17B-DFlash)
- [z-lab/Qwen3.5-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.5-35B-A3B-DFlash)

Qwen3.8 + MTP
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
    cd pyhip
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

## Qwen3.5 Target Only Evaluation
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

## Qwen3.5 + DFlash Evaluation
### Launch server
Target model: [sammysun0711/Qwen3.5-397B-A17B-PTPC-FP8](https://huggingface.co/sammysun0711/Qwen3.5-397B-A17B-PTPC-FP8)
Draft model: [z-lab/Qwen3.5-397B-A17B-DFlash](https://huggingface.co/z-lab/Qwen3.5-397B-A17B-DFlash)
```bash
./launch_qwen3.5-397B-fp8_tp8_dflash_with_prefix_cache.sh
```

Target model: [sammysun0711/Qwen3.5-35B-A3B-PTPC-FP8](https://huggingface.co/sammysun0711/Qwen3.5-35B-A3B-PTPC-FP8)
Draft model: [z-lab/Qwen3.5-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.5-35B-A3B-DFlash)
```bash
./launch_qwen3.5-35B-fp8_tp1_dflash_with_prefix_cache_mfs0.7_bs32.sh
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


## Qwen3.8 + MTP Evaluation

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
