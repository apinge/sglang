# Qwen3.5 & Qwen3.8 PTPC Model Evaluation Guide

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


<details>
<summary>Qwen 3.5 & Qwen 3.8 CI Overview</summary>
CI runs each selected case in its own GitHub Actions matrix job using `evaluation/Dockerfile.ci` and the existing launch, benchmark, and `run_gsm8k.sh` scripts.
Non-draft fork PRs require the `run-qwen4-ci` label to run GPU cases. Adding the label triggers a new run through the `labeled` event.
Case jobs run serially on the shared host, each with its own image container, console logs, and result artifact. Docker builds reuse cached layers.
The default case order is Qwen3.8 Flash TP2 base, Qwen3.8 Flash TP2 MTP, Qwen3.5 35B TP4 base, Qwen3.5 35B TP1 DFlash, Qwen3.5 397B TP8 base, then Qwen3.5 397B TP8 DFlash. Local runs and the Actions matrix use this order. Direct local runs can evaluate multiple cases in one container and share its normal kernel cache:

```bash
python3 scripts/ci/qwen4_ci.py --models-root /raid/models \
  --client-root /path/to/pinned-client --output /tmp/qwen4-results-new
```

The default command runs **performance first, then GSM8K accuracy** against the
same server. Use it without `--accuracy-only` or `--smoke` for the full workload.
CI captures one complete log per phase: `server.log`,
`run_qwen*_pure_text_bench.sh.log`, and `run_gsm8k.sh.log`. These logs stream to
the console and are saved in the result artifact.
Performance uses 10 warmup requests and 128 random requests at concurrency one;
Qwen3.5 uses 8,000 input / 500 output tokens, and Qwen3.8 uses 12,000 / 350.
Accuracy scores 200 examples and enforces the recorded gate.

Build the local image from the checkout once, or rebuild after local code changes:

```bash
mkdir -p /tmp/qwen4-buildx
BUILDX_CONFIG=/tmp/qwen4-buildx docker build --no-cache --network=host --progress=plain \
  -f evaluation/Dockerfile.ci -t qwen4-ci:local .
```

Use that image to run the local benchmark and accuracy checks:

```bash
(
  set -euo pipefail
  qwen4_work=$(mktemp -d /tmp/qwen4-local.XXXXXX)
  qwen4_container="$(basename "$qwen4_work")"
  echo "Results: $qwen4_work"
  touch /tmp/qwen4-mi308x.lock
  docker run -d --name "$qwen4_container" --device /dev/kfd --device /dev/dri \
    --ipc=host --network=host --group-add video --ulimit nofile=65536:65536 \
    --security-opt seccomp=unconfined \
    -v /raid/models:/models:ro -v "$qwen4_work:/results" \
    -v /tmp/qwen4-mi308x.lock:/tmp/qwen4-mi308x.lock \
    qwen4-ci:local sleep infinity
  cleanup() {
    docker exec "$qwen4_container" chown -R "$(id -u):$(id -g)" /results || true
    docker rm -f "$qwen4_container" >/dev/null || true
  }
  trap cleanup EXIT
  docker cp "$qwen4_container:/ci/build.json" "$qwen4_work/build.json"
  docker exec "$qwen4_container" python3 scripts/ci/qwen4_ci.py \
    --case Qwen3.5-397B-TP8-DFlash --gpus 0,1,2,3,4,5,6,7 \
    --port 17880 --output /results/evaluation \
    2>&1 | tee "$qwen4_work/console.log"
  cat "$qwen4_work/evaluation/summary.md"
)
```

Remove `--case Qwen3.5-397B-TP8-DFlash` to run all six cases sequentially, or pass
comma-separated case IDs from the table below.

The command exits nonzero for launch errors, incomplete requests, invalid results, or an accuracy regression. Logs, native JSON/HTML reports, and `summary.md` are written under the fresh output directory. Build provenance lists the source commits and installed package versions. Keep `QWEN4_WORK_ROOT` outside `/home` on hosts with limited home storage; Docker layers remain in the daemon's data root.
Use a fresh `--output-dir` for each direct evaluator run. Ordinary evaluations retain model-specific report filenames; strict GSM8K runs use `gsm8k.json/html`.

Default runs evaluate all six cases with 128 fixed-length performance requests and 200 GSM8K examples each. The accuracy gate is `score >= recorded_reference * 0.98`; performance thresholds are advisory.


| Case ID                 | Reference | Minimum correct / 200 |
| ----------------------- | --------- | --------------------- |
| Qwen3.8-Flash-TP2-Base  | 98.0%     | 193                   |
| Qwen3.8-Flash-TP2-MTP   | 98.5%     | 194                   |
| Qwen3.5-35B-TP4-Base    | 97.0%     | 191                   |
| Qwen3.5-35B-TP1-DFlash  | 97.0%     | 191                   |
| Qwen3.5-397B-TP8-Base   | 99.0%     | 195                   |
| Qwen3.5-397B-TP8-DFlash | 98.5%     | 194                   |


Use `--case Qwen3.5-35B-TP1-DFlash --gpus 2` to run one selected case on a free
GPU. Comma-separated case IDs and GPU indices are supported. A selected subset
reports its coverage and does not reuse prior results. `--accuracy-only` runs the
200-example gate without the performance benchmark. `--smoke` uses two requests
and two examples and reports diagnostic results with the accuracy gate disabled.
`--port` selects an unused server port.
</details>
