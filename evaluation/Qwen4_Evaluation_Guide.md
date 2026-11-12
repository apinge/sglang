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

The script enables the sequence-parallel hyper-connection residual for prefills of at least 2048 tokens, with its
reduce-scatter and all-gathers overlapped with compute over SDMA through mori (see
`benchmark/qwen4_seq_parallel/README.md`). mori is not in the base image:

```bash
apt-get update && apt-get install -y libibverbs-dev   # build-time only
pip install "amd_mori @ git+https://github.com/ROCm/mori.git@de97ceacefa96a6ba45b5ed3276abdb4a3bf7fd6"
```

Without mori the aiter collectives run, with a warning. `SGLANG_QWEN4_SP_SDMA_RS=0`, `SGLANG_QWEN4_SP_SDMA_AG=0` and
`SGLANG_QWEN4_SP_SDMA_AG_MOE=0` keep the corresponding collective on aiter; `SGLANG_QWEN4_SEQ_PARALLEL=0` restores
the all-reduce.



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

### TP1 startup

Run from `evaluation/`. TP1 enables PLE offload and disables PyHIP QSA by default, without MTP:

```bash
CUDA_VISIBLE_DEVICES=6 HIP_VISIBLE_DEVICES=6 ./launch_qwen38_flash_next_fp8_mi308x_pure_tp1.sh
```

### 1M context

The long-context script defaults to TP2 and `CONTEXT_LEN=1048576`. It sets `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` to allow a context length beyond the checkpoint's configured limit. `ROPE_TYPE=default` keeps the checkpoint RoPE; `ROPE_TYPE=yarn` applies YaRN with factor 4.0 and original context 262144 through a runtime override.

Start the server from `evaluation/`:

```bash
GPU_DEVICES=6,7 ROPE_TYPE=yarn ./launch_qwen38_flash_next_fp8_mi308x_long_server_1m.sh
```

Send one 1,000,000-token input with 350 output tokens:

```bash
python3 -m sglang.benchmark.serving \
  --backend sglang \
  --model /models/Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16 \
  --host 127.0.0.1 --port 7080 \
  --dataset-name random --tokenize-prompt \
  --random-input-len 1000000 \
  --random-output-len 350 \
  --random-range-ratio 1.0 \
  --num-prompts 1 --max-concurrency 1 \
  --warmup-requests 0 --flush-cache \
  --cache-report
```


<details>
<summary>Qwen 3.5 & Qwen 3.8 CI Overview</summary>
### PR test:
- CI runs each selected case in its own GitHub Actions matrix job using `evaluation/Dockerfile.ci` and the existing launch, benchmark, `run_gsm8k.sh`, and `run_multimodal_test.sh` scripts.
- The default image is `rocm/ali-private:ubuntu22.04_rocm7.14.0_cp310_torch2.12.0_sglang_4d63dea_aiter_514c166_20261105_v2`. All Qwen3.5/Qwen3.8 launchers select `--mm-feature-transport cuda_ipc`. CI enables rocJPEG and supplies the ROCm libva driver settings.
- Non-draft fork PRs require the `run-qwen4-ci` label to run GPU cases. Adding the label triggers a new run through the `labeled` event.
- Case jobs run serially on the shared host, each with its own image container, console logs, and result artifact. Docker builds reuse cached layers.
- CI skips Docker registry login and requires the configured base image (`QWEN4_BASE_IMAGE`, or the workflow default) in each GPU runner's Docker image store. A missing image fails the case before the build. The default Docker builder uses the cached image with `--pull=false`, and case containers use `--pull=never`.
- Each new CI run resolves AITER `qwen38-flash-next-dev` and pyhip `dev` to their latest commits. Cases and GPU-only retries use the same resolved SHAs, recorded in `build.json`. Manual and reusable workflows accept `aiter_ref` and `pyhip_ref` overrides; companion-repository callers default to their PR or workflow commit.
- Each GPU case job has a 60-minute timeout, including the image build. Cancellation stops its container, and the container also exits after 60 minutes from creation so its server workers cannot keep holding GPUs if the runner disconnects. Timed-out cases fail the final CI gate.
- The final **Qwen4 CI** job calls `python3 scripts/ci/qwen4_ci.py summarize` to publish one combined table and the number of passing cases (for example, **7/7**).
- The default case order is Qwen3.8 Flash TP2 base, Qwen3.8 Flash TP2 MTP, Qwen3.8 Flash TP2 R-Fork, Qwen3.5 35B TP4 base, Qwen3.5 35B TP1 DFlash, Qwen3.5 397B TP8 base, then Qwen3.5 397B TP8 DFlash. Local runs and the Actions matrix use this order. PR runs skip `Qwen3.8-Flash-TP2-RFork`; it runs in the daily schedule, or when selected with the `case` input. Direct local runs can evaluate multiple cases in one container and share its normal kernel cache:

```bash
python3 scripts/ci/qwen4_ci.py --models-root /raid/models \
  --client-root /path/to/pinned-client --output /tmp/qwen4-results-new
```

For serving cases, the default command runs **performance, GSM8K accuracy, then a multimodal smoke test** against the
same server. The R-Fork case checks startup, transferred weights, and GSM8K instead.
Use it without `--accuracy-only` or `--smoke` for the full workload.
Performance uses 10 warmup requests and 128 random requests at concurrency one;
Qwen3.5 uses 8,000 input / 500 output tokens, and Qwen3.8 uses 12,000 / 350.
Accuracy scores 200 examples and enforces the recorded gate.

Resolve both development branches before building the local image. Passing their
SHAs refreshes changed sources while preserving Docker's cache for unchanged layers:

```bash
mkdir -p /tmp/qwen4-buildx
aiter_sha=$(git ls-remote https://github.com/apinge/aiter.git refs/heads/qwen38-flash-next-dev | awk '{print $1}')
pyhip_sha=$(git ls-remote https://github.com/tingqli/pyhip.git refs/heads/dev | awk '{print $1}')
BUILDX_CONFIG=/tmp/qwen4-buildx docker build --network=host --progress=plain \
  --build-arg "AITER_REF=$aiter_sha" --build-arg "PYHIP_REF=$pyhip_sha" \
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

Remove `--case Qwen3.5-397B-TP8-DFlash` to run all cases sequentially, or pass
comma-separated case IDs from the table below.

The command exits nonzero for launch errors, incomplete requests, invalid results, or an accuracy regression. Logs, native JSON/HTML reports, and `summary.md` are written under the fresh output directory. Build provenance lists the source commits and installed package versions. Keep `QWEN4_WORK_ROOT` outside `/home` on hosts with limited home storage; Docker layers remain in the daemon's data root.
Use a fresh `--output-dir` for each direct evaluator run. Ordinary evaluations retain model-specific report filenames; strict GSM8K runs use `gsm8k.json/html`.

Default runs evaluate all cases with 128 fixed-length performance requests and 200 GSM8K examples each (the R-Fork case runs GSM8K only). The accuracy gate permits at most a 3% relative drop: `score >= recorded_reference * 0.97`; performance thresholds are advisory.


| Case ID                 | Reference | Minimum correct / 200 |
| ----------------------- | --------- | --------------------- |
| Qwen3.8-Flash-TP2-Base  | 98.0%     | 191                   |
| Qwen3.8-Flash-TP2-MTP   | 98.5%     | 192                   |
| Qwen3.8-Flash-TP2-RFork | 98.0%     | 191                   |
| Qwen3.5-35B-TP4-Base    | 97.0%     | 189                   |
| Qwen3.5-35B-TP1-DFlash  | 97.0%     | 189                   |
| Qwen3.5-397B-TP8-Base   | 99.0%     | 193                   |
| Qwen3.5-397B-TP8-DFlash | 98.5%     | 192                   |


Each case runs on the first idle GPUs (under 1 GiB used) among `--gpus`, which
defaults to all eight; use `--case Qwen3.5-35B-TP1-DFlash --gpus 2,3` to restrict
the candidates. Comma-separated case IDs and GPU indices are supported. A selected subset
reports its coverage and does not reuse prior results. `--accuracy-only` runs the
200-example gate without the performance benchmark. `--smoke` uses two requests
and two examples and reports diagnostic results with the accuracy gate disabled.
`--port` selects an unused server port.

`Qwen3.8-Flash-TP2-RFork` checks R-Fork scale-out within one container. CI runs it daily, not on PRs; select it with the `case` input to run it on demand. It uses four idle GPUs:
1. It starts a TP2 seed on the first two, loaded from disk on `--port`.
2. It starts a second TP2 instance on the other two, on `--port + 1`, with `--load-format remote_instance` over RCCL. Transfer group ports start at 62000.
3. The case fails if any of these hold:
   - the new instance does not load via R-Fork;
   - any rank's per-tensor checksums (`SGLANG_DEBUG_WEIGHT_CHECKSUM_DIR`) differ from the seed's;
   - the new instance fails the GSM8K gate.
4. The summary also reports the new instance's startup time and `load_weight`. The seed log is `seed.log`; the new instance's log is `server.log`.

R-Fork does not run the multimodal smoke test; its multimodal summary column is
`not applicable`. Serving cases require a successful multimodal response and
rocJPEG request-batch decoding, including in daily pressure runs.


### Daily Pressure Test
- The daily run starts at **04:00 China time** and runs pressure and multimodal checks for serving cases. The R-Fork case checks loading and weight checksums instead, then the seed and server logs.
- Pressure uses `benchmark_example-20261013-363695ea/concurrency/run.sh` with `LLM_AUTO_SEQ_SIGMA=4`. Qwen3.8 uses `tk_in:12000,tk_out:350,conc:1-2-4-8-12-16-32`; Qwen3.5 uses `tk_in:8000,tk_out:500,conc:1-2-4-8-12-16-32`. Fixed concurrency levels use 8, 8, 8, 16, 32, 32, and 64 main requests respectively, with repost warmups disabled.
- CI checks the complete server log for error, core-dump, segmentation-fault, and GPU memory-access-fault markers. Matches fail the case. The daily summary shows each case's status, multimodal status, and any failure messages. The original `server.log`, pressure console log, and multimodal response are retained for debugging.
- To run one pressure case locally, add `--pressure` to the driver command. `--smoke --pressure` runs two requests at concurrency one. The CI image adds the pinned concurrency-client packages without upgrading the serving runtime;
</details>
