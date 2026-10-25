# Qwen3.5 & Qwen3.8 PTPC Model Evaluation Guide


**Unified CI and local evaluation.** The two CI entrypoints are [the workflow](../.github/workflows/qwen4-ci.yml) and [the driver](../scripts/ci/qwen4_ci.py). The driver calls the existing launch, benchmark, and GSM8K scripts. It builds one image and creates three sequential target containers; each container runs its two cases with a server restart. The fixed matrix is 35B TP4 base / TP1 DFlash, 397B TP8 base / DFlash, and Qwen3.8 TP2 MTP off / on. The case-to-script table and dependency/dataset pins are together at the top of the driver.

The default GSM8K test scores **200 examples**, with five separate few-shot examples excluded from scoring. It writes JSON, HTML, and per-example JSONL and fails on incomplete requests. Completed token-limit answers remain in the score denominator. Set `GSM8K_DATA_PATH` to the pinned local test file and choose a fresh `GSM8K_OUTPUT_DIR` for each evaluation. Performance uses 128 requests, 10 warmups, and concurrency 1. An optional `smoke` manual run uses two performance requests, one warmup, and two scored GSM8K examples per case; its report is explicitly diagnostic.

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

**Self-hosted runner setup.** GPU work defaults to `runs-on: self-hosted`, matching the previous Qwen3.5 runner. The driver requires eight idle MI308X/gfx942 GPUs and reserves `/var/lock/qwen4-mi308x.lock`. Both repositories' runner registrations must share that lock and a user/group able to open it. The runner needs Python 3.10+, Git, and access to the Docker daemon. Model storage is mounted `/raid/models:/models:ro`; each target container has its own results mount and private network/IPC namespaces. Only resources belonging to this run are removed.

Set repository secret `DOCKER_PASSWORD` for the supplied private base image; optional `QWEN4_DOCKER_USERNAME` defaults to `alishared`. Repository variables `QWEN4_BASE_IMAGE`, `QWEN4_MODELS_ROOT`, `QWEN4_RUNNER_LABELS` (a JSON array), and `QWEN4_CI_LABEL` override the image, model root, runner labels, and label name. The manual `base_image` input overrides the image for one run. The driver default is the ROCm image listed below. Its resolved digest and the built image ID are recorded in the result artifact. Registry credentials are used by the host and are not mounted into evaluation containers.

Stage `ShareGPT_V3_unfiltered_cleaned_split.json` under `/raid/models/benchmark_datasets` and the pinned GSM8K file under its `gsm8k` subdirectory. The driver checks both dataset checksums before building. If GSM8K is not already staged, provision it on the host:

```bash
mkdir -p /raid/models/benchmark_datasets/gsm8k
curl --fail --location \
  https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl \
  --output /raid/models/benchmark_datasets/gsm8k/test.jsonl
```

**Triggers and first installation.** A `pull_request` event runs only hosted CPU validation, including while introducing this workflow in a PR. GPU authorization uses `pull_request_target`, which requires the workflow and driver to be installed on `qwen4_v0.5.21` first. A ready PR with a maintainer-applied `run-qwen4-ci` label runs the full suite, as do subsequent pushes while the label remains. Manual dispatch can explicitly select a draft PR through `pr_number` for a pilot. There is one SGLang-owned nightly at 20:00 UTC. The nightly and dispatch entrypoints require installation on the default branch; starting the runner alone does not register a PR-only workflow.

SGLang PRs test their exact head with the latest AITER `qwen38-flash-next-dev` head, resolved once by the preparation job. AITER's thin caller invokes this same workflow at a pinned SGLang commit, with the AITER PR as candidate and SGLang's target head as companion. Neither a model pair nor a later target re-resolves sources. pyhip and the evaluator source remain pinned. A prepared source record is passed through native Actions job outputs, without a separate plan/retry framework.

The initial gates fail on build/startup errors, failed requests, incomplete evaluation, malformed/missing results, and cleanup errors. Numerical accuracy/performance thresholds are advisory. The always-run hosted report produces a six-row table on the Actions Summary page, with logs, native JSON/HTML, source SHAs, image and container identities as artifacts. Comment commands, baseline enforcement, calibration, and CI profiling are deferred. Use manual dispatch or Actions reruns. Results are stored per GPU attempt; report-only retries use the artifact name preserved in the GPU job's native output. Missing artifacts or a failed GPU job still fail the report. To reproduce an image across attempts, select its digest; a rebuilt attempt can otherwise resolve an updated tag or previously unpinned missing dependency.

**Local host commands.** From the SGLang checkout, with `gh` configured for repository reads and Docker authenticated:

```bash
python3 scripts/ci/qwen4_ci.py prepare --output /tmp/qwen4-plan
python3 scripts/ci/qwen4_ci.py run --plan /tmp/qwen4-plan/plan.json \
  --models-root /raid/models --output /tmp/qwen4-run
```

Local `prepare` defaults to both repositories' target branch heads. Prefix it with `INPUT_PR_NUMBER=65` to select that SGLang PR, or set `INPUT_SGLANG_REF`/`INPUT_AITER_REF` for explicit revisions. Local commands do not publish GitHub commit statuses. Use a fresh results directory for each attempt. Add `--smoke` to `prepare` for a diagnostic run. For execution-only validation inside an existing target-image container, the `container` command runs a selected target pair with `--group`, `--models-root`, `--datasets-root`, and `--client-root` pointing to the pinned evaluator checkout. That mode does not validate image construction or three distinct containers. CPU tests are `python3 -m pytest scripts/ci/test_qwen4_ci.py -q`; the three native evaluator integration cases are skipped if the SGLang runtime is not installed.

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
