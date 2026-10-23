**Qwen4 MI308X CI**

This harness orchestrates the existing scripts in `evaluation/`. It does not implement another benchmark or GSM8K scorer. The six cases map to the four Qwen3.5 launch scripts and the Qwen3.8 launcher with `MTP=0/1`. Performance runs through the existing `run_qwen*_pure_text_bench.sh` scripts; accuracy runs through `run_gsm8k.sh` and `sglang.test.run_eval`.

The initial policy is `observe`: build, launch, malformed/missing results, failed requests, and incomplete evaluation fail CI. Valid numerical accuracy/performance results are advisory. GSM8K scores 200 fixed examples after excluding five prompt examples. Performance uses 128 prompts, ten warmups, concurrency 1, and 8000/500 tokens for Qwen3.5 or 12000/350 for Qwen3.8.

**GitHub workflow.** `qwen4-ci.yml` is the label/nightly/manual entrypoint. `_qwen4-eval.yml` contains the shared jobs:

1. `prepare` runs on a hosted CPU runner, checks authorization, validates the harness, resolves source SHAs once, and uploads the immutable plan.
2. `gpu` reserves the MI308X host with a filesystem lock, builds one image, and creates one container for each of the three target models. Each container runs its base/speculative pair sequentially with a server restart. Model storage is mounted `/raid/models:/models:ro`.
3. `report` runs even after a GPU-job failure, reconciles all expected cases, and writes tables/details to `GITHUB_STEP_SUMMARY`. Raw logs, JSON, and HTML are uploaded as artifacts.

Apply the maintainer label `run-qwen4-ci`, or use manual dispatch. There is one SGLang-owned nightly at 20:00 UTC. The optional `qwen4-ci-comments.yml` controller accepts `/ci run` and `/ci run_failed` from users with write/maintain/admin permission. It dispatches or retries the same workflow and never runs PR code on the controller. A failed GPU-job retry reruns the six-case suite; a report-only retry reuses the earlier result artifact. The original source plan and base-image digest are restored for retries.

`/ci run` is a one-off request for the current PR head. The maintainer-added label separately enables automatic runs for subsequent pushes. The CPU report job posts the `Qwen4 CI` commit status on the tested PR SHA, because the trusted `pull_request_target` workflow itself runs from the base branch. Numerical results remain advisory in that status until enforcement is enabled.

Install AITER's thin callers after publishing the SGLang harness commit:

```bash
python3 evaluation/qwen4/install_aiter_workflows.py \
  --harness-sha FULL_SGLANG_COMMIT_SHA --aiter-dir /opt/aiter
```

The generated AITER workflows pin both the reusable workflow and controller checkout to that commit, forward AITER PR identity to the shared runner, and add no second nightly. Both repositories need the configured runner/registry access. Publish the SGLang commit before activating the AITER callers; an unpublished local commit cannot be fetched by Actions.

**Runner configuration.** The default label set is `["self-hosted","linux","mi308x"]`. Configure both caller repositories to reach the same host and obey `/var/lock/qwen4-mi308x.lock`. Repository variables are `QWEN4_RUNNER_LABELS` (JSON array), `QWEN4_CI_LABEL`, `QWEN4_BASE_IMAGE`, `QWEN4_MODELS_ROOT`, `QWEN4_GSM8K_PATH`, `QWEN4_SHAREGPT_PATH`, and `QWEN4_DOCKER_USERNAME`. The optional `DOCKER_PASSWORD` secret is used only for the host's temporary registry configuration and is not passed to candidate containers.

The default base image is `rocm/ali-private:ubuntu22.04_rocm7.14.0_cp310_torch2.12.0_sglang_d0a5b96_aiter_b7aed05_20261020`, defined once in `profiles.yaml`. Set repository variable `QWEN4_BASE_IMAGE` to change the workflow default for PR, nightly, and manual runs. Leave the manual `base_image` input blank to use that configured default, or supply a tag/digest for a one-off override. SGLang PRs use the latest AITER `qwen38-flash-next-dev` head, resolved once. AITER PRs must supply their PR head as the AITER candidate; the companion SGLang defaults to `qwen4_v0.5.21`. pyhip and the evaluator are pinned in `dependencies.lock.json`.

**Stage datasets before CI.** The large ShareGPT file must already exist. This command downloads the pinned GSM8K file and verifies both dataset hashes; it does not overwrite a mismatched dataset:

```bash
python3 evaluation/qwen4/prepare_assets.py --root /raid/models/benchmark_datasets
```

Model preflight verifies configs, shard presence/sizes, metadata hashes, and available Hugging Face revision metadata. `assets.model_manifest(..., hash_weights=True)` can produce full weight hashes during provisioning. Normal runs do not reread hundreds of GB solely to hash weights; the recorded identity strength is explicit in `assets.json`.

The image build checks installed module resolution without importing GPU-dependent AITER modules. Client isolation uses a virtual environment without bundled pip/setuptools, retaining the constrained base-image packages. Build provenance and installed package versions are saved as artifacts. Failed-job retries reuse the recorded Python dependency constraints as well as the original base-image digest.

**Local tests in an existing target-image container.** This mode validates actual model execution without claiming a fresh Docker build or three distinct container lifetimes. Create a separate evaluator checkout at the pinned client revision so candidate server changes do not change the client:

```bash
client_sha=$(python3 -c 'import json; print(json.load(open("evaluation/qwen4/dependencies.lock.json"))["client_sha"])')
git worktree add --detach /opt/qwen4-eval-client "$client_sha"
python3 evaluation/qwen4/run.py plan --sources local --profile full \
  --run-id local-full --output /tmp/qwen4-plan.json
python3 evaluation/qwen4/run.py execute --backend existing \
  --plan /tmp/qwen4-plan.json --output /tmp/qwen4-local-results \
  --models-root /models --client-root /opt/qwen4-eval-client \
  --gsm8k /models/benchmark_datasets/gsm8k/test.jsonl \
  --sharegpt /models/benchmark_datasets/ShareGPT_V3_unfiltered_cleaned_split.json
```

Use a new output directory for every run. A short diagnostic can use `--profile smoke --cases '["q35-35b-tp4-base"]'` during planning. `full` and `calibration` require all six cases. `profile` requires one selected case and keeps profiler timings outside numerical regression gates. `--examples` is an explicit protocol override with a different comparison identity.

**Docker execution on the bare-metal runner.** Resolve sources through the GitHub API, then execute the saved plan. `gh` needs read access to source repositories; Docker needs access to the configured registry image.

```bash
python3 evaluation/qwen4/run.py plan --origin integration --profile full \
  --run-id manual-integration --output /tmp/qwen4-plan.json
python3 evaluation/qwen4/run.py execute --plan /tmp/qwen4-plan.json \
  --output /tmp/qwen4-docker-results --models-root /raid/models \
  --gsm8k /raid/models/benchmark_datasets/gsm8k/test.jsonl \
  --sharegpt /raid/models/benchmark_datasets/ShareGPT_V3_unfiltered_cleaned_split.json
```

Only containers/images labeled or named for this run are removed. No host-wide process kill or Docker prune is performed. A busy GPU, occupied endpoint, or failed cleanup stops the affected session and remains visible in the report.

**Validation and reports.** Run CPU contracts with `python3 -m pytest evaluation/qwen4/tests -q`. Native evaluator HTTP tests run when SGLang is installed and otherwise report a skip. Recipe tests execute snapshots of the original shell scripts with fake Python/server commands; they never load a model. Docker lifecycle tests simulate the Docker command boundary and are not evidence of a real image build.

```bash
python3 evaluation/qwen4/report.py --plan /tmp/qwen4-plan.json \
  --output /tmp/qwen4-local-results
```

`summary.json` is the machine-readable aggregate and `summary.md` is the Actions presentation. A baseline document has a `cases` mapping keyed by case ID; each entry contains `comparison_key`, `metrics`, `accuracy_floor`, `accuracy_drop`, and `performance_tolerance`. An explicit `--gate-mode enforce` plan plus `execute --baseline PATH` enables numerical gates. Missing/incompatible baselines remain advisory in `observe` and fail in `enforce`. No production baseline or threshold is shipped with this implementation.
