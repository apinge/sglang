# Fast startup on MI308X: R-Fork, kernel cache and storage loaders

How to start a Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16 instance faster on MI308X, and how much each option saves.
Measured on a single node with 8x MI308X (XGMI, no RDMA NIC), ROCm 7.14, at TP2 and TP4.

| Option | What it does | Effect (launch to ready) |
|---|---|---|
| Kernel cache | reuse `~/.flydsl`, `~/.cache/sglang`, `~/.cache/comgr` | 266-278 s -> 126 s |
| R-Fork (`remote_instance`) | copy processed weights from a running instance over XGMI (RCCL or Mooncake) | 126 s -> 66-67 s (TP2), 78-79 s (TP4)² |
| fastsafetensors / runai streamer | faster checkpoint reads | up to -19 s, only with a cold page cache; see below |

## Results

² TP4 R-Fork numbers were measured on 2026-10-26 at `378743602`, before this branch was trimmed. The trim removed
an RCCL pre-transfer manifest check and debug timers. The TP4 rerun on the final code and the TP4 cold-page-cache
R-Fork cells are still pending, because GPUs 0-3 are busy with other jobs. TP2 R-Fork numbers are from the final
code.

Mean of 2 runs, in seconds. The kernel cache is warm unless noted. The page cache is "warm" when the model files
are fully in host memory and "cold" when they were dropped first. Parentheses give the delta vs disk at the same TP
and page cache.

### Launch to ready

| Config | TP2 warm page cache | TP2 cold page cache | TP4 warm page cache | TP4 cold page cache |
|---|---|---|---|---|
| disk (default) | 126.9 | 142.2 | 125.9 | 137.9 |
| + prefetch | 133.7 (+6.8) | 146.2 (+4.0) | 124.9 (-1.0) | 145.4 (+7.5) |
| fastsafetensors | OOM | OOM | 126.7 (+0.8) | 126.2 (-11.8) |
| runai streamer | 117.4 (-9.5) | 123.4 (-18.8) | 139.2 (+13.3) | 132.7 (-5.3) |
| R-Fork RCCL | 66.8 (-60.1) | 68.1 (-74.1) | 79.1 (-46.8)² | rerun pending |
| R-Fork Mooncake | 67.1 (-59.8) | 66.6 (-75.6) | 77.6 (-48.3)² | rerun pending |

- R-Fork does not read the weight files, so the page-cache state does not matter for it. The R-Fork rows come
  from a later batch than the other rows (same code paths, next day).
- fastsafetensors cannot start at TP2: it copies each file whole onto one GPU, and this checkpoint's 95 GiB first
  shard does not fit next to the 110 GiB of weights per GPU.
- Differences under about 10 s are within run-to-run variation; for example, the two TP4 runai warm runs took
  148.7 s and 129.7 s.

### Kernel cache (disk loading, page cache warm)

| TP | Empty cache | Warm cache | Where the time goes with an empty cache |
|---|---|---|---|
| TP2 | 266.4 | 126.9 | PyHIP warmup +64 s, first-batch JIT +52 s, graph capture +25 s |
| TP4 | 278.2 | 125.9 | PyHIP warmup +65 s, first-batch JIT +57 s, graph capture +28 s |

### Correctness of R-Fork

Each rank's per-tensor checksums (1449 parameters + 54 buffers) were compared against the seed, and full GSM8K
(1314 questions) was run once per instance.

| TP | Disk-loaded seed GSM8K | R-Fork RCCL: checksums / GSM8K | R-Fork Mooncake: checksums / GSM8K |
|---|---|---|---|
| TP2 | 0.9779 | 2 x 1503 match / 0.9749 | 2 x 1503 match / 0.9810 |
| TP4² | 0.9810 | 4 x 1503 match / 0.9772 | 4 x 1503 match / 0.9802 |

## How to use

```bash
export PYTHONPATH=<this repo>/python
export SGLANG_USE_AITER=1 SGLANG_GR_READ_FLYDSL=1 SGLANG_USE_PYHIP_QSA=1 AITER_MOE_PADDING_SIZE=64
```

Baseline, TP4 on GPU0-3. For TP2, use `--tp-size 2 --mem-fraction-static 0.95` and two GPUs.
```bash
HIP_VISIBLE_DEVICES=0,1,2,3 sglang serve \
  --model-path /models/Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16 \
  --served-model-name Qwen/Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16 \
  --host 0.0.0.0 --port 7090 --tp-size 4 \
  --attention-backend aiter --moe-runner-backend aiter --kv-cache-dtype auto \
  --chunked-prefill-size 16384 --watchdog-timeout 1200 --disable-radix-cache \
  --mem-fraction-static 0.85 --max-running-requests 32 --cuda-graph-max-bs-decode 32 \
  --no-ple-offload-embedding
```

Each option is the baseline command (own port/GPUs) plus the following:

| Option | Add | Notes |
|---|---|---|
| R-Fork RCCL | new instance: `--load-format remote_instance --remote-instance-weight-loader-backend nccl --remote-instance-weight-loader-seed-instance-ip <seed ip> --remote-instance-weight-loader-seed-instance-service-port 7090 --remote-instance-weight-loader-send-weights-group-ports '[62000,62001,62002,62003]'` | The seed is a plain baseline instance with the same TP and model. Group ports are a JSON list with one port per rank, outside `/proc/sys/net/ipv4/ip_local_port_range` |
| R-Fork Mooncake | env `MOONCAKE_PROTOCOL=hip`. Seed: `--base-gpu-id 0 --remote-instance-weight-loader-start-seed-via-transfer-engine`. New instance: `--base-gpu-id 4 --load-format remote_instance --remote-instance-weight-loader-backend transfer_engine --remote-instance-weight-loader-seed-instance-ip <seed ip> --remote-instance-weight-loader-seed-instance-service-port 7090` | `pip install mooncake-transfer-engine-rocm` and `apt install libibverbs1 librdmacm1`. Do not set `HIP_VISIBLE_DEVICES`: HIP IPC needs the peer GPU visible |
| fastsafetensors | `--load-format fastsafetensors --model-loader-extra-config '{"enable_gds": false}'` | `pip install fastsafetensors`; GDS must be off on ROCm; TP4 only for this checkpoint |
| runai streamer | env `RUNAI_STREAMER_FS_STRATEGY=libaio_direct`; `--load-format runai_streamer` | `pip install runai-model-streamer`; the default io_uring fails on kernel 5.10 |
| Kernel cache | `tar -C $HOME -cf cache.tar .flydsl .cache/sglang .cache/comgr` on a warmed node; untar into `$HOME` on the new one | no argument change; `~/.flydsl` is the part that matters |

## Test scripts

```bash
# Startup time per config and phase. TP = number of --gpus; R-Fork configs start a seed on --seed-gpus first.
# Writes results.csv, summary.md and logs/ to --out.
python3 benchmark/rfork_mi308x/bench_startup.py \
  --configs disk,disk_prefetch,fastsafetensors,runai,rfork_nccl,rfork_mooncake \
  --kernel-cache warm --page-cache warm,cold --repeat 2 --prime \
  --gpus 4,5,6,7 --seed-gpus 0,1,2,3 --out /tmp/startup_tp4
#   TP2: --gpus 6,7 --seed-gpus 4,5
#   empty kernel cache: --configs disk --kernel-cache cold --page-cache warm
#   --dry-run lists the runs; --reparse --out <dir> rebuilds summary.md from existing logs

# R-Fork correctness: disk seed, then an RCCL and a Mooncake instance; per-rank checksum diff and full GSM8K for each.
python3 benchmark/rfork_mi308x/verify_rfork.py --tp 4 --seed-gpus 0,1,2,3 --gpus 4,5,6,7 --out /tmp/verify_tp4

# Show or drop the page cache of one model's files only.
python3 benchmark/rfork_mi308x/evict_model_cache.py /models/Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16 [--evict]
```

- `bench_startup.py` aborts a run if the requested page-cache state is not reached (warm >=99% resident, cold
  <=1%). It refuses GPUs that hold more than 2 GiB of other jobs' memory.
- `mooncake_hip_offset_repro.py` is a standalone check of the Mooncake HIP transport. Registered ranges that start
  inside an allocation must read correctly with `--layout offset --raw-register` before the whole-segment
  registration used on ROCm can be dropped.

## Manual check

1. Start a seed with the baseline command, and wait for `The server is fired up`.
2. Start a second instance on other GPUs with the R-Fork options above. Set `SGLANG_DEBUG_WEIGHT_CHECKSUM_DIR` to a
   different directory for the seed and for the new instance. The new instance should log `Load weight end` with a
   load_weight of a few seconds.
3. Compare the checksums for each rank. No output means the weights are identical:
   `diff <(cut -f1-4,6 /tmp/ck_seed/target_rank0.tsv) <(cut -f1-4,6 /tmp/ck_new/target_rank0.tsv)`
4. Send a request to both instances, then run
   `python -m sglang.test.run_eval --eval-name gsm8k --num-examples 1319 --max-tokens 16384 --port <port>`.
   Expect about 0.975-0.982 here.

Limitations: tested on a single node at TP2 and TP4 with this model. The seed must be running and have the same TP
size and model. Not tested: cross-node, TP8, MTP draft models.
