# Fast startup on MI308X (ModelExpress-style): Qwen3.8-Flash-Next-PTPC-FP8, TP2 and TP4

NV ModelExpress shortens cold start / scale-out by (1) pulling processed GPU weights from an online replica,
(2) accelerated storage loading, (3) shipping compiled-kernel caches. This directory validates the AMD/SGLang
counterparts on MI308X and holds the scripts used. All numbers below come from one full rerun on 2026-10-26
(12:50-17:34 UTC) at commit `378743602`; TP2 and TP4 use the same scripts and table layout.

- 8x MI308X (gfx942, XGMI), single node, no RDMA NIC; GPUs shared with other containers
- ROCm 7.14, torch 2.12.0+rocm7.14; mooncake-transfer-engine-rocm 0.3.13.post1, fastsafetensors 0.5.0,
  runai-model-streamer 0.17.0
- Model `/models/Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16`: 221 GB, 149,318 tensors, largest shard 102 GB
- Server args = `scripts/launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh` defaults
  (`--mem-fraction-static` 0.95 at TP2, 0.85 at TP4)

## Summary

| Capability | MI308X counterpart | Result |
|---|---|---|
| Compiled-kernel cache transfer | keep / ship `~/.flydsl`, `~/.cache/sglang`, `~/.cache/comgr` | **Largest gain**: TP2 266.4 -> 126.9 s, TP4 278.2 -> 125.9 s |
| P2P weight pull from a live replica | R-Fork `remote_instance`, `nccl` (RCCL over XGMI) | **Works with this branch's fixes**: TP2 126.9 -> 68.1 s, TP4 125.9 -> 79.1 s; checksums identical |
| Same, other transport | R-Fork `transfer_engine` (Mooncake, `MOONCAKE_PROTOCOL=hip`) | **Works with the workaround**, 1-2 s faster than RCCL (66.3 / 77.6 s); silently wrong weights without it |
| Accelerated storage loading | fastsafetensors / runai_streamer / prefetch | **Small, situational**: only stable with a cold page cache (TP2 runai -18.8 s, TP4 fastsafetensors -11.8 s); prefetch no gain; fastsafetensors OOMs at TP2 |
| Same-GPU resident weights | `--weight-cache-mode` | Not supported for compressed-tensors (earlier result) |

## Method

- Time = launch to `fired up`; phases from log timestamps (1 s resolution). "Weight load" = load_weight minus
  PyHIP GR-read warmup. 2 runs per cell (mean shown); differences under about 10 s are not conclusive.
- Kernel cache warm = existing `$HOME`, cold = empty `HOME` per run. Page cache warm = model files read in and
  checked >=99% resident; cold = `posix_fadvise(DONTNEED)` on the model files only, checked <=1% (measured: warm
  >=99.98%, cold 0%).
- Each launch is refused if any of its GPUs has >2 GiB in use; KV capacity was identical across runs of each
  config (e.g. TP4 disk 5,939,200 tokens).
- R-Fork: the seed is a normal disk-loaded instance already serving; its startup is not counted.

## Results

### Launch to ready (s, kernel cache warm; delta vs disk at the same TP and page cache)

| Config | TP2 warm page cache | TP2 cold page cache | TP4 warm page cache | TP4 cold page cache |
|---|---|---|---|---|
| disk (default) | 126.9 | 142.2 | 125.9 | 137.9 |
| + prefetch | 133.7 (+6.8) | 146.2 (+4.0) | 124.9 (-1.0) | 145.4 (+7.5) |
| fastsafetensors | OOM (0/2) | OOM (0/2) | 126.7 (+0.8) | 126.2 (-11.8) |
| runai streamer | 117.4 (-9.5) | 123.4 (-18.8) | 139.2 (+13.3) | 132.7 (-5.3) |
| R-Fork RCCL | 68.1 (-58.8) | not run¹ | 79.1 (-46.8) | not run¹ |
| R-Fork Mooncake | 66.3 (-60.6) | not run¹ | 77.6 (-48.3) | not run¹ |

¹ An R-Fork client copies weights from the live seed over XGMI and does not read the safetensors files (only config/tokenizer), so the page-cache state barely matters; these cells were not measured.

### Weight load (s, excl. PyHIP warmup; same layout)

| Config | TP2 warm page cache | TP2 cold page cache | TP4 warm page cache | TP4 cold page cache |
|---|---|---|---|---|
| disk (default) | 68.7 | 83.7 | 55.1 | 67.3 |
| + prefetch | 74.8 (+6.2) | 86.2 (+2.5) | 55.7 (+0.6) | 74.2 (+6.9) |
| fastsafetensors | OOM (0/2) | OOM (0/2) | 56.0 (+0.8) | 56.0 (-11.2) |
| runai streamer | 57.9 (-10.8) | 63.9 (-19.9) | 67.7 (+12.6) | 62.2 (-5.1) |
| R-Fork RCCL | 5.3 (-63.4) | - | 4.8 (-50.3) | - |
| R-Fork Mooncake | 4.0 (-64.7) | - | 3.0 (-52.2) | - |

### Phases (s, kernel cache and page cache warm)

| TP | Config | Launch to ready | Start + imports | Init (incl. dist) | Weight load | PyHIP warmup | Graph capture | Rest | Per run |
|---|---|---|---|---|---|---|---|---|---|
| TP2 | disk | 126.9 | 12.7 | 23.5 | 68.7 | 4.0 | 10.1 | 7.2 | 127.2 / 126.7 |
| TP2 | + prefetch | 133.7 | 13.2 | 23.0 | 74.8 | 3.7 | 10.4 | 7.5 | 140.2 / 127.2 |
| TP2 | fastsafetensors | OOM | - | - | - | - | - | - | - |
| TP2 | runai | 117.4 | 13.8 | 23.0 | 57.9 | 3.7 | 10.6 | 7.8 | 118.7 / 116.2 |
| TP2 | R-Fork RCCL | 68.1 | 14.0 | 25.5 | 5.3 | 3.6 | 10.9 | 8.3 | 66.6 / 69.6 |
| TP2 | R-Fork Mooncake | 66.3 | 13.6 | 25.5 | 4.0 | 3.6 | 10.7 | 8.2 | 66.6 / 66.1 |
| TP4 | disk | 125.9 | 13.7 | 34.0 | 55.1 | 4.1 | 10.6 | 7.6 | 123.2 / 128.7 |
| TP4 | + prefetch | 124.9 | 13.3 | 33.0 | 55.7 | 3.6 | 10.6 | 8.2 | 124.2 / 125.7 |
| TP4 | fastsafetensors | 126.7 | 13.4 | 33.5 | 56.0 | 4.0 | 10.9 | 8.1 | 127.2 / 126.2 |
| TP4 | runai | 139.2 | 13.3 | 34.0 | 67.7 | 3.6 | 11.1 | 8.5 | 148.7 / 129.7 |
| TP4 | R-Fork RCCL | 79.1 | 14.3 | 36.0 | 4.8 | 3.6 | 11.0 | 8.5 | 79.1 / 79.1 |
| TP4 | R-Fork Mooncake | 77.6 | 13.7 | 36.0 | 3.0 | 3.6 | 11.1 | 9.3 | 79.6 / 75.6 |

### Kernel cache (disk, page cache warm, s)

| TP | Kernel cache | Launch to ready | Start + imports | Init | Weight load | PyHIP warmup | Graph capture | Rest | Per run |
|---|---|---|---|---|---|---|---|---|---|
| TP2 | cold (empty HOME) | 266.4 | 14.2 | 23.5 | 66.7 | 67.8 | 34.8 | 58.7 | 266.9 / 265.9 |
| TP2 | warm | 126.9 | 12.7 | 23.5 | 68.7 | 4.0 | 10.1 | 7.2 | 127.2 / 126.7 |
| TP4 | cold (empty HOME) | 278.2 | 13.9 | 35.0 | 56.6 | 68.8 | 38.4 | 64.6 | 283.0 / 273.4 |
| TP4 | warm | 125.9 | 13.7 | 34.0 | 55.1 | 4.1 | 10.6 | 7.6 | 123.2 / 128.7 |

### Correctness

| TP | Load path | Per-tensor checksums vs seed | GSM8K (1314 questions, 1 run) |
|---|---|---|---|
| TP2 | disk (seed) | reference | 0.9817 |
| TP2 | R-Fork RCCL | 2 ranks x 1503, 0 mismatches | 0.9779 |
| TP2 | R-Fork Mooncake | 2 ranks x 1503, 0 mismatches | 0.9787 |
| TP4 | disk (seed) | reference | 0.9810 |
| TP4 | R-Fork RCCL | 4 ranks x 1503, 0 mismatches | 0.9772 |
| TP4 | R-Fork Mooncake | 4 ranks x 1503, 0 mismatches | 0.9802 |

Checksums cover 1449 parameters + 54 buffers per rank (name, shape, dtype, byte sum, position-weighted byte sum).
One GSM8K run each cannot separate 0.977-0.982.

## Findings

- **R-Fork** replaces the per-tensor walk over 149,318 checkpoint tensors with a copy of 1449 already-processed
  parameters (116.6 GB per rank at TP2). Weight load drops to 3-5 s at both TPs. It saves more at TP2 (46-48%)
  than at TP4 (37-38%), because disk loading gets slower at lower TP while XGMI transfer does not.
- **Kernel cache** saves 139-152 s: PyHIP warmup about 68 -> 4 s, graph capture 35-38 -> 10-11 s, and the
  first-batch lazy JIT ("rest") 59-65 -> 7-8 s. An earlier ablation showed `~/.flydsl` is the part that matters.
- **Storage loaders.** A cold page cache costs the default loader +15.3 s (TP2) / +12.0 s (TP4), which caps what
  they can recover.
  - fastsafetensors: TP4 cold -11.8 s, both runs 126.2 s vs disk 138.7 / 137.2 s. It uses O_DIRECT, so cold and
    warm load the same. No gain warm. At TP2 it cannot start: it copies a whole file to one GPU, and the 95.37 GiB
    first shard does not fit next to 109.79 GiB of weights. That is 4/4 OOM with no other process on the GPU.
  - runai (libaio_direct): every TP2 run is faster than every disk run (-9.5 s warm, -18.8 s cold). At TP4 cold,
    both runs are faster (-5.3 s). At TP4 warm it is slower (148.7 / 129.7 s vs 123.2 / 128.7 s).
  - prefetch: no gain in any cell.
  - Why the ceiling is low (earlier TP4 analysis): the pure disk read is 4-10 s per rank (fastsafetensors reads
    13-21 GB/s vs 0.4-1.4 GB/s for mmap). The rest is the single-threaded `load_weights` loop over 147,456
    per-expert tensors. That loop does a linear name lookup in the 1536-entry `expert_params_mapping` (about 14 s),
    TP slicing with small H2D copies (about 9 s), and per-call weight_loader overhead (about 30 s, estimated by
    subtraction).
  - Configurations to avoid (earlier TP4 runs): fastsafetensors with GDS on (113 s cold);
    `--weight-loader-disable-mmap` (363 s).
- **Remaining floor** with R-Fork Mooncake: start + init + PyHIP warmup + graph capture + rest is 61.6 s (TP2) /
  73.7 s (TP4). None of these optimizations touch those phases.

## Fixes in this branch

| # | Symptom without the fix | Cause | Fix |
|---|---|---|---|
| 1 | R-Fork client crashes (`PyHIP GR read weights not prepared`), FP8 dtype mismatch | remote_instance loader skipped `process_weights_after_loading` | client runs it before receiving, matching the seed's processed layout |
| 2 | 48 `gemma_weight` buffers differ | derived in weight_loader hooks, skipped by a raw copy | `_rebind_parameter_aliases` after receiving |
| 3 | FP8 KV cache with non-unit scales: `k/v_scale_float` stay 1.0 after receiving | floats were computed from the -1 placeholders before the copy | refresh them from the received tensors (`bc044ab43`); CPU check 1.0/1.0 -> 4.0/6.0 = seed |
| 4 | RCCL deadlock | seed grows its RoPE cache at runtime, so the tensor lists differ | send parameters only; both sides compare a manifest digest first and fail cleanly |
| 5 | seed crashes with `KeyError` | send group failed to init (`EADDRINUSE`) | return an error; client checks `build_group` |
| 6 | `EADDRINUSE` | group ports inside `ip_local_port_range` (32768-60999) | client script uses 62000+ |
| 7 | Mooncake: 913 of 1503 tensors wrong, no error | HIP transport resolves remote addresses from the IPC allocation base | on ROCm, register whole allocator segments; repro `mooncake_hip_offset_repro.py` |
| 8 | bench_startup mislabels runs | "warm" never warmed; stale log read as ready; fastsafetensors ignores `--base-gpu-id` | prepare and verify page cache, delete old logs, use `HIP_VISIBLE_DEVICES` (`378743602`) |

Not fixed: if a client dies after triggering the seed's send-group init but before joining, the seed blocks in the
rendezvous for up to 30 min (reproduced at TP2 earlier).

## How to run

Environment for every command:
```bash
export PYTHONPATH=<repo>/python   # this branch
export SGLANG_USE_AITER=1 SGLANG_GR_READ_FLYDSL=1 SGLANG_USE_PYHIP_QSA=1 AITER_MOE_PADDING_SIZE=64
```

Baseline, TP4 on GPU0-3. For TP2 use `--tp-size 2 --mem-fraction-static 0.95` and two GPUs.
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

Each optimization = the baseline command (same args, own port/GPUs) plus:

| Optimization | Extra env / args | Notes |
|---|---|---|
| R-Fork RCCL (client) | `--load-format remote_instance --remote-instance-weight-loader-backend nccl --remote-instance-weight-loader-seed-instance-ip 127.0.0.1 --remote-instance-weight-loader-seed-instance-service-port 7090 --remote-instance-weight-loader-send-weights-group-ports '[62000,62001,62002,62003]'` | seed = plain baseline; group ports are a JSON list of TP-size ports outside `ip_local_port_range` |
| R-Fork Mooncake | env `MOONCAKE_PROTOCOL=hip`; seed: `--base-gpu-id 0 --remote-instance-weight-loader-start-seed-via-transfer-engine`; client: `--base-gpu-id 4 --load-format remote_instance --remote-instance-weight-loader-backend transfer_engine --remote-instance-weight-loader-seed-instance-ip 127.0.0.1 --remote-instance-weight-loader-seed-instance-service-port 7090` | no `HIP_VISIBLE_DEVICES` (HIP IPC needs the peer GPU visible); needs `apt install libibverbs1 librdmacm1` |
| fastsafetensors | `--load-format fastsafetensors --model-loader-extra-config '{"enable_gds": false}'` | GDS must be off on ROCm; OOMs at TP2 with this checkpoint |
| runai streamer | env `RUNAI_STREAMER_FS_STRATEGY=libaio_direct`; `--load-format runai_streamer` | default io_uring fails on kernel 5.10; about +34 GB load-time peak per GPU, freed before KV sizing |
| kernel cache | `tar -C $HOME -cf cache.tar .flydsl .cache/sglang .cache/comgr`, untar into the new `$HOME` | no arg change |

### Scripts in this directory

```bash
# Startup matrix (configs x kernel cache x page cache); TP = number of --gpus.
# R-Fork configs start a seed on --seed-gpus first. Writes results.csv, summary.md, logs/.
python3 benchmark/rfork_mi308x/bench_startup.py \
  --configs disk,disk_prefetch,fastsafetensors,runai,rfork_nccl,rfork_mooncake \
  --kernel-cache warm --page-cache warm,cold --repeat 2 --prime \
  --gpus 4,5,6,7 --seed-gpus 0,1,2,3 --out /tmp/startup_tp4
# TP2: --gpus 6,7 --seed-gpus 4,5.  Cold kernel cache: --configs disk --kernel-cache cold --page-cache warm
# --dry-run lists the runs; --reparse --out <dir> rebuilds summary.md from existing logs.

# R-Fork correctness: disk seed, then RCCL and Mooncake clients; per-rank checksum diff + full GSM8K each.
python3 benchmark/rfork_mi308x/verify_rfork.py --tp 4 --seed-gpus 0,1,2,3 --gpus 4,5,6,7 --out /tmp/verify_tp4

# Page-cache residency / eviction for one model only.
python3 benchmark/rfork_mi308x/evict_model_cache.py /models/Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16 --evict
```

- `bench_startup.py` guards:
  - it prepares and verifies the page-cache state before every run;
  - it refuses busy GPUs (>2 GiB used) and deletes stale logs;
  - it places non-R-Fork runs with `HIP_VISIBLE_DEVICES`;
  - for R-Fork, seed and client GPU ranges must be contiguous, the same size and disjoint.
- Debug switches:
  - `SGLANG_DEBUG_WEIGHT_CHECKSUM_DIR=<dir>`: per-rank tensor checksums;
  - `SGLANG_DEBUG_WEIGHT_ITER_TIMING=1`: read vs `load_weights` time split.
- Other tools:
  - `mooncake_hip_offset_repro.py`: standalone repro of the Mooncake offset bug;
  - `fst_raw_throughput.py`: raw file-to-GPU read speed;
  - `first_requests_latency.py`: lazy-JIT latency of the first requests;
  - `time_startup.sh`: single launch plus timing;
  - `greedy_compare.py`: compares greedy outputs of two servers;
  - `weight_cache_shim/`: starts the weight-cache daemon with the launch script's args.

## Limitations

- Single node, TP2/TP4 only. Not tested: cross-node, cross-container, TP8, MTP draft models, ModelExpress itself.
- 2 timing runs and 1 GSM8K run per configuration, on GPUs shared with other containers.
- The seed-stall issue above is open. The impact of a transfer on the seed's live traffic was not measured.
- The ROCm Mooncake workaround also exposes non-weight memory in the registered allocator segments.
