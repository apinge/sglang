# ModelExpress-style fast startup on MI308X: Qwen3.8-Flash-Next-PTPC-FP8 validation report

NV ModelExpress shortens inference-instance cold start / scale-out with: (1) pulling processed GPU weights from an
online replica, (2) accelerated storage loading, (3) shipping compiled-kernel caches. This report validates the
AMD/SGLang counterparts of each on MI308X, plus SGLang's same-GPU weight cache. All numbers come from logs of real
runs on 2026-10-23; failures are reported as observed.

## Environment
- 8x AMD Instinct MI308X (gfx942), single node, XGMI full mesh; no RDMA NIC in the container; kernel 5.10
- ROCm 7.14, torch 2.12.0+rocm7.14; code = this branch (based on `c93b9c195`)
- Model: `/models/Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16` (220 GiB, 149,318 checkpoint tensors of which 147,458 are
  per-expert MoE weights; about 61 GB of weights per rank at TP4); storage is an ext4 RAID (`/dev/md127`)
- All runs TP4 unless noted, launched through `scripts/launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh`
- The GPUs are shared with other containers (`rocm-smi --showpids` showed foreign-PID-namespace processes, e.g. 28 GB on
  GPU7 at times). Throughput and latency numbers are a rough reference only.

## Summary
| ModelExpress capability | MI308X counterpart | Result |
|---|---|---|
| P2P weight pull from an online replica | R-Fork `remote_instance`, backend `nccl` (RCCL over XGMI) | **Works** after 5 fixes. Weights byte-identical; load_weight about 60 s to 8 s |
| Same, alternative transport | R-Fork backend `transfer_engine` (Mooncake, `MOONCAKE_PROTOCOL=hip`) | **Works** after working around a Mooncake HIP-transport offset bug (silently wrong weights before the fix). load_weight 6.6-6.8 s |
| Compiled-kernel cache transfer (`MX_ARTIFACT_TRANSFER`) | Pack and restore `~/.flydsl`, `~/.cache/sglang`, `~/.cache/comgr` | **Works**. Startup 267 s cold to 127 s with a migrated 1.2 GB bundle; `~/.flydsl` is the part that matters |
| Accelerated storage loading | `fastsafetensors` (HIP, nogds), `runai_streamer`, prefetch, no-mmap | **Little headroom.** Best cold load 65 s vs 72 s default; warm about 58-64 s for all. The cost is the per-tensor `load_weights` loop, not I/O |
| (not in MX) same-GPU resident weights | `--weight-cache-mode` daemon + IPC | **Not supported for this model**: `compressed-tensors` is not on the IPC allowlist, by design |
| ModelExpress itself | needs ROCm NIXL/UCX; MX uses CUDA VMM / GDS | Not attempted |

Measured with the same timing script (`time_startup.sh`, launch to "fired up" wall time, warm page cache, warm kernel cache):
disk load about 130 s; R-Fork RCCL 77-81 s; R-Fork Mooncake 76-77 s. A completely cold node (empty kernel cache) needs about 267 s.

## 1. P2P weight pull: R-Fork over RCCL
### Usage
```bash
# seed: an ordinary instance
HIP_VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 PORT=7090 bash scripts/launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh
# client: pulls weights from the seed
HIP_VISIBLE_DEVICES=4,5,6,7 TP_SIZE=4 PORT=7091 SEED_PORT=7090 bash scripts/launch_qwen38_flash_next_fp8_mi308x_rfork_client.sh
```
Debug: set `SGLANG_DEBUG_WEIGHT_CHECKSUM_DIR=<dir>` on both instances. After loading, each rank writes the shape, dtype
and byte checksum of every parameter and buffer, plus the manifest checked before transfer. Compare with
`diff <(cut -f1-4,6 seed/target_rank0.tsv) <(cut -f1-4,6 client/target_rank0.tsv)`.

### Problems found with the unmodified branch, and fixes
| # | Symptom (as observed) | Root cause | Fix |
|---|---|---|---|
| 1 | Client crashes: `Model loader did not prepare PyHIP GR read weights` | The remote_instance loader never runs `process_weights_after_loading` | Client runs each module's `quant_method.process_weights_after_loading` before receiving, so its layout matches the post-processed seed (same approach as `ShardedStateLoader`) |
| 2 | Checksums without fix 1: 175-240 entries differ per rank. FP8 bytes match but dtype is `float8_e4m3fn` vs the seed's `float8_e4m3fnuz`; 48 `gemma_weight` buffers per rank differ | fnuz conversion etc. happen in post-processing. `GemmaRMSNorm.gemma_weight` (= weight + 1) is computed in the weight_loader hook, which a raw copy skips | Fix 1, plus the client calls the existing `PreshardedModelLoader._rebind_parameter_aliases` after receiving |
| 3 | After also transferring buffers, seed and client deadlock (seed in `destroy_process_group`, client in `synchronize`) | The seed grows its RoPE `cos_sin_cache` at runtime (8192->262400, 262144->262400), so broadcast sizes differ; RCCL hangs instead of raising | Transfer parameters only. Before transferring, both sides `all_gather` a manifest digest (name/shape/dtype + count + bytes) and fail cleanly on mismatch (verified: client errors in 3 s, seed `/health` stays 200) |
| 4 | Seed instance crashes on `KeyError` | A seed rank's send group failed to init (`EADDRINUSE`), and the send request indexed the dict directly | Return an error instead of crashing the seed; the client also checks `build_group`'s result |
| 5 | `EADDRINUSE` (cause of 4) | `ip_local_port_range` here is 32768-60999, so group ports 41000/42000 collide with ephemeral sockets | Client script defaults to ports 62000+ (`GROUP_PORT_BASE`) |
| - | `--remote-instance-weight-loader-send-weights-group-ports` must be a JSON list | Argument format | Generated by the client script |

### Results
- Checksums: **0 mismatches out of 1503 entries (1449 params + 54 buffers) on each of 4 ranks**.
- gsm8k (`scripts/run_gsm8k_qwen38.sh`, 1314 questions, one run each): disk-load seed 0.9787 (baseline) / 0.9795 (rerun, same period as client); R-Fork client 0.9741.
  The 0.54-point gap is within noise (standard error of a two-run difference is about 0.006).
- Greedy fixed-prompt comparison: client vs seed 1/8 identical, but seed vs itself is also 1/8, so greedy output is nondeterministic here and this test cannot judge correctness.
- load_weight 8.2-8.5 s (4 runs: 8.26, 8.23, 8.45, 8.50) vs 58-64 s from disk. Broadcast of 65.26 GB per rank takes about 5 s.

## 2. P2P weight pull: R-Fork over Mooncake TransferEngine (HIP transport)
### Setup
- `pip install mooncake-transfer-engine-rocm` (0.3.13.post1). Importing it needed `apt install libibverbs1 librdmacm1`.
- `MOONCAKE_PROTOCOL=hip` (intra-node HIP IPC). Both instances see all 8 GPUs and are placed with `--base-gpu-id 0 / 4`.
- Seed: `--remote-instance-weight-loader-start-seed-via-transfer-engine`.
  Client: `--load-format remote_instance --remote-instance-weight-loader-backend transfer_engine --remote-instance-weight-loader-seed-instance-ip 127.0.0.1 --remote-instance-weight-loader-seed-instance-service-port 7090`.

### Bug found: silently wrong weights
The first TP2 run started and served, but **913 of 1503 tensors per rank differed from the seed**. For example, the
client's `input_mix_weight_up` held exactly the seed's `input_mix_weight_down` bytes.
`mooncake_hip_offset_repro.py` reproduces it without SGLang (two processes, GPU4 to GPU6):
- registered region starts at its allocation base: reads are correct;
- registered region starts mid-segment (a live non-weight tensor in front): the read returns data shifted by that offset
  (`--layout offset --raw-register` gives WRONG DATA).

Mooncake's HIP transport resolves remote addresses from the IPC allocation base rather than the registered address.
SGLang's `register_memory_region_v2` registers weight blocks inside PyTorch allocator segments, so this hits real models.
Workaround in this branch: on ROCm, register whole allocator segments that contain weights (`--layout offset` without
`--raw-register` then reads OK). The client path also recomputes derived buffers (`_rebind_parameter_aliases`) as in section 1.

### Results (after fix)
- Checksums: TP2 **0/1503 mismatches on 2 ranks**; TP4 **0/1503 mismatches on 4 ranks**.
- gsm8k on the Mooncake-loaded TP4 client: 0.9795 (one run; seed baseline 0.9787 / 0.9795).
- Same seed, same timing script, 2 runs each:

| Backend | load_weight | launch -> ready (wall) |
|---|---|---|
| transfer_engine (Mooncake, hip) | 6.56 / 6.79 s | 76.2 / 77.2 s |
| nccl (RCCL) | 8.45 / 8.50 s | 81.2 / 77.2 s |

- No cross-node test (no RDMA NIC). The official wheel is reported to lack `ENABLE_MULTI_PROTOCOL`, which matters for cross-node.

## 3. Compiled-kernel cache transfer
Kernel caches on this stack live in several places, not only `SGLANG_CACHE_DIR`: `~/.flydsl` (FlyDSL kernels),
`~/.cache/comgr` (AMD code objects), `~/.cache/sglang/{triton,inductor,jit}`. AITER modules are prebuilt `.so` files
under `/opt/aiter` and were left untouched; `/tmp/aiter_configs` (tuned CSVs) also untouched.
The test points `HOME` at a directory so that all of the above start empty, then compares.
`first_requests_latency.py` sends 3 rounds of 16 mixed-length requests right after startup.

| Cache state | launch -> ready | load_weight | decode graph | first round | 2nd / 3rd round |
|---|---|---|---|---|---|
| Cold (empty HOME) | 266.6 s | 120.5 s | 35.2 s | 6.90 s | 2.35 / 1.37 s |
| Warm (same HOME, second start) | 131.3 s | 63.6 s | 11.2 s | 1.53 s | 3.34 / 2.30 s |
| Migrated (tar of the cold run's cache unpacked at a new path) | 127.3 s | 58.9 s | 11.2 s | 1.45 s | 2.12 / 1.20 s |
| Migrated without `~/.cache/comgr` | 125.3 s | 55.8 s | 12.2 s | 1.68 s | 1.07 / 2.37 s |
| Migrated without `~/.flydsl` | 204.5 s | 122.3 s | 13.9 s | 5.87 s | 2.08 / 1.28 s |

- Bundle: 1.18 GB (comgr 928 MB, FlyDSL 155 MB, sglang 41 MB); tar about 1.2 s, untar about 1.1 s.
- Cache paths are relocatable (a different HOME works).
- `~/.flydsl` is the part that matters for startup; comgr made no measurable difference in these runs.

## 4. Accelerated storage loading
Cold = model files evicted with `evict_model_cache.py --evict` (per-file `posix_fadvise(DONTNEED)`, residency checked
with `mincore`: 0.0% before each cold run). Single-stream cold read of one shard: 2.26 GiB/s.

| Load mode | cold load_weight | cold wall | warm load_weight | warm wall | GPU mem after load | gsm8k |
|---|---|---|---|---|---|---|
| default (`auto`, mmap) | 72.3 s | 145.3 s | 62.4 s | 130.3 s | 62.0 GB | 0.9787 / 0.9795 (section 1) |
| `--weight-loader-prefetch-checkpoints` | 74.3 s | 142.3 s | 64.5 s | 133.4 s | 62.0 GB | not run |
| `--weight-loader-disable-mmap` | 363.0 s | 436.1 s | 363.4 s | 432.3 s | 96.2 GB | not run |
| `fastsafetensors` (SGLang default `enable_gds=true`) | 113.4 s | 181.4 s | not run | not run | 62.6 GB | not run |
| `fastsafetensors` + `{"enable_gds": false}` | **65.0 s** | 134.4 s | 63.5 s | 132.4 s | 62.6 GB | 0.9779 |
| `runai_streamer` (default io_uring) | fails | - | fails | - | - | - |
| `runai_streamer` + `RUNAI_STREAMER_FS_STRATEGY=libaio_direct` | 66.9 s | 134.3 s | **57.9 s** | 125.4 s | 96.2 GB | 0.9772 |

Raw file-to-GPU throughput of one file on one GPU (`fst_raw_throughput.py`; compare with fastsafetensors'
[docs/amd-perf.md](https://github.com/foundation-model-stack/fastsafetensors/blob/main/docs/amd-perf.md), which reports
2-6 GB/s nogds vs about 1 GB/s mmap for small GPT-2 files on MI300X / fastsafetensors 0.1.15):

| File | Method (threads / buffer) | Cold | Warm |
|---|---|---|---|
| shard 2 (20.0 GB) | fastsafetensors nogds 16 / 16 MB (library default, the doc's best) | 12.68-13.01 GB/s | 13.10 GB/s |
| shard 2 | fastsafetensors nogds 32 / 32 MB | 10.94 GB/s | 11.55 GB/s |
| shard 2 | fastsafetensors nogds 8 / 8 MB | 11.68 GB/s | 11.12 GB/s |
| shard 2 | safetensors mmap + `.to("cuda")` | 0.41 GB/s | 1.41 GB/s |
| shard 1 (102.4 GB) | fastsafetensors nogds 16 / 16 MB | 21.02 GB/s | not run |

The storage is a 4x NVMe RAID0 (`/dev/md127`). fastsafetensors reads with O_DIRECT: residency stayed at 0.0% after a cold read, so these are real disk reads.
As in the doc, 16 threads / 16 MB is best, and nogds is much faster than mmap (about 9-30x here).

Findings:
- Raw I/O is not the bottleneck at this scale: at 13-21 GB/s the whole 220 GiB could be read in roughly 10-17 s, yet
  SGLang's end-to-end fastsafetensors load is 65 s. Making storage acceleration pay off first needs a faster
  `load_weights` path for the 147k per-expert tensors (e.g. loading experts in batches).
- The warm-cache floor is about 55-64 s for every disk loader. fastsafetensors warm already has the data on GPU, yet its
  iterator still needs about 54 s. That time is `model.load_weights` calling a weight_loader for each of the
  149k checkpoint tensors (147k of them per-expert). Storage acceleration can only remove the 7-10 s I/O part.
  R-Fork avoids it by moving 1449 already-merged tensors.
- fastsafetensors 0.5.0 runs on ROCm (HIP mode, `is_hip_found()`); use `enable_gds=false`. With the default
  (GDS on) it loads but is slower (113 s cold).
- runai-model-streamer 0.17.0 fails on this kernel with `File access error` (its default `io_uring_direct` strategy).
  `libaio_direct` or `sync_buffered` work.
- `disable-mmap` and `runai_streamer` leave about 34 GB more GPU memory in use per rank after loading (96 vs 62 GB),
  which reduces KV-cache capacity. Not investigated further.

## 5. Same-GPU resident weights (`weight_cache`)
Standalone daemon started via `weight_cache_shim/sglang` (turns the launch script's `sglang serve ARGS` into
`python -m sglang.srt.weight_cache.daemon ARGS`). Result:
`UnsupportedQuantForIPCError: quantization method 'compressed-tensors' is not verified for CUDA IPC zero-copy weight sharing`.
The IPC client builds the model on the meta device and maps the daemon's tensors without running
`process_weights_after_loading`. This model's post-processing also leaves Python-side state (e.g. PyHIP GR read's
`_gr_read_weights`, which references the packed tensors), so mapping tensors alone would leave that state missing or
stale. Supporting it needs a per-quant-method "rebuild post-processing state after mapping" step. That is not done here,
and the guard was not bypassed.

## Tools in this directory
- `time_startup.sh`: launch one instance, wait for ready, print timings (optionally evicting model cache first).
- `evict_model_cache.py`: per-model page-cache residency and eviction.
- `first_requests_latency.py`: latency of the first request rounds (lazy JIT).
- `mooncake_hip_offset_repro.py`: standalone Mooncake HIP transport offset repro.
- `fst_raw_throughput.py`: raw fastsafetensors / mmap file-to-GPU throughput.
- `greedy_compare.py`: fixed-prompt greedy output comparison between two servers.
- `weight_cache_shim/sglang`: PATH shim to start the weight-cache daemon with the launch script's args.

## Not done / known limitations
- Single node only. Not validated: cross-node (RCCL or Mooncake RDMA), TP2/TP8 for RCCL, MTP/NEXTN draft models, ModelExpress itself.
- Process-weight post-processing on the client was only exercised on this model's quant/PyHIP path. Other models and
  quant schemes need checking (the checksum switch exposes problems).
- Derived-buffer recompute relies on `_rebind_parameter_aliases` (gemma_weight and a few linear-attn aliases).
- If an RCCL client dies after the send group is built, the seed can be left in a bad state (observed once, not fixed).
- Interference of a transfer with the seed's online requests was not measured.
- The ROCm segment-registration workaround registers some non-weight memory in the seed's allocator segments for remote reads.
- gsm8k was run once per configuration.
