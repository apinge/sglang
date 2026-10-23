# R-Fork (remote_instance, RCCL) on MI308X: Qwen3.8-Flash-Next-PTPC-FP8 validation report

This is the AMD counterpart of NV ModelExpress's "new instance pulls GPU weights from an online replica" capability.
It uses SGLang's built-in R-Fork (`--load-format remote_instance --remote-instance-weight-loader-backend nccl`),
which runs over RCCL on ROCm. Everything below was measured on 2026-10-23, and every number comes from a log of an actual run.

## Environment
- 8x AMD Instinct MI308X (gfx942), single node, XGMI full mesh; no RDMA NIC visible in the container
- ROCm 7.14, torch 2.12.0+rocm7.14; the code is this branch (based on `c93b9c195`)
- Model: `/models/Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16` (221 GB on disk, about 61 GB of weights per rank at TP4)
- Topology: seed on GPU0-3 (TP4, port 7090), client on GPU4-7 (TP4, port 7091), both on the same node
- Launch scripts: `scripts/launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh` (seed) and
  `scripts/launch_qwen38_flash_next_fp8_mi308x_rfork_client.sh` (client, new in this branch)
- Interference: the GPUs are shared with other containers; `rocm-smi --showpids` showed processes from another PID
  namespace using GPU memory, e.g. about 28 GB on GPU7. Throughput and latency numbers are therefore only a rough reference.

## Usage
```bash
# seed: an ordinary instance
HIP_VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 PORT=7090 bash scripts/launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh
# client: pulls weights from the seed
HIP_VISIBLE_DEVICES=4,5,6,7 TP_SIZE=4 PORT=7091 SEED_PORT=7090 bash scripts/launch_qwen38_flash_next_fp8_mi308x_rfork_client.sh
```
Debug: set `SGLANG_DEBUG_WEIGHT_CHECKSUM_DIR=<dir>` on both instances. After loading, each rank writes the shape, dtype
and byte checksum of every parameter and buffer, plus the manifest checked before transfer. Compare with
`diff <(cut -f1-4,6 seed/target_rank0.tsv) <(cut -f1-4,6 client/target_rank0.tsv)`.

## Problems found with the unmodified branch, and fixes
| # | Symptom (as observed) | Root cause | Fix |
|---|---|---|---|
| 1 | Client crashes: `Model loader did not prepare PyHIP GR read weights` | The remote_instance loader never runs `process_weights_after_loading` | Client runs each module's `quant_method.process_weights_after_loading` before receiving, so its layout matches the post-processed seed (same approach as `ShardedStateLoader`) |
| 2 | Checksums with fix 1 not yet applied (from the debug run): 175-240 entries differ per rank. FP8 weight bytes match but dtype is `float8_e4m3fn` vs the seed's `float8_e4m3fnuz`, and 48 `gemma_weight` buffers per rank differ | Same as 1: fnuz conversion and similar steps happen in post-processing. `GemmaRMSNorm.gemma_weight` (= weight + 1) is computed in the weight_loader hook, which a raw copy skips | Fix 1 resolves the dtype issue. After receiving, the client calls the existing `PreshardedModelLoader._rebind_parameter_aliases` to recompute `gemma_weight` |
| 3 | After also transferring buffers, seed and client deadlock: seed in `destroy_process_group`, client in `synchronize` | The seed expands its RoPE `cos_sin_cache` at runtime (8192->262400, 262144->262400), so the two sides broadcast different sizes. RCCL hangs instead of raising an error | Transfer parameters only (as originally designed). Before transferring, both sides `all_gather` a manifest digest (name/shape/dtype + count + bytes). On mismatch, both sides fail with a clear error (verified: client errors out in 3s, seed `/health` still 200) |
| 4 | Seed instance crashes on `KeyError` | One seed rank's send group failed to init with `EADDRINUSE` on 42001, and the later send request indexed the dict directly | Return an error when the group is missing instead of crashing the seed; the client also checks `build_group`'s return value |
| 5 | `EADDRINUSE` (cause of 4) | `ip_local_port_range` on this machine is 32768-60999, so group ports 41000/42000 can be taken by ephemeral sockets | Client script uses ports 62000+ by default (`GROUP_PORT_BASE`) |
| - | `--remote-instance-weight-loader-send-weights-group-ports` must be a JSON list such as `'[62000,62001,62002,62003]'` | Argument format | The client script generates it automatically |

## Results
### Consistency (run r7)
- Per-rank parameter and buffer checksums of client vs seed after loading: **0 mismatches out of 1503 entries (1449 params + 54 buffers) on each of 4 ranks**
  - The seed checksum is taken at seed load time, before serving. Runtime-expanded caches such as RoPE are excluded from the comparison by design.
- gsm8k (`scripts/run_gsm8k_qwen38.sh`, 1314 questions, one run each):

| Instance | score | latency |
|---|---|---|
| Disk-load seed (01:25, baseline) | 0.9787 | 630 s |
| Disk-load seed, rerun (02:44, same conditions as client) | 0.9795 | 758 s |
| R-Fork client (02:23) | 0.9741 | 712 s |

  The client trails the seed rerun by 0.54 points (about 7 questions). With one run each, the standard error of the difference is about 0.006, so this is within noise.
  Latency differs a lot between periods and is affected by external GPU load; the same seed went from 630 s to 758 s.
- Greedy comparison on fixed prompts (temperature=0, 8 prompts, 128 tokens): client vs seed 1/8 identical, **and seed vs itself is also 1/8 identical**.
  Greedy output is nondeterministic on this stack, so this test cannot judge correctness and is recorded only.

### Startup time (TP4, measured from each log's first timestamped line to "fired up"; excludes about 15 s of Python import time before that)
| Mode | load_weight | first log -> ready | Runs |
|---|---|---|---|
| Disk (page cache presumably warm, cold cache not tested) | 58.2-62.4 s | 115-121 s | 5 |
| R-Fork RCCL, no checksum dump | 7.8-8.3 s | 65 s | 2 |

Client phase breakdown (client_timing1, TP0): distributed init 17.7 s, weight load 7.9 s (broadcast of 65.26 GB per rank about 5 s,
second-resolution logs), decode CUDA graph capture 12.7 s, remainder (warmup / tokenizer etc.) about 25 s.
Weight load is no longer the main cost; next steps should target init, graph capture and JIT (see "Not done").

## Not done / known limitations
- Only single-node TP4 + RCCL (XGMI) was validated. Not validated: TP2/TP8, cross-node, `transfer_engine` (Mooncake), `modelexpress`, MTP/NEXTN draft models.
- Process-weight post-processing changes only went through this model's quant/PyHIP path. Other models and quant schemes running post-processing on uninitialized data still need verification.
- Recompute of derived buffers relies on `_rebind_parameter_aliases` (currently covers gemma_weight and a few linear-attn aliases). Other models with weight-derived buffers need per-model checks; enabling `SGLANG_DEBUG_WEIGHT_CHECKSUM_DIR` exposes them.
- If the client dies after the send group is built, the seed may stay in a bad state. One case was observed (interrupted client), but the root cause is not resolved.
- Interference with the seed's online requests during transfer was not measured.
- Disk-load baseline was not tested with a cold page cache.
