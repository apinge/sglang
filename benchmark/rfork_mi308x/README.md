# R-Fork scale-out on MI308X (Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16)

> Cherry-picked from PR #67 on `qwen38-flash-next-amd-current-fix-install`; the results below were measured on that
> branch. On `qwen4_v0.5.21` the same-container TP2 path is covered by the Qwen4 CI case `Qwen3.8-Flash-TP2-RFork`.

R-Fork (`--load-format remote_instance`) starts a new instance by copying the processed weights from a running
instance (the seed) over RCCL, instead of reading and processing the checkpoint. This directory holds the scripts used
to measure it, and the results on a single node with 8x MI308X (XGMI, ROCm 7.14).

## Effect

Launch to ready, in seconds, mean of 2 runs. Kernel caches are warm. "Page cache" is whether the model files were in
host memory before the launch. The new instance does not read the weight files, so its time does not depend on it.

| New instance loads weights from | TP2, page cache warm | TP2, page cache cold | TP4, page cache warm | TP4, page cache cold |
|---|---|---|---|---|
| disk (default) | 139.5¹ | 142.2 | 125.9 | 137.9 |
| R-Fork, RCCL, from a running seed | 67.1¹ | 68.1² | 79.1 | not measured |

- Weight loading drops from 77.2 s (TP2) / 55.1 s (TP4) to about 5 s. The rest is process start, init and graph
  capture, which R-Fork does not change.
- The seed's own startup is not included. The seed is an ordinary instance that is already serving.

¹ Measured at this PR's head (`ebb48a000d`): disk 138.7 / 140.2 s, R-Fork 66.1 / 68.1 s. Another job was running on
other GPUs of the node during this run.
² Measured at `dde1734b3`. The other cells were measured at `378743602`. The NCCL path did not change between these
commits.

**Correctness:** every rank's per-tensor checksums (1449 parameters + 54 buffers) match the seed. Full GSM8K, one run
each:

| TP | Disk-loaded seed | R-Fork instance |
|---|---|---|
| TP2 (`ebb48a000d`) | 0.9779 | 0.9810 |
| TP4 (`378743602`) | 0.9810 | 0.9772 |

## Dependencies

None beyond the SGLang ROCm image:
- The RCCL backend uses `torch.distributed`'s `nccl` backend. On ROCm PyTorch, that is RCCL, which ships with ROCm (`/opt/rocm/lib/librccl.so`; RCCL 2.30.4 in the tested image).
- The HTTP calls to the seed use `requests`, which is already installed.

Deployment requirements:
- The new instance must use the same model and TP size as the seed.
- `--remote-instance-weight-loader-send-weights-group-ports` takes a JSON list with one port per TP rank. The seed listens on these ports, so keep them outside `/proc/sys/net/ipv4/ip_local_port_range`.

## How to use

```bash
export PYTHONPATH=<this repo>/python
# seed (GPU0-3): an ordinary instance
HIP_VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 PORT=7090 MODE=seed bash evaluation/launch_qwen38_flash_next_fp8_mi308x_rfork.sh
# new instance (GPU4-7): pulls its weights from the seed
HIP_VISIBLE_DEVICES=4,5,6,7 TP_SIZE=4 PORT=7091 MODE=rfork SEED_PORT=7090 bash evaluation/launch_qwen38_flash_next_fp8_mi308x_rfork.sh
```

`MODE=rfork` adds these arguments to the regular launch command:
```
--load-format remote_instance --remote-instance-weight-loader-backend nccl
--remote-instance-weight-loader-seed-instance-ip <seed ip> --remote-instance-weight-loader-seed-instance-service-port 7090
--remote-instance-weight-loader-send-weights-group-ports '[62000,62001,62002,62003]'
```

For a new instance in its own container on the same node, see [CROSS_CONTAINER.md](CROSS_CONTAINER.md) and
`evaluation/launch_qwen38_flash_next_fp8_mi308x_rfork_cross_container.sh`.

The seed and the new instance are independent servers with their own ports. Put a load balancer or router in front if
clients should see one address.

## Test scripts

```bash
# Startup time, disk vs R-Fork, per phase. TP = number of --gpus; the seed runs on --seed-gpus.
python3 benchmark/rfork_mi308x/bench_startup.py --prime --repeat 2 \
  --gpus 4,5,6,7 --seed-gpus 0,1,2,3 --out /tmp/rfork_tp4
# TP2: --gpus 2,3 --seed-gpus 0,1.  --dry-run lists the runs; --reparse --out <dir> rebuilds summary.md.

# Correctness: disk seed + R-Fork instance; per-rank checksum diff and full GSM8K on both.
python3 benchmark/rfork_mi308x/verify_rfork.py --tp 4 --seed-gpus 0,1,2,3 --gpus 4,5,6,7 --out /tmp/verify_tp4
```

- `bench_startup.py` prepares and checks the page-cache state before every run, using `evict_model_cache.py`.
- It refuses GPUs that already hold more than 2 GiB.
- `SGLANG_DEBUG_WEIGHT_CHECKSUM_DIR=<dir>` makes every rank write its per-tensor checksums after loading. `verify_rfork.py` uses this; it can also be diffed by hand.

## Known limitations

- If a new instance fails after asking the seed to set up the transfer group but before joining it, the seed blocks in the rendezvous for up to 30 minutes (`/health` returns 503). Restart the seed after a failed new instance.
- Tested on a single node at TP2 and TP4 with this model only.
- The effect of a transfer on the seed's live traffic was not measured.
