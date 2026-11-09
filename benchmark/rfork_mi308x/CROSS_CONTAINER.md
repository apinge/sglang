# R-Fork across containers on one node

A new instance in its own container can pull weights from a seed instance that runs in
another container on the same node. The R-Fork arguments are the same as within one
container; only the container setup differs.

## Requirements

| Requirement | Setting | Why |
|---|---|---|
| Both containers on the host network | `--network=host` | Group setup and weight transfer requests go over TCP. With the same hostname, RCCL treats both sides as one node and uses P2P/XGMI instead of sockets |
| Shared IPC namespace | `--ipc=host` | RCCL intra-node P2P exchanges shared-memory/IPC handles |
| GPU access | `--privileged --device /dev/kfd --device /dev/dri` | Each container sees all GPUs; placement is done with `HIP_VISIBLE_DEVICES` or `--base-gpu-id` |
| Same software in both containers | see below | The new container must run the same code and have the same caches as the seed container |

The new container must match the seed container's environment. Anything installed into the seed container after the image was built is missing from a fresh container:
- **Code and Python packages:**
  - this branch's code, via `PYTHONPATH=<repo>/python`;
  - packages added later, such as PyHIP. SGLang reads PyHIP's version, so its `dist-info` must be on the path too.
- **Kernel caches:**
  - `~/.flydsl`, `~/.cache/sglang` and `~/.cache/comgr`;
  - aiter JIT modules built at runtime under `/opt/aiter/aiter/jit/`. Without them, the first launch rebuilds them. Here that cost about 220 s: `module_gemm_a8w8_bpreshuffle_cktile` took 164.6 s and `mha_varlen_fwd` took 52.1 s.
- Committing the seed container (`docker commit`) and using that image avoids all of this.

## Usage

Seed, inside the existing container (TP2 on GPU0-1):
```bash
HIP_VISIBLE_DEVICES=0,1 TP_SIZE=2 PORT=7090 MODE=seed \
  bash scripts/launch_qwen38_flash_next_fp8_mi308x_rfork.sh
```

New instance, from the host (TP2 on GPU2-3). The script starts a container with the settings above and runs `MODE=rfork`:
```bash
IMAGE=<seed container image> HOST_MODELS=<host model dir> \
HIP_VISIBLE_DEVICES=2,3 TP_SIZE=2 PORT=7091 SEED_PORT=7090 \
  bash scripts/launch_qwen38_flash_next_fp8_mi308x_rfork_cross_container.sh
```
- The new container runs the SGLang install at `REPO_DIR` (default `/opt/sglang`, the image's own). To run a checkout from the host instead, mount it with `DOCKER_ARGS="-v <host checkout>:/src"` and set `REPO_DIR=/src`.
- Packages the image lacks go in `EXTRA_PYTHONPATH`. One-time setup goes in `SETUP_CMD`, for example `tar -C /root -xf <kernel cache tar>; cp -n <dir>/*.so /opt/aiter/aiter/jit/`.

Check that the transfer used P2P by starting the new instance with `NCCL_DEBUG=INFO`. The log should show `via P2P/IPC` and no `via NET`. To verify the weights, set `SGLANG_DEBUG_WEIGHT_CHECKSUM_DIR` on both instances and diff the per-rank files.

## Result (TP2, RCCL, 1 run)

Single node, 8x MI308X. The seed ran in container A, which was already up. The new instance ran in a fresh container B from the same image, with the environment items above restored.

| | Cross-container | Same container (2 runs, for reference) |
|---|---|---|
| Launch to ready | 71.5 s | 66.6 / 69.6 s |
| load_weight | 13.09 s | 8.75-8.8 s |
| of which group setup to seed done sending | 3 s | 3 s |
| Per-tensor checksums vs seed | 2 x 1503, identical | identical |
| RCCL transport | `via P2P/IPC` only | P2P |

- The transfer itself took as long as within one container.
- The extra ~4 s in load_weight came after the transfer. That run had `NCCL_DEBUG=INFO` enabled (5x more log lines) and the same-container runs did not; whether the debug logging explains the difference was not checked.
- This is a single run. The isolated case (bridge network, no `--ipc=host`) was not measured.

## Caveat

If a new instance fails after asking the seed to set up the transfer group but before joining it, the seed blocks (up to 30 min, `/health` returns 503). This happened during these tests when the new container lacked PyHIP. A failed attempt can also leave a stale request queued on the seed. Restart the seed after any failed fork.
