# Sequence-parallel hyper-connection residual for Qwen4-Exp TP prefill

Qwen4-Exp (Qwen3.8-Flash-Next) keeps a 4-lane hyper-connection residual stream. Between the TP-sharded
blocks, three per-token ops run on it: `hc_combine` (inject the block output), `hc_norm`
(grouped Gemma RMSNorm, variance per token and per 2560-wide lane) and the low-rank `mix` that builds the
next block's input. With the usual layout every TP rank repeats them on all tokens; in a 12k-token
prefill step that is ~165 ms per rank (31% of the step at TP4, 22% at TP2).

With `SGLANG_QWEN4_SEQ_PARALLEL=1` (the default) each rank keeps only its 1/TP slice of the token rows of the residual
stream during prefill:

```
partial block output [T, 2560]
  -> reduce-scatter            (this rank's T/TP rows, summed)
  -> hc_combine, hc_norm, mix  (on T/TP rows)
  -> all-gather                (mixed input, [T, 2560])
  -> attention / GDN / MoE     (unchanged, on all T rows; MoE skips its own all-reduce)
```

The reduce-scatter plus all-gather (aiter custom kernels) move the same bytes as the all-reduce they
replace, and AG(RS(x)) is bit-identical to AR(x). The residual is sharded after the last PLE layer
(PLE's short conv mixes neighbouring tokens). At the end the final mixer runs on the local rows and only
its [T, 2560] output is gathered; the [T, 4 x 2560] residual (HC state) is gathered only when the forward
captures hidden states (speculative decoding, `return_hidden_states`).

| Knob | Default | Effect |
|---|---|---|
| `SGLANG_QWEN4_SEQ_PARALLEL` | 1 | sequence-parallel residual for prefill; 0 keeps the all-reduce |
| `SGLANG_QWEN4_SEQ_PARALLEL_MIN_TOKENS` | 2048 | smaller batches keep the all-reduce |

The path covers eager EXTEND / MIXED / TARGET_VERIFY forwards with at least
`SGLANG_QWEN4_SEQ_PARALLEL_MIN_TOKENS` tokens. Decode, CUDA-graph-captured batches (including captured
target verify), draft-model forwards, DP attention and EP all-to-all keep the original path.

## Scripts

| Script | What it measures |
|---|---|
| `bench_rs_ag.py` | aiter all-reduce vs reduce-scatter + all-gather, and their bit-identity |
| `bench_qwen35_sdma_all_reduce.py` | Qwen3.5 TP2 SDMA all-reduce correctness, and a layer stand-in vs RCCL |

## Results (8x MI308X, 12000 in / 350 out, `scripts/run_pure_text_bench.sh`)

12k-token prefill step (mean of 4): TP4 542.0 -> 434.9 ms, TP2 748.7 -> 691.8 ms. Serving req/s, two
interleaved rounds (TP4 and TP2 run one after the other):

| conc | TP4 base | TP4 SP | TP2 base | TP2 SP |
|---|---|---|---|---|
| 1 | 0.342 / 0.339 | 0.349 / 0.353 | 0.302 / 0.302 | 0.310 / 0.309 |
| 4 | 0.776 / 0.769 | 0.821 / 0.836 | 0.594 / 0.597 | 0.619 / 0.616 |
| 8 | 1.003 / 1.012 | 1.087 / 1.122 | 0.798 / 0.792 | 0.834 / 0.826 |
| 16 | 1.211 / 1.205 | 1.359 / 1.373 | 0.930 / 0.931 | 0.991 / 0.993 |
| 32 | 1.353 / 1.347 | 1.541 / 1.553 | 1.008 / 1.006 | 1.079 / 1.081 |

Greedy outputs of a long-context probe are bit-identical to the baseline; the long-context check is
48/48 and gsm8k stays within run-to-run noise.

## SDMA overlap (ROCm, mori)

On top of sequence parallelism, each of its communications moves to the SDMA copy engines and runs
under compute (`layers/sdma_reduce_scatter.py`, `layers/sdma_mori.py`):

```
RS       o_proj / out_proj in row chunks, peers' blocks first, own block last;
         each chunk is pushed to its owner as soon as it is computed       -> overlaps the next chunk's GEMM
         (the MoE output, computed whole, is pushed in the same chunks)
attn AG  mix -> push this rank's rows; in_proj runs on the local rows,
         then on each peer's block as it lands                             -> overlaps in_proj on local rows
MoE AG   combine / hc_norm / mix in row ranges, each range pushed as
         soon as it is mixed; the MoE waits for every range                -> overlaps the next range's HC ops
```

The copies are posted by one-wave kernels on the compute stream, which write the SDMA packets directly
(mori CCO `put_ptr`); the copy engine bumps the receiver's counter when the bytes land and the consumer
waits with a one-wave spin kernel. No side streams or cross-stream events are involved. Destinations are
hipIpc mappings, not mori's symmetric windows, so hosts whose kernel lacks `CONFIG_DMABUF_MOVE_NOTIFY`
work too. When mori cannot run in the process, the server logs one warning and keeps the aiter
reduce-scatter / all-gather of plain sequence parallelism.

| Knob | Default | Effect |
|---|---|---|
| `SGLANG_QWEN4_SP_SDMA_RS` | 1 | reduce-scatters over SDMA |
| `SGLANG_QWEN4_SP_SDMA_AG` | 1 | attention / GDN input all-gather over SDMA, consumed per block by the input projection |
| `SGLANG_QWEN4_SP_SDMA_AG_MOE` | 1 | MoE input all-gather over SDMA, pushed per HC row range |
| `SGLANG_QWEN4_SP_SDMA_RS_TRANSPORT` | bf16 | `bf16` (exact) or `fp8` (peers' rows as fp8 + a bf16 scale per 32; lossy) |
| `SGLANG_QWEN4_SP_SDMA_RS_CHUNKS` | auto | row chunks per owner block; auto = 2 at TP2, 1 above |
| `SGLANG_QWEN4_SP_SDMA_AG_MOE_CHUNKS` | auto | minimum HC row ranges; auto = 4 at TP2, 2 above |

All of them act only with `SGLANG_QWEN4_SEQ_PARALLEL=1`.

### TP2 prefill: AG zero copy, two-row hc_combine (MI308X)

- gfx942 AG consumer zero copy: each landed block's PTPC GEMM writes its rows of the gathered output
  through `aiter.gemm_a8w8_bpreshuffle(out=...)`; an aiter without `out=` keeps the per-block copy.
- gfx942 two-row `hc_combine` for rows of at most 4 x 2560 elements, bit-identical to the one-row kernel.

With aiter's TP2 prefill GEMM rows (`qwen38-flash-next-dev`), TP2 serving at 12000 in / 350 out goes
from 0.916 / 1.043 to 0.959 / 1.088 req/s at c8 / c12 (+4.6 % / +4.3 %).

### Qwen3.5 MoE TP2: pipelined SDMA all-reduce

Qwen3.5 (`qwen3_5_moe_text`, e.g. Qwen3.5-35B-A3B) has no hyper-connection residual, so it keeps the
replicated residual and only borrows the SDMA overlap (`models/qwen3_5_sdma_all_reduce.py`). At TP2 an
all-reduce is "push the partial to the peer, then add", the same bytes per rank as RS + AG:

```
o_proj / out_proj in row chunks, each pushed to the peer as soon as it is computed
per landed chunk: rank 0 rows + rank 1 rows (bit-identical on both ranks)
    -> residual add + RMSNorm -> MoE on that chunk (no all-reduce) -> push
next layer: per landed chunk -> residual add + input RMSNorm
```

It covers eager TP2 EXTEND batches of at least `SGLANG_QWEN4_SEQ_PARALLEL_MIN_TOKENS` tokens and shares
the Qwen4 switches, independently of `SGLANG_QWEN4_SEQ_PARALLEL`: `SGLANG_QWEN4_SP_SDMA_RS=0` keeps the
all-reduce, `SGLANG_QWEN4_SP_SDMA_RS_CHUNKS` sets the row chunks (default 4 for Qwen3.5; 2 and 8 were
5-12 % slower at 12k-token prefill). Other models, Qwen4-Exp included, and other TP sizes are unchanged.

`bench_qwen35_sdma_all_reduce.py` checks the all-reduce (bit-identical ranks, delayed producer) and times
a layer stand-in.

MI308X TP2, `launch_qwen3.5-35B-fp8_tp2_disable_prefix_cache.sh`, two interleaved rounds against the launcher
before this change (which also lacked the FlyDSL GDN decode; the middle column adds only that):

| Workload | req/s before | + GDN decode | + SDMA all-reduce | TTFT | TPOT |
|---|---|---|---|---|---|
| 12000 in / 500 out, c8 | 1.211 | 1.281 | 1.399 (+15.5 %) | -9.1 % | -15.2 % |
| 12000 in / 500 out, c16 | 1.660 | 1.702 | 1.946 (+17.2 %) | -13.4 % | -15.3 % |
| 12000 in / 500 out, c32 | 2.000 | 2.062 | 2.382 (+19.1 %) | -16.4 % | -15.8 % |
| 8000 in / 500 out, c8 x 256 (CI) | 1.425 | 1.477 | 1.602 (+12.4 %) | +0.8 % | -14.5 % |

One full-attention layer of a 32k-token prefill (o_proj to the next GDN core) takes 11.6 ms instead of 16.1 ms:
5.9 ms of RCCL all-reduce leave the critical path, 0.4 ms of SDMA wait remain.

### mori dependency

Everything except mori and the ibverbs headers comes with the ROCm 7.14 image; keep the image's versions.

| Component | Version | Source | Why |
|---|---|---|---|
| ROCm (incl. `rocm_sysdeps` libdrm) | 7.14.0 | image | HIP runtime; the libdrm copy mori must share with it |
| torch | 2.12.0+rocm7.14.0 | image | |
| FlyDSL | 0.3.2 | image | mori's device API compiles the push / wait kernels with it |
| libibverbs-dev | distro package | **install** | build time only: mori's `setup.py` stops without the ibverbs headers |
| mori | ROCm/mori `de97cea` | **install** | SDMA queues and engine-side completion signals |
| PyHIP | `8027faa` | base branch | required by the base branch's QSA path, not by this feature |

mori is pinned in `python/pyproject_other.toml` (`srt_hip`, next to PyHIP) as
`amd_mori @ git+https://github.com/ROCm/mori.git@de97ceacefa96a6ba45b5ed3276abdb4a3bf7fd6`. That upstream
commit includes the MI308X per-link SDMA engine balancing (ROCm/mori#921; without it, two of a GPU's xGMI links
can share one SDMA engine at TP4).

Install:

```bash
# 1. ibverbs headers: not in the ROCm base image, needed only to build mori.
apt-get update && apt-get install -y libibverbs-dev       # RHEL/Fedora: dnf install -y rdma-core-devel
# 2. mori: pulled in by the srt_hip extras, or on its own into the current environment
#    (add --target DIR and put DIR on PYTHONPATH to keep it separate). Do not use the
#    [flydsl] extra: it is unpinned and would replace the image's FlyDSL.
pip install "amd_mori @ git+https://github.com/ROCm/mori.git@de97ceacefa96a6ba45b5ed3276abdb4a3bf7fd6"
```

The pip build uses mori's defaults (`BUILD_CCO_SDMA=ON`, local GPU arch); the built libraries do not link
libibverbs, so the package is not needed at run time. Runtime requirements, all set by the MI308X launch script:

- `MORI_ENABLE_SDMA=1`, `MORI_SOCKET_IFNAME=lo`.
- `LD_PRELOAD` of ROCm's `rocm_sysdeps/lib/librocm_sysdeps_drm{,_amdgpu}`: mori links the system libdrm,
  the HIP runtime its private copy, and two copies cannot share the GPU VM (mori's KFD setup would exit
  the process; sglang checks this and falls back before touching mori).

Check, with the launch script's libdrm preload (without it, importing mori next to FlyDSL can crash):

```bash
LD_PRELOAD=/opt/rocm/lib/rocm_sysdeps/lib/librocm_sysdeps_drm.so.2:/opt/rocm/lib/rocm_sysdeps/lib/librocm_sysdeps_drm_amdgpu.so.1 \
  python -c "import mori.cco, flydsl.compiler"
```

and look for `push_one_0` / `push_all_0` kernels in a prefill trace.

| Script | What it measures |
|---|---|
| `bench_sdma_rs.py` | `SdmaReduceScatter` vs an fp32 reference (bf16 / fp8, uneven T), and GEMM + RS + combine vs aiter |
