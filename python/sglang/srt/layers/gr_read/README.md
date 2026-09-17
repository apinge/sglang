# ROCm Qwen GR read

This package integrates the final pyhip `down_wave_splitk_pipeline + up_gate`
implementation for gfx942, BF16 HC weights and `C=4, H=2560, R=320`.
The kernel source and configuration selectors come from pyhip `d3510c6`
(algorithm `a716010`); the production package has no imports from pyhip or
its tests. FlyDSL 0.3.1 and AITER are runtime dependencies.

## Loading and dispatch

`GatedResidual` installs `GRReadMethod` as its loader `quant_method`.
`process_weights_after_loading` runs after original checkpoint loading and
replaces WD/WU parameter storage with AITER `shuffle_weight(..., (16,16))`
storage. WU first reorders output rows from `c*H+h` to `h*C+c`.
There is one persistent weight layout, marked
`gr_read_bf16_hc_interleaved_preshuffle_v1`; the original storage is released.

`GatedResidual.mix` keeps its existing normalization and residual tuple.
The normalized input has physical shape `[B,10240]`:

- B=0 retains the existing empty-input return.
- B=1..32 launches the two FlyDSL kernels with the source configuration selectors.
- B>32 calls `compiled_mix_packed`, which decodes the logical weight layout
  inside the original Torch computation. Decoded weights are call-local
  temporaries and must not be cached as a second persistent weight set.

Dispatch uses tensor rows, including graph dummy rows, rather than request
count or a prefill/decode flag. TP2/TP4 do not shard these HC dimensions.

## Preparation and lifetime

`ModelRunner.maybe_precompile_model_kernels_after_loading` prepares all small
row counts for the initialization and forward streams, before KV sizing.
`DecodeCudaGraphRunner._capture_one_stream` prepares the actual capture token
counts on its capture stream before any graph is recorded.

Each module/B/stream owns independent FP32 P and BF16 Y. Read-only packed
weights and compiled launch code are shared. Preparation is explicit;
forward raises for an unprepared small B/stream and never silently compiles
or allocates a workspace during capture. Stream references, packed weights,
handles and workspaces remain alive with the module.

For B<=16, P has `4*16*320` elements; for B17..32 it has `4*32*320`.
The kernel overwrites the whole P on every call. The 324-wide hidden LDS
layout does not change the 320-wide global P stride. Y always has B rows;
live-token slicing belongs to the model runner. Dummy input rows need not
be zero and cannot affect other rows.

Small-B output aliases the prepared Y. Standalone callers retaining an
output across another call on the same module/B/stream must copy it, and
cross-stream consumers must obey their own event/wait lifetime ordering.
SGLang serializes Python forwards per worker; compiled launch argument
storage is not a host-thread concurrency API.

## Configuration and validation

Set `SGLANG_USE_AITER=1`. `SGLANG_GR_READ_FLYDSL=1` is the default for this
shape on ROCm; set it to 0 before constructing a model to run the original
unpacked baseline. The layout cannot be switched on a live model.
The supported integration configuration is ordinary TP2/TP4 decode on gfx942.

```bash
# Integrated packed-weight/FlyDSL path (default).
SGLANG_GR_READ_FLYDSL=1 TP_SIZE=2 bash /opt/launch_qwen38_flash_next_fp8_mi308x_2.sh
# Original weight layout and computation, for comparison or rollback.
SGLANG_GR_READ_FLYDSL=0 TP_SIZE=2 bash /opt/launch_qwen38_flash_next_fp8_mi308x_2.sh
```

Set the switch before starting the server and restart to change it. Both
settings retain `SGLANG_USE_AITER=1`; the switch controls the whole GR read
integration, including load-time packing and the packed Torch fallback.

Online weight updates and post-pack device/dtype moves require a fresh model
and graphs and are rejected. Source checkpoints remain in original layout;
exporting packed serving parameters as an ordinary source checkpoint is not
supported.

`test/registered/kernels/test_gr_read_rocm.py` supplies three standalone suites:

```bash
export CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 SGLANG_USE_AITER=1
python test/registered/kernels/test_gr_read_rocm.py --suite torch --weights 100 --output /tmp/gr_torch_new.jsonl
python test/registered/kernels/test_gr_read_rocm.py --suite bucket --synthetic --weights 2 --output /tmp/gr_synthetic_new.jsonl
python test/registered/kernels/test_gr_read_rocm.py --suite bucket --weights 100 --tails stale --output /tmp/gr_checkpoint_new.jsonl
python test/registered/kernels/test_gr_read_rocm.py --suite lifecycle --synthetic --weights 1 --output /tmp/gr_lifecycle_new.jsonl
```

Output paths must be new. Torch equivalence is checked directly against the
original compiled math, including bitwise output equality. Kernel tests use
an independent FP64 reference, all B1..32 and every live M, guard storage,
poisoned P/Y and changing dummy rows. Lifecycle tests cover one-time packing,
residual preservation, input rejection and simultaneous replay on separate
streams. Whole-service accuracy and performance are separate acceptance steps.
