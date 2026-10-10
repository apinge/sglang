"""Fused HC low-rank mix for decode-size batches.

`GatedResidual._mix_compute` lowers to a five-kernel chain per call
(down GEMV + splitK reduce, silu, up GEMV, sigmoid-mul-mean); at bs=1
speculative decode that chain runs ~100 times per iteration on the GPU
critical path between allreduces.

Three ordinary launches replace the chain.  ``out_j`` needs the whole
low-rank vector, so the down and up projections cannot share a kernel
without a device-wide dependency; the split expresses that dependency
with launch boundaries instead of a software grid barrier, which frees
the grid to be sized for bandwidth rather than pinned to one CTA per CU:

* down  — one CTA per (n-block, k-chunk) tile of ``x @ W_down^T``,
  writing per-chunk fp32 partials (no atomics, no accumulator zeroing)
* reduce — sums the partials and applies ``silu(t_raw / hc)``
* up    — one CTA per output block:
  ``out_j = mean_g(sigmoid(t @ W_up[g,j]^T) * x[g,j])``

Row counts beyond ``_FUSED_MIX_MAX_ROWS`` (prefill) keep the
torch.compile path, which uses proper GEMM kernels.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_FUSED_MIX_MAX_ROWS = 32

_DEFAULT_DOWN_CONFIG = dict(BLOCK_N=32, BLOCK_K=256, num_warps=4)
_DEFAULT_REDUCE_CONFIG = dict(BLOCK_R=64, num_warps=4)
_DEFAULT_UP_CONFIG = dict(BLOCK_J=32, BLOCK_R=64, num_warps=4)

# Tuned on an 80-CU MI308X against Qwen3.8's BF16 HC weights (hc=4, hs=2560,
# lowrank=320, k=10240); re-tune when the shape changes. num_warps=1 (one
# wave64) wins on both projections: the grids are already 200-320 CTAs, so
# extra warps only add per-CTA scheduling without adding memory parallelism.
_GFX942_MFMA_CONFIG = dict(
    num_stages=1,
    waves_per_eu=1,
    matrix_instr_nonkdim=16,
    kpack=2,
    WEIGHT_CACHE_MODIFIER=".cg",
)
_GFX942_DOWN_CONFIG = dict(BLOCK_N=32, BLOCK_K=512, num_warps=1, **_GFX942_MFMA_CONFIG)
# The 32-row tile doubles the MFMA work per CTA; one wave stops covering it.
_GFX942_DOWN_CONFIG_WIDE = dict(
    BLOCK_N=32, BLOCK_K=512, num_warps=2, **_GFX942_MFMA_CONFIG
)
_GFX942_REDUCE_CONFIG = dict(BLOCK_R=32, num_warps=2, num_stages=1)
_GFX942_UP_CONFIG = dict(BLOCK_J=8, BLOCK_R=64, num_warps=1, **_GFX942_MFMA_CONFIG)


@triton.jit
def _hc_mix_down_kernel(
    x_ptr,
    w_down_ptr,
    partials_ptr,
    K,
    LOWRANK,
    num_rows,
    ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    WEIGHT_CACHE_MODIFIER: tl.constexpr = "",
):
    n_blocks = tl.cdiv(LOWRANK, BLOCK_N)
    pid = tl.program_id(0)
    nb = pid % n_blocks
    kc = pid // n_blocks

    offs_m = tl.arange(0, ROWS)
    mask_m = offs_m < num_rows
    n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n < LOWRANK
    k = kc * BLOCK_K + tl.arange(0, BLOCK_K)

    xt = tl.load(
        x_ptr + offs_m[:, None] * K + k[None, :],
        mask=mask_m[:, None],
        other=0.0,
    )
    w = tl.load(
        w_down_ptr + n[:, None] * K + k[None, :],
        mask=mask_n[:, None],
        other=0.0,
        cache_modifier=WEIGHT_CACHE_MODIFIER,
    )
    acc = tl.dot(xt, tl.trans(w))
    tl.store(
        partials_ptr + kc * (ROWS * LOWRANK) + offs_m[:, None] * LOWRANK + n[None, :],
        acc,
        mask=mask_n[None, :],
    )


@triton.jit
def _hc_mix_reduce_kernel(
    partials_ptr,
    t_ptr,
    LOWRANK,
    inv_hc,
    ROWS: tl.constexpr,
    KSPLIT: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    offs_m = tl.arange(0, ROWS)
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    mask_r = r < LOWRANK
    idx = offs_m[:, None] * LOWRANK + r[None, :]

    acc = tl.zeros((ROWS, BLOCK_R), dtype=tl.float32)
    for s in tl.static_range(KSPLIT):
        acc += tl.load(
            partials_ptr + s * (ROWS * LOWRANK) + idx, mask=mask_r[None, :], other=0.0
        )
    a = acc * inv_hc
    t = a * tl.sigmoid(a)
    tl.store(t_ptr + idx, t.to(t_ptr.dtype.element_ty), mask=mask_r[None, :])


@triton.jit
def _hc_mix_up_kernel(
    x_ptr,
    w_up_ptr,
    t_ptr,
    out_ptr,
    LOWRANK,
    HS,
    num_rows,
    inv_hc,
    ROWS: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_R: tl.constexpr,
    WEIGHT_CACHE_MODIFIER: tl.constexpr = "",
):
    offs_m = tl.arange(0, ROWS)
    mask_m = offs_m < num_rows
    offs_g = tl.arange(0, HC)
    j = tl.program_id(0) * BLOCK_J + tl.arange(0, BLOCK_J)
    mask_j = j < HS

    gj_flat = tl.reshape(offs_g[:, None] * HS + j[None, :], (HC * BLOCK_J,))
    mask_gj = tl.reshape(
        tl.broadcast_to(mask_j[None, :], (HC, BLOCK_J)), (HC * BLOCK_J,)
    )

    acc = tl.zeros((ROWS, HC * BLOCK_J), dtype=tl.float32)
    for r0 in range(0, LOWRANK, BLOCK_R):
        r = r0 + tl.arange(0, BLOCK_R)
        mask_r = r < LOWRANK
        t = tl.load(
            t_ptr + offs_m[:, None] * LOWRANK + r[None, :],
            mask=mask_r[None, :],
            other=0.0,
        )
        w = tl.load(
            w_up_ptr + gj_flat[:, None] * LOWRANK + r[None, :],
            mask=mask_gj[:, None] & mask_r[None, :],
            other=0.0,
            cache_modifier=WEIGHT_CACHE_MODIFIER,
        )
        acc = tl.dot(t, tl.trans(w), acc)

    gate = tl.sigmoid(tl.reshape(acc, (ROWS, HC, BLOCK_J)))
    xg = tl.load(
        x_ptr
        + offs_m[:, None, None] * (HC * HS)
        + offs_g[None, :, None] * HS
        + j[None, None, :],
        mask=mask_m[:, None, None] & mask_j[None, None, :],
        other=0.0,
    ).to(tl.float32)
    out = tl.sum(gate * xg, axis=1) * inv_hc
    tl.store(
        out_ptr + offs_m[:, None] * HS + j[None, :],
        out.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_j[None, :],
    )


_deterministic_inference_cached = None


def _deterministic_inference() -> bool:
    global _deterministic_inference_cached
    if _deterministic_inference_cached is None:
        try:
            from sglang.srt.server_args import get_global_server_args

            _deterministic_inference_cached = bool(
                get_global_server_args().enable_deterministic_inference
            )
        except Exception:
            _deterministic_inference_cached = False
    return _deterministic_inference_cached


def fused_hc_mix_supported(
    hyper_input_normed: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor
) -> bool:
    # Reproducible run to run, but the launch config is picked from the row
    # count, so the summation order is not batch-invariant.
    if _deterministic_inference():
        return False
    return (
        hyper_input_normed.is_cuda
        and hyper_input_normed.dtype in (torch.bfloat16, torch.float16)
        and w_down.dtype == hyper_input_normed.dtype
        and w_up.dtype == hyper_input_normed.dtype
        and hyper_input_normed.shape[0] <= _FUSED_MIX_MAX_ROWS
        and hyper_input_normed.dim() == 2
        and hyper_input_normed.shape[1] % 2048 == 0
        and hyper_input_normed.is_contiguous()
        and w_down.is_contiguous()
        and w_up.is_contiguous()
    )


def _select_configs(
    *,
    props,
    rows_pad: int,
    dtype: torch.dtype,
    hc: int,
    hs: int,
    lowrank: int,
    k: int,
):
    if (
        torch.version.hip is not None
        and props.gcnArchName.split(":", 1)[0] == "gfx942"
        and dtype == torch.bfloat16
        and (hc, hs, lowrank, k) == (4, 2560, 320, 10240)
    ):
        down = _GFX942_DOWN_CONFIG if rows_pad <= 16 else _GFX942_DOWN_CONFIG_WIDE
        return down, _GFX942_REDUCE_CONFIG, _GFX942_UP_CONFIG
    return _DEFAULT_DOWN_CONFIG, _DEFAULT_REDUCE_CONFIG, _DEFAULT_UP_CONFIG


def fused_hc_mix(
    hyper_input_normed: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    hc: int,
    hs: int,
) -> torch.Tensor:
    rows, k = hyper_input_normed.shape
    lowrank = w_down.shape[0]
    device = hyper_input_normed.device
    dtype = hyper_input_normed.dtype
    out = torch.empty((rows, hs), dtype=dtype, device=device)
    if rows == 0:
        return out

    # tl.dot needs a 16-row operand; below that the MFMA tile is padded anyway.
    rows_pad = max(16, triton.next_power_of_2(rows))
    down_cfg, reduce_cfg, up_cfg = _select_configs(
        props=torch.cuda.get_device_properties(device),
        rows_pad=rows_pad,
        dtype=dtype,
        hc=hc,
        hs=hs,
        lowrank=lowrank,
        k=k,
    )
    k_chunks = triton.cdiv(k, down_cfg["BLOCK_K"])

    partials = torch.empty(
        (k_chunks, rows_pad, lowrank), dtype=torch.float32, device=device
    )
    t = torch.empty((rows_pad, lowrank), dtype=dtype, device=device)

    n_blocks = triton.cdiv(lowrank, down_cfg["BLOCK_N"])
    _hc_mix_down_kernel[(n_blocks * k_chunks,)](
        hyper_input_normed,
        w_down,
        partials,
        k,
        lowrank,
        rows,
        ROWS=rows_pad,
        **down_cfg,
    )
    _hc_mix_reduce_kernel[(triton.cdiv(lowrank, reduce_cfg["BLOCK_R"]),)](
        partials,
        t,
        lowrank,
        1.0 / hc,
        ROWS=rows_pad,
        KSPLIT=k_chunks,
        **reduce_cfg,
    )
    _hc_mix_up_kernel[(triton.cdiv(hs, up_cfg["BLOCK_J"]),)](
        hyper_input_normed,
        w_up,
        t,
        out,
        lowrank,
        hs,
        rows,
        1.0 / hc,
        ROWS=rows_pad,
        HC=hc,
        **up_cfg,
    )
    return out
