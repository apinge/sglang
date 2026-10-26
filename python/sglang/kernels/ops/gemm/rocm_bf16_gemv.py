"""Skinny bf16 GEMM (``y = x @ w.T``) for weight-streaming shapes on gfx942."""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

_MAX_M = 32


@triton.jit
def _gemv_m1_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    N,
    K,
    stride_wn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CACHE_MODIFIER: tl.constexpr,
):
    pid = tl.program_id(0)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rn_mask = rn < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        w = tl.load(
            w_ptr + rn[:, None] * stride_wn + rk[None, :],
            mask=rn_mask[:, None],
            other=0.0,
            cache_modifier=CACHE_MODIFIER,
        )
        x = tl.load(x_ptr + rk)
        acc += tl.sum(w.to(tl.float32) * x.to(tl.float32)[None, :], axis=1)
    tl.store(y_ptr + rn, acc.to(y_ptr.dtype.element_ty), mask=rn_mask)


@triton.jit
def _skinny_mfma_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_wn,
    stride_ym,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CACHE_MODIFIER: tl.constexpr,
):
    pid = tl.program_id(0)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = tl.arange(0, BLOCK_M)
    rn_mask = rn < N
    rm_mask = rm < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        # Load W row-major ([BLOCK_N, BLOCK_K], contiguous along K) and
        # transpose in registers; loading it pre-transposed halves bandwidth.
        w = tl.load(
            w_ptr + rn[:, None] * stride_wn + rk[None, :],
            mask=rn_mask[:, None],
            other=0.0,
            cache_modifier=CACHE_MODIFIER,
        )
        x = tl.load(
            x_ptr + rm[:, None] * stride_xm + rk[None, :],
            mask=rm_mask[:, None],
            other=0.0,
        )
        acc += tl.dot(x, tl.trans(w))
    tl.store(
        y_ptr + rm[:, None] * stride_ym + rn[None, :],
        acc.to(y_ptr.dtype.element_ty),
        mask=rm_mask[:, None] & rn_mask[None, :],
    )


def _gemv(block_n, block_k, num_warps, num_stages, waves_per_eu, cache_modifier):
    return dict(
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        waves_per_eu=waves_per_eu,
        cache_modifier=cache_modifier,
    )


def _mfma(
    block_m, block_n, block_k, num_warps, num_stages, waves_per_eu, cache_modifier
):
    return dict(
        _gemv(block_n, block_k, num_warps, num_stages, waves_per_eu, cache_modifier),
        BLOCK_SIZE_M=block_m,
        matrix_instr_nonkdim=16,
        kpack=2,
    )


# {(N, K): {M upper bound: config}}. Only listed shapes use these kernels; untuned
# shapes measured slower than hipBLASLt for some M (N <= 24576, or K=5120).
# Tuned on MI308X with benchmark/kernels/rocm_bf16_gemv/tune_rocm_bf16_gemv.py
# for the Qwen3.8-Flash-Next lm_head shards (TP1/2/4/8).
_GFX942_CONFIGS = {
    (248320, 2560): {
        1: _gemv(8, 512, 1, 1, 0, ".cg"),
        2: _mfma(16, 64, 256, 1, 1, 2, ".cg"),
        4: _mfma(16, 64, 256, 1, 1, 2, ".cg"),
        8: _mfma(16, 64, 256, 1, 1, 2, ".cg"),
        16: _mfma(16, 64, 256, 1, 1, 2, ".cg"),
        24: _mfma(32, 64, 256, 2, 1, 0, ".cg"),
        32: _mfma(32, 64, 256, 2, 1, 2, ".cg"),
    },
    (124160, 2560): {
        1: _gemv(32, 512, 8, 1, 1, ".cg"),
        2: _mfma(16, 64, 256, 1, 1, 2, ".cg"),
        4: _mfma(16, 64, 256, 1, 1, 2, ".cg"),
        8: _mfma(16, 64, 256, 1, 1, 2, ".cg"),
        16: _mfma(16, 64, 256, 1, 1, 2, ".cg"),
        24: _mfma(32, 64, 256, 2, 1, 0, ".cg"),
        32: _mfma(32, 64, 256, 2, 1, 3, ".cg"),
    },
    (62080, 2560): {
        1: _gemv(16, 512, 4, 1, 0, ".cg"),
        2: _mfma(16, 64, 256, 1, 1, 2, ".cg"),
        4: _mfma(16, 64, 256, 1, 1, 2, ".cg"),
        8: _mfma(16, 64, 256, 1, 1, 2, None),
        16: _mfma(16, 64, 256, 1, 1, 2, None),
        24: _mfma(32, 64, 256, 2, 1, 0, ".cg"),
        32: _mfma(32, 64, 256, 2, 1, 0, ".cg"),
    },
    (31040, 2560): {
        1: _gemv(8, 512, 1, 1, 0, None),
        2: _mfma(16, 64, 256, 1, 1, 0, ".cg"),
        4: _mfma(16, 32, 256, 1, 1, 2, ".cg"),
        8: _mfma(16, 64, 256, 2, 1, 0, ".cg"),
        16: _mfma(16, 64, 256, 1, 1, 0, ".cg"),
        24: _mfma(32, 64, 128, 2, 1, 4, ".cg"),
        32: _mfma(32, 128, 256, 4, 1, 3, ".cg"),
    },
}


@functools.lru_cache(maxsize=None)
def _get_config(m: int, n: int, k: int) -> dict:
    buckets = _GFX942_CONFIGS[(n, k)]
    cfg = buckets[min(b for b in buckets if m <= b)]
    assert k % cfg["BLOCK_SIZE_K"] == 0, (m, n, k, cfg)  # K loop is unmasked
    return cfg


def use_rocm_bf16_gemv(x: torch.Tensor, w: torch.Tensor) -> bool:
    """Whether ``rocm_bf16_gemv(x, w)`` applies; callers gate on gfx942."""
    return (
        x.dim() == 2
        and w.dim() == 2
        and x.dtype == torch.bfloat16
        and w.dtype == torch.bfloat16
        and x.is_cuda
        and 1 <= x.shape[0] <= _MAX_M
        and x.shape[1] == w.shape[1]
        and (w.shape[0], w.shape[1]) in _GFX942_CONFIGS
        and x.stride(1) == 1
        and w.stride(1) == 1
    )


def _launch(x: torch.Tensor, w: torch.Tensor, cfg: dict) -> torch.Tensor:
    m, k = x.shape
    n = w.shape[0]
    y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(n, cfg["BLOCK_SIZE_N"]),)
    launch_opts = dict(
        BLOCK_N=cfg["BLOCK_SIZE_N"],
        BLOCK_K=cfg["BLOCK_SIZE_K"],
        CACHE_MODIFIER=cfg["cache_modifier"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
        waves_per_eu=cfg["waves_per_eu"],
    )
    if m == 1:
        _gemv_m1_kernel[grid](x, w, y, n, k, w.stride(0), **launch_opts)
    else:
        _skinny_mfma_kernel[grid](
            x,
            w,
            y,
            m,
            n,
            k,
            x.stride(0),
            w.stride(0),
            y.stride(0),
            BLOCK_M=cfg["BLOCK_SIZE_M"],
            matrix_instr_nonkdim=cfg["matrix_instr_nonkdim"],
            kpack=cfg["kpack"],
            **launch_opts,
        )
    return y


def rocm_bf16_gemv(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """y[M, N] = x[M, K] @ w[N, K]^T in bf16 with fp32 accumulation."""
    m, k = x.shape
    cfg = _get_config(m, w.shape[0], k)
    return _launch(x, w, cfg)
