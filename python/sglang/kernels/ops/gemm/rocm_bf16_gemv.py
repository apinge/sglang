"""Skinny bf16 GEMM (``y = x @ w.T``) for weight-streaming shapes on gfx942."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_MAX_M = 32
_K_ALIGN = 512  # every config's BLOCK_K divides this; kernels skip K masking


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
        and w.shape[1] % _K_ALIGN == 0
        and x.stride(1) == 1
        and w.stride(1) == 1
    )


def rocm_bf16_gemv(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """y[M, N] = x[M, K] @ w[N, K]^T in bf16 with fp32 accumulation."""
    m, k = x.shape
    n = w.shape[0]
    y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if m == 1:
        block_n = 32
        _gemv_m1_kernel[(triton.cdiv(n, block_n),)](
            x,
            w,
            y,
            n,
            k,
            w.stride(0),
            BLOCK_N=block_n,
            BLOCK_K=512,
            num_warps=4,
        )
        return y
    # (BLOCK_M, BLOCK_N): the 16-row tile is fastest up to M=16; past that a
    # 32-row tile needs a narrower N tile to stay within registers.
    block_m, block_n = (16, 128) if m <= 16 else (32, 64)
    _skinny_mfma_kernel[(triton.cdiv(n, block_n),)](
        x,
        w,
        y,
        m,
        n,
        k,
        x.stride(0),
        w.stride(0),
        y.stride(0),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=256,
        num_warps=2,
        num_stages=1,
        matrix_instr_nonkdim=16,
        kpack=2,
        waves_per_eu=2,
    )
    return y
