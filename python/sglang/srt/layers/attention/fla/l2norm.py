# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/l2norm.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl

from sglang.srt.layers.attention.fla.utils import input_guard

BT_LIST = [8, 16, 32, 64, 128]


@triton.jit
def fused_l2norm_qk_kernel(
    q,
    k,
    q_out,
    k_out,
    eps,
    T: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0)
    rows = i_t * BT + tl.arange(0, BT)
    cols = tl.arange(0, BD)
    mask = (rows[:, None] < T) & (cols[None, :] < D)
    offsets = rows[:, None] * D + cols[None, :]

    q_block = tl.load(q + offsets, mask=mask, other=0.0).to(tl.float32)
    q_block = q_block / tl.sqrt(tl.sum(q_block * q_block, axis=1) + eps)[:, None]
    tl.store(q_out + offsets, q_block.to(q_out.dtype.element_ty), mask=mask)

    k_block = tl.load(k + offsets, mask=mask, other=0.0).to(tl.float32)
    k_block = k_block / tl.sqrt(tl.sum(k_block * k_block, axis=1) + eps)[:, None]
    tl.store(k_out + offsets, k_block.to(k_out.dtype.element_ty), mask=mask)


@triton.jit
def fused_l2norm_qk_kernel1(q, k, q_out, k_out, D, BD: tl.constexpr, eps):
    row = tl.program_id(0)
    cols = tl.arange(0, BD)
    mask = cols < D
    offsets = row * D + cols

    q_block = tl.load(q + offsets, mask=mask, other=0.0).to(tl.float32)
    q_block = q_block / tl.sqrt(tl.sum(q_block * q_block, axis=0) + eps)
    tl.store(q_out + offsets, q_block.to(q_out.dtype.element_ty), mask=mask)

    k_block = tl.load(k + offsets, mask=mask, other=0.0).to(tl.float32)
    k_block = k_block / tl.sqrt(tl.sum(k_block * k_block, axis=0) + eps)
    tl.store(k_out + offsets, k_block.to(k_out.dtype.element_ty), mask=mask)


def fused_l2norm_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_shape, k_shape = q.shape, k.shape
    q_flat = q.view(-1, q.shape[-1])
    k_flat = k.view(-1, k.shape[-1])
    if q_flat.shape != k_flat.shape:
        raise ValueError(
            f"Fused Q/K L2Norm requires matching shapes, got {q_shape} and {k_shape}"
        )

    if output_dtype is None:
        q_out, k_out = torch.empty_like(q_flat), torch.empty_like(k_flat)
    else:
        q_out = torch.empty_like(q_flat, dtype=output_dtype)
        k_out = torch.empty_like(k_flat, dtype=output_dtype)

    tokens, dim = q_flat.shape
    max_fused_size = 65536 // q.element_size()
    block_dim = min(max_fused_size, triton.next_power_of_2(dim))
    if dim > block_dim:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    if dim <= 512:
        block_tokens = 16
        fused_l2norm_qk_kernel[(triton.cdiv(tokens, block_tokens),)](
            q_flat,
            k_flat,
            q_out,
            k_out,
            eps,
            T=tokens,
            D=dim,
            BT=block_tokens,
            BD=block_dim,
            num_warps=8,
            num_stages=3,
        )
    else:
        fused_l2norm_qk_kernel1[(tokens,)](
            q_flat,
            k_flat,
            q_out,
            k_out,
            D=dim,
            BD=block_dim,
            eps=eps,
            num_warps=8,
            num_stages=3,
        )

    return q_out.view(q_shape), k_out.view(k_shape)


# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8, 16, 32]
#     ],
#     key=["D"],
# )
@triton.jit
def l2norm_fwd_kernel1(
    x,
    y,
    D,
    BD: tl.constexpr,
    eps,
):
    i_t = tl.program_id(0)
    x += i_t * D
    y += i_t * D
    # Compute mean and variance
    cols = tl.arange(0, BD)
    mask = cols < D
    b_x = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=0)
    b_rstd = 1 / tl.sqrt(b_var + eps)
    # tl.store(Rstd + i_t, rstd)
    # Normalize and apply linear transformation
    b_y = b_x * b_rstd
    tl.store(y + cols, b_y, mask=mask)


# @triton.autotune(
#     configs=[
#         triton.Config({"BT": BT}, num_warps=num_warps)
#         for num_warps in [1, 2, 4, 8, 16]
#         for BT in BT_LIST
#     ],
#     key=["D", "NB"],
# )
@triton.jit
def l2norm_fwd_kernel(
    x,
    y,
    eps,
    NB: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0)
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=1)
    b_y = b_x / tl.sqrt(b_var + eps)[:, None]
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


def l2norm_fwd(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None
):
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])
    # allocate output
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]
    # rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    if D <= 512:
        NB = triton.cdiv(T, 2048)

        def grid(meta):
            return (triton.cdiv(T, meta["BT"]),)

        l2norm_fwd_kernel[grid](
            x,
            y,
            eps,
            NB=NB,
            T=T,
            D=D,
            BD=BD,
            BT=16,
            num_warps=8,
            num_stages=3,
        )
    else:
        l2norm_fwd_kernel1[(T,)](
            x,
            y,
            eps=eps,
            D=D,
            BD=BD,
            num_warps=8,
            num_stages=3,
        )

    return y.view(x_shape_og)


class L2NormFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(ctx, x, eps=1e-6, output_dtype=None):
        return l2norm_fwd(x, eps, output_dtype)


def l2norm(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    return L2NormFunction.apply(x, eps, output_dtype)


l2_norm = l2norm


class L2Norm(nn.Module):
    def __init__(self, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.eps = eps
        self.output_dtype = output_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2norm(x, self.eps, self.output_dtype)
