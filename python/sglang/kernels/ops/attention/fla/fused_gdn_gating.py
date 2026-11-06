from typing import Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs


@triton.jit(do_not_specialize=["rows"])
def _gfx942_gdn_gating(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    rows,
    stride_a: tl.constexpr,
    stride_b: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, head = index // 24, index % 24
    mask = row < rows
    alog = tl.load(A_log + head).to(tl.float32)
    aa = tl.load(a + row * stride_a + head, mask, other=0).to(tl.float32)
    bb = tl.load(b + row * stride_b + head, mask, other=0).to(tl.float32)
    dt = tl.load(dt_bias + head).to(tl.float32)
    x = aa + dt
    softplus = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    value = -tl.exp(alog) * softplus
    tl.store(g + index, value, mask)
    # Preserve the existing BF16 sigmoid boundary before storing FP32 beta.
    tl.store(
        beta_output + index, tl.sigmoid(bb).to(b.dtype.element_ty).to(tl.float32), mask
    )


# g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
# beta_output = b.sigmoid()
@triton.jit
def fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    stride_a,
    stride_b,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + i_b * stride_a + head_off, mask=mask)
    blk_b = tl.load(b + i_b * stride_b + head_off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(beta_output + off, blk_beta_output.to(b.dtype.element_ty), mask=mask)


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, num_heads = a.shape
    seq_len = 1
    stride_a = a.stride(0)
    stride_b = b.stride(0)
    grid = (batch, seq_len, triton.cdiv(num_heads, 8))
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, batch, num_heads, dtype=torch.float32, device=b.device)
    if (
        torch.version.hip is not None
        and envs.SGLANG_GFX942_GDN_GATING.get()
        and num_heads == 24
        and a.stride(1) == b.stride(1) == 1
        and torch.cuda.get_device_properties(a.device).gcnArchName.startswith("gfx942")
    ):
        _gfx942_gdn_gating[(triton.cdiv(batch * 24, 256),)](
            g,
            beta_output,
            A_log,
            a,
            b,
            dt_bias,
            batch,
            stride_a,
            stride_b,
            beta=beta,
            threshold=threshold,
            BLOCK=256,
            num_warps=1,
        )
        return g, beta_output
    fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        stride_a,
        stride_b,
        num_heads,
        beta,
        threshold,
        8,
        num_warps=1,
    )
    return g, beta_output
