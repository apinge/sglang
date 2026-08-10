from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_fp8_set_kv_buffer_kernel(
    k_ptr,
    v_ptr,
    k_cache_ptr,
    v_cache_ptr,
    cache_loc_ptr,
    k_scale_ptr,
    v_scale_ptr,
    k_stride_t: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_d: tl.constexpr,
    v_stride_t: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_d: tl.constexpr,
    k_cache_stride_t: tl.constexpr,
    k_cache_stride_h: tl.constexpr,
    k_cache_stride_d: tl.constexpr,
    v_cache_stride_t: tl.constexpr,
    v_cache_stride_h: tl.constexpr,
    v_cache_stride_d: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    K_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    HAS_K_SCALE: tl.constexpr,
    HAS_V_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_id = tl.program_id(1)
    kv_id = tl.program_id(2)
    dim_offsets = tl.arange(0, BLOCK_D)
    cache_loc = tl.load(cache_loc_ptr + token_id).to(tl.int64)

    if cache_loc >= 0:
        if kv_id == 0:
            if head_id < NUM_K_HEADS:
                mask = dim_offsets < K_HEAD_DIM
                values = tl.load(
                    k_ptr
                    + token_id * k_stride_t
                    + head_id * k_stride_h
                    + dim_offsets * k_stride_d,
                    mask=mask,
                    other=0.0,
                )
                if HAS_K_SCALE:
                    values = values.to(tl.float32) / tl.load(k_scale_ptr)
                tl.store(
                    k_cache_ptr
                    + cache_loc * k_cache_stride_t
                    + head_id * k_cache_stride_h
                    + dim_offsets * k_cache_stride_d,
                    values.to(k_cache_ptr.dtype.element_ty),
                    mask=mask,
                )
        else:
            if head_id < NUM_V_HEADS:
                mask = dim_offsets < V_HEAD_DIM
                values = tl.load(
                    v_ptr
                    + token_id * v_stride_t
                    + head_id * v_stride_h
                    + dim_offsets * v_stride_d,
                    mask=mask,
                    other=0.0,
                )
                if HAS_V_SCALE:
                    values = values.to(tl.float32) / tl.load(v_scale_ptr)
                tl.store(
                    v_cache_ptr
                    + cache_loc * v_cache_stride_t
                    + head_id * v_cache_stride_h
                    + dim_offsets * v_cache_stride_d,
                    values.to(v_cache_ptr.dtype.element_ty),
                    mask=mask,
                )


def _scale_ptr(
    scale: Optional[torch.Tensor], ref: torch.Tensor
) -> tuple[torch.Tensor, bool]:
    if scale is None:
        return ref, False
    if isinstance(scale, torch.Tensor):
        if scale.numel() != 1:
            raise ValueError(
                "AITER fused FP8 KV write expects k_scale/v_scale to be scalar "
                "tensors (numel == 1)"
            )
        return scale.to(device=ref.device, dtype=torch.float32), True
    if float(scale) == 1.0:
        return ref, False
    return torch.tensor([float(scale)], dtype=torch.float32, device=ref.device), True


def fused_fp8_set_kv_buffer(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_loc: torch.Tensor,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
) -> None:
    if k_cache.ndim != 3 or v_cache.ndim != 3:
        raise ValueError("AITER fused FP8 KV write expects 3D K/V cache tensors")
    if k.ndim != 3 or v.ndim != 3:
        raise ValueError("AITER fused FP8 KV write expects 3D K/V input tensors")
    if cache_loc.numel() == 0:
        return

    num_tokens = k.shape[0]
    num_k_heads = k.shape[1]
    num_v_heads = v.shape[1]
    k_head_dim = k.shape[2]
    v_head_dim = v.shape[2]

    if k_cache.shape[1] != num_k_heads or k_cache.shape[2] != k_head_dim:
        raise ValueError("K cache shape does not match K tensor shape")
    if v_cache.shape[1] != num_v_heads or v_cache.shape[2] != v_head_dim:
        raise ValueError("V cache shape does not match V tensor shape")
    if cache_loc.shape[0] != num_tokens:
        raise ValueError("cache_loc length must match number of K/V tokens")

    k_scale_ptr, has_k_scale = _scale_ptr(k_scale, k)
    v_scale_ptr, has_v_scale = _scale_ptr(v_scale, v)

    max_head_dim = max(k_head_dim, v_head_dim)
    block_d = triton.next_power_of_2(max_head_dim)
    grid = (num_tokens, max(num_k_heads, num_v_heads), 2)

    _fused_fp8_set_kv_buffer_kernel[grid](
        k,
        v,
        k_cache,
        v_cache,
        cache_loc,
        k_scale_ptr,
        v_scale_ptr,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        NUM_K_HEADS=num_k_heads,
        NUM_V_HEADS=num_v_heads,
        K_HEAD_DIM=k_head_dim,
        V_HEAD_DIM=v_head_dim,
        HAS_K_SCALE=has_k_scale,
        HAS_V_SCALE=has_v_scale,
        BLOCK_D=block_d,
    )
