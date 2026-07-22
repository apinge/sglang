# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional

import torch

from aiter.ops.shuffle import shuffle_weight as _aiter_shuffle_weight

try:
    from aiter.ops.shuffle import shuffle_scale as _aiter_shuffle_scale
except ImportError:
    _aiter_shuffle_scale = None


def shuffle_weight(
    x: torch.Tensor,
    layout=(16, 16),
    use_int4: bool = False,
    is_guinterleave: bool = False,
    gate_up: bool = False,
) -> torch.Tensor:
    if not is_guinterleave:
        return _aiter_shuffle_weight(x, layout=layout, use_int4=use_int4)

    x_type = x.dtype
    if hasattr(torch, "float4_e2m1fn_x2") and x_type == torch.float4_e2m1fn_x2:
        x = x.view(torch.uint8)

    experts_cnt, n, k_pk = x.shape
    if gate_up:
        n = n // 2

    n_lane, k_pack = layout
    k_lane = 64 // n_lane
    n0 = n // n_lane
    k0 = k_pk // (k_lane * k_pack)

    if gate_up:
        x_ = x.view(experts_cnt, 2, n0, n_lane, k0, k_lane, k_pack)
        x_ = x_.permute(0, 2, 1, 4, 5, 3, 6).contiguous()
    else:
        x_ = x.view(experts_cnt, n0, n_lane, k0, k_lane, k_pack)
        x_ = x_.permute(0, 1, 3, 4, 2, 5).contiguous()

    x_ = x_.view(*x.shape).contiguous().view(x_type)
    x_.is_shuffled = True
    return x_


def shuffle_weight_a16w4(
    src: torch.Tensor, n_lane: int, gate_up: bool
) -> torch.Tensor:
    return shuffle_weight(
        src, layout=(n_lane, 16), is_guinterleave=True, gate_up=gate_up
    )


def shuffle_scale(
    src: Optional[torch.Tensor],
    experts_cnt: Optional[int] = None,
    is_guinterleave: bool = False,
    gate_up: bool = False,
) -> Optional[torch.Tensor]:
    if _aiter_shuffle_scale is not None:
        return _aiter_shuffle_scale(
            src,
            experts_cnt=experts_cnt,
            is_guinterleave=is_guinterleave,
            gate_up=gate_up,
        )

    if src is None:
        return src
    if src.dtype == torch.float32:
        return src
    assert src.ndim == 2, "scale must be a 2D tensor"

    if not is_guinterleave:
        m, n = src.shape
        scale_padded = torch.empty(
            (m + 255) // 256 * 256,
            (n + 7) // 8 * 8,
            dtype=src.dtype,
            device=src.device,
        )

        scale_padded[:m, :n] = src
        scale = scale_padded
        sm, sn = scale.shape
        scale = scale.view(sm // 32, 2, 16, sn // 8, 2, 4)
        scale = scale.permute(0, 3, 5, 2, 4, 1).contiguous()
        return scale.view(sm, sn)

    if experts_cnt is None:
        raise ValueError("experts_cnt is required when is_guinterleave=True")

    n_experts, k_ = src.shape
    n_ = n_experts // experts_cnt
    k_pack = 2
    n_pack = 2
    n_lane = 16
    k_lane = 64 // n_lane

    k1 = k_ // k_pack // k_lane
    n1 = n_ // n_lane // n_pack
    real_k = 32 * k_ * k_pack * k_lane
    assert real_k >= 256, f"K {real_k} must be larger than Tile_K(256)"

    if gate_up:
        shfl_scale = src.view(experts_cnt, n_pack, n1, n_lane, k1, k_pack, k_lane)
        shfl_scale = shfl_scale.permute(0, 2, 4, 6, 3, 5, 1).contiguous()
    else:
        shfl_scale = src.view(experts_cnt, n1, n_pack, n_lane, k1, k_pack, k_lane)
        shfl_scale = shfl_scale.permute(0, 1, 4, 6, 3, 5, 2).contiguous()

    return shfl_scale.view(*src.shape).contiguous()


def shuffle_scale_a16w4(
    src: torch.Tensor, experts_cnt: int, gate_up: bool
) -> torch.Tensor:
    return shuffle_scale(
        src,
        experts_cnt=experts_cnt,
        is_guinterleave=True,
        gate_up=gate_up,
    )
