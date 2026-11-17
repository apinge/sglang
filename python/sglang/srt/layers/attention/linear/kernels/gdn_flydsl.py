import functools
from typing import Optional

import torch

from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.utils.common import is_gfx942_supported, rank0_log


@functools.lru_cache(maxsize=1)
def flydsl_gdn_decode_ops():
    """aiter FlyDSL GDN decode entry points, or None when unavailable."""
    if not is_gfx942_supported():
        return None
    try:
        from aiter.ops.flydsl.gdn_decode import (
            fused_qkvzba_conv1d_gated_delta_rule_decode_flydsl,
            fused_recurrent_gated_delta_rule_packed_decode_flydsl,
        )
    except ImportError:
        return None
    return (
        fused_recurrent_gated_delta_rule_packed_decode_flydsl,
        fused_qkvzba_conv1d_gated_delta_rule_decode_flydsl,
    )


def _state_ok(ssm_states: torch.Tensor) -> bool:
    # [slot, HV, V, K] fp32 with K = V = 128 contiguous inside a slot.
    return (
        ssm_states.dtype == torch.float32
        and ssm_states.dim() == 4
        and ssm_states.shape[-2:] == (128, 128)
        and ssm_states.stride()[-3:] == (128 * 128, 128, 1)
    )


def _params_ok(A_log: torch.Tensor, dt_bias: torch.Tensor) -> bool:
    return A_log.dtype == torch.float32 and dt_bias.dtype in (
        torch.bfloat16,
        torch.float32,
    )


def _packed_decode_ok(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    num_v_heads: int,
    head_v_dim: int,
) -> bool:
    # The FlyDSL kernel reads bf16 activations, fp32 A_log and K = V = 128 heads.
    if not (
        _state_ok(ssm_states)
        and head_v_dim == 128
        and ssm_states.shape[1] == num_v_heads
        and cache_indices.dtype == torch.int32
        and mixed_qkv.dtype == a.dtype == b.dtype == torch.bfloat16
        and mixed_qkv.stride(-1) == a.stride(-1) == b.stride(-1) == 1
        and _params_ok(A_log, dt_bias)
        and abs(scale - 128**-0.5) < 1e-12
    ):
        return False
    qk_dim = mixed_qkv.shape[1] - num_v_heads * 128
    num_k_heads = qk_dim // 256
    return qk_dim > 0 and qk_dim % 256 == 0 and num_v_heads % num_k_heads == 0


class FlyDSLGDNKernel(LinearAttnKernelBase):
    """gfx942 FlyDSL GDN decode with the Triton kernel's interface.

    ``packed_decode`` is a drop-in for ``TritonGDNKernel.packed_decode``;
    ``fused_qkvzba_conv1d_decode`` serves the fused decode projection/Conv1D
    protocol (in_proj outputs in, ``(core_attn_out, z)`` out). Anything the
    FlyDSL kernels do not cover runs on the Triton kernel.
    """

    supports_packed_decode = True
    supports_strided_target_verify_qkv = (
        TritonGDNKernel.supports_strided_target_verify_qkv
    )

    def __init__(self, fallback_kernel: TritonGDNKernel):
        self.fallback_kernel = fallback_kernel
        self._packed_decode, self._fused_decode = flydsl_gdn_decode_ops()
        self._logged = False

    def _log_once(self, what: str) -> None:
        if not self._logged:
            rank0_log(f"GDN decode: using aiter FlyDSL {what}.")
            self._logged = True

    def packed_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_v_heads: int,
        head_v_dim: int,
        **kwargs,
    ) -> torch.Tensor:
        if kwargs.get("replayssm_d") is not None or not _packed_decode_ok(
            mixed_qkv,
            a,
            b,
            A_log,
            dt_bias,
            scale,
            ssm_states,
            cache_indices,
            num_v_heads,
            head_v_dim,
        ):
            return self.fallback_kernel.packed_decode(
                mixed_qkv,
                a,
                b,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=scale,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                num_v_heads=num_v_heads,
                head_v_dim=head_v_dim,
                **kwargs,
            )
        self._log_once("packed decode")
        out = mixed_qkv.new_empty(mixed_qkv.shape[0], 1, num_v_heads, head_v_dim)
        self._packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=ssm_states,
            out=out,
            ssm_state_indices=cache_indices,
            use_qk_l2norm_in_kernel=True,
        )
        return out.transpose(0, 1)

    def supports_fused_qkvzba_conv1d_decode(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        conv_weights: torch.Tensor,
        bias: Optional[torch.Tensor],
        activation: str,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        head_k_dim: int,
        head_v_dim: int,
    ) -> bool:
        return (
            head_k_dim == 128
            and head_v_dim == 128
            and _params_ok(A_log, dt_bias)
            and bias is None
            and activation in ("silu", "swish")
            and conv_weights.dim() == 2
            and conv_weights.shape[-1] == 4
            and conv_weights.dtype == torch.bfloat16
            and conv_weights.is_contiguous()
            and conv_states.dim() == 3
            and conv_states.shape[-1] == 3
            and conv_states.dtype == torch.bfloat16
            and _state_ok(ssm_states)
        )

    def fused_qkvzba_conv1d_decode(
        self,
        projected_qkvz: torch.Tensor,
        projected_ba: torch.Tensor,
        *,
        conv_states: torch.Tensor,
        conv_weights: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_k_heads: int,
        num_v_heads: int,
        head_v_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns core_attn_out [1, B, HV, V] and a contiguous z [B, HV, V]."""
        self._log_once("fused qkvz/ba unpack + conv1d + recurrent decode")
        B = projected_qkvz.shape[0]
        out = projected_qkvz.new_empty(B, 1, num_v_heads, head_v_dim)
        z = projected_qkvz.new_empty(B, num_v_heads, head_v_dim)
        if cache_indices.dtype != torch.int32:
            cache_indices = cache_indices.to(torch.int32)
        self._fused_decode(
            projected_qkvz=projected_qkvz,
            projected_ba=projected_ba,
            conv_states=conv_states,
            conv_weights=conv_weights,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=ssm_states,
            out=out,
            ssm_state_indices=cache_indices,
            num_k_heads=num_k_heads,
            z_out=z,
        )
        return out.transpose(0, 1), z

    def decode(self, *args, **kwargs):
        return self.fallback_kernel.decode(*args, **kwargs)

    def extend(self, *args, **kwargs):
        return self.fallback_kernel.extend(*args, **kwargs)

    def target_verify(self, *args, **kwargs):
        return self.fallback_kernel.target_verify(*args, **kwargs)


@functools.lru_cache(maxsize=1)
def flydsl_gdn_prefill_op():
    """aiter FlyDSL GDN prefill front end (conv + split + q/k l2norm), or None."""
    if not is_gfx942_supported():
        return None
    try:
        from aiter.ops.flydsl.gdn_prefill import flydsl_gdn_prefill_front
    except ImportError:
        return None
    return flydsl_gdn_prefill_front
