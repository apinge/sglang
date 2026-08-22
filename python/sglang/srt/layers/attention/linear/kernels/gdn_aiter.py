from __future__ import annotations

import inspect
import logging
import os
from typing import Callable, Optional

import torch

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.utils.common import is_gfx942_supported


logger = logging.getLogger(__name__)
_UNSET = object()

_HIP_GDN_SUPPORTED_LOCAL_HEAD_SHAPES = frozenset(
    {
        (2, 8),
    }
)


def supports_hip_gdn_decode_local_heads(
    local_num_k_heads: int, local_num_v_heads: int
) -> bool:
    return (local_num_k_heads, local_num_v_heads) in (
        _HIP_GDN_SUPPORTED_LOCAL_HEAD_SHAPES
    )


def should_use_hip_gdn_decode(
    local_num_k_heads: int,
    local_num_v_heads: int,
    batch_size: int,
) -> bool:
    if not supports_hip_gdn_decode_local_heads(
        local_num_k_heads, local_num_v_heads
    ):
        return False
    return batch_size >= 24


def supports_hip_gdn_decode_runtime(
    *,
    local_num_k_heads: int,
    local_num_v_heads: int,
    q_dtype: torch.dtype,
    k_dtype: torch.dtype,
    v_dtype: torch.dtype,
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
    dt_bias_dtype: torch.dtype,
    state_dtype: torch.dtype,
    head_k_dim: int,
    head_v_dim: int,
    state_shape: tuple[int, ...],
) -> bool:
    if not supports_hip_gdn_decode_local_heads(local_num_k_heads, local_num_v_heads):
        return False
    if head_k_dim != 128 or head_v_dim != 128:
        return False
    if any(
        dtype != torch.bfloat16
        for dtype in (q_dtype, k_dtype, v_dtype, a_dtype, b_dtype, dt_bias_dtype)
    ):
        return False
    if state_dtype != torch.float32 or len(state_shape) != 4:
        return False
    return tuple(state_shape[-3:]) == (local_num_v_heads, 128, 128)


def supports_flydsl_gdn_decode_runtime(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_states: torch.Tensor,
) -> bool:
    if any(tensor.dtype != torch.bfloat16 for tensor in (q, k, v, a, b, dt_bias)):
        return False
    if q.shape[-1] != 128 or k.shape[-1] != 128 or v.shape[-1] != 128:
        return False
    if ssm_states.dtype != torch.float32 or not ssm_states.is_contiguous():
        return False
    if ssm_states.ndim != 4 or tuple(ssm_states.shape[-2:]) != (128, 128):
        return False
    return ssm_states.shape[1] == v.shape[2]


def select_aiter_gdn_decode_backend(
    local_num_k_heads: int,
    local_num_v_heads: int,
    *,
    hip_available: bool,
    fly_available: bool,
    batch_size: int = 1,
) -> str:
    if hip_available and should_use_hip_gdn_decode(
        local_num_k_heads,
        local_num_v_heads,
        batch_size,
    ):
        return "hip"
    if fly_available and should_use_flydsl_decode(
        local_num_k_heads,
        local_num_v_heads,
        batch_size,
    ):
        return "flydsl"
    return "triton"


def should_use_flydsl_decode(
    local_num_k_heads: int,
    local_num_v_heads: int,
    batch_size: int,
) -> bool:
    del batch_size
    return (
        local_num_k_heads > 0
        and local_num_v_heads >= local_num_k_heads
        and local_num_v_heads % local_num_k_heads == 0
    )


def _load_aiter_decode_ops():
    hip_decode = None
    fly_decode = None
    reset_sort_cache = None
    try:
        # The HIP extension's host-side sorted-index cache cannot be replayed
        # safely by CUDA Graph. Keep the graph-safe unsorted kernel path.
        os.environ["HIP_GDN_SORT_IDX_BS"] = "0"
        from aiter.ops.hip.gated_delta_net import (
            hip_fused_sigmoid_gating_delta_rule_update,
            hip_gdn_decode_reset_sort_cache,
        )
        from aiter.ops.hip.gated_delta_net.hip_gdn_decode import _load_extension

        _load_extension()
        hip_decode = hip_fused_sigmoid_gating_delta_rule_update
        reset_sort_cache = hip_gdn_decode_reset_sort_cache
    except Exception as exc:
        logger.info("AITER HIP GDN decode is unavailable; using FlyDSL: %s", exc)

    try:
        from aiter.ops.flydsl.linear_attention_kernels import flydsl_gdr_decode

        def official_flydsl_decode(
            *,
            A_log,
            a,
            dt_bias,
            q,
            k,
            v,
            b,
            initial_state_source,
            initial_state_indices,
            use_qk_l2norm_in_kernel,
            **_kwargs,
        ):
            q_launch = q.transpose(0, 1).contiguous()
            k_launch = k.transpose(0, 1).contiguous()
            v_launch = v.transpose(0, 1).contiguous()
            if a.ndim == 2:
                a_launch = a.unsqueeze(1)
                b_launch = b.unsqueeze(1)
            else:
                a_launch = a.transpose(0, 1)
                b_launch = b.transpose(0, 1)
            output = torch.empty_like(v_launch)
            flydsl_gdr_decode(
                q_launch,
                k_launch,
                v_launch,
                a_launch,
                b_launch,
                dt_bias,
                A_log,
                initial_state_indices,
                initial_state_source,
                output,
                use_qk_l2norm=use_qk_l2norm_in_kernel,
                need_shuffle_state=False,
            )
            return output.transpose(0, 1)

        fly_decode = official_flydsl_decode
    except Exception:
        try:
            from aiter.ops.hip.gated_delta_net.hip_gdn_decode_flydsl import (
                flydsl_fused_sigmoid_gating_delta_rule_update,
            )

            fly_decode = flydsl_fused_sigmoid_gating_delta_rule_update
        except Exception as exc:
            logger.warning("AITER FlyDSL GDN decode is unavailable: %s", exc)

    return hip_decode, fly_decode, reset_sort_cache


def _load_aiter_decode_conv_split():
    try:
        from aiter.ops.triton._triton_kernels.gated_delta_rule.decode import (
            causal_conv1d_split_qkv,
        )

        return causal_conv1d_split_qkv.causal_conv1d_update_split_qkv
    except Exception as exc:
        logger.info("AITER fused decode conv-split is unavailable: %s", exc)
        return None


def _load_aiter_prefill_op():
    try:
        from aiter.ops.triton.gated_delta_net.gated_delta_rule import (
            chunk_gated_delta_rule_opt_vk,
        )

        parameters = inspect.signature(chunk_gated_delta_rule_opt_vk).parameters
        required = {"initial_state_indices", "inplace_final_state"}
        if not required.issubset(parameters):
            logger.info(
                "AITER indexed high-level prefill is unavailable; "
                "using the low-level K1-K6 chain. Missing parameters: %s",
                sorted(required - set(parameters)),
            )
            return None
        return chunk_gated_delta_rule_opt_vk
    except Exception as exc:
        logger.warning("AITER GDN prefill is unavailable: %s", exc)
        return None


def _load_aiter_prefill_intermediate_ops():
    try:
        from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_delta_h import (
            chunk_gated_delta_rule_fwd_h_opt_vk,
        )
        from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_o import (
            chunk_fwd_o_opt_vk,
        )
        from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.fused_cumsum_kkt import (
            fused_chunk_local_cumsum_scaled_dot_kkt_fwd,
        )
        from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.fused_solve_tril_recompute import (
            fused_solve_tril_recompute_w_u,
        )

        ops = {
            "cumsum": fused_chunk_local_cumsum_scaled_dot_kkt_fwd,
            "solve": fused_solve_tril_recompute_w_u,
            "chunk_h": chunk_gated_delta_rule_fwd_h_opt_vk,
            "chunk_o": chunk_fwd_o_opt_vk,
        }
        try:
            from aiter.ops.chunk_gated_delta_rule_fwd_h import (
                chunk_gated_delta_rule_fwd_h_hip_fn,
            )

            ops["chunk_h_hip"] = chunk_gated_delta_rule_fwd_h_hip_fn
        except Exception:
            pass
        return ops
    except Exception as exc:
        logger.warning("AITER intermediate-h GDN prefill is unavailable: %s", exc)
        return None


def aiter_prefill_min_tokens(local_num_v_heads: int) -> int:
    return 4096 if local_num_v_heads <= 8 else 1024


def supports_aiter_gdn_prefill_runtime(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> bool:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return False
    if q.is_cuda and torch.cuda.is_current_stream_capturing():
        return False
    if q.shape[0] != 1 or q.shape[:2] != k.shape[:2] or q.shape[:2] != v.shape[:2]:
        return False
    if q.shape[2] != k.shape[2] or v.shape[2] % k.shape[2] != 0:
        return False
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        return False
    if q.shape[-1] != 128 or k.shape[-1] != 128 or v.shape[-1] != 128:
        return False
    if g.dtype != torch.float32 or beta.dtype != torch.float32:
        return False
    if g.shape != beta.shape or g.shape != (1, q.shape[1], v.shape[2]):
        return False
    if ssm_states.dtype not in (torch.float32, torch.bfloat16):
        return False
    if ssm_states.ndim != 4 or tuple(ssm_states.shape[-2:]) != (128, 128):
        return False
    if not ssm_states.is_contiguous():
        return False
    if ssm_states.shape[1] != v.shape[2]:
        return False
    if cache_indices.dtype != torch.int32 or query_start_loc.dtype != torch.int32:
        return False
    if cache_indices.ndim != 1 or query_start_loc.ndim != 1:
        return False
    if query_start_loc.numel() != cache_indices.numel() + 1:
        return False
    tensors = (q, k, v, g, beta, ssm_states, cache_indices, query_start_loc)
    if any(tensor.device != q.device for tensor in tensors[1:]):
        return False
    return True


class AiterGDNKernel(LinearAttnKernelBase):
    supports_packed_decode = True

    def __init__(
        self,
        fallback_kernel: Optional[LinearAttnKernelBase] = None,
        *,
        hip_decode: Optional[Callable] | object = _UNSET,
        fly_decode: Optional[Callable] | object = _UNSET,
        reset_sort_cache: Optional[Callable] | object = _UNSET,
        prefill_vk: Optional[Callable] | object = _UNSET,
        prefill_intermediate_ops: Optional[dict[str, Callable]] | object = _UNSET,
        l2norm: Optional[Callable] | object = _UNSET,
        l2norm_qk: Optional[Callable] | object = _UNSET,
        hip_arch_supported: Optional[bool] = None,
        decode_conv_split: Optional[Callable] | object = _UNSET,
    ):
        if fallback_kernel is None:
            from sglang.srt.layers.attention.linear.kernels.gdn_triton import (
                TritonGDNKernel,
            )

            fallback_kernel = TritonGDNKernel()
        self.fallback_kernel = fallback_kernel

        if hip_decode is _UNSET or fly_decode is _UNSET or reset_sort_cache is _UNSET:
            loaded_hip, loaded_fly, loaded_reset = _load_aiter_decode_ops()
            if hip_decode is _UNSET:
                hip_decode = loaded_hip
            if fly_decode is _UNSET:
                fly_decode = loaded_fly
            if reset_sort_cache is _UNSET:
                reset_sort_cache = loaded_reset

        self.hip_decode = hip_decode
        self.fly_decode = fly_decode
        self.reset_sort_cache = reset_sort_cache
        self.hip_arch_supported = (
            is_gfx942_supported() if hip_arch_supported is None else hip_arch_supported
        )
        self.decode_conv_split_op = (
            _load_aiter_decode_conv_split()
            if decode_conv_split is _UNSET
            else decode_conv_split
        )
        self.prefill_vk = (
            _load_aiter_prefill_op() if prefill_vk is _UNSET else prefill_vk
        )
        self.prefill_intermediate_ops = (
            _load_aiter_prefill_intermediate_ops()
            if prefill_intermediate_ops is _UNSET
            else prefill_intermediate_ops
        )
        if l2norm is _UNSET:
            from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd

            l2norm = l2norm_fwd
        self.l2norm = l2norm
        if l2norm_qk is _UNSET:
            from sglang.srt.layers.attention.fla.l2norm import fused_l2norm_qk

            l2norm_qk = fused_l2norm_qk
        self.l2norm_qk = l2norm_qk

    def reset_decode_cache(self):
        if self.reset_sort_cache is not None:
            self.reset_sort_cache()

    def decode_conv_split(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor],
        activation: Optional[str],
        conv_state_indices: torch.Tensor,
        key_dim: int,
        value_dim: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        active_batch_size: Optional[int] = None,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        batch_size = x.shape[0]
        dispatch_batch_size = (
            batch_size if active_batch_size is None else active_batch_size
        )
        can_use_hip = (
            self.hip_decode is not None
            and self.hip_arch_supported
            and should_use_hip_gdn_decode(
                num_k_heads,
                num_v_heads,
                dispatch_batch_size,
            )
        )
        can_use_fly = self.fly_decode is not None and should_use_flydsl_decode(
            num_k_heads,
            num_v_heads,
            dispatch_batch_size,
        )
        if (
            self.decode_conv_split_op is None
            or not (can_use_hip or can_use_fly)
            or x.dtype != torch.bfloat16
            or conv_state.dtype != torch.bfloat16
            or weight.dtype != torch.bfloat16
            or head_k_dim != 128
            or head_v_dim != 128
        ):
            return None

        query, key, value = self.decode_conv_split_op(
            x,
            conv_state,
            weight,
            key_dim,
            value_dim,
            bias=bias,
            activation=activation,
            conv_state_indices=conv_state_indices,
        )
        query = query.squeeze(-1).view(batch_size, 1, num_k_heads, head_k_dim)
        key = key.squeeze(-1).view(batch_size, 1, num_k_heads, head_k_dim)
        value = value.squeeze(-1).view(batch_size, 1, num_v_heads, head_v_dim)
        return (
            query.transpose(0, 1),
            key.transpose(0, 1),
            value.transpose(0, 1),
        )

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
    ) -> Optional[torch.Tensor]:
        key_dim = (mixed_qkv.shape[-1] - num_v_heads * head_v_dim) // 2
        num_k_heads = key_dim // head_v_dim
        active_batch_size = int(
            kwargs.get("active_batch_size", mixed_qkv.shape[0])
        )
        can_use_hip = (
            self.hip_decode is not None
            and self.hip_arch_supported
            and should_use_hip_gdn_decode(
                num_k_heads,
                num_v_heads,
                active_batch_size,
            )
            and ssm_states.dtype == torch.float32
            and ssm_states.is_contiguous()
        )
        can_use_fly = (
            self.fly_decode is not None
            and should_use_flydsl_decode(
                num_k_heads,
                num_v_heads,
                active_batch_size,
            )
            and ssm_states.dtype == torch.float32
            and ssm_states.is_contiguous()
        )
        use_triton_packed = not (can_use_hip or can_use_fly) or any(
            kwargs.get(name) is not None
            for name in ("replayssm_d", "replayssm_k", "replayssm_g")
        )
        if not use_triton_packed:
            return None
        return self.fallback_kernel.packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            **kwargs,
        )

    def _fallback_decode(self, **kwargs):
        return self.fallback_kernel.decode(**kwargs)

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        fallback_kwargs = dict(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

        if any(
            kwargs.get(name) is not None
            for name in ("replayssm_d", "replayssm_k", "replayssm_g")
        ):
            return self._fallback_decode(**fallback_kwargs)

        total_batch_size = q.shape[1]
        active_batch_size = int(kwargs.get("active_batch_size", total_batch_size))
        if active_batch_size < 0 or active_batch_size > total_batch_size:
            raise ValueError(
                f"Invalid AITER GDN active batch size {active_batch_size}; "
                f"expected 0..{total_batch_size}"
            )
        if active_batch_size == 0:
            return torch.zeros_like(v)

        full_output = None
        launch_q, launch_k, launch_v = q, k, v
        launch_a, launch_b = a, b
        launch_indices = cache_indices
        launch_start_loc = query_start_loc
        if active_batch_size < total_batch_size:
            full_output = torch.zeros_like(v)
            launch_q = q[:, :active_batch_size]
            launch_k = k[:, :active_batch_size]
            launch_v = v[:, :active_batch_size]
            if a.ndim == 3:
                launch_a = a[:, :active_batch_size]
                launch_b = b[:, :active_batch_size]
            else:
                launch_a = a[:active_batch_size]
                launch_b = b[:active_batch_size]
            launch_indices = cache_indices[:active_batch_size]
            launch_start_loc = query_start_loc[: active_batch_size + 1]

        use_hip = (
            self.hip_decode is not None
            and self.hip_arch_supported
            and ssm_states.is_contiguous()
            and should_use_hip_gdn_decode(
                launch_q.shape[2],
                launch_v.shape[2],
                active_batch_size,
            )
            and supports_hip_gdn_decode_runtime(
                local_num_k_heads=launch_q.shape[2],
                local_num_v_heads=launch_v.shape[2],
                q_dtype=launch_q.dtype,
                k_dtype=launch_k.dtype,
                v_dtype=launch_v.dtype,
                a_dtype=launch_a.dtype,
                b_dtype=launch_b.dtype,
                dt_bias_dtype=dt_bias.dtype,
                state_dtype=ssm_states.dtype,
                head_k_dim=launch_q.shape[-1],
                head_v_dim=launch_v.shape[-1],
                state_shape=tuple(ssm_states.shape),
            )
        )
        use_fly = (
            not use_hip
            and self.fly_decode is not None
            and should_use_flydsl_decode(
                launch_q.shape[2],
                launch_v.shape[2],
                active_batch_size,
            )
            and supports_flydsl_gdn_decode_runtime(
                q=launch_q,
                k=launch_k,
                v=launch_v,
                a=launch_a,
                b=launch_b,
                dt_bias=dt_bias,
                ssm_states=ssm_states,
            )
        )
        decode_impl = (
            self.hip_decode if use_hip else self.fly_decode if use_fly else None
        )
        if decode_impl is None:
            return self._fallback_decode(**fallback_kwargs)

        output = decode_impl(
            A_log=A_log,
            a=launch_a,
            dt_bias=dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=launch_q,
            k=launch_k,
            v=launch_v,
            b=launch_b,
            initial_state_source=ssm_states,
            initial_state_indices=launch_indices,
            scale=launch_q.shape[-1] ** -0.5,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=launch_start_loc,
            is_kda=False,
        )
        if full_output is not None:
            full_output[:, :active_batch_size].copy_(output)
            return full_output
        return output

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> tuple:
        fallback_kwargs = dict(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )
        if (
            (self.prefill_vk is None and self.prefill_intermediate_ops is None)
            or kwargs.get("has_padding", False)
            or kwargs.get("mamba_cache_chunk_size", 64) != 64
            or query_start_loc.numel() != 2
            or q.shape[1] < aiter_prefill_min_tokens(v.shape[2])
            or not supports_aiter_gdn_prefill_runtime(
                q,
                k,
                v,
                g,
                beta,
                ssm_states,
                cache_indices,
                query_start_loc,
            )
        ):
            return self.fallback_kernel.extend(**fallback_kwargs)

        return_intermediate_h = kwargs.get("return_intermediate_h", False)
        seq_lens_cpu = kwargs.get("seq_lens_cpu")
        prefill_metadata = kwargs.get("prefill_metadata")
        metadata_kwargs = (
            {
                "seq_lens_cpu": seq_lens_cpu,
                "prefill_metadata": prefill_metadata,
            }
            if prefill_metadata is not None
            else {}
        )
        prefill_metadata_kwargs = (
            {"prefill_metadata": prefill_metadata}
            if prefill_metadata is not None
            else {}
        )
        use_low_level = return_intermediate_h or self.prefill_vk is None
        if use_low_level:
            autotune_enabled = os.getenv(
                "GATED_DELTA_RULE_TRITON_AUTOTUNE", "0"
            ).lower() in ("1", "true", "yes", "on")
            if self.prefill_intermediate_ops is None or autotune_enabled:
                return self.fallback_kernel.extend(**fallback_kwargs)

        launch_q = q.contiguous()
        launch_k = k.contiguous()
        launch_v = v.contiguous()
        launch_g = g.contiguous()
        launch_beta = beta.contiguous()
        launch_indices = cache_indices.contiguous()
        launch_start_loc = query_start_loc.contiguous()
        launch_q, launch_k = self.l2norm_qk(launch_q, launch_k)
        if use_low_level:
            g_cumsum, A = self.prefill_intermediate_ops["cumsum"](
                k=launch_k,
                beta=launch_beta,
                g=launch_g,
                cu_seqlens=launch_start_loc,
                use_exp2=True,
                **prefill_metadata_kwargs,
            )
            w, u = self.prefill_intermediate_ops["solve"](
                A_raw=A,
                k=launch_k,
                v=launch_v,
                beta=launch_beta,
                g_cumsum=g_cumsum,
                cu_seqlens=launch_start_loc,
                use_exp2=True,
                **prefill_metadata_kwargs,
            )
            chunk_h = self.prefill_intermediate_ops["chunk_h"]
            chunk_h_kwargs = {}
            if (
                not return_intermediate_h
                and launch_start_loc.numel() == 2
                and "chunk_h_hip" in self.prefill_intermediate_ops
            ):
                chunk_h = self.prefill_intermediate_ops["chunk_h_hip"]
                chunk_h_kwargs["g_head_major"] = True
            h, v_new, _ = chunk_h(
                k=launch_k,
                w=w,
                u=u,
                g=g_cumsum,
                initial_state=ssm_states,
                initial_state_indices=launch_indices,
                output_final_state=True,
                inplace_final_state=True,
                cu_seqlens=launch_start_loc,
                state_dtype=ssm_states.dtype,
                use_exp2=True,
                **prefill_metadata_kwargs,
                **chunk_h_kwargs,
            )
            output = self.prefill_intermediate_ops["chunk_o"](
                q=launch_q,
                k=launch_k,
                v=v_new,
                o=launch_v.new_empty(launch_v.shape),
                h=h,
                g=g_cumsum,
                scale=launch_k.shape[-1] ** -0.5,
                cu_seqlens=launch_start_loc,
                use_exp2=True,
                **prefill_metadata_kwargs,
            )
            return output, None, h if return_intermediate_h else None

        output, _ = self.prefill_vk(
            q=launch_q,
            k=launch_k,
            v=launch_v,
            o=launch_v.new_empty(launch_v.shape),
            g=launch_g,
            beta=launch_beta,
            scale=launch_k.shape[-1] ** -0.5,
            initial_state=ssm_states,
            initial_state_indices=launch_indices,
            output_final_state=True,
            inplace_final_state=True,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=launch_start_loc,
            use_chunk_hip=launch_start_loc.numel() == 2,
            state_dtype=ssm_states.dtype,
            use_exp2=True,
            **metadata_kwargs,
        )
        return output, None, None
