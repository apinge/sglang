"""AITER's three-stage GDN prefill with SGLang's indexed VK state pool."""

import functools
import logging

import torch

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)
CHUNK_SIZE = 64


@functools.lru_cache(maxsize=None)
def _is_gfx942(device: torch.device) -> bool:
    if not is_hip() or device.type != "cuda":
        return False
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    if arch != "gfx942":
        logger.warning("AITER GDN prefill requires gfx942; using Triton on %s.", arch)
        return False
    return True


@functools.lru_cache(maxsize=1)
def _aiter_ops():
    # Only the selected AITER backend loads these optional dependencies.
    try:
        from aiter.ops.flydsl.linear_attention_prefill_kernels import (
            chunk_gated_delta_rule_fwd_h_flydsl_opt,
            gdn_prepare_fwd_flydsl,
        )
        from aiter.ops.prefill_batch_metadata import (
            build_gated_delta_rule_prefill_metadata,
        )

        try:
            from aiter.ops.triton._triton_kernels.gated_delta_net.prefill.chunk_o import (
                chunk_fwd_o_opt_vk,
            )
        except ModuleNotFoundError:
            # AITER versions before the gated_delta_net directory rename.
            from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_o import (
                chunk_fwd_o_opt_vk,
            )
    except ImportError as exc:
        logger.warning(
            "AITER GDN prefill is unavailable; using Triton. "
            "Install AITER with FlyDSL GDN K5 opt support: %s",
            exc,
        )
        return None
    return (
        build_gated_delta_rule_prefill_metadata,
        gdn_prepare_fwd_flydsl,
        chunk_gated_delta_rule_fwd_h_flydsl_opt,
        chunk_fwd_o_opt_vk,
    )


def _build_prefill_metadata(forward_batch, forward_metadata, build_metadata):
    """Accept only eager ordinary prefill with real, unpadded request rows.

    The scheduler supplies valid physical state indices for these rows. K5
    does not support the -1 padding sentinel; check host metadata without
    synchronizing the GPU indices or cumulative sequence lengths.
    """
    if forward_batch.forward_mode != ForwardMode.EXTEND:
        return None
    cu = forward_metadata.query_start_loc
    indices = forward_metadata.mamba_cache_indices
    lengths = forward_batch.extend_seq_lens_cpu
    bs = forward_batch.batch_size
    real_bs = forward_batch._original_batch_size
    if (
        cu is None
        or not _is_gfx942(cu.device)
        or torch.cuda.is_current_stream_capturing()
        or not lengths
        or len(lengths) != bs
        or any(length <= 0 for length in lengths)
        or (real_bs is not None and real_bs != bs)
        or cu.shape != (bs + 1,)
        or indices.shape != (bs,)
        or sum(lengths) != forward_batch.input_ids.numel()
    ):
        return None
    return build_metadata(lengths, cu_seqlens=cu, chunk_size=CHUNK_SIZE)


def aiter_gdn_prefill_supported(q, k, v, g, beta, states, indices, cu) -> bool:
    """Check tensor metadata for the AITER prefill kernels.

    The forward metadata certifies real request rows.
    """
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        return False
    if (
        q.shape[0] != 1
        or q.shape[1] == 0
        or q.shape[-1] != 128
        or v.shape[:2] != q.shape[:2]
        or v.shape[-1] != 128
        or q.shape[2] == 0
        or v.shape[2] % q.shape[2] != 0
        or v.numel() >= 2**31
        or g.shape != v.shape[:3]
        or beta.shape != g.shape
        or states.ndim != 4
        or states.shape[1:] != (v.shape[2], 128, 128)
        or states.dtype not in (torch.float32, torch.bfloat16)
        or not states.is_contiguous()
        or states.stride()[-2:] != (128, 1)
        or indices.ndim != 1
        or cu.shape != (indices.numel() + 1,)
        or indices.dtype not in (torch.int32, torch.int64)
        or cu.dtype not in (torch.int32, torch.int64)
        or g.dtype != torch.float32
        or beta.dtype not in (torch.bfloat16, torch.float32)
        or not _is_gfx942(q.device)
    ):
        return False
    tensors = (q, k, v, g, beta, states, indices, cu)
    return (
        all(t.device == q.device for t in tensors)
        and all(t.dtype == torch.bfloat16 for t in (q, k, v))
        and all(t.is_contiguous() for t in (q, k, v, g, beta, indices, cu))
    )


@functools.lru_cache(maxsize=1)
def _log_use():
    logger.info("Using AITER FlyDSL GDN prefill (chunk size 64).")


@functools.lru_cache(maxsize=None)
def _log_fallback(reason: str):
    # Fixed reasons keep this bounded and avoid repeating warnings per layer/batch.
    logger.warning(
        "AITER GDN prefill is falling back to Triton: %s. "
        "This reason is logged once per process.",
        reason,
    )


def _run_aiter_gdn_prefill(
    q,
    k,
    v,
    g,
    beta,
    *,
    ssm_states,
    cache_indices,
    query_start_loc,
    prefill_metadata,
    ops,
    l2norm,
):
    _, prepare, recurrent, output = ops
    _log_use()
    q = l2norm(q)
    k = l2norm(k)
    w, u, g_cumsum = prepare(
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=query_start_loc,
        BT=CHUNK_SIZE,
        use_exp2=True,
        prefill_metadata=prefill_metadata,
    )
    h, v_new, _ = recurrent(
        k=k,
        w=w,
        u=u,
        g=g_cumsum,
        initial_state=ssm_states,
        initial_state_indices=cache_indices,
        output_final_state=True,
        inplace_final_state=True,
        cu_seqlens=query_start_loc,
        state_dtype=ssm_states.dtype,
        snapshot_dtype=q.dtype,
        # Match Triton's round-to-nearest conversion, including the recurrent
        # matmul operands. AITER defaults to truncation for HIP compatibility.
        bf16_convert_trunc=False,
        chunk_size=CHUNK_SIZE,
        use_exp2=True,
        g_head_major=True,
        prefill_metadata=prefill_metadata,
    )
    o = output(
        q=q,
        k=k,
        v=v_new,
        o=torch.empty_like(v),
        h=h,
        g=g_cumsum,
        cu_seqlens=query_start_loc,
        chunk_size=CHUNK_SIZE,
        use_exp2=True,
        prefill_metadata=prefill_metadata,
    )
    # The live pool was updated in place; radix tracking consumes the BF16
    # state before each chunk, in flattened request/chunk order.
    return o, None, h


class AiterGDNKernel(LinearAttnKernelBase):
    """AITER GDN prefill, with Triton for unsupported batches and other modes.

    Per-batch metadata belongs to ForwardMetadata, never to this instance.
    The full chunk snapshots use the existing tracking protocol, rather than
    the selective checkpoint protocol (uses_state_checkpoints stays False).
    """

    def __init__(self, fallback_kernel: LinearAttnKernelBase):
        self.fallback_kernel = fallback_kernel
        self._ops = _aiter_ops()
        if self._ops is not None:
            from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd

            self._l2norm = l2norm_fwd

    def build_prefill_metadata(self, forward_batch, forward_metadata):
        if self._ops is None:
            return None
        return _build_prefill_metadata(forward_batch, forward_metadata, self._ops[0])

    def decode(self, *args, **kwargs):
        # Required by the abstract base; the dispatcher selects Triton for decode.
        return self.fallback_kernel.decode(*args, **kwargs)

    def target_verify(self, *args, **kwargs):
        return self.fallback_kernel.target_verify(*args, **kwargs)

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
        aiter_gdn_prefill_metadata=None,
        **kwargs,
    ) -> tuple:
        fallback_reason = None
        if self._ops is None:
            fallback_reason = "required AITER/FlyDSL ops are unavailable"
        elif aiter_gdn_prefill_metadata is None:
            fallback_reason = (
                "compatible prefill metadata is unavailable for this batch"
            )
        elif not aiter_gdn_prefill_supported(
            q, k, v, g, beta, ssm_states, cache_indices, query_start_loc
        ):
            fallback_reason = "unsupported tensor shape, dtype, device or layout"
        if fallback_reason is not None:
            _log_fallback(fallback_reason)
            return self.fallback_kernel.extend(
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
        # Do not retry a failed launch: K5 may already have updated the pool.
        return _run_aiter_gdn_prefill(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            prefill_metadata=aiter_gdn_prefill_metadata,
            ops=self._ops,
            l2norm=self._l2norm,
        )
