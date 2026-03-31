"""Fused GDN input projection for Qwen3.5.

BF16: full fusion (qkv+z+b+a -> 1 GEMM).
FP8/PTPC: qkv+z -> 1 GEMM, b+a -> 1 GEMM.
HIP/ROCm only.
"""

GEMM_N_ALIGN = 128

from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl
from einops import rearrange

from sglang.srt.configs.qwen3_5 import Qwen3_5TextConfig
from sglang.srt.layers.attention.fla.layernorm_gated import RMSNorm as RMSNormGated
from sglang.srt.layers.attention.mamba.mamba import mamba_v2_sharded_weight_loader
from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.parameter import (
    BlockQuantScaleParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import sharded_weight_loader
from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet
from sglang.srt.utils import add_prefix, set_weight_attrs

FUSED_PROJ_WEIGHT_MAPPING = [
    ("in_proj_ba.", "in_proj_b.", 0),
    ("in_proj_ba.", "in_proj_a.", 1),
    ("in_proj_qkvz.", "in_proj_qkv.", (0, 1, 2)),
    ("in_proj_qkvz.", "in_proj_z.", 3),
    ("in_proj_fused.", "in_proj_qkv.", (0, 1, 2)),
    ("in_proj_fused.", "in_proj_z.", 3),
    ("in_proj_fused.", "in_proj_b.", 4),
    ("in_proj_fused.", "in_proj_a.", 5),
]


@triton.jit
def _scatter_fused_proj_kernel(
    fused_ptr,
    qkv_ptr,
    z_ptr,
    b_ptr,
    a_ptr,
    stride_fused,
    qkv_size: tl.constexpr,
    z_size: tl.constexpr,
    ba_size: tl.constexpr,
    BLK_LARGE: tl.constexpr,
    BLK_SMALL: tl.constexpr,
):
    row = tl.program_id(0)
    src = row * stride_fused

    for off in range(0, qkv_size, BLK_LARGE):
        col = off + tl.arange(0, BLK_LARGE)
        mask = col < qkv_size
        tl.store(
            qkv_ptr + row * qkv_size + col,
            tl.load(fused_ptr + src + col, mask=mask),
            mask=mask,
        )

    z_off = qkv_size
    for off in range(0, z_size, BLK_LARGE):
        col = off + tl.arange(0, BLK_LARGE)
        mask = col < z_size
        tl.store(
            z_ptr + row * z_size + col,
            tl.load(fused_ptr + src + z_off + col, mask=mask),
            mask=mask,
        )

    ba_off = qkv_size + z_size
    col_s = tl.arange(0, BLK_SMALL)
    b_mask = col_s < ba_size
    tl.store(
        b_ptr + row * ba_size + col_s,
        tl.load(fused_ptr + src + ba_off + col_s, mask=b_mask),
        mask=b_mask,
    )
    a_mask = col_s < ba_size
    tl.store(
        a_ptr + row * ba_size + col_s,
        tl.load(fused_ptr + src + ba_off + ba_size + col_s, mask=a_mask),
        mask=a_mask,
    )


def scatter_fused_proj(fused_out, qkv_size, z_size, b_size, a_size):
    """Scatter fused output into 4 contiguous buffers."""
    assert b_size == a_size
    rows = fused_out.shape[0]
    qkv = torch.empty(rows, qkv_size, device=fused_out.device, dtype=fused_out.dtype)
    z = torch.empty(rows, z_size, device=fused_out.device, dtype=fused_out.dtype)
    b = torch.empty(rows, b_size, device=fused_out.device, dtype=fused_out.dtype)
    a = torch.empty(rows, a_size, device=fused_out.device, dtype=fused_out.dtype)

    BLK_LARGE = triton.next_power_of_2(max(qkv_size, z_size))
    BLK_SMALL = triton.next_power_of_2(max(b_size, a_size))
    _scatter_fused_proj_kernel[(rows,)](
        fused_out,
        qkv,
        z,
        b,
        a,
        fused_out.stride(0),
        qkv_size,
        z_size,
        b_size,
        BLK_LARGE,
        BLK_SMALL,
        num_warps=4,
    )
    return qkv, z, b, a


@triton.jit
def _scatter_partial_fused_kernel(
    fused_ptr,
    ba_ptr,
    qkv_ptr,
    z_ptr,
    b_ptr,
    a_ptr,
    stride_fused,
    stride_ba,
    qkv_size: tl.constexpr,
    z_size: tl.constexpr,
    ba_size: tl.constexpr,
    BLK_LARGE: tl.constexpr,
    BLK_SMALL: tl.constexpr,
):
    row = tl.program_id(0)

    src = row * stride_fused
    for off in range(0, qkv_size, BLK_LARGE):
        col = off + tl.arange(0, BLK_LARGE)
        mask = col < qkv_size
        tl.store(
            qkv_ptr + row * qkv_size + col,
            tl.load(fused_ptr + src + col, mask=mask),
            mask=mask,
        )

    z_off = qkv_size
    for off in range(0, z_size, BLK_LARGE):
        col = off + tl.arange(0, BLK_LARGE)
        mask = col < z_size
        tl.store(
            z_ptr + row * z_size + col,
            tl.load(fused_ptr + src + z_off + col, mask=mask),
            mask=mask,
        )

    src_ba = row * stride_ba
    col_s = tl.arange(0, BLK_SMALL)
    b_mask = col_s < ba_size
    tl.store(
        b_ptr + row * ba_size + col_s,
        tl.load(ba_ptr + src_ba + col_s, mask=b_mask),
        mask=b_mask,
    )
    a_mask = col_s < ba_size
    tl.store(
        a_ptr + row * ba_size + col_s,
        tl.load(ba_ptr + src_ba + ba_size + col_s, mask=a_mask),
        mask=a_mask,
    )


def scatter_partial_fused(fused_out, ba_out, qkv_size, z_size, ba_size):
    """Scatter fused_out(qkv+z) and ba_out(b+a) into 4 buffers in one kernel."""
    rows = fused_out.shape[0]
    qkv = torch.empty(rows, qkv_size, device=fused_out.device, dtype=fused_out.dtype)
    z = torch.empty(rows, z_size, device=fused_out.device, dtype=fused_out.dtype)
    b = torch.empty(rows, ba_size, device=ba_out.device, dtype=ba_out.dtype)
    a = torch.empty(rows, ba_size, device=ba_out.device, dtype=ba_out.dtype)

    BLK_LARGE = triton.next_power_of_2(max(qkv_size, z_size))
    BLK_SMALL = triton.next_power_of_2(ba_size)
    _scatter_partial_fused_kernel[(rows,)](
        fused_out,
        ba_out,
        qkv,
        z,
        b,
        a,
        fused_out.stride(0),
        ba_out.stride(0),
        qkv_size,
        z_size,
        ba_size,
        BLK_LARGE,
        BLK_SMALL,
        num_warps=4,
    )
    return qkv, z, b, a


class Qwen3_5GatedDeltaNetFusedProj(Qwen3_5GatedDeltaNet):

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)

        self.config = config
        self.attn_tp_rank = get_attention_tp_rank()
        self.attn_tp_size = get_attention_tp_size()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.alt_stream = alt_stream
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_id = layer_id
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
            quant_config=None,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("conv1d", prefix),
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        self._partial_fuse = quant_config is not None

        if self._partial_fuse:
            self.in_proj_qkvz = MergedColumnParallelLinear(
                input_size=self.hidden_size,
                output_sizes=[
                    self.key_dim,
                    self.key_dim,
                    self.value_dim,
                    self.value_dim,
                ],
                bias=False,
                quant_config=quant_config,
                tp_rank=self.attn_tp_rank,
                tp_size=self.attn_tp_size,
                prefix=add_prefix("in_proj_qkvz", prefix),
            )
            self._bind_packed_weight_loaders(self.in_proj_qkvz)

            self.in_proj_ba = MergedColumnParallelLinear(
                input_size=self.hidden_size,
                output_sizes=[self.num_v_heads, self.num_v_heads],
                bias=False,
                quant_config=quant_config,
                tp_rank=self.attn_tp_rank,
                tp_size=self.attn_tp_size,
                prefix=add_prefix("in_proj_ba", prefix),
            )
            self._bind_packed_weight_loaders(self.in_proj_ba)
        else:
            output_sizes = [
                self.key_dim,
                self.key_dim,
                self.value_dim,
                self.value_dim,
                self.num_v_heads,
                self.num_v_heads,
            ]

            self.in_proj_fused = MergedColumnParallelLinear(
                input_size=self.hidden_size,
                output_sizes=output_sizes,
                bias=False,
                quant_config=quant_config,
                tp_rank=self.attn_tp_rank,
                tp_size=self.attn_tp_size,
                prefix=add_prefix("in_proj_fused", prefix),
            )
            self._bind_packed_weight_loaders(self.in_proj_fused)

        self.qkv_size_local = (
            self.key_dim // self.attn_tp_size * 2
            + self.value_dim // self.attn_tp_size
        )
        self.z_size_local = self.value_dim // self.attn_tp_size
        self.b_size_local = self.num_v_heads // self.attn_tp_size
        self.a_size_local = self.num_v_heads // self.attn_tp_size

        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)
        delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(
            self.conv1d.weight,
            {
                "weight_loader": mamba_v2_sharded_weight_loader(
                    [query_key_settings, query_key_settings, value_settings],
                    self.attn_tp_size,
                    self.attn_tp_rank,
                )
            },
        )

        self.dt_bias = nn.Parameter(
            torch.ones(self.num_v_heads // self.attn_tp_size),
        )
        self.A_log = nn.Parameter(
            torch.empty(self.num_v_heads // self.attn_tp_size),
        )
        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        self.attn = RadixLinearAttention(
            layer_id=layer_id,
            num_q_heads=self.num_k_heads // self.attn_tp_size,
            num_k_heads=self.num_k_heads // self.attn_tp_size,
            num_v_heads=self.num_v_heads // self.attn_tp_size,
            head_q_dim=self.head_k_dim,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            conv_weights=conv_weights,
            bias=self.conv1d.bias,
            activation=self.activation,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
        )

        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=True,
            device=torch.get_device_module().current_device(),
            dtype=config.torch_dtype,
        )

        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            reduce_results=False,
            quant_config=quant_config,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("out_proj", prefix),
        )

    @staticmethod
    def _override_weight_loader(param, loader):
        if hasattr(param, "_weight_loader"):
            param._weight_loader = loader
            return
        if hasattr(param, "weight_loader"):
            param.weight_loader = loader
            return
        set_weight_attrs(param, {"weight_loader": loader})

    def _bind_packed_weight_loaders(self, module):
        for attr_name in ("weight", "weight_scale_inv", "weight_scale", "input_scale"):
            param = getattr(module, attr_name, None)
            if param is None:
                continue
            original_loader = getattr(param, "weight_loader", None)
            if original_loader is None:
                continue
            wrapped_loader = self._make_packed_weight_loader(module, original_loader)
            self._override_weight_loader(param, wrapped_loader)

    @staticmethod
    def _get_split_sizes_for_param(module, param, loaded_shard_id):
        if isinstance(param, BlockQuantScaleParameter):
            block_n, _ = module.quant_method.quant_config.weight_block_size
            block_n = 1 if getattr(param, "format_ue8m0", False) else block_n
            return [
                (module.output_sizes[idx] + block_n - 1) // block_n
                for idx in loaded_shard_id
            ]

        if isinstance(param, PerTensorScaleParameter):
            return [1 for _ in loaded_shard_id]

        return [module.output_sizes[idx] for idx in loaded_shard_id]

    @classmethod
    def _make_packed_weight_loader(cls, module, original_weight_loader):
        def weight_loader(param, loaded_weight, loaded_shard_id=None):
            if isinstance(loaded_shard_id, tuple):
                split_sizes = cls._get_split_sizes_for_param(
                    module, param, loaded_shard_id
                )

                if len(loaded_weight.shape) == 0:
                    assert len(split_sizes) == 1 and split_sizes[0] == 1, (
                        f"Unexpected scalar for tuple shard load: "
                        f"{loaded_shard_id=}, {split_sizes=}"
                    )
                    chunks = [loaded_weight.reshape(1)]
                else:
                    split_dim = getattr(param, "output_dim", 0)
                    chunks = loaded_weight.split(split_sizes, dim=split_dim)

                assert len(chunks) == len(loaded_shard_id), (
                    f"Chunk/shard mismatch: {len(chunks)=}, "
                    f"{len(loaded_shard_id)=}, {split_sizes=}"
                )

                for idx, chunk in zip(loaded_shard_id, chunks):
                    original_weight_loader(param, chunk, idx)
                return

            return original_weight_loader(param, loaded_weight, loaded_shard_id)

        return weight_loader

    def _forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        seq_len, _ = hidden_states.shape

        if self._partial_fuse:
            projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
            projected_states_ba, _ = self.in_proj_ba(hidden_states)
            mixed_qkv, z, b, a = scatter_partial_fused(
                projected_states_qkvz,
                projected_states_ba,
                self.qkv_size_local,
                self.z_size_local,
                self.b_size_local,
            )
        else:
            fused_out, _ = self.in_proj_fused(hidden_states)
            if seq_len >= 4:
                mixed_qkv, z, b, a = scatter_fused_proj(
                    fused_out,
                    self.qkv_size_local,
                    self.z_size_local,
                    self.b_size_local,
                    self.a_size_local,
                )
            else:
                qkv_end = self.qkv_size_local
                z_end = qkv_end + self.z_size_local
                b_end = z_end + self.b_size_local
                a_end = b_end + self.a_size_local
                mixed_qkv = fused_out[:, :qkv_end].contiguous()
                z = fused_out[:, qkv_end:z_end].contiguous()
                b = fused_out[:, z_end:b_end].contiguous()
                a = fused_out[:, b_end:a_end].contiguous()

        z = z.reshape(hidden_states.size(0), -1, self.head_v_dim)

        core_attn_out = self.attn.forward(
            forward_batch=forward_batch,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
        )

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output, _ = self.out_proj(core_attn_out)
        return output
