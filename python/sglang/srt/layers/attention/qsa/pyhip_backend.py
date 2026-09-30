"""Direct gfx942 PyHIP operators; SGLang retains projection and cache ownership."""

from __future__ import annotations

import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa.metadata import build_rope_position_matrix
from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    get_tc_piecewise_forward_context,
)
from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode

logger = logging.getLogger(__name__)


class PyHIPAttention:
    def __init__(self):
        from pyhip.ops.qsa.flydsl.attention import attention

        self.attention = attention
        self.validate = envs.SGLANG_TEST_PYHIP_QSA.get()
        self.checked_layouts = set()

    @staticmethod
    def supports(*, q, k, v, indices, layer, forward_mode, kwargs):
        return (
            forward_mode == ForwardMode.EXTEND
            and not get_is_capture_mode()
            and not torch.compiler.is_compiling()
            and not torch.cuda.is_current_stream_capturing()
            and get_tc_piecewise_forward_context() is None
            and layer.tp_q_head_num in (12, 6, 3)
            and layer.tp_k_head_num == layer.tp_v_head_num == 1
            and layer.head_dim == layer.v_head_dim == 256
            and layer.logit_cap == 0
            and layer.sliding_window_size == -1
            and not layer.is_cross_attention
            and layer.pos_encoding_mode == "NONE"
            and not any(value is not None for value in kwargs.values())
            and indices.shape == (q.shape[0], 2051)
            and q.shape[0] > 0
            and all(
                t is not None
                and t.ndim == 3
                and t.shape[-1] == 256
                and t.dtype == torch.bfloat16
                and t.device == q.device
                and not t.requires_grad
                and t.numel() * t.element_size() < 2**31
                for t in (q, k, v)
            )
        )

    def forward(self, *, q, k, v, indices, query_lens, prefix_lens, scale, layer_id):
        output = self.attention(
            q,
            k,
            v,
            indices,
            query_lens=query_lens,
            prefix_lens=prefix_lens,
            softmax_scale=scale,
        )
        layout = (layer_id, tuple(query_lens), tuple(prefix_lens), q.shape[1], scale)
        if self.validate and layout not in self.checked_layouts:
            from sglang.test.qsa_validation import check_attention

            check_attention(
                q=q,
                k=k,
                v=v,
                indices=indices,
                output=output,
                query_lens=query_lens,
                prefix_lens=prefix_lens,
                scale=scale,
            )
            self.checked_layouts.add(layout)
            logger.info(
                "PyHIP QSA attention checked: layer=%s queries=%s prefixes=%s",
                layer_id,
                query_lens,
                prefix_lens,
            )
        return output


class PyHIPIndexer:
    def __init__(self):
        from pyhip.ops.qsa.flydsl import indexer

        self.runtime = indexer
        self.validation = None
        if envs.SGLANG_TEST_PYHIP_QSA.get():
            from sglang.test.qsa_validation import IndexerValidation

            self.validation = IndexerValidation()

    @classmethod
    def create(cls, *, indexer):
        if (
            indexer.index_n_heads,
            indexer.index_kv_heads,
            indexer.index_head_dim,
            indexer.compress_ratio,
            indexer.block_topk,
            indexer.token_topk,
        ) != (4, 1, 128, 4, 512, 2048):
            return None
        rotary = indexer.rotary_emb
        if type(rotary) is MRotaryEmbedding:
            if rotary.mrope_interleaved_glm or len(rotary.mrope_section or ()) not in (
                0,
                3,
            ):
                return None
        elif type(rotary) is not RotaryEmbedding:
            return None
        if (
            not rotary.is_neox_style
            or rotary.rotary_dim != 64
            or not envs.SGLANG_USE_AITER.get()
        ):
            return None
        return cls()

    @staticmethod
    def _supports_frame(*, indexer, hidden, positions):
        cache = indexer.rotary_emb.cos_sin_cache
        return (
            hidden.is_cuda
            and hidden.dtype == torch.bfloat16
            and len(hidden) > 0
            and not torch.compiler.is_compiling()
            and torch.cuda.get_device_properties(hidden.device).gcnArchName.startswith(
                "gfx942"
            )
            and positions.dtype == torch.int64
            and positions.ndim in (1, 2)
            and positions.shape[-1] == len(hidden)
            and positions.stride(-1) == 1
            and (positions.ndim == 1 or positions.shape[0] == 3)
            # FP32-cache native prep has different rounding; retain that path.
            and cache.dtype == torch.bfloat16
            and cache.shape[1] == 64
            and cache.device == hidden.device
            and cache.is_contiguous()
            and all(
                w.dtype == torch.bfloat16
                and w.device == hidden.device
                and w.is_contiguous()
                for w in (indexer.q_layernorm.weight, indexer.k_layernorm.weight)
            )
        )

    @staticmethod
    def _common_inputs(*, indexer, hidden, positions, metadata, state_slots):
        pool = metadata.token_to_kv_pool
        key_state = pool.get_qsa_key_state_buffer(indexer.layer_id)
        compressed = pool.get_qsa_compressed_k_buffer(indexer.layer_id)
        rope_state = pool.qsa_rope_position_buffer
        if (
            any(
                t.dtype != torch.bfloat16
                or tuple(t.shape[1:]) != (1, 128)
                or not t.is_contiguous()
                or t.device != hidden.device
                for t in (key_state, compressed)
            )
            or rope_state.dtype != torch.int64
            or tuple(rope_state.shape[1:]) != (3,)
            or not rope_state.is_contiguous()
            or rope_state.device != hidden.device
        ):
            return None
        return dict(
            positions=positions,
            state_slots=state_slots[: len(hidden)].contiguous(),
            key_state=key_state,
            rope_state=rope_state,
            compressed=compressed,
            cos_sin_cache=indexer.rotary_emb.cos_sin_cache,
            axis_map=indexer._rope_axis_map(hidden.device),
            q_weight=indexer.q_layernorm.weight.data,
            k_weight=indexer.k_layernorm.weight.data,
            q_eps=indexer.q_layernorm.variance_epsilon,
            k_eps=indexer.k_layernorm.variance_epsilon,
        )

    def forward(
        self, *, indexer, hidden, positions, logical, batch, metadata, state_slots
    ):
        if not self._supports_frame(
            indexer=indexer, hidden=hidden, positions=positions
        ):
            return None
        if batch.forward_mode == ForwardMode.EXTEND:
            return self._prefill(
                indexer=indexer,
                hidden=hidden,
                positions=positions,
                logical=logical,
                batch=batch,
                metadata=metadata,
                state_slots=state_slots,
            )
        if batch.forward_mode == ForwardMode.DECODE and metadata.is_cuda_graph:
            return self._decode(
                indexer=indexer,
                hidden=hidden,
                positions=positions,
                logical=logical,
                metadata=metadata,
                state_slots=state_slots,
            )
        return None

    def _prefill(
        self, *, indexer, hidden, positions, logical, batch, metadata, state_slots
    ):
        if (
            metadata.is_cuda_graph
            or get_is_capture_mode()
            or torch.cuda.is_current_stream_capturing()
            or get_tc_piecewise_forward_context() is not None
            or batch.seq_lens_cpu is None
            or batch.extend_seq_lens_cpu is None
            or any(
                t is None
                for t in (
                    metadata.write_locs,
                    metadata.compress_member_rows,
                    metadata.compress_sequence_ids,
                    metadata.compress_group_positions,
                )
            )
        ):
            return None
        lengths = tuple(int(n) for n in batch.seq_lens_cpu)
        extends = tuple(int(n) for n in batch.extend_seq_lens_cpu)
        if (
            not lengths
            or len(lengths) != len(extends)
            or sum(extends) != len(hidden)
            or len(lengths) != metadata.sequence_lengths.numel()
            or any(e < 0 or s < e or (s - e) % 4 for s, e in zip(lengths, extends))
            or max(lengths)
            > min(
                indexer.rotary_emb.cos_sin_cache.shape[0],
                metadata.token_slot_table.shape[1],
            )
            or max(lengths) // 4 > self.runtime.MAX_COMPRESSED_KEYS
        ):
            return None
        inputs = self._common_inputs(
            indexer=indexer,
            hidden=hidden,
            positions=positions,
            metadata=metadata,
            state_slots=state_slots,
        )
        if inputs is None:
            return None
        rope = metadata.extend_rope_matrix
        if rope is None:
            rope = build_rope_position_matrix(positions, len(hidden))
        inputs.update(
            qk=indexer.index_qk_proj(hidden)[0].contiguous(),
            heads=4,
            logical_positions=logical.contiguous(),
            write_locs=metadata.write_locs,
            member_rows=metadata.compress_member_rows,
            group_sequences=metadata.compress_sequence_ids,
            group_ends=metadata.compress_group_positions,
            rope_matrix=rope[: len(hidden)].contiguous(),
            token_slot_table=metadata.token_slot_table,
            seq_lens=lengths,
            extend_lens=extends,
        )
        with torch.profiler.record_function("pyhip_qsa.indexer.prefill"):
            if (
                self.validation is not None
                and (lengths, extends) not in self.validation.prefill_layouts
            ):
                return self.validation.prefill(
                    indexer=indexer,
                    runtime=self.runtime,
                    hidden=hidden,
                    positions=positions,
                    logical=logical,
                    metadata=metadata,
                    inputs=inputs,
                )
            return self.runtime.prefill_indexer(**inputs)

    @staticmethod
    def _supports_decode(*, rows, device, cache, table, lengths):
        return (
            0 < rows < 65536
            and cache.dtype == torch.bfloat16
            and cache.ndim == 4
            and tuple(cache.shape[1:]) == (16, 1, 128)
            and cache.is_contiguous()
            and cache.numel() * cache.element_size() < 2**31
            and table.dtype == torch.int32
            and table.ndim == 2
            and table.shape[0] == rows
            and table.is_contiguous()
            and table.shape[1] > 0
            and lengths.dtype == torch.int32
            and lengths.shape == (rows,)
            and lengths.is_contiguous()
            and all(t.device == device for t in (cache, table, lengths))
        )

    def _decode(self, *, indexer, hidden, positions, logical, metadata, state_slots):
        if any(
            t is None
            for t in (
                metadata.graph_write_locs,
                metadata.graph_ring_group_locs,
                metadata.decode_logical_positions,
                metadata.graph_compressed_page_table,
                metadata.graph_compressed_lengths,
            )
        ):
            return None
        cache, table, lengths, _ = metadata.get_decode_mqa_inputs(indexer.layer_id)
        rows = len(hidden)
        groups, writes = (
            metadata.graph_ring_group_locs[:rows],
            metadata.graph_write_locs[:rows],
        )
        if (
            not self._supports_decode(
                rows=rows,
                device=hidden.device,
                cache=cache,
                table=table,
                lengths=lengths,
            )
            or state_slots.dtype != torch.int64
            or not state_slots.is_contiguous()
            or groups.dtype != torch.int32
            or groups.shape != (rows, 4)
            or not groups.is_contiguous()
            or writes.dtype != torch.int32
            or not writes.is_contiguous()
        ):
            return None
        inputs = self._common_inputs(
            indexer=indexer,
            hidden=hidden,
            positions=positions,
            metadata=metadata,
            state_slots=state_slots,
        )
        if inputs is None:
            return None
        inputs.update(
            qk=indexer.index_qk_proj(hidden)[0].contiguous(),
            group_locs=groups,
            write_locs=writes,
            cache=cache,
            page_table=table,
            lengths=lengths,
            query_positions=metadata.decode_logical_positions[:rows],
            sequence_lengths=metadata.get_seqlens_int32(),
        )
        with torch.profiler.record_function("pyhip_qsa.indexer.decode"):
            if self.validation is not None:
                return self.validation.decode(
                    indexer=indexer,
                    runtime=self.runtime,
                    hidden=hidden,
                    positions=positions,
                    logical=logical,
                    metadata=metadata,
                    inputs=inputs,
                )
            return self.runtime.decode_forward(**inputs)
