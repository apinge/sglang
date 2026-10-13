"""Direct PyHIP wiring: cache ownership, ragged prefill, graph decode and graph verify."""

import importlib.util
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa.metadata import (
    QSAIndexerMetadata,
    build_group_ring_slots,
    build_pending_ring_slots,
    qsa_ring_slots_per_request,
)
from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd")


def _config():
    return SimpleNamespace(
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
        hidden_size=2560,
        rms_norm_eps=1e-6,
    )


def _pool(device):
    ring = 3 * qsa_ring_slots_per_request(4)
    pool = QSATokenToKVPool.__new__(QSATokenToKVPool)
    pool.full_attention_layer_id_mapping = {3: 0}
    pool.qsa_key_state_buffer_pool = [
        torch.zeros((ring, 1, 128), dtype=torch.bfloat16, device=device)
    ]
    pool.qsa_compressed_k_buffer_pool = [
        torch.randn((2112, 1, 128), dtype=torch.bfloat16, device=device)
    ]
    pool.qsa_rope_position_buffer = torch.zeros(
        (ring, 3), dtype=torch.int64, device=device
    )
    pool.qsa_compress_ratio = 4
    pool.qsa_index_head_dim = 128
    pool.qsa_index_kv_heads = 1
    pool.qsa_compressed_page_size = 16
    pool.qsa_block_topk = 512
    pool.qsa_token_topk = 2048
    return pool


def _indexer(device):
    rotary = MRotaryEmbedding(
        head_size=256,
        rotary_dim=64,
        max_position_embeddings=8192,
        base=1000000,
        is_neox_style=True,
        dtype=torch.bfloat16,
        mrope_section=[11, 11, 10],
        mrope_interleaved=True,
    )
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        module = QSAIndexer(_config(), layer_id=3, rotary_emb=rotary).to(device)
    finally:
        torch.set_default_dtype(previous)
    with torch.no_grad():
        module.index_qk_proj.weight.normal_(std=0.02)
        module.q_layernorm.weight.normal_(std=0.1)
        module.k_layernorm.weight.normal_(std=0.1)
    return module


def _ring_slots(requests, logical):
    """Graph rows' pending-ring store slots and group member slots, as SGLang builds them."""
    rows = torch.arange(len(logical), device=logical.device)
    slots = build_pending_ring_slots(
        token_to_batch_idx=rows,
        req_pool_indices=requests,
        sequence_lengths=logical + 1,
        logical_positions=logical,
        compress_ratio=4,
        is_extend=False,
    )
    groups = build_group_ring_slots(
        req_pool_indices=requests,
        group_end_positions=logical.long(),
        sequence_ids=rows,
        compress_ratio=4,
    )
    return slots, groups.int()


class _AttentionPool:
    def __init__(self, keys, values):
        self.keys = keys
        self.values = values
        self.write_calls = 0
        self.read_calls = 0

    def set_kv_buffer(self, layer, locations, keys, values):
        self.keys.index_copy_(0, locations.long(), keys)
        self.values.index_copy_(0, locations.long(), values)
        self.write_calls += 1

    def get_key_buffer(self, layer_id):
        self.read_calls += 1
        return self.keys

    def get_value_buffer(self, layer_id):
        self.read_calls += 1
        return self.values


@unittest.skipUnless(torch.version.hip is not None, "ROCm only")
class TestPyHIPQSA(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not torch.cuda.get_device_properties().gcnArchName.startswith("gfx942"):
            raise unittest.SkipTest("PyHIP QSA requires gfx942")
        if any(importlib.util.find_spec(name) is None for name in ("pyhip", "flydsl")):
            raise unittest.SkipTest("Install the qsa_pyhip optional dependencies")
        if not envs.SGLANG_USE_AITER.get():
            raise unittest.SkipTest("PyHIP indexer prep requires SGLANG_USE_AITER=1")
        __import__("pyhip.ops.qsa.flydsl.attention")
        publish(ServerArgs(model_path="dummy", attention_backend="aiter"), role="test")
        cls.device = torch.device("cuda", torch.cuda.current_device())

    @classmethod
    def tearDownClass(cls):
        reset_context()
        super().tearDownClass()

    def setUp(self):
        super().setUp()
        torch.manual_seed(42)

    @torch.no_grad()
    def test_attention_cache_and_padding(self):
        from pyhip.testing.qsa_reference import reference

        for heads in (12, 6):
            for queries, prefixes in (((33,), (0,)), ((7, 0, 9), (2048, 0, 12))):
                with self.subTest(heads=heads, queries=queries, prefixes=prefixes):
                    rows = sum(queries)
                    lengths = tuple(
                        count + prefix for count, prefix in zip(queries, prefixes)
                    )
                    width = max(lengths) + 64
                    table = (
                        torch.arange(len(queries) * width, device=self.device)
                        .int()
                        .reshape(len(queries), width)
                    )
                    table = table.roll(64, dims=1)
                    shape = (len(queries) * width + 2, 1, 256)
                    pool = _AttentionPool(
                        torch.randn(shape, dtype=torch.bfloat16, device=self.device),
                        torch.randn(shape, dtype=torch.bfloat16, device=self.device),
                    )
                    req_pool = SimpleNamespace(req_to_token=table)
                    runner = SimpleNamespace(
                        device=self.device,
                        token_to_kv_pool=pool,
                        req_to_token_pool=req_pool,
                        model_config=SimpleNamespace(
                            hf_text_config=_config(), context_len=8192
                        ),
                        is_draft_worker=False,
                        kv_cache_dtype=torch.bfloat16,
                        ps=SimpleNamespace(attn_cp_size=1, attn_dcp_size=1),
                    )
                    with envs.SGLANG_USE_PYHIP_QSA.override(
                        True
                    ), envs.SGLANG_TEST_PYHIP_QSA.override(True):
                        backend = QwenSparseAttnBackend(runner)
                    self.assertIsNotNone(backend._pyhip_attention)
                    locations = torch.cat(
                        [
                            table[request, prefix : prefix + count]
                            for request, (count, prefix) in enumerate(
                                zip(queries, prefixes)
                            )
                        ]
                        + [
                            torch.arange(
                                shape[0] - 2, shape[0], device=self.device
                            ).int()
                        ]
                    )
                    query = torch.randn(
                        (rows + 2, heads, 256), dtype=torch.bfloat16, device=self.device
                    )
                    keys = torch.randn(
                        (rows + 2, 1, 256), dtype=torch.bfloat16, device=self.device
                    )
                    values = torch.randn_like(keys)
                    indices = torch.full(
                        (rows, 2051), -1, dtype=torch.int32, device=self.device
                    )
                    row = 0
                    for count, prefix in zip(queries, prefixes):
                        for position in range(prefix, prefix + count):
                            visible = position + 1
                            complete = min(visible // 4, 512) * 4
                            indices[row, :complete] = torch.arange(
                                complete, device=self.device
                            )
                            indices[row, complete : complete + visible % 4] = (
                                torch.arange(
                                    visible // 4 * 4, visible, device=self.device
                                )
                            )
                            row += 1
                    batch = SimpleNamespace(
                        forward_mode=ForwardMode.EXTEND,
                        out_cache_loc=locations,
                        extend_seq_lens_cpu=queries,
                        seq_lens_cpu=lengths,
                        extend_seq_lens=torch.tensor(queries, device=self.device),
                        req_pool_indices=torch.arange(len(queries), device=self.device),
                    )
                    layer = SimpleNamespace(
                        tp_q_head_num=heads,
                        tp_k_head_num=1,
                        tp_v_head_num=1,
                        head_dim=256,
                        v_head_dim=256,
                        scaling=1 / 16,
                        layer_id=3,
                        logit_cap=0,
                        sliding_window_size=-1,
                        is_cross_attention=False,
                        pos_encoding_mode="NONE",
                    )
                    output = backend.forward_extend(
                        query, keys, values, layer, batch, topk_indices=indices
                    )
                    self.assertEqual(pool.write_calls, 1)
                    self.assertEqual(pool.read_calls, 2 if any(prefixes) else 0)
                    packed_keys = torch.cat(
                        [
                            pool.keys[table[request, :length].long()]
                            for request, length in enumerate(lengths)
                        ]
                    )
                    packed_values = torch.cat(
                        [
                            pool.values[table[request, :length].long()]
                            for request, length in enumerate(lengths)
                        ]
                    )
                    expected = reference(
                        query[:rows],
                        packed_keys,
                        packed_values,
                        indices,
                        query_lens=queries,
                        prefix_lens=prefixes,
                        scale=1 / 16,
                    )
                    torch.testing.assert_close(
                        output[:rows].reshape_as(query[:rows]).float(),
                        expected,
                        rtol=0.02,
                        atol=0.02,
                    )
                    self.assertEqual(output.shape, (rows + 2, heads * 256))
                    self.assertTrue(bool((output[rows:] == 0).all()))
                    self.assertEqual(len(backend._pyhip_attention.checked_layouts), 1)
                    self.assertFalse(
                        backend._pyhip_attention.supports(
                            q=query[:rows],
                            k=keys[:rows],
                            v=values[:rows],
                            indices=indices,
                            layer=layer,
                            forward_mode=ForwardMode.TARGET_VERIFY,
                            kwargs={},
                        )
                    )

    def test_selection_validator_rejects_invalid_output(self):
        from sglang.test.qsa_validation import _selection_valid

        logits = torch.arange(520, 0, -1, device=self.device).float()[None]
        lengths = torch.tensor([520], dtype=torch.int32, device=self.device)
        tokens = torch.cat(
            [
                torch.arange(2048, device=self.device),
                torch.arange(2080, 2083, device=self.device),
            ]
        ).int()[None]
        real = torch.ones(1, dtype=torch.bool, device=self.device)

        def valid(actual, scores=logits):
            return bool(
                _selection_valid(
                    logits=scores,
                    lengths=lengths,
                    actual=actual,
                    expected=tokens,
                    real=real,
                )
            )

        self.assertTrue(valid(tokens))
        repeated = tokens.clone()
        repeated[0, 1] = 0
        self.assertFalse(valid(repeated))
        suboptimal = tokens.clone()
        suboptimal[0, :4] = torch.arange(2076, 2080, device=self.device)
        self.assertFalse(valid(suboptimal))
        self.assertTrue(
            bool(
                _selection_valid(
                    logits=logits,
                    lengths=lengths,
                    actual=tokens,
                    expected=suboptimal,
                    real=real,
                )
            )
        )
        nonfinite = logits.clone()
        nonfinite[0, 0] = float("nan")
        self.assertFalse(valid(tokens, nonfinite))

    @torch.no_grad()
    def test_prefill_and_graph_decode(self):
        with envs.SGLANG_USE_PYHIP_QSA.override(
            True
        ), envs.SGLANG_TEST_PYHIP_QSA.override(True):
            module = _indexer(self.device)
        self.assertIsNotNone(module._pyhip_indexer)
        pool = _pool(self.device)
        lengths, extends = (2055, 15), (7, 15)
        prefix = torch.tensor([2048, 0], device=self.device)
        seq = torch.tensor(lengths, dtype=torch.int32, device=self.device)
        ext = torch.tensor(extends, device=self.device)
        table = torch.stack(
            [
                64 + torch.arange(4096, dtype=torch.int32, device=self.device),
                4224 + torch.arange(4096, dtype=torch.int32, device=self.device),
            ]
        )
        logical = torch.cat(
            [
                torch.arange(2048, 2055, device=self.device),
                torch.arange(15, device=self.device),
            ]
        )
        positions = torch.stack([logical, logical + 3, logical + 7])
        writes, ends, sequences, members = QwenSparseAttnBackend._qsa_write_plan(
            token_slot_table=table,
            start_blocks=prefix // 4,
            end_blocks=seq.long() // 4,
            capacity=sum(extends) // 4 + 2,
            compress_ratio=4,
            row_token_starts=torch.cumsum(ext, 0) - ext,
            prefix_lens=prefix,
        )
        metadata = QSAIndexerMetadata(
            sequence_lengths=seq,
            token_to_batch_idx=torch.repeat_interleave(
                torch.arange(2, device=self.device).int(), ext
            ),
            token_slot_table=table,
            out_cache_loc=torch.zeros(
                sum(extends), dtype=torch.int64, device=self.device
            ),
            token_to_kv_pool=pool,
            compress_ratio=4,
            block_topk=512,
            req_pool_indices=torch.tensor([1, 2], device=self.device),
            write_locs=writes,
            compress_group_positions=ends,
            compress_sequence_ids=sequences,
            compress_member_rows=members,
        )
        hidden = torch.randn(
            (sum(extends), 2560), device=self.device, dtype=torch.bfloat16
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            positions=logical,
            seq_lens_cpu=torch.tensor(lengths),
            extend_seq_lens_cpu=extends,
        )
        output = module.forward_cuda(hidden, positions, batch, metadata)
        self.assertEqual(output.shape, (sum(extends), 2051))
        self.assertEqual(module._pyhip_indexer.validation.prefill_checks, 1)
        self._check_graph_decode(module, pool, table, lengths)
        self._check_graph_verify(module, pool, table, (lengths[0] + 2, lengths[1] + 2))

    def _check_graph_decode(self, module, pool, table, lengths):
        seq = torch.tensor(
            [n + 1 for n in lengths], dtype=torch.int32, device=self.device
        )
        logical = seq - 1
        requests = torch.tensor([1, 2], dtype=torch.int64, device=self.device)
        slots, groups = _ring_slots(requests, logical.long())
        writes = (table[torch.arange(2, device=self.device), logical.long()] // 4).int()
        metadata = QSAIndexerMetadata(
            sequence_lengths=seq,
            token_to_batch_idx=torch.arange(2, dtype=torch.int32, device=self.device),
            token_slot_table=table,
            out_cache_loc=torch.zeros(2, dtype=torch.int64, device=self.device),
            token_to_kv_pool=pool,
            compress_ratio=4,
            block_topk=512,
            req_pool_indices=requests,
            is_cuda_graph=True,
            graph_write_locs=writes,
            graph_compressed_page_table=(table[:, ::64] // 64).contiguous(),
            graph_compressed_lengths=seq // 4,
            decode_logical_positions=logical,
            pending_ring_slots=slots,
            graph_ring_group_locs=groups,
        )
        hidden = torch.randn((2, 2560), device=self.device, dtype=torch.bfloat16)
        positions = torch.stack(
            [logical.long(), logical.long() + 3, logical.long() + 7]
        )
        batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
        module.forward_cuda(hidden, positions, batch, metadata)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with model_capture_mode(), torch.cuda.graph(graph):
            output = module.forward_cuda(hidden, positions, batch, metadata)
        seq.add_(1)
        logical.add_(1)
        for buffer, value in zip(
            (slots, groups), _ring_slots(requests, logical.long())
        ):
            buffer.copy_(value)
        writes.zero_()
        positions.add_(1)
        hidden.neg_()
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(output.shape, (2, 2051))
        self.assertGreater(
            int(module._pyhip_indexer.validation.decode_checks.item()), 1
        )

    def _check_graph_verify(self, module, pool, table, bases):
        """Four-row TARGET_VERIFY windows that cross a compression boundary must match
        native prep, ring writes, compression and selection, eager and in a graph."""
        rows = 8
        owner = torch.arange(2, device=self.device).repeat_interleave(4)
        requests = owner + 1
        offsets = torch.arange(4, device=self.device).repeat(2)
        seq = torch.empty(rows, dtype=torch.int32, device=self.device)
        logical = torch.empty(rows, dtype=torch.int64, device=self.device)
        slots = torch.empty(rows, dtype=torch.int64, device=self.device)
        groups = torch.empty((rows, 4), dtype=torch.int32, device=self.device)
        writes = torch.empty(rows, dtype=torch.int32, device=self.device)
        compressed = torch.empty(rows, dtype=torch.int32, device=self.device)
        positions = torch.empty((3, rows), dtype=torch.int64, device=self.device)
        hidden = torch.empty((rows, 2560), device=self.device, dtype=torch.bfloat16)

        def step(window_bases):
            seq.copy_(
                torch.tensor(window_bases, device=self.device).repeat_interleave(4)
                + 1
                + offsets
            )
            logical.copy_(seq - 1)
            for buffer, value in zip((slots, groups), _ring_slots(requests, logical)):
                buffer.copy_(value)
            boundary = table[owner, logical] // 4
            writes.copy_(torch.where(seq % 4 == 0, boundary, 0))
            compressed.copy_(seq // 4)
            positions.copy_(torch.stack([logical, logical + 3, logical + 7]))
            hidden.normal_()

        step(bases)
        metadata = QSAIndexerMetadata(
            sequence_lengths=seq,
            token_to_batch_idx=torch.arange(
                rows, dtype=torch.int32, device=self.device
            ),
            token_slot_table=table[owner].contiguous(),
            out_cache_loc=torch.zeros(rows, dtype=torch.int64, device=self.device),
            token_to_kv_pool=pool,
            compress_ratio=4,
            block_topk=512,
            req_pool_indices=requests,
            is_cuda_graph=True,
            graph_write_locs=writes,
            graph_compressed_page_table=(table[owner, ::64] // 64).contiguous(),
            graph_compressed_lengths=compressed,
            decode_logical_positions=logical,
            pending_ring_slots=slots,
            graph_ring_group_locs=groups,
        )
        batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
        checks = int(module._pyhip_indexer.validation.decode_checks.item())
        module.forward_cuda(hidden, positions, batch, metadata)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with model_capture_mode(), torch.cuda.graph(graph):
            output = module.forward_cuda(hidden, positions, batch, metadata)
        # One request accepts its whole window, the other rewrites its rejected tail.
        step((bases[0] + 4, bases[1] + 1))
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(output.shape, (rows, 2051))
        self.assertEqual(
            int(module._pyhip_indexer.validation.decode_checks.item()), checks + 2
        )


if __name__ == "__main__":
    unittest.main()
