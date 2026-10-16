"""Direct PyHIP wiring: cache ownership, ragged prefill, graph decode and graph verify."""

import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa.metadata import (
    QSAIndexerMetadata,
    build_group_ring_slots,
    build_pending_ring_slots,
    qsa_ring_slots_per_request,
)
from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QSAMTPSharedSparseIndices,
    QwenSparseAttnBackend,
    QwenSparseAttnMetadata,
)
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
        self._check_attention_cache_and_padding(is_draft_worker=False)

    @torch.no_grad()
    def test_draft_prefill_attention_cache_and_padding(self):
        self._check_attention_cache_and_padding(is_draft_worker=True)

    def _check_attention_cache_and_padding(self, *, is_draft_worker):
        from pyhip.testing.qsa_reference import reference

        for heads in (12, 6, 3):
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
                        is_draft_worker=is_draft_worker,
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
                    # Supply the full-context slot plan that serving metadata
                    # prepares before attention; the fixture has no indexer pool.
                    backend.forward_metadata = SimpleNamespace(
                        packed_kv_slots=torch.cat(
                            [
                                table[request, :length]
                                for request, length in enumerate(lengths)
                            ]
                        )
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
                        backend._pyhip_attention.can_use_for_forward(
                            q=query[:rows],
                            k=keys[:rows],
                            v=values[:rows],
                            indices=indices,
                            layer=layer,
                            forward_mode=ForwardMode.TARGET_VERIFY,
                            kwargs={},
                        )
                    )
                    self.assertFalse(
                        backend._pyhip_attention.can_use_for_forward(
                            q=query[:rows],
                            k=keys[:rows],
                            v=values[:rows],
                            indices=indices,
                            layer=layer,
                            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
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

    @torch.no_grad()
    def test_draft_extend_indexer_eager_and_graph(self):
        from sglang.srt.layers.attention.qsa.graph_metadata import launch_graph_metadata

        with envs.SGLANG_USE_PYHIP_QSA.override(
            True
        ), envs.SGLANG_TEST_PYHIP_QSA.override(False):
            module = _indexer(self.device)
        wrapper = module._pyhip_indexer
        self.assertIsNotNone(wrapper)
        for bases in ((4,), (5,), (6,), (7,), (2046, 14)):
            with self.subTest(bases=bases):
                wrapper.validation = None
                pool = _pool(self.device)
                table = torch.stack(
                    [
                        torch.zeros(4096, dtype=torch.int32, device=self.device),
                        64 + torch.arange(4096, dtype=torch.int32, device=self.device),
                        4224
                        + torch.arange(4096, dtype=torch.int32, device=self.device),
                    ]
                )
                bs, window = len(bases), 4
                reqs = torch.arange(1, bs + 1, device=self.device)
                extends = torch.full(
                    (bs,), window, dtype=torch.int32, device=self.device
                )
                logical = torch.cat(
                    [
                        torch.arange(base, base + window, device=self.device)
                        for base in bases
                    ]
                )
                rows = len(logical)
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.DRAFT_EXTEND_V2,
                    positions=logical,
                    seq_lens=torch.tensor(
                        [b + window for b in bases], device=self.device
                    ),
                    seq_lens_cpu=torch.tensor([b + window for b in bases]),
                    extend_seq_lens=extends,
                    req_pool_indices=reqs,
                    batch_size=bs,
                    out_cache_loc=torch.zeros(
                        rows, dtype=torch.int64, device=self.device
                    ),
                    spec_info=SimpleNamespace(
                        draft_token_num=window,
                        num_front_tokens=0,
                        num_accept_tokens=torch.tensor(
                            [min(i + 2, window) for i in range(bs)], device=self.device
                        ),
                    ),
                    _original_forward_mode=None,
                )
                backend = QwenSparseAttnBackend(
                    SimpleNamespace(
                        device=self.device,
                        token_to_kv_pool=pool,
                        req_to_token_pool=SimpleNamespace(req_to_token=table),
                        model_config=SimpleNamespace(
                            hf_text_config=_config(), context_len=4096
                        ),
                        ps=SimpleNamespace(attn_cp_size=1, attn_dcp_size=1),
                        kv_cache_dtype=torch.bfloat16,
                        is_draft_worker=True,
                    )
                )
                eager = backend._metadata_from_forward_batch(batch).indexer_metadata
                positions = torch.stack([logical, logical + 3, logical + 7])
                hidden = torch.randn(
                    (rows, 2560), dtype=torch.bfloat16, device=self.device
                )

                def snapshot():
                    return tuple(
                        t.clone()
                        for t in (
                            pool.get_qsa_key_state_buffer(3),
                            pool.qsa_rope_position_buffer,
                            pool.get_qsa_compressed_k_buffer(3),
                        )
                    )

                def restore(saved):
                    for buffer, value in zip(
                        (
                            pool.get_qsa_key_state_buffer(3),
                            pool.qsa_rope_position_buffer,
                            pool.get_qsa_compressed_k_buffer(3),
                        ),
                        saved,
                    ):
                        buffer.copy_(value)

                def native(metadata):
                    module._pyhip_indexer = None
                    try:
                        with patch.object(
                            module,
                            "select_decode_tokens",
                            wraps=module.select_decode_tokens,
                        ) as select:
                            output = module.forward_cuda(
                                hidden, positions, batch, metadata
                            )
                            query = select.call_args.args[0][:, :4].double()
                        cache, pages, counts, _ = metadata.get_decode_mqa_inputs(3)
                        slots = pages.long()[:, :, None] * 16 + torch.arange(
                            16, device=self.device
                        )
                        keys = (
                            cache.reshape(-1, 128)[slots.flatten()]
                            .reshape(len(query), -1, 128)
                            .double()
                        )
                        logits = (
                            torch.einsum("rhd,rnd->rhn", query, keys).relu().sum(1)
                            / 128**0.5
                        )
                        return output, logits, counts
                    finally:
                        module._pyhip_indexer = wrapper

                def check(output, expected, state):
                    from sglang.test.qsa_validation import _selection_valid

                    tokens, logits, counts = expected
                    # Judge near-tied top-k boundaries against independent
                    # FP64 logits instead of requiring native token identity.
                    real = torch.ones(len(output), dtype=torch.bool, device=self.device)
                    self.assertTrue(
                        bool(
                            _selection_valid(
                                logits=logits,
                                lengths=counts,
                                actual=output,
                                expected=tokens,
                                real=real,
                            )
                        )
                    )
                    for index, (actual, wanted) in enumerate(zip(snapshot(), state)):
                        # Reserved ring/request 0 and compressed slot 0 are
                        # inert sinks, so their concurrent writes are ignored.
                        first = 1 if index == 2 else qsa_ring_slots_per_request(4)
                        torch.testing.assert_close(
                            actual[first:], wanted[first:], atol=0, rtol=0
                        )

                initial = snapshot()
                expected = native(eager)
                expected_state = snapshot()
                restore(initial)
                with patch.object(
                    wrapper.runtime,
                    "decode_indexer",
                    wraps=wrapper.runtime.decode_indexer,
                ) as decode:
                    output = module.forward_cuda(hidden, positions, batch, eager)
                    self.assertEqual(decode.call_count, 1)
                    self.assertTrue(decode.call_args.kwargs["verify"])
                torch.cuda.synchronize()
                check(output, expected, expected_state)

                # Unsupported widened windows must decline before mutating
                # the pending ring or compressed cache.
                before = snapshot()
                batch.spec_info.num_front_tokens = 1
                self.assertIsNone(
                    wrapper.forward(
                        indexer=module,
                        hidden=hidden,
                        positions=positions,
                        logical=eager.decode_logical_positions,
                        batch=batch,
                        metadata=eager,
                        state_slots=eager.pending_ring_slots,
                    )
                )
                for actual, wanted in zip(snapshot(), before):
                    torch.testing.assert_close(actual, wanted, atol=0, rtol=0)
                batch.spec_info.num_front_tokens = 0

                from sglang.test.qsa_validation import IndexerValidation

                wrapper.validation = IndexerValidation()
                module.forward_cuda(hidden, positions, batch, eager)
                torch.cuda.synchronize()
                self.assertEqual(int(wrapper.validation.decode_checks), 1)
                wrapper.validation = None

                # Graph buckets include one padding request (slot 0). Build
                # row metadata from stable GPU request lengths inside the
                # graph, just as the serving prologue does.
                graph_bs, graph_rows = bs + 1, (bs + 1) * window
                hidden = torch.randn(
                    (graph_rows, 2560), dtype=torch.bfloat16, device=self.device
                )
                positions = torch.empty(
                    (3, graph_rows), dtype=torch.int64, device=self.device
                )
                end_lengths = torch.tensor(
                    [b + window for b in bases] + [1],
                    dtype=torch.int32,
                    device=self.device,
                )
                graph_reqs = torch.cat([reqs, reqs.new_zeros(1)])
                graph_extends = torch.full(
                    (graph_bs,), window, dtype=torch.int32, device=self.device
                )
                row_lengths = torch.empty(
                    graph_rows, dtype=torch.int32, device=self.device
                )
                row_reqs = torch.empty_like(row_lengths)
                graph_metadata = QSAIndexerMetadata(
                    sequence_lengths=row_lengths,
                    token_to_batch_idx=torch.arange(
                        graph_rows, dtype=torch.int32, device=self.device
                    ),
                    token_slot_table=torch.zeros(
                        (graph_rows, 1), dtype=torch.int32, device=self.device
                    ),
                    out_cache_loc=torch.zeros(
                        graph_rows, dtype=torch.int64, device=self.device
                    ),
                    token_to_kv_pool=pool,
                    compress_ratio=4,
                    block_topk=512,
                    req_pool_indices=row_reqs,
                    is_cuda_graph=True,
                    graph_write_locs=torch.empty_like(row_lengths),
                    graph_compressed_lengths=torch.empty_like(row_lengths),
                    graph_compressed_page_table=torch.empty(
                        (graph_rows, 64), dtype=torch.int32, device=self.device
                    ),
                    graph_prefix_lengths=torch.empty_like(row_lengths),
                    decode_logical_positions=torch.empty_like(row_lengths),
                    pending_ring_slots=torch.empty(
                        graph_rows, dtype=torch.int64, device=self.device
                    ),
                    graph_ring_group_locs=torch.empty(
                        (graph_rows, 4), dtype=torch.int32, device=self.device
                    ),
                )
                metadata = QwenSparseAttnMetadata(
                    sequence_lengths=row_lengths,
                    token_to_batch_idx=graph_metadata.token_to_batch_idx,
                    token_slot_table=graph_metadata.token_slot_table,
                    indexer_metadata=graph_metadata,
                    row_req_pool_indices=row_reqs,
                    is_cuda_graph=True,
                )
                state = QSAMTPSharedSparseIndices(
                    layer_ids=[3],
                    num_requests=3,
                    token_topk=2051,
                    tail_width=4,
                    device=self.device,
                )
                backend.set_mtp_shared_sparse_indices(state)
                batch.batch_size = graph_bs
                batch.req_pool_indices = graph_reqs
                batch.extend_seq_lens = graph_extends
                batch.spec_info.num_accept_tokens = torch.tensor(
                    [2] * bs + [4], device=self.device
                )

                def prepare():
                    launch_graph_metadata(
                        mode=2,
                        bs=graph_bs,
                        num_rows=graph_rows,
                        seq_lens=end_lengths,
                        req_pool_indices=graph_reqs,
                        extend_lens=graph_extends,
                        extend_len=0,
                        num_padding=1,
                        metadata=metadata,
                        req_to_token=table,
                        pool=pool,
                    )
                    logical = graph_metadata.decode_logical_positions.long()
                    positions.copy_(torch.stack([logical, logical + 3, logical + 7]))

                def run():
                    prepare()
                    output = module.forward_cuda(
                        hidden, positions, batch, graph_metadata
                    )
                    backend.capture_mtp_sparse_indices(output, batch, 3, graph_metadata)
                    return output

                run()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with model_capture_mode(), torch.cuda.graph(graph):
                    output = run()
                for accepted in (1, 3, 4):
                    end_lengths[:bs].add_(accepted)
                    batch.spec_info.num_accept_tokens[:bs].fill_(accepted)
                    hidden.normal_()
                    prepare()
                    initial = snapshot()
                    expected = native(graph_metadata)
                    expected_state = snapshot()
                    restore(initial)
                    graph.replay()
                    torch.cuda.synchronize()
                    check(output, expected, expected_state)
                    for request in range(bs):
                        anchor = request * window + accepted - 1
                        torch.testing.assert_close(
                            state.indices[0, request + 1, :2051],
                            output[anchor],
                            atol=0,
                            rtol=0,
                        )
                        self.assertEqual(
                            int(state.captured_len[0, request + 1]),
                            int(row_lengths[anchor]),
                        )

                # The opt-in native comparison must itself record and replay.
                wrapper.validation = IndexerValidation()
                run()
                torch.cuda.synchronize()
                validation_graph = torch.cuda.CUDAGraph()
                with model_capture_mode(), torch.cuda.graph(validation_graph):
                    run()
                end_lengths[:bs].add_(1)
                hidden.neg_()
                validation_graph.replay()
                torch.cuda.synchronize()
                self.assertEqual(int(wrapper.validation.decode_checks), 2)
                wrapper.validation = None


if __name__ == "__main__":
    unittest.main()
