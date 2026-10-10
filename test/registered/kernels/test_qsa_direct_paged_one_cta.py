from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.elementwise.fast_topk import fast_topk
from sglang.srt.layers.attention import qwen_sparse_attn_backend as backend_module
from sglang.srt.layers.attention.qsa import direct_paged_one_cta as direct_module
from sglang.srt.layers.attention.qsa import sparse_attn as sparse_module
from sglang.srt.layers.attention.qsa.direct_paged_one_cta import (
    is_sparse_gqa_direct_paged_one_cta_supported,
)
from sglang.srt.layers.attention.qsa.kernel import expand_qsa_block_indices
from sglang.srt.layers.attention.qsa.metadata import QSAIndexerMetadata
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QwenSparseAttnBackend,
    QwenSparseAttnMetadata,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, stage="stage-b", runner_config="1-gpu-large-amd")


Q_HEADS = 12
KV_HEADS = 1
HEAD_DIM = 256
PAGE_SIZE = 64
COMPRESS_RATIO = 4
BLOCK_TOPK = 512
TOPK = BLOCK_TOPK * COMPRESS_RATIO + COMPRESS_RATIO - 1
CACHE_TOKENS = 2112
SCALE = HEAD_DIM**-0.5


def _is_mi308x() -> bool:
    return bool(
        torch.cuda.is_available()
        and torch.version.hip is not None
        and torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0] == "gfx942"
        and "MI308X" in torch.cuda.get_device_name(0)
    )


def _valid_count(visible: int) -> int:
    groups = min(max(visible, 0) // COMPRESS_RATIO, BLOCK_TOPK)
    return min(TOPK, groups * COMPRESS_RATIO + max(visible, 0) % COMPRESS_RATIO)


def _fill_indices(indices: torch.Tensor, lengths: list[int], seed: int) -> None:
    torch.manual_seed(seed)
    indices.fill_(-1)
    offsets = torch.arange(COMPRESS_RATIO, dtype=torch.int64, device=indices.device)
    for row, length in enumerate(lengths):
        count = _valid_count(length)
        tail = length % COMPRESS_RATIO
        selected_groups = (count - tail) // COMPRESS_RATIO
        if selected_groups:
            groups = torch.randperm(
                length // COMPRESS_RATIO,
                dtype=torch.int64,
                device=indices.device,
            )[:selected_groups]
            expanded = (groups[:, None] * COMPRESS_RATIO + offsets[None, :]).reshape(-1)
        else:
            expanded = indices.new_empty(0, dtype=torch.int64)
        if tail:
            expanded = torch.cat(
                (
                    expanded,
                    torch.arange(
                        length - tail,
                        length,
                        dtype=torch.int64,
                        device=indices.device,
                    ),
                )
            )
        indices[row, :count].copy_(expanded.to(torch.int32))


class _PagedPool:
    qsa_compressed_page_size = PAGE_SIZE // COMPRESS_RATIO
    qsa_compress_ratio = COMPRESS_RATIO
    qsa_block_topk = BLOCK_TOPK

    def __init__(self, keys, values):
        self.keys = keys
        self.values = values

    def get_key_buffer(self, layer_id):
        return self.keys[layer_id]

    def get_value_buffer(self, layer_id):
        return self.values[layer_id]


def _make_backend_state(batch: int, lengths: list[int], seed: int, layers: int = 1):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    pages_per_request = CACHE_TOKENS // PAGE_SIZE
    num_pages = batch * pages_per_request
    page_table = (
        torch.randperm(num_pages, dtype=torch.int64, device=device)
        .view(batch, pages_per_request)
        .to(torch.int32)
    )
    row_to_page_table = torch.randperm(batch, dtype=torch.int64, device=device).to(
        torch.int32
    )
    sequence_lens = torch.tensor(lengths, dtype=torch.int32, device=device)
    indices = torch.empty(batch, TOPK, dtype=torch.int32, device=device)
    _fill_indices(indices, lengths, seed + 10000)
    q = torch.randn(batch, Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    keys = {
        layer_id: torch.randn(
            num_pages * PAGE_SIZE,
            KV_HEADS,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        for layer_id in range(layers)
    }
    values = {layer_id: torch.randn_like(keys[layer_id]) for layer_id in keys}
    pool = _PagedPool(keys, values)
    indexer = QSAIndexerMetadata(
        sequence_lengths=sequence_lens,
        token_to_batch_idx=row_to_page_table,
        token_slot_table=torch.zeros(batch, 1, dtype=torch.int32, device=device),
        out_cache_loc=torch.zeros(batch, dtype=torch.int64, device=device),
        token_to_kv_pool=pool,
        compress_ratio=COMPRESS_RATIO,
        block_topk=BLOCK_TOPK,
        req_pool_indices=torch.arange(batch, dtype=torch.int32, device=device),
        is_cuda_graph=True,
        graph_compressed_page_table=page_table,
        graph_compressed_lengths=torch.div(
            sequence_lens, COMPRESS_RATIO, rounding_mode="floor"
        ),
    )
    metadata = QwenSparseAttnMetadata(
        sequence_lengths=sequence_lens,
        token_to_batch_idx=row_to_page_table,
        token_slot_table=indexer.token_slot_table,
        indexer_metadata=indexer,
        row_req_pool_indices=indexer.req_pool_indices,
        is_cuda_graph=True,
    )
    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend.token_to_kv_pool = pool
    backend.forward_metadata = metadata
    backend._fa2_scratch = {}
    backend._cuda_graph_max_tokens = batch
    backend.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.zeros(batch, CACHE_TOKENS, dtype=torch.int32, device=device)
    )
    return SimpleNamespace(
        backend=backend,
        pool=pool,
        metadata=metadata,
        page_table=page_table,
        row_to_page_table=row_to_page_table,
        sequence_lens=sequence_lens,
        indices=indices,
        q=q,
        forward_batch=SimpleNamespace(),
    )


def _update_state(state, lengths: list[int], seed: int) -> None:
    torch.manual_seed(seed)
    state.q.copy_(torch.randn_like(state.q))
    for layer_id in state.pool.keys:
        state.pool.keys[layer_id].copy_(torch.randn_like(state.pool.keys[layer_id]))
        state.pool.values[layer_id].copy_(torch.randn_like(state.pool.values[layer_id]))
    state.page_table.copy_(
        torch.randperm(
            state.page_table.numel(), dtype=torch.int64, device=state.q.device
        ).view_as(state.page_table)
    )
    state.row_to_page_table.copy_(
        torch.randperm(state.q.shape[0], dtype=torch.int64, device=state.q.device).to(
            torch.int32
        )
    )
    state.sequence_lens.copy_(
        torch.tensor(lengths, dtype=torch.int32, device=state.q.device)
    )
    _fill_indices(state.indices, lengths, seed + 10000)


def _reference(state, layer_id: int) -> torch.Tensor:
    rows = []
    keys = state.pool.keys[layer_id].view(-1, PAGE_SIZE, KV_HEADS, HEAD_DIM)
    values = state.pool.values[layer_id].view_as(keys)
    num_pages = keys.shape[0]
    for row in range(state.q.shape[0]):
        page_row = int(state.row_to_page_table[row].item())
        length = int(state.sequence_lens[row].item())
        logical = state.indices[row].long()
        logical = logical[(logical >= 0) & (logical < length)]
        if page_row < 0 or page_row >= state.page_table.shape[0] or not logical.numel():
            rows.append(torch.zeros_like(state.q[row], dtype=torch.float32))
            continue
        page_columns = torch.div(logical, PAGE_SIZE, rounding_mode="floor")
        in_table = page_columns < state.page_table.shape[1]
        logical = logical[in_table]
        physical_pages = (
            state.page_table[page_row].long().index_select(0, page_columns[in_table])
        )
        page_valid = (physical_pages >= 0) & (physical_pages < num_pages)
        logical = logical[page_valid]
        physical_pages = physical_pages[page_valid]
        if not logical.numel():
            rows.append(torch.zeros_like(state.q[row], dtype=torch.float32))
            continue
        page_offsets = logical % PAGE_SIZE
        row_keys = keys[physical_pages, page_offsets, 0].float()
        row_values = values[physical_pages, page_offsets, 0].float()
        scores = state.q[row].float() @ row_keys.T * SCALE
        rows.append(torch.softmax(scores, dim=-1) @ row_values)
    return torch.stack(rows)


def _packed_inputs(state, layer_id: int):
    """Build the compact representation consumed by the original kernel."""

    keys = state.pool.keys[layer_id].view(-1, PAGE_SIZE, KV_HEADS, HEAD_DIM)
    values = state.pool.values[layer_id].view_as(keys)
    packed_keys = []
    packed_values = []
    counts = []
    relative_indices = torch.full_like(state.indices, -1)
    for row in range(state.q.shape[0]):
        page_row = int(state.row_to_page_table[row].item())
        length = int(state.sequence_lens[row].item())
        logical = state.indices[row].long()
        logical = logical[(logical >= 0) & (logical < length)]
        page_columns = torch.div(logical, PAGE_SIZE, rounding_mode="floor")
        physical_pages = state.page_table[page_row].long().index_select(0, page_columns)
        page_offsets = logical % PAGE_SIZE
        packed_keys.append(keys[physical_pages, page_offsets])
        packed_values.append(values[physical_pages, page_offsets])
        count = logical.numel()
        counts.append(count)
        relative_indices[row, :count] = torch.arange(
            count, dtype=torch.int32, device=state.q.device
        )
    cu_k = torch.tensor(
        [0, *torch.tensor(counts).cumsum(0).tolist()],
        dtype=torch.int32,
        device=state.q.device,
    )
    cu_q = torch.arange(state.q.shape[0] + 1, dtype=torch.int32, device=state.q.device)
    kv_lens = torch.tensor(counts, dtype=torch.int32, device=state.q.device)
    return (
        torch.cat(packed_keys).contiguous(),
        torch.cat(packed_values).contiguous(),
        relative_indices,
        cu_q,
        cu_k,
        kv_lens,
    )


def test_qsa_packed_one_cta_mi308x_launch_config(monkeypatch):
    fallback = (32, 8, 2, {})
    monkeypatch.setattr(sparse_module, "_get_best_config", lambda total_q: fallback[:3])
    target = (1, TOPK, Q_HEADS, KV_HEADS, HEAD_DIM)
    dtypes = (torch.bfloat16, torch.bfloat16, torch.bfloat16)

    monkeypatch.setattr(sparse_module, "is_gfx942_supported", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda index: "AMD Instinct MI308X"
    )
    assert sparse_module._get_sparse_gqa_packed_decode_launch_config(
        *target, *dtypes
    ) == (64, 4, 1, {"kpack": 2})

    monkeypatch.setattr(sparse_module, "is_gfx942_supported", lambda: False)
    assert (
        sparse_module._get_sparse_gqa_packed_decode_launch_config(*target, *dtypes)
        == fallback
    )


def test_qsa_packed_one_cta_launch_config_is_forwarded(monkeypatch):
    class KernelRecorder:
        def __getitem__(self, grid):
            self.grid = grid
            return self

        def __call__(self, *args, **kwargs):
            self.kwargs = kwargs

    recorder = KernelRecorder()
    monkeypatch.setattr(sparse_module, "_sparse_gqa_chunk_prefill", recorder)
    monkeypatch.setattr(
        sparse_module,
        "_get_sparse_gqa_packed_decode_launch_config",
        lambda *args: (64, 4, 1, {"kpack": 2}),
    )
    q = torch.empty(2, Q_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    k = torch.empty(2, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    v = torch.empty_like(k)
    indices = torch.zeros(2, TOPK, dtype=torch.int32)
    cu = torch.tensor([0, 1, 2], dtype=torch.int32)
    kv_lens = torch.tensor([1, 1], dtype=torch.int32)

    out = sparse_module.sparse_gqa_packed_decode_triton(
        q, k, v, indices, cu, cu, kv_lens, SCALE
    )

    assert out.shape == q.shape
    assert recorder.grid == (1, 2)
    assert recorder.kwargs["num_warps"] == 4
    assert recorder.kwargs["num_stages"] == 1
    assert recorder.kwargs["kpack"] == 2


def test_qsa_same_named_chunk_prefill_jits_have_distinct_cache_keys():
    packed_kernel = sparse_module._sparse_gqa_chunk_prefill
    direct_kernel = direct_module._sparse_gqa_chunk_prefill

    assert packed_kernel is not direct_kernel
    assert packed_kernel.__name__ == direct_kernel.__name__
    assert packed_kernel.__name__ == "_sparse_gqa_chunk_prefill"
    assert packed_kernel.cache_key != direct_kernel.cache_key


def test_qsa_direct_paged_one_cta_gate_is_narrow(monkeypatch):
    device = torch.device("cuda:3")

    def tensor(shape, dtype):
        return SimpleNamespace(
            is_cuda=True,
            device=device,
            ndim=len(shape),
            shape=shape,
            dtype=dtype,
            is_contiguous=lambda: True,
        )

    q = tensor((4, Q_HEADS, HEAD_DIM), torch.bfloat16)
    k = tensor((4096, KV_HEADS, HEAD_DIM), torch.bfloat16)
    v = tensor(k.shape, k.dtype)
    indices = tensor((4, TOPK), torch.int32)
    page_table = tensor((4, 64), torch.int32)
    row_map = tensor((4,), torch.int32)
    lengths = tensor((4,), torch.int32)
    monkeypatch.setattr(direct_module, "is_gfx942_supported", lambda: True)
    seen = []
    monkeypatch.setattr(
        torch.cuda,
        "get_device_name",
        lambda index: seen.append(index) or "AMD Instinct MI308X",
    )
    args = (q, k, v, indices, page_table, row_map, lengths)
    kwargs = dict(
        full_kv_page_size=PAGE_SIZE,
        compress_ratio=COMPRESS_RATIO,
        block_topk=BLOCK_TOPK,
    )
    assert is_sparse_gqa_direct_paged_one_cta_supported(*args, **kwargs)
    assert seen == [device]

    bad_q = tensor((9, Q_HEADS, HEAD_DIM), torch.bfloat16)
    assert not is_sparse_gqa_direct_paged_one_cta_supported(
        bad_q, k, v, indices, page_table, row_map, lengths, **kwargs
    )
    assert not is_sparse_gqa_direct_paged_one_cta_supported(
        *args, **{**kwargs, "full_kv_page_size": 128}
    )
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda index: "AMD Instinct MI300X"
    )
    assert not is_sparse_gqa_direct_paged_one_cta_supported(*args, **kwargs)


@pytest.mark.skipif(not _is_mi308x(), reason="requires MI308X ROCm")
def test_qsa_same_named_chunk_prefill_kernels_do_not_collide_in_one_process():
    state = _make_backend_state(4, [2051, 1024, 17, 1], 800000)
    expected = _reference(state, 0)
    packed_k, packed_v, relative_indices, cu_q, cu_k, kv_lens = _packed_inputs(state, 0)

    def run_packed():
        return sparse_module.sparse_gqa_packed_decode_triton(
            state.q,
            packed_k,
            packed_v,
            relative_indices,
            cu_q,
            cu_k,
            kv_lens,
            SCALE,
        )

    def run_direct():
        return direct_module.sparse_gqa_direct_paged_decode_one_cta_triton(
            state.q,
            state.pool.keys[0],
            state.pool.values[0],
            state.indices,
            state.page_table,
            state.row_to_page_table,
            state.sequence_lens,
            SCALE,
            full_kv_page_size=PAGE_SIZE,
            compress_ratio=COMPRESS_RATIO,
            block_topk=BLOCK_TOPK,
        )

    torch.cuda.synchronize()
    runners = {"packed": run_packed, "direct": run_direct}
    order = ("packed", "direct", "direct", "packed")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        outputs = [(name, runners[name]()) for name in order]
        torch.cuda.synchronize()

    for name, output in outputs:
        torch.testing.assert_close(
            output.float(),
            expected,
            rtol=2e-2,
            atol=2e-2,
            msg=lambda message, name=name: f"{name} kernel mismatch: {message}",
        )

    gpu_kernel_names = [event.name for event in profiler.events()]
    assert gpu_kernel_names.count("_sparse_gqa_chunk_prefill.kd") == len(order)
    assert not any(
        "splitk" in name.lower() or "merge" in name.lower() for name in gpu_kernel_names
    )


@pytest.mark.skipif(not _is_mi308x(), reason="requires MI308X ROCm")
def test_qsa_fast_topk_expand_dispatches_direct_paged_one_cta(monkeypatch):
    """Exercise the decode selection output directly through sparse attention."""

    lengths = [CACHE_TOKENS, 2051, 1025, 17]
    state = _make_backend_state(len(lengths), lengths, 802000)
    layer = SimpleNamespace(layer_id=0, scaling=SCALE)
    compressed_lengths = state.metadata.indexer_metadata.graph_compressed_lengths
    query_positions = state.sequence_lens.to(torch.int64) - 1
    logits = torch.randn(
        len(lengths),
        CACHE_TOKENS // COMPRESS_RATIO,
        dtype=torch.float32,
        device="cuda",
    )
    columns = torch.arange(logits.shape[1], device="cuda")
    logits.masked_fill_(columns[None, :] >= compressed_lengths[:, None], 1.0e9)

    def fallback_must_not_run(*args, **kwargs):
        pytest.fail("direct-paged backend dispatch reached a compact/fallback path")

    monkeypatch.setattr(
        backend_module, "_resolve_trtllm_sparse_decode", fallback_must_not_run
    )
    monkeypatch.setattr(
        backend_module, "qwen_sparse_fa2_cu_seqlens_triton", fallback_must_not_run
    )
    monkeypatch.setattr(
        backend_module,
        "qwen_sparse_kv_extraction_compact_triton",
        fallback_must_not_run,
    )
    monkeypatch.setattr(
        backend_module, "sparse_gqa_packed_decode_triton", fallback_must_not_run
    )

    def select_and_attend():
        block_indices = fast_topk(
            logits,
            compressed_lengths,
            topk=BLOCK_TOPK,
            row_starts=None,
        )
        token_indices = expand_qsa_block_indices(
            block_indices,
            query_positions,
            state.sequence_lens,
            compress_ratio=COMPRESS_RATIO,
            token_topk=BLOCK_TOPK * COMPRESS_RATIO,
        )
        output = state.backend._forward_paged_attention(
            state.q, layer, state.forward_batch, token_indices
        )
        return block_indices, token_indices, output

    # Compile all three kernels before profiling the steady-state decode chain.
    select_and_attend()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        block_indices, token_indices, output = select_and_attend()
        torch.cuda.synchronize()

    for row, compressed_length in enumerate(compressed_lengths.tolist()):
        valid_blocks = block_indices[row][block_indices[row] >= 0]
        assert valid_blocks.numel() == min(compressed_length, BLOCK_TOPK)
        assert torch.all(valid_blocks < compressed_length)

    state.indices.copy_(token_indices)
    torch.testing.assert_close(
        output.view_as(state.q).float(),
        _reference(state, 0),
        rtol=2e-2,
        atol=2e-2,
    )

    gpu_kernel_names = [event.name for event in profiler.events()]
    assert (
        sum("_expand_qsa_block_indices_kernel" in name for name in gpu_kernel_names)
        == 1
    )
    assert gpu_kernel_names.count("_sparse_gqa_chunk_prefill.kd") == 1
    assert not any(
        forbidden in name.lower()
        for name in gpu_kernel_names
        for forbidden in ("compact", "split", "merge")
    )


def test_qsa_backend_direct_paged_helper_falls_back_without_inputs(monkeypatch):
    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    monkeypatch.setattr(backend_module, "is_hip", lambda: True)
    indexer = SimpleNamespace(
        graph_compressed_page_table=None,
        decode_page_table=None,
        token_to_kv_pool=SimpleNamespace(qsa_compressed_page_size=16),
        compress_ratio=COMPRESS_RATIO,
        block_topk=BLOCK_TOPK,
    )
    metadata = SimpleNamespace(
        is_cuda_graph=True,
        indexer_metadata=indexer,
        token_to_batch_idx=torch.zeros(1, dtype=torch.int32),
        sequence_lengths=torch.ones(1, dtype=torch.int32),
    )
    q = torch.zeros(1, Q_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    k = torch.zeros(PAGE_SIZE, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    indices = torch.full((1, TOPK), -1, dtype=torch.int32)
    assert (
        backend._forward_direct_paged_one_cta(
            q,
            k,
            k,
            SimpleNamespace(scaling=SCALE),
            metadata,
            indices,
        )
        is None
    )


def test_qsa_backend_direct_page_table_selects_graph_and_eager_inputs():
    graph_page_table = object()
    eager_page_table = object()
    indexer = SimpleNamespace(
        graph_compressed_page_table=graph_page_table,
        decode_page_table=eager_page_table,
    )

    assert (
        QwenSparseAttnBackend._get_direct_full_kv_page_table(
            SimpleNamespace(is_cuda_graph=True, indexer_metadata=indexer)
        )
        is graph_page_table
    )
    assert (
        QwenSparseAttnBackend._get_direct_full_kv_page_table(
            SimpleNamespace(is_cuda_graph=False, indexer_metadata=indexer)
        )
        is eager_page_table
    )


@pytest.mark.skipif(not _is_mi308x(), reason="requires MI308X ROCm")
def test_qsa_backend_direct_gate_miss_keeps_packed_one_cta_fallback(monkeypatch):
    state = _make_backend_state(1, [2051], 805000)
    indexer = SimpleNamespace(
        graph_compressed_page_table=None,
        decode_page_table=None,
        token_to_kv_pool=state.pool,
        compress_ratio=COMPRESS_RATIO,
        block_topk=BLOCK_TOPK,
    )
    state.backend.forward_metadata = SimpleNamespace(
        is_cuda_graph=True,
        indexer_metadata=indexer,
        token_to_batch_idx=state.row_to_page_table,
        sequence_lengths=state.sequence_lens,
        row_req_pool_indices=torch.zeros(1, dtype=torch.int32, device="cuda"),
        fa2_valid_counts=torch.zeros(1, dtype=torch.int32, device="cuda"),
        fa2_cu_seqlens_k=torch.zeros(2, dtype=torch.int32, device="cuda"),
        fa2_cu_seqlens_q=torch.arange(2, dtype=torch.int32, device="cuda"),
    )
    monkeypatch.setattr(backend_module, "_resolve_trtllm_sparse_decode", lambda: None)
    monkeypatch.setattr(
        backend_module, "qwen_sparse_fa2_cu_seqlens_triton", lambda *args: None
    )
    monkeypatch.setattr(
        backend_module,
        "qwen_sparse_kv_extraction_compact_triton",
        lambda *args: None,
    )
    calls = []

    def fake_packed_one_cta(q, *args, **kwargs):
        calls.append(q.shape[0])
        return q + 1

    monkeypatch.setattr(
        backend_module, "sparse_gqa_packed_decode_triton", fake_packed_one_cta
    )
    output = state.backend._forward_paged_attention(
        state.q,
        SimpleNamespace(layer_id=0, scaling=SCALE),
        state.forward_batch,
        state.indices,
    )
    assert calls == [1]
    torch.testing.assert_close(output.view_as(state.q), state.q + 1)


@pytest.mark.skipif(not _is_mi308x(), reason="requires MI308X ROCm")
@pytest.mark.parametrize(
    "primary_lengths,alternate_lengths",
    [
        ([2051], [17]),
        ([2051, 1, 1, 1], [17, 512, 1024, 2048]),
        (
            [2051, 1, 1, 1, 1, 1, 1, 1],
            [1, 17, 512, 1024, 2048, 2051, 1, 17],
        ),
    ],
)
def test_qsa_backend_direct_paged_one_cta_graph_replay(
    monkeypatch, primary_lengths, alternate_lengths
):
    state = _make_backend_state(
        len(primary_lengths), primary_lengths, 810000 + len(primary_lengths)
    )
    layer = SimpleNamespace(layer_id=0, scaling=SCALE)

    def compact_must_not_run(*args, **kwargs):
        pytest.fail("direct-paged backend dispatch reached compact metadata")

    monkeypatch.setattr(
        backend_module, "qwen_sparse_fa2_cu_seqlens_triton", compact_must_not_run
    )
    monkeypatch.setattr(
        backend_module, "qwen_sparse_kv_extraction_compact_triton", compact_must_not_run
    )

    for _ in range(3):
        state.backend._forward_paged_attention(
            state.q, layer, state.forward_batch, state.indices
        )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = state.backend._forward_paged_attention(
            state.q, layer, state.forward_batch, state.indices
        )

    _update_state(state, alternate_lengths, 820000 + len(primary_lengths))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        captured.view_as(state.q).float(),
        _reference(state, 0),
        rtol=2e-2,
        atol=2e-2,
    )
    _update_state(state, primary_lengths, 830000 + len(primary_lengths))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        captured.view_as(state.q).float(),
        _reference(state, 0),
        rtol=2e-2,
        atol=2e-2,
    )

    if len(primary_lengths) == 4:
        invalid_lengths = [2051, 2051, 2051, 0]
        state.sequence_lens.copy_(
            torch.tensor(invalid_lengths, dtype=torch.int32, device=state.q.device)
        )
        _fill_indices(state.indices, invalid_lengths, 835000)
        state.indices[0].fill_(-1)
        state.row_to_page_table[1] = -1
        invalid_page_row = int(state.row_to_page_table[2].item())
        state.page_table[invalid_page_row].fill_(state.pool.keys[0].shape[0])
        state.sequence_lens[3] = 0
        graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(captured).all()
        assert torch.equal(captured, torch.zeros_like(captured))


@pytest.mark.skipif(not _is_mi308x(), reason="requires MI308X ROCm")
def test_qsa_backend_direct_paged_one_cta_two_layers_do_not_alias():
    state = _make_backend_state(4, [2051, 1, 1, 1], 840000, layers=2)
    layer0 = SimpleNamespace(layer_id=0, scaling=SCALE)
    layer1 = SimpleNamespace(layer_id=1, scaling=SCALE)
    for _ in range(3):
        state.backend._forward_paged_attention(
            state.q, layer0, state.forward_batch, state.indices
        )
        state.backend._forward_paged_attention(
            state.q, layer1, state.forward_batch, state.indices
        )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        layer0_output = state.backend._forward_paged_attention(
            state.q, layer0, state.forward_batch, state.indices
        )
        layer1_output = state.backend._forward_paged_attention(
            state.q, layer1, state.forward_batch, state.indices
        )
    assert layer0_output.data_ptr() != layer1_output.data_ptr()

    _update_state(state, [17, 512, 1024, 2048], 850000)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        layer0_output.view_as(state.q).float(),
        _reference(state, 0),
        rtol=2e-2,
        atol=2e-2,
    )
    torch.testing.assert_close(
        layer1_output.view_as(state.q).float(),
        _reference(state, 1),
        rtol=2e-2,
        atol=2e-2,
    )
