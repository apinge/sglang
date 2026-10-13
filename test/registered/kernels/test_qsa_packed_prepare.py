from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention import qwen_sparse_attn_backend as backend_module
from sglang.srt.layers.attention.qsa import direct_paged_one_cta as direct_module
from sglang.srt.layers.attention.qsa.packed_prepare import prepare_qsa_packed_kv
from sglang.srt.layers.attention.qsa.sparse_attn_decode import (
    SparsePackedGQAWorkspace,
    qsa_sparse_packed_gqa,
)
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, stage="stage-b", runner_config="1-gpu-large-amd")


Q_HEADS = 12
KV_HEADS = 1
HEAD_DIM = 256
PAGE_SIZE = 64
COMPRESS_RATIO = 4
BLOCK_TOPK = 512
TOPK = BLOCK_TOPK * COMPRESS_RATIO + COMPRESS_RATIO - 1
ROW_STRIDE = TOPK + 1
CACHE_TOKENS = 2112
SCALE = HEAD_DIM**-0.5
PREPARE_KWARGS = dict(
    full_kv_page_size=PAGE_SIZE,
    compress_ratio=COMPRESS_RATIO,
    block_topk=BLOCK_TOPK,
)


def _is_mi308x() -> bool:
    return bool(
        torch.cuda.is_available()
        and torch.version.hip is not None
        and torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0] == "gfx942"
        and "MI308X" in torch.cuda.get_device_name(0)
    )


requires_mi308x = pytest.mark.skipif(not _is_mi308x(), reason="requires MI308X ROCm")


def _selection_capacity(length: int) -> int:
    length = max(length, 0)
    return min(
        TOPK,
        min(length // COMPRESS_RATIO, BLOCK_TOPK) * COMPRESS_RATIO
        + length % COMPRESS_RATIO,
    )


def _fill_indices(state, lengths: list[int]) -> None:
    state.indices.fill_(-1)
    offsets = torch.arange(COMPRESS_RATIO, device=state.q.device)
    for row, length in enumerate(lengths):
        length = max(length, 0)
        groups = min(length // COMPRESS_RATIO, BLOCK_TOPK)
        if groups:
            selected = torch.randperm(length // COMPRESS_RATIO, device=state.q.device)[
                :groups
            ]
            expanded = selected[:, None] * COMPRESS_RATIO + offsets[None, :]
            state.indices[row, : groups * COMPRESS_RATIO].copy_(expanded.reshape(-1))
        tail = length % COMPRESS_RATIO
        if tail:
            tail_start = groups * COMPRESS_RATIO
            state.indices[row, tail_start : tail_start + tail] = torch.arange(
                length - tail, length, device=state.q.device
            )


def _update_state(state, lengths: list[int], seed: int) -> None:
    torch.manual_seed(seed)
    state.q.copy_(torch.randn_like(state.q))
    state.k.copy_(torch.randn_like(state.k))
    state.v.copy_(torch.randn_like(state.v))
    state.sequence_lens.copy_(state.sequence_lens.new_tensor(lengths))
    state.page_table.copy_(
        torch.randperm(state.page_table.numel(), device=state.q.device).view_as(
            state.page_table
        )
    )
    state.row_to_page_table.copy_(
        torch.randperm(state.q.shape[0], device=state.q.device)
    )
    _fill_indices(state, lengths)


def _make_state(lengths: list[int], seed: int):
    batch = len(lengths)
    device = torch.device("cuda")
    state = SimpleNamespace(
        q=torch.empty(batch, Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device),
        k=torch.empty(
            batch * CACHE_TOKENS,
            KV_HEADS,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        ),
        v=torch.empty(
            batch * CACHE_TOKENS,
            KV_HEADS,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        ),
        indices=torch.empty(batch, TOPK, dtype=torch.int32, device=device),
        page_table=torch.empty(
            batch, CACHE_TOKENS // PAGE_SIZE, dtype=torch.int32, device=device
        ),
        row_to_page_table=torch.empty(batch, dtype=torch.int32, device=device),
        sequence_lens=torch.empty(batch, dtype=torch.int32, device=device),
    )
    _update_state(state, lengths, seed)
    return state


def _selected_slots(state) -> list[tuple[list[int], list[int]]]:
    """Independent host oracle for valid original slots and physical cache rows."""

    tables = state.page_table.cpu().tolist()
    row_map = state.row_to_page_table.cpu().tolist()
    lengths = state.sequence_lens.cpu().tolist()
    indices = state.indices.cpu().tolist()
    num_pages = state.k.shape[0] // PAGE_SIZE
    selected_rows = []
    for row, length in enumerate(lengths):
        selected_positions, physical_tokens = [], []
        table_row = row_map[row]
        if 0 <= table_row < len(tables):
            for position, logical in enumerate(
                indices[row][: _selection_capacity(length)]
            ):
                if not 0 <= logical < length:
                    continue
                column, offset = divmod(logical, PAGE_SIZE)
                if column >= len(tables[table_row]):
                    continue
                page = tables[table_row][column]
                if not 0 <= page < num_pages:
                    continue
                selected_positions.append(position)
                physical_tokens.append(page * PAGE_SIZE + offset)
        selected_rows.append((selected_positions, physical_tokens))
    return selected_rows


def _reference(state) -> torch.Tensor:
    output = []
    for row, (_, tokens) in enumerate(_selected_slots(state)):
        if not tokens:
            output.append(torch.zeros_like(state.q[row], dtype=torch.float32))
            continue
        physical = torch.tensor(tokens, dtype=torch.long, device=state.q.device)
        keys = state.k.index_select(0, physical)[:, 0].float()
        values = state.v.index_select(0, physical)[:, 0].float()
        scores = state.q[row].float() @ keys.T * SCALE
        output.append(torch.softmax(scores, dim=-1) @ values)
    return torch.stack(output)


def _prepare(state):
    return prepare_qsa_packed_kv(
        state.k,
        state.v,
        state.indices,
        state.page_table,
        state.row_to_page_table,
        state.sequence_lens,
        **PREPARE_KWARGS,
    )


def _assert_prepared(state, prepared) -> None:
    packed_k, packed_v, relative_indices, cu_q, cu_k, kv_lens = prepared
    batch = state.q.shape[0]
    assert packed_k.shape == (batch * ROW_STRIDE, KV_HEADS, HEAD_DIM)
    assert packed_v.shape == packed_k.shape
    assert packed_k.dtype == state.k.dtype
    assert packed_v.dtype == state.v.dtype
    assert relative_indices.shape == state.indices.shape
    assert all(tensor.is_contiguous() for tensor in prepared)
    assert all(tensor.dtype == torch.int32 for tensor in prepared[2:])
    assert all(tensor.device == state.q.device for tensor in prepared)
    assert packed_k.data_ptr() != state.k.data_ptr()
    assert packed_v.data_ptr() != state.v.data_ptr()
    assert packed_k.data_ptr() != packed_v.data_ptr()
    assert cu_q.cpu().tolist() == list(range(batch + 1))
    assert cu_k.cpu().tolist() == [row * ROW_STRIDE for row in range(batch + 1)]
    relative = relative_indices.cpu().tolist()
    lengths = kv_lens.cpu().tolist()
    for row, (positions, tokens) in enumerate(_selected_slots(state)):
        expected_prefix = positions if positions else [TOPK]
        assert lengths[row] == len(expected_prefix)
        assert relative[row][: len(expected_prefix)] == expected_prefix
        assert relative[row][len(expected_prefix) :] == [-1] * (
            TOPK - len(expected_prefix)
        )
        sentinel = row * ROW_STRIDE + TOPK
        assert torch.count_nonzero(packed_k[sentinel]).item() == 0
        assert torch.count_nonzero(packed_v[sentinel]).item() == 0
        if positions:
            destination = (
                torch.tensor(positions, dtype=torch.long, device=state.q.device)
                + row * ROW_STRIDE
            )
            source = torch.tensor(tokens, dtype=torch.long, device=state.q.device)
            # K/V holes are unspecified; only slots named by relative_indices matter.
            torch.testing.assert_close(
                packed_k.index_select(0, destination),
                state.k.index_select(0, source),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                packed_v.index_select(0, destination),
                state.v.index_select(0, source),
                rtol=0,
                atol=0,
            )


def _add_holes_and_dirty_tails(state) -> None:
    for row, length in enumerate(state.sequence_lens.cpu().tolist()):
        capacity = _selection_capacity(length)
        # These are valid logical tokens, but are outside this row's selection.
        state.indices[row, capacity:] = 0
        state.indices[row, 7:capacity:19] = -1
    if _selection_capacity(int(state.sequence_lens[0].item())) > 128:
        # The original packed kernel cannot begin with an entirely masked tile.
        state.indices[0, :128] = -1


def _make_backend(state):
    pool = SimpleNamespace(
        qsa_compressed_page_size=PAGE_SIZE // COMPRESS_RATIO,
        get_key_buffer=lambda layer_id: state.k,
        get_value_buffer=lambda layer_id: state.v,
    )
    indexer = SimpleNamespace(
        graph_compressed_page_table=state.page_table,
        decode_page_table=state.page_table,
        token_to_kv_pool=pool,
        compress_ratio=COMPRESS_RATIO,
        block_topk=BLOCK_TOPK,
    )
    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend.token_to_kv_pool = pool
    backend.forward_metadata = SimpleNamespace(
        is_cuda_graph=True,
        indexer_metadata=indexer,
        token_to_batch_idx=state.row_to_page_table,
        sequence_lengths=state.sequence_lens,
    )
    return backend


@requires_mi308x
@pytest.mark.parametrize(
    "case,batch",
    [
        ("permuted_pages", 1),
        ("permuted_pages", 4),
        ("permuted_pages", 8),
        ("holes_dirty_tails", 4),
        ("invalid_metadata", 8),
        ("all_empty", 4),
    ],
)
def test_qsa_packed_prepare_matches_selected_kv_and_fp32_attention(case, batch):
    lengths = [2051] * batch
    if case == "permuted_pages":
        lengths = [CACHE_TOKENS, 2051, 1025, 17, 128, 65, 3, 1][:batch]
    elif case == "holes_dirty_tails":
        lengths = [2051, 17, 128, 65]
    elif case == "all_empty":
        lengths = [0, 2051, 2051, -7]
    state = _make_state(lengths, 910000 + batch)
    if case == "holes_dirty_tails":
        _add_holes_and_dirty_tails(state)
        state.indices[2, :2] = state.indices.new_tensor([-7, 128])
    elif case == "invalid_metadata":
        state.row_to_page_table[0] = -1
        state.row_to_page_table[1] = batch
        state.page_table[state.row_to_page_table[2].long()] = -1
        num_pages = state.k.shape[0] // PAGE_SIZE
        state.page_table[state.row_to_page_table[3].long()] = num_pages
        state.page_table[state.row_to_page_table[4].long(), ::2] = -1
        state.sequence_lens[5] = CACHE_TOKENS + 128
        state.indices[5, :128] = torch.arange(
            CACHE_TOKENS, CACHE_TOKENS + 128, device=state.q.device
        )
        state.indices[6, :2] = state.indices.new_tensor([-7, lengths[6]])
    elif case == "all_empty":
        state.indices[0].fill_(0)
        state.indices[1].fill_(-1)
        state.row_to_page_table[2] = -1
        state.indices[3].fill_(0)

    prepared = _prepare(state)
    _assert_prepared(state, prepared)
    output = qsa_sparse_packed_gqa(state.q, *prepared, SCALE)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.float(), _reference(state), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("is_graph", [False, True])
def test_qsa_backend_passes_only_prepared_inputs_to_packed_attention(
    monkeypatch, is_graph
):
    q = torch.empty(1, Q_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    k = torch.empty(PAGE_SIZE, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    v = torch.empty_like(k)
    indices = torch.zeros(1, TOPK, dtype=torch.int32)
    graph_table = torch.zeros(1, 1, dtype=torch.int32)
    eager_table = torch.ones(1, 1, dtype=torch.int32)
    row_map = torch.zeros(1, dtype=torch.int32)
    lengths = torch.ones(1, dtype=torch.int32)
    metadata = SimpleNamespace(
        is_cuda_graph=is_graph,
        token_to_batch_idx=row_map,
        sequence_lengths=lengths,
        indexer_metadata=SimpleNamespace(
            graph_compressed_page_table=graph_table,
            decode_page_table=eager_table,
            token_to_kv_pool=SimpleNamespace(
                qsa_compressed_page_size=PAGE_SIZE // COMPRESS_RATIO
            ),
            compress_ratio=COMPRESS_RATIO,
            block_topk=BLOCK_TOPK,
        ),
    )
    prepared = tuple(torch.empty(1) for _ in range(6))
    expected_output = torch.empty_like(q)
    calls = []

    def fake_prepare(*args, **kwargs):
        calls.append("prepare")
        expected = (
            k,
            v,
            indices,
            graph_table if is_graph else eager_table,
            row_map,
            lengths,
        )
        assert len(args) == len(expected)
        assert all(actual is wanted for actual, wanted in zip(args, expected))
        assert kwargs == PREPARE_KWARGS
        return prepared

    def fake_packed(attn_q, *args, workspace):
        calls.append("packed")
        assert isinstance(workspace, SparsePackedGQAWorkspace)
        assert workspace is backend._packed_gqa_workspace
        assert attn_q is q
        assert len(args) == len(prepared) + 1
        assert all(actual is wanted for actual, wanted in zip(args[:-1], prepared))
        assert args[-1] == SCALE
        return expected_output

    def direct_must_not_run(*args, **kwargs):
        pytest.fail("backend invoked the old packed or direct-paged attention")

    monkeypatch.setattr(backend_module, "is_hip", lambda: True)
    monkeypatch.setattr(
        backend_module,
        "is_sparse_gqa_direct_paged_one_cta_supported",
        lambda *a, **k: True,
    )
    monkeypatch.setattr(backend_module, "prepare_qsa_packed_kv", fake_prepare)
    monkeypatch.setattr(backend_module, "qsa_sparse_packed_gqa", fake_packed)
    monkeypatch.setattr(
        backend_module, "sparse_gqa_packed_decode_triton", direct_must_not_run
    )
    monkeypatch.setattr(
        backend_module,
        "sparse_gqa_direct_paged_decode_one_cta_triton",
        direct_must_not_run,
        raising=False,
    )
    monkeypatch.setattr(
        direct_module,
        "sparse_gqa_direct_paged_decode_one_cta_triton",
        direct_must_not_run,
    )
    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    output = backend._forward_packed_decode(
        q, k, v, SimpleNamespace(scaling=SCALE), metadata, indices
    )
    assert calls == ["prepare", "packed"]
    assert output is expected_output
    workspace = backend._packed_gqa_workspace
    assert (
        backend._forward_packed_decode(
            q, k, v, SimpleNamespace(scaling=SCALE), metadata, indices
        )
        is expected_output
    )
    assert backend._packed_gqa_workspace is workspace
    assert calls == ["prepare", "packed", "prepare", "packed"]


@requires_mi308x
@pytest.mark.parametrize("batch", [1, 4, 8])
def test_qsa_backend_launches_prepare_pr40_attention_and_aiter_reduce(
    monkeypatch, batch
):
    state = _make_state([2051] * batch, 920000 + batch)
    _add_holes_and_dirty_tails(state)
    backend = _make_backend(state)
    layer = SimpleNamespace(layer_id=0, scaling=SCALE)
    calls = []

    def record_prepare(*args, **kwargs):
        calls.append("prepare")
        return prepare_qsa_packed_kv(*args, **kwargs)

    def record_packed(*args, **kwargs):
        calls.append("packed")
        return qsa_sparse_packed_gqa(*args, **kwargs)

    def other_path_must_not_run(*args, **kwargs):
        pytest.fail("backend left the prepare + PR40 packed attention path")

    monkeypatch.setattr(backend_module, "prepare_qsa_packed_kv", record_prepare)
    monkeypatch.setattr(backend_module, "qsa_sparse_packed_gqa", record_packed)
    for name in (
        "sparse_gqa_packed_decode_triton",
        "_resolve_trtllm_sparse_decode",
        "qwen_sparse_fa2_cu_seqlens_triton",
        "qwen_sparse_kv_extraction_compact_triton",
        "sparse_gqa_direct_paged_decode_one_cta_triton",
    ):
        monkeypatch.setattr(
            backend_module, name, other_path_must_not_run, raising=False
        )
    monkeypatch.setattr(
        direct_module,
        "sparse_gqa_direct_paged_decode_one_cta_triton",
        other_path_must_not_run,
    )

    def run():
        return backend._forward_paged_attention(
            state.q, layer, SimpleNamespace(), state.indices
        )

    run()
    torch.cuda.synchronize()
    calls.clear()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        output = run()
        torch.cuda.synchronize()
    assert calls == ["prepare", "packed"]
    kernels = [
        event.name
        for event in profiler.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    assert len(kernels) == 3, kernels
    assert sum("_prepare_qsa_packed_kv" in name for name in kernels) == 1
    assert sum("_qsa_sparse_packed_gqa_kernel" in name for name in kernels) == 1
    assert sum("_mla_decode_fwd_reduce_kernel" in name for name in kernels) == 1
    assert not any("_sparse_gqa_chunk_prefill" in name for name in kernels)
    torch.testing.assert_close(
        output.view_as(state.q).float(), _reference(state), rtol=2e-2, atol=2e-2
    )


@requires_mi308x
@pytest.mark.parametrize("batch", [1, 4, 8])
def test_qsa_packed_prepare_graph_replay_refreshes_metadata_and_empty_sentinel(batch):
    primary_lengths = [2051] * batch
    alternate_lengths = [17, 0, 1, 65, 1024, 2048, 2051, 128][:batch]
    state = _make_state(primary_lengths, 930000 + batch)
    workspace = SparsePackedGQAWorkspace()

    def prepare_and_attend():
        prepared = _prepare(state)
        return prepared, qsa_sparse_packed_gqa(
            state.q, *prepared, SCALE, workspace=workspace
        )

    for _ in range(3):
        prepare_and_attend()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepared, captured = prepare_and_attend()
    pointers = [tensor.data_ptr() for tensor in (*prepared, captured)]

    for step, lengths in enumerate(
        [alternate_lengths, [0] * batch, primary_lengths, alternate_lengths]
    ):
        _update_state(state, lengths, 940000 + batch * 10 + step)
        _add_holes_and_dirty_tails(state)
        if step == 3:
            state.row_to_page_table[-1] = -1
        for partials, maxima, sums, widths in workspace.buffers.values():
            partials.fill_(float("nan"))
            maxima.fill_(float("nan"))
            sums.fill_(float("nan"))
            widths.fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        assert [tensor.data_ptr() for tensor in (*prepared, captured)] == pointers
        _assert_prepared(state, prepared)
        assert torch.isfinite(captured).all()
        torch.testing.assert_close(
            captured.float(), _reference(state), rtol=2e-2, atol=2e-2
        )


@requires_mi308x
def test_qsa_packed_prepare_calls_and_attention_outputs_do_not_alias():
    first = _make_state([2051, 17, 0, 65], 950000)
    second = _make_state([128, 0, 2048, 1], 950001)
    first_prepared = _prepare(first)
    first_output = qsa_sparse_packed_gqa(first.q, *first_prepared, SCALE)
    second_prepared = _prepare(second)
    second_output = qsa_sparse_packed_gqa(second.q, *second_prepared, SCALE)
    assert all(
        original.data_ptr() != later.data_ptr()
        for original, later in zip(first_prepared, second_prepared)
    )
    assert first_output.data_ptr() != second_output.data_ptr()
    _assert_prepared(first, first_prepared)
    _assert_prepared(second, second_prepared)
    torch.testing.assert_close(
        first_output.float(), _reference(first), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        second_output.float(), _reference(second), rtol=2e-2, atol=2e-2
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
