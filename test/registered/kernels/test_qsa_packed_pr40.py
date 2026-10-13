"""PR40 split-K attention over the separate packed-prepare representation."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import pytest
import torch

from sglang.srt.layers.attention.qsa.sparse_attn import sparse_gqa_packed_decode_triton
from sglang.srt.layers.attention.qsa.sparse_attn_decode import (
    SparsePackedGQAWorkspace,
    qsa_sparse_packed_gqa,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, stage="stage-b", runner_config="1-gpu-large-amd")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is None
    or torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0] != "gfx942",
    reason="requires gfx942 ROCm BF16 attention",
)


@dataclass
class PackedInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    indices: torch.Tensor
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    kv_lens: torch.Tensor

    @property
    def scale(self):
        return self.q.shape[-1] ** -0.5

    def args(self):
        return (
            self.q,
            self.k,
            self.v,
            self.indices,
            self.cu_q,
            self.cu_k,
            self.kv_lens,
            self.scale,
        )


def _strided_copy(tensor):
    storage = tensor.new_empty(tuple(2 * size for size in tensor.shape))
    view = storage[tuple(slice(None, None, 2) for _ in tensor.shape)]
    view.copy_(tensor)
    return view


def _make_inputs(
    batch=4,
    width=2051,
    *,
    heads=12,
    kv_heads=1,
    dim=256,
    strided=False,
    metadata_dtype=torch.int32,
    seed=231140,
):
    torch.manual_seed(seed)
    device = "cuda"
    # Include distinct row extents, as well as the fixed-stride prepare layout.
    spans = [width + 1 + 7 * (row % 3) for row in range(batch)]
    boundaries = [0]
    for span in spans:
        boundaries.append(boundaries[-1] + span)
    inputs = PackedInputs(
        q=torch.randn(batch, heads, dim, device=device, dtype=torch.bfloat16),
        k=torch.randn(
            boundaries[-1], kv_heads, dim, device=device, dtype=torch.bfloat16
        ),
        v=torch.randn(
            boundaries[-1], kv_heads, dim, device=device, dtype=torch.bfloat16
        ),
        indices=torch.full((batch, width), -1, device=device, dtype=metadata_dtype),
        cu_q=torch.arange(batch + 1, device=device, dtype=metadata_dtype),
        cu_k=torch.tensor(boundaries, device=device, dtype=metadata_dtype),
        kv_lens=torch.full((batch,), width, device=device, dtype=metadata_dtype),
    )
    _refresh(inputs, [width] * batch, seed + 1)
    if strided:
        for name in inputs.__dataclass_fields__:
            setattr(inputs, name, _strided_copy(getattr(inputs, name)))
    return inputs


def _refresh(inputs, lengths, seed, *, dirty_tails=False, rotate_segments=False):
    """Change all packed payloads in place, keeping graph input pointers stable."""
    torch.manual_seed(seed)
    inputs.q.normal_()
    inputs.k.normal_()
    inputs.v.normal_()
    if rotate_segments and len(lengths) > 1:
        spans = (inputs.cu_k[1:] - inputs.cu_k[:-1]).roll(1)
        inputs.cu_k[1:].copy_(spans.cumsum(0))
    inputs.kv_lens.copy_(inputs.kv_lens.new_tensor(lengths))
    inputs.indices.fill_(-1)
    boundaries = inputs.cu_k.cpu().tolist()
    for row, length in enumerate(lengths):
        start, end = boundaries[row : row + 2]
        span = end - start
        count = min(max(length, 0), inputs.indices.shape[1])
        # Prefix values retain their original slot numbers after compaction;
        # a slot can legitimately exceed the number of selected tokens.
        if count:
            selected = torch.randperm(span - 1, device=inputs.q.device)[:count]
            inputs.indices[row, :count].copy_(selected)
        inputs.k[end - 1].zero_()
        inputs.v[end - 1].zero_()
        if dirty_tails:
            inputs.indices[row, count:] = span - 2


def _selected_tokens(inputs):
    """Interpret selection counts independently of physical packed row extents."""
    boundaries = inputs.cu_k.cpu().tolist()
    lengths = inputs.kv_lens.cpu().tolist()
    relative = inputs.indices.cpu().tolist()
    query_boundaries = inputs.cu_q.cpu().tolist()
    rows = []
    for row, length in enumerate(lengths):
        start, end = boundaries[row : row + 2]
        if query_boundaries[row : row + 2] != [row, row + 1] or not (
            0 <= start <= end <= inputs.k.shape[0]
        ):
            rows.append([])
            continue
        count = min(max(length, 0), len(relative[row]))
        rows.append(
            [
                start + slot
                for slot in relative[row][:count]
                if 0 <= slot < end - start and 0 <= start + slot < inputs.k.shape[0]
            ]
        )
    return rows


def _poison_unused_kv(inputs):
    unused = torch.ones(inputs.k.shape[0], dtype=torch.bool, device=inputs.q.device)
    selected = [token for row in _selected_tokens(inputs) for token in row]
    if selected:
        unused[torch.tensor(selected, device=inputs.q.device)] = False
    inputs.k[unused] = float("nan")
    inputs.v[unused] = float("nan")


def _reference(inputs):
    expected = torch.zeros_like(inputs.q, dtype=torch.float32)
    group_size = inputs.q.shape[1] // inputs.k.shape[1]
    for row, tokens in enumerate(_selected_tokens(inputs)):
        if not tokens:
            continue
        physical = torch.tensor(tokens, dtype=torch.long, device=inputs.q.device)
        keys = inputs.k.index_select(0, physical).float()
        values = inputs.v.index_select(0, physical).float()
        keys = keys.repeat_interleave(group_size, dim=1)
        values = values.repeat_interleave(group_size, dim=1)
        scores = torch.einsum("hd,nhd->hn", inputs.q[row].float(), keys)
        probabilities = torch.softmax(scores * inputs.scale, dim=-1)
        expected[row] = torch.einsum("hn,nhd->hd", probabilities, values)
    return expected


def _assert_reference(actual, inputs):
    assert actual.shape == inputs.q.shape
    assert actual.dtype == inputs.q.dtype
    assert torch.isfinite(actual).all()
    # PR40 rounds the scaled Q and softmax probabilities to BF16. The original
    # PR40 direct kernel has the same 3.476e-3 FP32 error on the short-row replay
    # diagnostic, with bitwise-identical packed/eager/graph outputs.
    torch.testing.assert_close(actual.float(), _reference(inputs), atol=4e-3, rtol=2e-2)


def _poison(workspace):
    assert workspace.buffers
    for partials, maxima, sums, widths in workspace.buffers.values():
        partials.fill_(float("nan"))
        maxima.fill_(float("nan"))
        sums.fill_(float("nan"))
        widths.fill_(-1)


@pytest.mark.parametrize(
    "batch,width,splits",
    [
        (0, 2051, 64),
        (1, 0, 64),
        (1, 1, 1),
        (1, 7, 64),
        (4, 31, 64),
        (4, 511, None),
        (1, 512, None),
        (4, 1025, 64),
        (4, 2048, 1),
        (1, 2051, None),
        (4, 2051, None),
        (8, 2051, None),
        (4, 2051, 1),
        (4, 2051, 8),
        (4, 2051, 64),
        (4, 4099, 64),
        (1, 8195, 1),
        (4, 8195, 64),
    ],
)
def test_qsa_packed_pr40_matches_fp32(batch, width, splits):
    inputs = _make_inputs(batch=batch, width=width)
    _assert_reference(qsa_sparse_packed_gqa(*inputs.args(), kv_splits=splits), inputs)


@pytest.mark.parametrize("splits", [1, 64])
@pytest.mark.parametrize(
    "case",
    [
        "ragged_dirty_tails",
        "holes",
        "sentinel",
        "all_empty",
        "invalid_cu_q",
        "invalid_cu_k",
    ],
)
def test_qsa_packed_pr40_selection_counts_and_slot_bounds(case, splits):
    inputs = _make_inputs(batch=4)
    if case == "ragged_dirty_tails":
        _refresh(inputs, [1, 17, 129, 2049], 240001, dirty_tails=True)
        inputs.indices[0, 0] = 2050
        assert inputs.indices[0, 0] > inputs.kv_lens[0]
    elif case == "holes":
        _refresh(inputs, [2051, 513, 65, 17], 240002, dirty_tails=True)
        inputs.indices[:, ::7] = -1
        inputs.indices[0, :128] = -1
        inputs.indices[1, 1] = inputs.cu_k[2] - inputs.cu_k[1]
        inputs.indices[2, 2] = torch.iinfo(inputs.indices.dtype).max
    elif case == "sentinel":
        inputs.kv_lens.fill_(1)
        inputs.indices.fill_(0)
        inputs.indices[:, 0] = inputs.cu_k[1:] - inputs.cu_k[:-1] - 1
    elif case == "all_empty":
        inputs.kv_lens.copy_(inputs.kv_lens.new_tensor([0, -1, 2051, 17]))
        inputs.indices[:2].fill_(0)
        inputs.indices[2].fill_(-1)
        inputs.indices[3].fill_(torch.iinfo(inputs.indices.dtype).max)
    elif case == "invalid_cu_q":
        inputs.cu_q[1] = 0
        inputs.cu_q[-1] = inputs.q.shape[0] + 1
    elif case == "invalid_cu_k":
        inputs.cu_k[0] = -1
        inputs.cu_k[2] = inputs.cu_k[1] - 1
        inputs.cu_k[-1] = inputs.k.shape[0] + 1
    _poison_unused_kv(inputs)
    output = qsa_sparse_packed_gqa(*inputs.args(), kv_splits=splits)
    _assert_reference(output, inputs)
    if case in ("sentinel", "all_empty"):
        assert torch.count_nonzero(output) == 0


@pytest.mark.parametrize(
    "heads,kv_heads,dim", [(12, 1, 256), (20, 1, 256), (24, 2, 128)]
)
@pytest.mark.parametrize("metadata_dtype", [torch.int32, torch.int64])
def test_qsa_packed_pr40_strided_payloads_and_metadata(
    heads, kv_heads, dim, metadata_dtype
):
    inputs = _make_inputs(
        batch=8,
        heads=heads,
        kv_heads=kv_heads,
        dim=dim,
        strided=True,
        metadata_dtype=metadata_dtype,
    )
    _refresh(inputs, [2051, 17, 0, 65, 1025, 1, 513, 2048], 240003, dirty_tails=True)
    assert all(not tensor.is_contiguous() for tensor in inputs.args()[:-1])
    _assert_reference(qsa_sparse_packed_gqa(*inputs.args(), kv_splits=64), inputs)


def _legacy_output_for_same_kv(inputs):
    """Give the old kernel the same effective KV, with its required clean padding."""
    keys, values, counts = [], [], []
    relative = torch.full_like(inputs.indices, -1)
    for row, tokens in enumerate(_selected_tokens(inputs)):
        if tokens:
            physical = torch.tensor(tokens, dtype=torch.long, device=inputs.q.device)
            keys.append(inputs.k.index_select(0, physical))
            values.append(inputs.v.index_select(0, physical))
        else:
            keys.append(torch.zeros_like(inputs.k[:1]))
            values.append(torch.zeros_like(inputs.v[:1]))
        count = max(1, len(tokens))
        counts.append(count)
        relative[row, :count] = torch.arange(count, device=inputs.q.device)
    lengths = inputs.kv_lens.new_tensor(counts)
    cu_k = torch.cat((lengths.new_zeros(1), lengths.cumsum(0))).to(torch.int32)
    return sparse_gqa_packed_decode_triton(
        inputs.q,
        torch.cat(keys),
        torch.cat(values),
        relative,
        inputs.cu_q,
        cu_k,
        lengths,
        inputs.scale,
    )


@pytest.mark.parametrize("batch", [1, 4, 8])
def test_qsa_packed_pr40_matches_original_packed_on_same_effective_kv(batch):
    inputs = _make_inputs(batch=batch)
    _refresh(inputs, [17, 0, 2051, 65, 1, 1025, 2048, 513][:batch], 240004)
    inputs.indices[:, 1:17:3] = -1
    expected = _legacy_output_for_same_kv(inputs)
    _assert_reference(expected, inputs)
    for splits in (1, 64):
        output = qsa_sparse_packed_gqa(*inputs.args(), kv_splits=splits)
        _assert_reference(output, inputs)
        torch.testing.assert_close(output, expected, atol=4e-3, rtol=2e-2)


@pytest.mark.parametrize("width", [1, 31, 1025, 2051])
def test_qsa_packed_pr40_overwrites_poisoned_empty_partitions(width):
    inputs = _make_inputs(batch=4, width=width)
    workspace = SparsePackedGQAWorkspace()
    workspace.get(inputs.q, 64)
    for step, length in enumerate((width, 0, min(width, 1), width)):
        _refresh(inputs, [length] * 4, 240010 + step, dirty_tails=True)
        _poison(workspace)
        output = qsa_sparse_packed_gqa(
            *inputs.args(), workspace=workspace, kv_splits=64
        )
        _assert_reference(output, inputs)
        for partials, maxima, sums, _ in workspace.buffers.values():
            assert torch.isfinite(partials).all()
            assert torch.isfinite(maxima).all()
            assert torch.isfinite(sums).all()


@pytest.mark.parametrize("batch", [1, 4, 8])
def test_qsa_packed_pr40_graph_replay_refreshes_all_inputs_and_workspace(batch):
    inputs = _make_inputs(batch=batch)
    workspace = SparsePackedGQAWorkspace()
    for _ in range(3):
        qsa_sparse_packed_gqa(*inputs.args(), workspace=workspace, kv_splits=64)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = qsa_sparse_packed_gqa(
            *inputs.args(), workspace=workspace, kv_splits=64
        )
    output_pointer = output.data_ptr()
    scratch_pointers = {
        key: tuple(tensor.data_ptr() for tensor in buffers)
        for key, buffers in workspace.buffers.items()
    }
    lengths_by_step = [
        [17, 0, 1, 65, 1024, 2048, 2051, 128][:batch],
        [0] * batch,
        [2051] * batch,
        [1, 2049, 33, 0, 513, 7, 128, 2051][:batch],
    ]
    for step, lengths in enumerate(lengths_by_step):
        _refresh(
            inputs,
            lengths,
            240100 + batch * 10 + step,
            dirty_tails=True,
            rotate_segments=True,
        )
        if step == 2:
            inputs.indices[0, :128] = -1
        _poison(workspace)
        graph.replay()
        torch.cuda.synchronize()
        assert output.data_ptr() == output_pointer
        assert {
            key: tuple(tensor.data_ptr() for tensor in buffers)
            for key, buffers in workspace.buffers.items()
        } == scratch_pointers
        _assert_reference(output, inputs)
        # Validate the poisoned graph replay before another launch refreshes
        # scratch, then require exact agreement with eager execution.
        graph_result = output.clone()
        eager = qsa_sparse_packed_gqa(*inputs.args(), workspace=workspace, kv_splits=64)
        assert eager.data_ptr() != output.data_ptr()
        torch.testing.assert_close(output, graph_result, atol=0, rtol=0)
        torch.testing.assert_close(graph_result, eager, atol=0, rtol=0)


def test_qsa_packed_pr40_outputs_do_not_alias_reused_workspace():
    first = _make_inputs(seed=240201)
    second = _make_inputs(seed=240202)
    workspace = SparsePackedGQAWorkspace()
    first_output = qsa_sparse_packed_gqa(
        *first.args(), workspace=workspace, kv_splits=64
    )
    second_output = qsa_sparse_packed_gqa(
        *second.args(), workspace=workspace, kv_splits=64
    )
    assert len(workspace.buffers) == 1
    assert first_output.data_ptr() != second_output.data_ptr()
    assert all(
        output.data_ptr() != scratch.data_ptr()
        for output in (first_output, second_output)
        for buffers in workspace.buffers.values()
        for scratch in buffers
    )
    _assert_reference(first_output, first)
    _assert_reference(second_output, second)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
