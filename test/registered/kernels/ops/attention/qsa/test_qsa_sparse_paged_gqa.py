"""Correctness and automatic dispatch for the gfx942 QSA decode path."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.qsa.sparse_attn_decode import (
    SparsePagedGQAWorkspace,
    _kernel_config,
    _kv_splits_heuristic,
    qsa_sparse_paged_gqa,
    supports_sparse_paged_gqa,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, stage="stage-b", runner_config="1-gpu-large-amd")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is None
    or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx942"),
    reason="MI300 BF16 paged QSA kernel",
)


def make_inputs(
    tokens=4, heads=12, kv_heads=1, dim=256, width=2051, context=12000, strided=False
):
    torch.manual_seed(2311)
    device = "cuda"
    step = 2 if strided else 1
    q = torch.randn(tokens, heads, dim * step, device=device, dtype=torch.bfloat16)[
        ..., ::step
    ]
    k = torch.randn(
        context + 17, kv_heads, dim * step, device=device, dtype=torch.bfloat16
    )[..., ::step]
    v = torch.randn_like(k)
    table = torch.randint(
        0, k.shape[0], (5, context * step), device=device, dtype=torch.int32
    )[:, ::step]
    indices = torch.randint(
        0, context, (tokens, width * step), device=device, dtype=torch.int32
    )[:, ::step]
    # Repeated requests with distinct visible lengths model flattened MTP rows.
    requests = (torch.arange(tokens, device=device, dtype=torch.int32) // 4 + 1) % 5
    lengths = torch.full((tokens,), context, device=device, dtype=torch.int32)
    if tokens:
        lengths[:] -= torch.arange(tokens, device=device, dtype=torch.int32) % 4
    if width >= 7:
        indices[:, :7] = torch.tensor(
            [-1, context, 0, 1, 2, 3, context - 1], device=device
        )
        table[:, 0] = -1
        table[:, 1] = k.shape[0] + 3
    return q, k, v, indices, table, requests, lengths


def reference(q, k, v, indices, table, requests, lengths):
    """FP32 attention, including request/slot bounds and query visibility."""
    out = torch.zeros_like(q, dtype=torch.float32)
    for row in range(q.shape[0]):
        req = int(requests[row])
        if req < 0 or req >= table.shape[0]:
            continue
        logical = indices[row].long()
        logical = logical[
            (logical >= 0) & (logical < lengths[row]) & (logical < table.shape[1])
        ]
        physical = table[req, logical].long()
        physical = physical[(physical >= 0) & (physical < k.shape[0])]
        if not physical.numel():
            continue
        keys = k[physical].float().repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        values = v[physical].float().repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        scores = torch.einsum("hd,nhd->hn", q[row].float(), keys) * q.shape[2] ** -0.5
        out[row] = torch.einsum("hn,nhd->hd", scores.softmax(-1), values)
    return out


def run(inputs, **kwargs):
    return qsa_sparse_paged_gqa(*inputs[:6], sequence_lengths=inputs[6], **kwargs)


def assert_reference(actual, inputs):
    expected = reference(*inputs)
    assert torch.isfinite(actual).all()
    # BF16 Q scaling / probabilities / output rounding in the source kernel.
    torch.testing.assert_close(actual.float(), expected, atol=3e-3, rtol=2e-2)


@pytest.mark.parametrize(
    "tokens,width,splits",
    [
        (0, 2051, None),
        (1, 0, 64),
        (1, 1, 1),
        (4, 31, 8),
        (4, 511, None),
        (1, 512, None),
        (4, 2048, None),
        (1, 2051, None),
        (4, 2051, None),
        (8, 2051, None),
        (16, 2051, None),
        (32, 2051, None),
        (128, 2051, None),
        (4, 2051, 1),
        (4, 2051, 8),
        (4, 2051, 16),
        (4, 2051, 32),
        (4, 2051, 64),
    ],
)
def test_sparse_paged_reference(tokens, width, splits):
    inputs = make_inputs(tokens=tokens, width=width)
    assert_reference(run(inputs, kv_splits=splits), inputs)


@pytest.mark.parametrize(
    "heads,kv_heads,dim,strided",
    [
        (24, 1, 256, False),
        (20, 1, 256, False),
        (24, 2, 128, False),
        (12, 1, 256, True),
    ],
)
def test_head_groups_and_strides(heads, kv_heads, dim, strided):
    inputs = make_inputs(
        tokens=8, heads=heads, kv_heads=kv_heads, dim=dim, strided=strided
    )
    assert_reference(run(inputs, kv_splits=64), inputs)


@pytest.mark.parametrize("length", [0, 1, 3, 4, 5, 7, 8, 9, 2047, 2048, 2049])
def test_visibility_and_invalid_rows(length):
    inputs = make_inputs(tokens=4)
    inputs[6].fill_(length)
    inputs[5][0] = -1
    inputs[5][1] = inputs[4].shape[0]
    inputs[3][2].fill_(-1)
    result = run(inputs, kv_splits=64)
    assert torch.count_nonzero(result[:3]) == 0
    assert_reference(result, inputs)


def test_graph_replay_refreshes_metadata_and_partials():
    inputs = make_inputs(tokens=8, width=2051)
    workspace = SparsePagedGQAWorkspace()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run(inputs, kv_splits=64, workspace=workspace)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = run(inputs, kv_splits=64, workspace=workspace)
    for iteration in range(4):
        for partials, maxima, sums, widths in workspace.buffers.values():
            partials.fill_(float("nan"))
            maxima.fill_(float("nan"))
            sums.fill_(float("nan"))
            widths.fill_(-1)
        inputs[0].normal_()
        inputs[3].random_(0, inputs[4].shape[1])
        inputs[4].random_(0, inputs[1].shape[0])
        inputs[5].copy_(
            torch.arange(8, device="cuda", dtype=torch.int32) // 4 + iteration % 3
        )
        inputs[6].copy_(torch.arange(8, device="cuda", dtype=torch.int32) % 4 + 11996)
        inputs[3][0].fill_(-1)
        if iteration == 2:
            inputs[6].zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert_reference(output, inputs)


def test_backend_dispatch_uses_real_pool_and_mtp_rows(monkeypatch):
    from sglang.srt.layers.attention import qwen_sparse_attn_backend as module

    inputs = make_inputs(tokens=8)
    q, k, v, indices, table, requests, lengths = inputs
    pool = SimpleNamespace(get_key_buffer=lambda _: k, get_value_buffer=lambda _: v)
    backend = module.QwenSparseAttnBackend(
        SimpleNamespace(
            token_to_kv_pool=pool,
            req_to_token_pool=SimpleNamespace(req_to_token=table),
        )
    )
    metadata = SimpleNamespace(
        sequence_lengths=lengths,
        row_req_pool_indices=requests,
        token_slot_table=torch.full((1, 1), -1, device="cuda", dtype=torch.int32),
        is_cuda_graph=True,
    )
    monkeypatch.setattr(backend, "_resolve_metadata", lambda _: metadata)

    def reject_compact(*args, **kwargs):
        pytest.fail("compatible input unexpectedly entered legacy compact path")

    monkeypatch.setattr(
        module, "qwen_sparse_kv_extraction_compact_triton", reject_compact
    )
    batch = SimpleNamespace(req_pool_indices=requests[::4])
    layer = SimpleNamespace(layer_id=0, scaling=q.shape[2] ** -0.5)
    output = backend._forward_paged_attention(q, layer, batch, indices).view_as(q)
    assert_reference(output, inputs)
    assert backend._paged_gqa_workspace is not None


@pytest.mark.parametrize(
    "heads,kv_heads,dim,supported",
    [
        (12, 1, 256, True),
        (24, 2, 128, True),
        (32, 1, 128, True),
        (33, 1, 128, False),
        (3, 2, 128, False),
        (12, 1, 64, False),
        (0, 1, 128, False),
        (12, 0, 128, False),
    ],
)
def test_supported_head_layouts(heads, kv_heads, dim, supported):
    q, k, v, *_ = make_inputs(heads=heads, kv_heads=kv_heads, dim=dim)
    assert supports_sparse_paged_gqa(q, k, v) == supported


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_unsupported_dtype(dtype):
    q, k, v, *_ = make_inputs()
    assert not supports_sparse_paged_gqa(q.to(dtype), k.to(dtype), v.to(dtype))


def test_unsupported_device_arch_and_layout(monkeypatch):
    from sglang.srt.layers.attention.qsa import sparse_attn_decode as module

    q, k, v, *_ = make_inputs()
    assert not supports_sparse_paged_gqa(q.cpu(), k, v)
    assert not supports_sparse_paged_gqa(q, k, v.to("meta"))
    assert not supports_sparse_paged_gqa(q, k.unsqueeze(0), v.unsqueeze(0))
    assert not supports_sparse_paged_gqa(q, k, v[:, :, :128])
    assert not supports_sparse_paged_gqa(q[:, :, :128], k, v)
    monkeypatch.setattr(module, "_device_info", lambda _: ("gfx90a", 110))
    assert not supports_sparse_paged_gqa(q, k, v)


def test_mtp_heuristic_counts_query_rows():
    assert _kv_splits_heuristic(4, 1, 2051, 80) == 64
    assert _kv_splits_heuristic(16, 1, 2051, 80) == 16
    assert _kv_splits_heuristic(4, 1, 511, 80) == 1
    assert _kernel_config(1, 1, 1, 24, 80) == (64, 4, 1, 0, 24)
    assert _kernel_config(8, 1, 64, 20, 80) == (32, 2, 1, 0, 8)


@pytest.mark.parametrize("supported", [True, False])
def test_automatic_dispatch_and_fallback_outputs(monkeypatch, supported):
    from sglang.srt.layers.attention import qwen_sparse_attn_backend as module
    from sglang.srt.layers.attention.qsa import sparse_attn_decode

    inputs = make_inputs(tokens=4)
    q, k, v, indices, table, requests, lengths = inputs
    # Legacy assumes physical slots from the allocator are valid.
    table[:, 0] = 5
    table[:, 1] = 6
    # Legacy compact requires a valid prefix followed by padding.
    valid = (indices >= 0) & (indices < lengths[:, None])
    indices.copy_(torch.sort(torch.where(valid, indices, table.shape[1]), dim=1).values)
    indices.masked_fill_(indices >= table.shape[1], -1)
    if not supported:
        monkeypatch.setattr(
            sparse_attn_decode, "supports_sparse_paged_gqa", lambda *args: False
        )
    backend = module.QwenSparseAttnBackend(
        SimpleNamespace(
            token_to_kv_pool=SimpleNamespace(
                get_key_buffer=lambda _: k, get_value_buffer=lambda _: v
            ),
            req_to_token_pool=SimpleNamespace(req_to_token=table),
        )
    )
    metadata = SimpleNamespace(
        sequence_lengths=lengths, row_req_pool_indices=None, is_cuda_graph=False
    )
    monkeypatch.setattr(backend, "_resolve_metadata", lambda _: metadata)
    layer = SimpleNamespace(layer_id=0, scaling=q.shape[2] ** -0.5)
    output = backend._forward_paged_attention(
        q, layer, SimpleNamespace(req_pool_indices=requests), indices
    ).view_as(q)
    assert_reference(output, inputs)
    assert bool(backend._fa2_scratch) == (not supported)
    assert (backend._paged_gqa_workspace is not None) == supported
