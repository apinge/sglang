"""Correctness of the split-KV sparse-GQA decode kernel against an explicit reference.

The kernel runs inside the decode CUDA graph, so capture/replay with mutated
valid counts is covered here too.
"""

import pytest
import torch

from sglang.srt.layers.attention.qsa.sparse_attn import (
    qwen_sparse_fa2_cu_seqlens_triton,
    sparse_gqa_packed_decode_triton,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, stage="stage-b", runner_config="1-gpu-large-amd")

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a GPU"
)


NUM_Q_HEADS = 24
NUM_KV_HEADS = 2
ATTN_HEAD_DIM = 256
TOPK = 2051
ATTN_SCALE = ATTN_HEAD_DIM**-0.5


def _packed_decode_inputs(valid_counts, identity, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    batch = len(valid_counts)
    counts = torch.tensor(valid_counts, dtype=torch.int32, device="cuda")
    cu_q = torch.arange(batch + 1, dtype=torch.int32, device="cuda")
    cu_k = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
    cu_k[1:] = torch.cumsum(counts, 0)
    packed = max(int(cu_k[-1].item()), 1)
    q = torch.randn(
        batch,
        NUM_Q_HEADS,
        ATTN_HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    k = torch.randn(
        packed,
        NUM_KV_HEADS,
        ATTN_HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    v = torch.randn(
        packed,
        NUM_KV_HEADS,
        ATTN_HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    indices = None
    if not identity:
        indices = torch.arange(TOPK, dtype=torch.int32, device="cuda").expand(batch, -1)
        indices = indices.masked_fill(indices >= counts[:, None], -1).contiguous()
    return q, k, v, indices, cu_q, cu_k, counts


def _explicit_gqa(q, k, v, cu_k, counts):
    expected = torch.empty_like(q)
    group_size = NUM_Q_HEADS // NUM_KV_HEADS
    for row, length in enumerate(counts.tolist()):
        start, end = int(cu_k[row]), int(cu_k[row + 1])
        for head in range(NUM_Q_HEADS):
            kv_head = head // group_size
            scores = (
                q[row, head].float() @ k[start:end][:length, kv_head].float().T
            ) * ATTN_SCALE
            expected[row, head] = (
                scores.softmax(-1) @ v[start:end][:length, kv_head].float()
            ).to(q.dtype)
    return expected


@requires_gpu
@pytest.mark.parametrize(
    "valid_counts",
    [
        [2048],
        [2048, 1],
        [2048, 1, 17, 64, 65, 1000, TOPK, 3],
        [1 + (index * 97) % TOPK for index in range(32)],
    ],
)
def test_split_decode_matches_explicit_gqa(valid_counts):
    """The flash-decoding rescale must reproduce one-pass softmax; a wrong
    combine shows up as a per-row scale error, not as garbage."""

    q, k, v, indices, cu_q, cu_k, counts = _packed_decode_inputs(
        valid_counts, identity=True
    )
    actual = sparse_gqa_packed_decode_triton(
        q, k, v, indices, cu_q, cu_k, counts, ATTN_SCALE, identity_topk=TOPK
    )
    torch.testing.assert_close(
        actual, _explicit_gqa(q, k, v, cu_k, counts), rtol=2e-2, atol=2e-2
    )


@requires_gpu
def test_split_decode_identity_path_matches_materialized_indices():
    """The identity fast path replaces a [batch, topk] index tensor the host
    used to build every replay; both must select the same rows."""

    valid_counts = [2048, 1, 17, 64, 65, 1000, TOPK, 3]
    args = _packed_decode_inputs(valid_counts, identity=False)
    q, k, v, indices, cu_q, cu_k, counts = args
    materialized = sparse_gqa_packed_decode_triton(
        q, k, v, indices, cu_q, cu_k, counts, ATTN_SCALE
    )
    identity = sparse_gqa_packed_decode_triton(
        q, k, v, None, cu_q, cu_k, counts, ATTN_SCALE, identity_topk=TOPK
    )
    assert torch.equal(identity, materialized)


@requires_gpu
def test_split_decode_accepts_non_contiguous_query():
    """The backend hands the kernel a query slice without a contiguity copy."""

    q, k, v, _, cu_q, cu_k, counts = _packed_decode_inputs([2048, 300], identity=True)
    padded = torch.zeros(
        q.shape[0], NUM_Q_HEADS, 2 * ATTN_HEAD_DIM, dtype=q.dtype, device=q.device
    )
    padded[:, :, :ATTN_HEAD_DIM] = q
    strided = padded[:, :, :ATTN_HEAD_DIM]
    assert not strided.is_contiguous()
    actual = sparse_gqa_packed_decode_triton(
        strided, k, v, None, cu_q, cu_k, counts, ATTN_SCALE, identity_topk=TOPK
    )
    torch.testing.assert_close(
        actual, _explicit_gqa(q, k, v, cu_k, counts), rtol=2e-2, atol=2e-2
    )


@requires_gpu
def test_split_decode_replays_under_cuda_graph():
    q, k, v, _, cu_q, cu_k, counts = _packed_decode_inputs([2048, 900], identity=True)

    def call():
        return sparse_gqa_packed_decode_triton(
            q, k, v, None, cu_q, cu_k, counts, ATTN_SCALE, identity_topk=TOPK
        )

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        call()
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    counts.copy_(torch.tensor([64, 2048], dtype=torch.int32, device="cuda"))
    cu_k.copy_(torch.tensor([0, 64, 2112], dtype=torch.int32, device="cuda"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        captured, _explicit_gqa(q, k, v, cu_k, counts), rtol=2e-2, atol=2e-2
    )


@requires_gpu
@pytest.mark.parametrize("batch", [1, 4, 8, 9, 16, 64])
def test_fa2_cu_seqlens_counts_and_prefix(batch):
    """The fused single-program path is selected only below a batch threshold;
    both sides of that switch must produce identical counts and offsets."""

    generator = torch.Generator(device="cuda").manual_seed(3)
    seq_lens = torch.randint(
        1, 12000, (batch,), dtype=torch.int32, device="cuda", generator=generator
    )
    indices = torch.randint(
        -1, 12000, (batch, TOPK), dtype=torch.int32, device="cuda", generator=generator
    )
    counts = torch.empty(batch, dtype=torch.int32, device="cuda")
    cu_k = torch.empty(batch + 1, dtype=torch.int32, device="cuda")
    qwen_sparse_fa2_cu_seqlens_triton(seq_lens, indices, counts, cu_k, batch, TOPK)

    expected_counts = ((indices >= 0) & (indices < seq_lens[:, None])).sum(1)
    assert torch.equal(counts, expected_counts.to(torch.int32))
    assert int(cu_k[0]) == 0
    assert torch.equal(cu_k[1:], torch.cumsum(expected_counts, 0).to(torch.int32))
