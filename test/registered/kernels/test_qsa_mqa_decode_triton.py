"""Correctness of the Triton paged QSA indexer MQA kernel against its torch reference.

The kernel runs inside the decode CUDA graph, so capture/replay with mutated
context lengths is covered here too.
"""

import math

import pytest
import torch

from sglang.srt.layers.attention.qsa.mqa import (
    torch_qsa_mqa_decode,
    triton_qsa_mqa_decode,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=60, stage="stage-b", runner_config="1-gpu-large-amd")

PAGE_SIZE = 64
HEAD_DIM = 128
NUM_HEADS = 4
CACHE_PAGES = 512

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a GPU"
)


def _decode_inputs(batch, max_pages, context_lens, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        batch,
        NUM_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    k_cache = torch.randn(
        CACHE_PAGES,
        PAGE_SIZE,
        1,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    page_table = torch.randint(
        0,
        CACHE_PAGES,
        (batch, max_pages),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )
    lengths = torch.tensor(context_lens, dtype=torch.int32, device="cuda")
    return q, k_cache, page_table, lengths


def _assert_matches_reference(q, k_cache, page_table, lengths, max_model_len):
    expected = torch_qsa_mqa_decode(q, k_cache, page_table, lengths, max_model_len)
    actual = triton_qsa_mqa_decode(q, k_cache, page_table, lengths, max_model_len)
    finite = torch.isfinite(expected)
    assert torch.equal(torch.isfinite(actual), finite)
    # Both paths multiply bf16 inputs and accumulate in fp32; only the
    # summation order differs, so the tolerance is at fp32 rounding level.
    torch.testing.assert_close(actual[finite], expected[finite], rtol=1e-5, atol=1e-4)


@requires_gpu
@pytest.mark.parametrize("batch", [1, 2, 8, 32])
def test_triton_mqa_decode_matches_torch_on_ragged_contexts(batch):
    max_pages = 24
    lengths = [1 + (index * 137) % (max_pages * PAGE_SIZE) for index in range(batch)]
    q, k_cache, page_table, context_lens = _decode_inputs(batch, max_pages, lengths)
    _assert_matches_reference(
        q, k_cache, page_table, context_lens, max_pages * PAGE_SIZE
    )


@requires_gpu
def test_triton_mqa_decode_masks_page_table_padding():
    """A negative page id is padding, not a cache slot: the torch reference
    clamps it to page 0 and the context length masks the row away."""

    max_pages = 8
    q, k_cache, page_table, context_lens = _decode_inputs(
        2, max_pages, [PAGE_SIZE * 3, 100]
    )
    page_table[:, 3:] = -1
    _assert_matches_reference(
        q, k_cache, page_table, context_lens, max_pages * PAGE_SIZE
    )


@requires_gpu
@pytest.mark.parametrize("max_model_len", [PAGE_SIZE, 320, 8 * PAGE_SIZE + 96])
def test_triton_mqa_decode_honours_output_width(max_model_len):
    """Rows are -inf past the page table even when the context reaches further,
    and are truncated when the output is narrower than the table."""

    max_pages = 8
    q, k_cache, page_table, context_lens = _decode_inputs(
        3, max_pages, [1, 4 * PAGE_SIZE, 1 << 20]
    )
    _assert_matches_reference(q, k_cache, page_table, context_lens, max_model_len)


@requires_gpu
def test_triton_mqa_decode_replays_under_cuda_graph():
    """Decode runs inside a replayed graph, so the kernel must not allocate on
    the host, sync, or read shapes off device tensors."""

    max_pages = 16
    max_model_len = max_pages * PAGE_SIZE
    q, k_cache, page_table, context_lens = _decode_inputs(
        4, max_pages, [PAGE_SIZE, 300, 900, 1024]
    )

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        triton_qsa_mqa_decode(q, k_cache, page_table, context_lens, max_model_len)
    torch.cuda.current_stream().wait_stream(warmup)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = triton_qsa_mqa_decode(
            q, k_cache, page_table, context_lens, max_model_len
        )
    context_lens.copy_(
        torch.tensor([1, 1023, 64, 512], dtype=torch.int32, device="cuda")
    )
    graph.replay()
    torch.cuda.synchronize()

    expected = torch_qsa_mqa_decode(q, k_cache, page_table, context_lens, max_model_len)
    finite = torch.isfinite(expected)
    assert torch.equal(torch.isfinite(captured), finite)
    torch.testing.assert_close(captured[finite], expected[finite], rtol=1e-5, atol=1e-4)


@requires_gpu
def test_triton_mqa_decode_applies_score_scale():
    max_pages = 4
    q, k_cache, page_table, context_lens = _decode_inputs(
        2, max_pages, [PAGE_SIZE, 200]
    )
    scale = 3.5
    actual = triton_qsa_mqa_decode(
        q, k_cache, page_table, context_lens, max_pages * PAGE_SIZE, score_scale=scale
    )
    default = triton_qsa_mqa_decode(
        q, k_cache, page_table, context_lens, max_pages * PAGE_SIZE
    )
    finite = torch.isfinite(default)
    torch.testing.assert_close(
        actual[finite],
        default[finite] * (math.sqrt(HEAD_DIM) / scale),
        rtol=1e-5,
        atol=1e-4,
    )
