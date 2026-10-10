"""Decode-shape benchmark for the QSA paged indexer MQA logits kernel.

It runs once per QSA layer, twelve times per decode step. Compares the Triton
kernel against the eager torch reference that runs when TileLang is missing.
"""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.srt.layers.attention.qsa.mqa import (
    torch_qsa_mqa_decode,
    triton_qsa_mqa_decode,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

# Qwen3.8-Flash-Next TP2 indexer: 4 heads, head_dim 128, compressed page size 64.
INDEX_HEADS = 4
INDEX_HEAD_DIM = 128
PAGE_SIZE = 64
MAX_PAGES = 1024
CACHE_PAGES = 4096
# 12000-token prompt at compress ratio 4.
COMPRESSED_CONTEXT = 3087


def _mqa_inputs(batch, device):
    generator = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(
        batch,
        INDEX_HEADS,
        INDEX_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    k_cache = torch.randn(
        CACHE_PAGES,
        PAGE_SIZE,
        1,
        INDEX_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    page_table = torch.randint(
        0,
        CACHE_PAGES,
        (batch, MAX_PAGES),
        dtype=torch.int32,
        device=device,
        generator=generator,
    )
    lengths = torch.full((batch,), COMPRESSED_CONTEXT, dtype=torch.int32, device=device)
    return q, k_cache, page_table, lengths


@marker.parametrize("batch", [1, 8, 32], [1, 32])
@marker.benchmark("impl", ["torch", "triton"], unit="us")
def benchmark_mqa_decode(batch: int, impl: str):
    device = torch.device("cuda")
    q, k_cache, page_table, lengths = _mqa_inputs(batch, device)
    call = torch_qsa_mqa_decode if impl == "torch" else triton_qsa_mqa_decode

    def fn():
        return call(q, k_cache, page_table, lengths, MAX_PAGES * PAGE_SIZE)

    return marker.do_bench(
        fn,
        input_args=(),
        graph_clone_args=None,
        memory_args=None,
        memory_output=None,
    )


if __name__ == "__main__":
    benchmark_mqa_decode.run()
