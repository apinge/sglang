"""Decode-shape benchmark for the split-KV sparse-GQA decode kernel.

It runs once per QSA layer, twelve times per decode step. Compares the split-KV
kernel against the one-program-per-request chunk-prefill kernel it replaced.
"""

import torch
import triton

from sglang.kernels.jit.benchmark import marker
from sglang.srt.layers.attention.qsa.sparse_attn import (
    _get_best_config,
    _sparse_gqa_chunk_prefill,
    sparse_gqa_packed_decode_triton,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, stage="jit-kernel-benchmark", runner_config="amd")

# Qwen3.8-Flash-Next TP2: 24 q / 2 kv heads, head_dim 256, indexer budget 2048
# (+ compress_ratio - 1 expansion slots).
NUM_Q_HEADS = 24
NUM_KV_HEADS = 2
HEAD_DIM = 256
TOPK = 2051
VALID_COUNT = 2048


def _gqa_inputs(batch, device):
    generator = torch.Generator(device=device).manual_seed(0)
    counts = torch.full((batch,), VALID_COUNT, dtype=torch.int32, device=device)
    cu_q = torch.arange(batch + 1, dtype=torch.int32, device=device)
    cu_k = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    cu_k[1:] = torch.cumsum(counts, 0)
    q = torch.randn(
        batch,
        NUM_Q_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    k = torch.randn(
        batch * VALID_COUNT,
        NUM_KV_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    v = torch.randn_like(k)
    indices = torch.arange(TOPK, dtype=torch.int32, device=device).expand(batch, -1)
    indices = indices.masked_fill(indices >= counts[:, None], -1).contiguous()
    return q, k, v, indices, cu_q, cu_k, counts


def _unsplit_gqa(q, k, v, indices, cu_q, cu_k, kv_lens, scale):
    """The pre-split-KV decode path: grid (1, batch * num_kv_heads)."""

    group_size = NUM_Q_HEADS // NUM_KV_HEADS
    block_n, warps, stages = _get_best_config(q.shape[0])
    out = torch.empty_like(q)
    _sparse_gqa_chunk_prefill[(1, (cu_q.shape[0] - 1) * NUM_KV_HEADS)](
        q,
        k,
        v,
        out,
        indices,
        cu_q,
        cu_k,
        kv_lens,
        scale,
        indices.shape[-1],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        indices.stride(0),
        0,
        indices.stride(1),
        NUM_KV_HEADS=NUM_KV_HEADS,
        GROUP_SIZE=group_size,
        BLOCK_M=max(16, triton.next_power_of_2(group_size)),
        BLOCK_N=block_n,
        HEAD_DIM=HEAD_DIM,
        num_warps=warps,
        num_stages=stages,
    )
    return out


@marker.parametrize("batch", [1, 8, 32], [1, 32])
@marker.benchmark("impl", ["unsplit", "split"], unit="us")
def benchmark_sparse_gqa_decode(batch: int, impl: str):
    device = torch.device("cuda")
    q, k, v, indices, cu_q, cu_k, counts = _gqa_inputs(batch, device)
    scale = HEAD_DIM**-0.5

    def fn():
        if impl == "unsplit":
            return _unsplit_gqa(q, k, v, indices, cu_q, cu_k, counts, scale)
        return sparse_gqa_packed_decode_triton(
            q, k, v, None, cu_q, cu_k, counts, scale, identity_topk=TOPK
        )

    return marker.do_bench(
        fn,
        input_args=(),
        graph_clone_args=None,
        memory_args=None,
        memory_output=None,
    )


if __name__ == "__main__":
    benchmark_sparse_gqa_decode.run()
