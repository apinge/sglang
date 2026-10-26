"""Benchmark the gfx942 Triton skinny bf16 GEMM against hipBLASLt (torch.matmul).

Shapes are the Qwen3.8-Flash-Next lm_head shards (vocab 248320, hidden 2560)
for TP1/2/4/8. Run on MI300-series (gfx942):
    python test/registered/kernels/benchmark/gemm/bench_rocm_bf16_gemv.py
"""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.srt.utils.common import is_gfx942_supported
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, stage="jit-kernel-benchmark", runner_config="amd")


def _triton(x, w):
    from sglang.kernels.ops.gemm.rocm_bf16_gemv import rocm_bf16_gemv

    return rocm_bf16_gemv(x, w)


def _hipblaslt(x, w):
    return torch.matmul(x, w.T)


FN_MAP = {"triton": _triton, "hipblaslt": _hipblaslt}


@marker.parametrize("n", [248320, 124160, 62080, 31040], [124160])
@marker.parametrize("k", [2560])
@marker.parametrize("m", [1, 2, 4, 8, 16, 24, 32], [1, 8, 32])
@marker.benchmark("provider", ["triton", "hipblaslt"])
def benchmark(n, k, m, provider):
    x = create_random(m, k, dtype=torch.bfloat16)
    w = create_random(n, k, dtype=torch.bfloat16)
    return marker.do_bench(FN_MAP[provider], input_args=(x, w))


if __name__ == "__main__":
    if not is_gfx942_supported():
        print("rocm_bf16_gemv is tuned for gfx942 only. Skipping benchmark.")
    else:
        benchmark.run()
