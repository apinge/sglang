#pragma once

#ifndef USE_ROCM
#error "shared_gate_append_hip.cuh requires ROCm"
#endif

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

// One block computes a token's BF16 gate and appends it to the routed top-k.
// All arithmetic stays FP32 until the final FP32 routing-weight store.
__global__ void shared_gate_append_hip_kernel(
    const uint16_t* __restrict__ hidden,
    const uint16_t* __restrict__ gate,
    const int32_t* __restrict__ ids,
    const float* __restrict__ weights,
    int32_t* __restrict__ out_ids,
    float* __restrict__ out_weights,
    int topk,
    int shared,
    int expert_base,
    float scale) {
  constexpr int kHidden = 4096;
  constexpr int kThreads = 64;
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  using bf16x8 = uint16_t __attribute__((ext_vector_type(8)));
  float sums[8] = {};
#pragma unroll
  for (int chunk = 0; chunk < kHidden / (kThreads * 8); ++chunk) {
    const int col = tid * 8 + chunk * (kThreads * 8);
    const auto xs = *reinterpret_cast<const bf16x8*>(hidden + row * kHidden + col);
    const auto ws = *reinterpret_cast<const bf16x8*>(gate + col);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float x = __uint_as_float(static_cast<uint32_t>(xs[i]) << 16);
      const float w = __uint_as_float(static_cast<uint32_t>(ws[i]) << 16);
      sums[i] += x * w;
    }
  }
  float sum = ((sums[0] + sums[1]) + (sums[2] + sums[3])) + ((sums[4] + sums[5]) + (sums[6] + sums[7]));
#pragma unroll
  for (int offset = 32; offset > 0; offset >>= 1) {
    sum += __shfl_xor(sum, offset, 64);
  }
  const float value = scale * __builtin_amdgcn_rcpf(1.0f + __builtin_amdgcn_exp2f(-sum * 1.4426950408889634f));

  const int out_base = row * (topk + shared);
  for (int col = tid; col < topk; col += kThreads) {
    out_ids[out_base + col] = ids[row * topk + col];
    out_weights[out_base + col] = weights[row * topk + col];
  }
  if (tid == 0) {
    for (int i = 0; i < shared; ++i) {
      out_ids[out_base + topk + i] = expert_base + i;
      out_weights[out_base + topk + i] = value;
    }
  }
}

struct SharedGateAppendHip {
  static void
  run(const tvm::ffi::TensorView hidden,
      const tvm::ffi::TensorView gate,
      const tvm::ffi::TensorView ids,
      const tvm::ffi::TensorView weights,
      const tvm::ffi::TensorView out_ids,
      const tvm::ffi::TensorView out_weights,
      int64_t expert_base,
      int64_t shared,
      double scale) {
    using namespace host;
    auto rows = SymbolicSize{"rows"};
    auto topk = SymbolicSize{"topk"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({rows, 4096}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(hidden);
    TensorMatcher({1, 4096}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(gate);
    TensorMatcher({rows, topk}).with_dtype<int32_t>().with_device(device).verify(ids);
    TensorMatcher({rows, topk}).with_dtype<float>().with_device(device).verify(weights);
    RuntimeCheck(topk.unwrap() > 0 && topk.unwrap() <= 256, "topk must be in [1, 256]");
    RuntimeCheck(shared > 0 && shared <= 4, "shared expert count must be in [1, 4]");
    TensorMatcher({rows, topk.unwrap() + shared}).with_dtype<int32_t>().with_device(device).verify(out_ids);
    TensorMatcher({rows, topk.unwrap() + shared}).with_dtype<float>().with_device(device).verify(out_weights);
    if (rows.unwrap() == 0) return;
    LaunchKernel(rows.unwrap(), 64, device.unwrap())(
        shared_gate_append_hip_kernel,
        static_cast<const uint16_t*>(hidden.data_ptr()),
        static_cast<const uint16_t*>(gate.data_ptr()),
        static_cast<const int32_t*>(ids.data_ptr()),
        static_cast<const float*>(weights.data_ptr()),
        static_cast<int32_t*>(out_ids.data_ptr()),
        static_cast<float*>(out_weights.data_ptr()),
        static_cast<int>(topk.unwrap()),
        static_cast<int>(shared),
        static_cast<int>(expert_base),
        static_cast<float>(scale));
  }
};

}  // namespace sglang
