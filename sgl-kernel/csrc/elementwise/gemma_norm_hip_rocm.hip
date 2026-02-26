/* Copyright 2025 SGLang Team. All Rights Reserved.
 * HIP Gemma RMSNorm BF16 kernels for ROCm (from pyhip/archive/norm).
 * Only bf16; data is assumed to be bf16.
 */
#include "elementwise/gemma_norm_hip.cuh"
#include "sgl_kernel_ops.h"
#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>

namespace gemma_norm_hip {

// -----------------------------------------------------------------------------
// BF16 Gemma RMSNorm kernel
// -----------------------------------------------------------------------------
__global__ void gemma_rmsnorm_bf16_kernel(__bf16* __restrict__ output,
                                          const __bf16* __restrict__ input,
                                          const __bf16* __restrict__ weight,
                                          uint32_t hidden_size, float eps) {
  const uint32_t d = hidden_size;
  const uint32_t stride_input = hidden_size;
  const uint32_t stride_output = hidden_size;
  const float weight_bias = 1.f;

  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t warp_size = 32;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * warp_size;
  const uint32_t num_threads = num_warps * warp_size;
  const uint32_t rounds = ceil_div(d, VEC_SIZE * num_threads);
  extern __shared__ float smem[];

  float sum_sq = 0.f;

  for (uint32_t i = 0; i < rounds; i++) {
    vec_bf16_8 input_vec;
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    } else {
      *reinterpret_cast<uint4*>(&input_vec.data) = uint4{0, 0, 0, 0};
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      float v = __bfloat162float(input_vec[j]);
      sum_sq += v * v;
    }
  }

#pragma unroll
  for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    sum_sq += shfl_xor_sync(sum_sq, offset);
  }

  smem[ty] = sum_sq;
  __syncthreads();
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
      sum_sq += shfl_xor_sync(sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();

  float rms_rcp = rsqrt(smem[0] / float(d) + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    vec_bf16_8 input_vec, weight_vec, output_vec;
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      weight_vec.load(weight + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    } else {
      *reinterpret_cast<uint4*>(&input_vec.data) = uint4{0, 0, 0, 0};
      *reinterpret_cast<uint4*>(&weight_vec.data) = uint4{0, 0, 0, 0};
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      output_vec[j] = __float2bfloat16(__bfloat162float(input_vec[j]) * rms_rcp *
                                       (weight_bias + __bfloat162float(weight_vec[j])));
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      output_vec.store(output + bx * stride_output + i * num_threads * VEC_SIZE +
                       thread_id * VEC_SIZE);
    }
  }
}

// -----------------------------------------------------------------------------
// BF16 Gemma FusedAdd RMSNorm kernel
// -----------------------------------------------------------------------------
__global__ void gemma_fused_add_rmsnorm_bf16_kernel(__bf16* __restrict__ input,
                                                   __bf16* __restrict__ residual,
                                                   const __bf16* __restrict__ weight,
                                                   uint32_t hidden_size, float eps) {
  const uint32_t d = hidden_size;
  const uint32_t stride_input = hidden_size;
  const uint32_t stride_residual = hidden_size;
  const float weight_bias = 1.f;

  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t warp_size = 32;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * warp_size;
  const uint32_t num_threads = num_warps * warp_size;
  const uint32_t rounds = ceil_div(d, VEC_SIZE * num_threads);
  extern __shared__ float smem[];
  float* smem_x = smem + ceil_div(num_warps, 4u) * 4;

  float sum_sq = 0.f;

  for (uint32_t i = 0; i < rounds; i++) {
    vec_bf16_8 input_vec, residual_vec;
    float x_buf[VEC_SIZE];
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      residual_vec.load(residual + bx * stride_residual + i * num_threads * VEC_SIZE +
                        thread_id * VEC_SIZE);
    } else {
      *reinterpret_cast<uint4*>(&input_vec.data) = uint4{0, 0, 0, 0};
      *reinterpret_cast<uint4*>(&residual_vec.data) = uint4{0, 0, 0, 0};
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      float x = __bfloat162float(input_vec[j]) + __bfloat162float(residual_vec[j]);
      sum_sq += x * x;
      residual_vec[j] = __float2bfloat16(x);
      x_buf[j] = x;
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      residual_vec.store(residual + bx * stride_residual + i * num_threads * VEC_SIZE +
                         thread_id * VEC_SIZE);
      for (uint32_t j = 0; j < VEC_SIZE; j++) {
        smem_x[i * num_threads * VEC_SIZE + thread_id * VEC_SIZE + j] = x_buf[j];
      }
    }
  }

#pragma unroll
  for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    sum_sq += shfl_xor_sync(sum_sq, offset);
  }

  smem[ty] = sum_sq;
  __syncthreads();
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
      sum_sq += shfl_xor_sync(sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();

  float rms_rcp = rsqrt(smem[0] / float(d) + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    vec_bf16_8 weight_vec, output_vec;
    float x_buf[VEC_SIZE];
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      weight_vec.load(weight + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      for (uint32_t j = 0; j < VEC_SIZE; j++) {
        x_buf[j] = smem_x[i * num_threads * VEC_SIZE + thread_id * VEC_SIZE + j];
      }
    } else {
      *reinterpret_cast<uint4*>(&weight_vec.data) = uint4{0, 0, 0, 0};
      for (uint32_t j = 0; j < VEC_SIZE; j++) x_buf[j] = 0.f;
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      output_vec[j] = __float2bfloat16(x_buf[j] * rms_rcp *
                                       (weight_bias + __bfloat162float(weight_vec[j])));
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      output_vec.store(input + bx * stride_input + i * num_threads * VEC_SIZE +
                       thread_id * VEC_SIZE);
    }
  }
}

}  // namespace gemma_norm_hip

// -----------------------------------------------------------------------------
// at::Tensor wrappers (same signature as sgl_kernel_ops.h for ROCm build)
// -----------------------------------------------------------------------------
void gemma_rmsnorm(at::Tensor& output, at::Tensor& input, at::Tensor& weight, double eps,
                   bool enable_pdl) {
  (void)enable_pdl;
  TORCH_CHECK(weight.is_contiguous(), "gemma_rmsnorm_hip: weight must be contiguous");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16,
              "gemma_rmsnorm_hip: only bf16 supported, got ", input.scalar_type());
  TORCH_CHECK(weight.scalar_type() == at::kBFloat16, "gemma_rmsnorm_hip: weight must be bf16");
  TORCH_CHECK(output.scalar_type() == at::kBFloat16, "gemma_rmsnorm_hip: output must be bf16");
  TORCH_CHECK(input.dim() == 2, "gemma_rmsnorm_hip: input must be 2D (batch, hidden)");
  TORCH_CHECK(weight.dim() == 1, "gemma_rmsnorm_hip: weight must be 1D");
  TORCH_CHECK(input.size(1) == weight.size(0), "gemma_rmsnorm_hip: input hidden != weight size");
  TORCH_CHECK(output.sizes() == input.sizes(), "gemma_rmsnorm_hip: output shape != input shape");

  at::Tensor input_contig = input.contiguous();
  at::Tensor output_use = output.is_contiguous() ? output : at::empty_like(input_contig);

  const uint32_t batch_size = static_cast<uint32_t>(input_contig.size(0));
  const uint32_t hidden_size = static_cast<uint32_t>(input_contig.size(1));

  const uint32_t num_warps = 8u;
  const uint32_t block_x = 32u;
  const uint32_t smem_rms = num_warps * sizeof(float);

  hipStream_t stream = at::hip::getCurrentHIPStream();
  hipLaunchKernelGGL(
      gemma_norm_hip::gemma_rmsnorm_bf16_kernel,
      dim3(batch_size),
      dim3(block_x, num_warps),
      smem_rms,
      stream,
      reinterpret_cast<__bf16*>(output_use.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __bf16*>(input_contig.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __bf16*>(weight.data_ptr<at::BFloat16>()),
      hidden_size,
      static_cast<float>(eps));
  TORCH_CHECK(hipGetLastError() == hipSuccess, "gemma_rmsnorm_hip: kernel launch failed");

  if (!output.is_contiguous()) {
    output.copy_(output_use);
  }
}

void gemma_fused_add_rmsnorm(at::Tensor& input, at::Tensor& residual, at::Tensor& weight,
                             double eps, bool enable_pdl) {
  (void)enable_pdl;
  TORCH_CHECK(weight.is_contiguous(), "gemma_fused_add_rmsnorm_hip: weight must be contiguous");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16,
              "gemma_fused_add_rmsnorm_hip: only bf16 supported, got ", input.scalar_type());
  TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "gemma_fused_add_rmsnorm_hip: residual must be bf16");
  TORCH_CHECK(weight.scalar_type() == at::kBFloat16, "gemma_fused_add_rmsnorm_hip: weight must be bf16");
  TORCH_CHECK(input.dim() == 2 && residual.dim() == 2, "gemma_fused_add_rmsnorm_hip: input/residual must be 2D");
  TORCH_CHECK(weight.dim() == 1, "gemma_fused_add_rmsnorm_hip: weight must be 1D");
  TORCH_CHECK(input.size(0) == residual.size(0) && input.size(1) == residual.size(1),
              "gemma_fused_add_rmsnorm_hip: input and residual shapes must match");
  TORCH_CHECK(input.size(1) == weight.size(0), "gemma_fused_add_rmsnorm_hip: hidden size mismatch");

  at::Tensor input_contig = input.contiguous();
  at::Tensor residual_contig = residual.contiguous();
  const bool need_copy_back = !input.is_contiguous() || !residual.is_contiguous();

  const uint32_t batch_size = static_cast<uint32_t>(input_contig.size(0));
  const uint32_t hidden_size = static_cast<uint32_t>(input_contig.size(1));

  const uint32_t num_warps = 8u;
  const uint32_t num_threads = num_warps * 32u;
  const uint32_t rounds = (hidden_size + gemma_norm_hip::VEC_SIZE * num_threads - 1) /
                          (gemma_norm_hip::VEC_SIZE * num_threads);
  const uint32_t smem_align = (gemma_norm_hip::ceil_div(num_warps, 4u) * 4u) * sizeof(float);
  const uint32_t smem_x = rounds * num_threads * gemma_norm_hip::VEC_SIZE * sizeof(float);
  const uint32_t smem_total = smem_align + smem_x;

  hipStream_t stream = at::hip::getCurrentHIPStream();
  hipLaunchKernelGGL(
      gemma_norm_hip::gemma_fused_add_rmsnorm_bf16_kernel,
      dim3(batch_size),
      dim3(32u, num_warps),
      smem_total,
      stream,
      reinterpret_cast<__bf16*>(input_contig.data_ptr<at::BFloat16>()),
      reinterpret_cast<__bf16*>(residual_contig.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __bf16*>(weight.data_ptr<at::BFloat16>()),
      hidden_size,
      static_cast<float>(eps));
  TORCH_CHECK(hipGetLastError() == hipSuccess, "gemma_fused_add_rmsnorm_hip: kernel launch failed");

  if (need_copy_back) {
    input.copy_(input_contig);
    residual.copy_(residual_contig);
  }
}
