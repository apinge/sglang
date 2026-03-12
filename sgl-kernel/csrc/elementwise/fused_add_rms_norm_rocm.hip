#include <ATen/hip/HIPContext.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include "hip/hip_math_def.h"

namespace {

constexpr uint32_t WARP_SIZE = 64;
constexpr uint32_t VEC_SIZE = 8;

template <typename T1, typename T2>
__forceinline__ __device__ __host__ T1 ceil_div(T1 x, T2 y) {
  return (x + y - 1) / y;
}

namespace rms_norm_rocm {

__forceinline__ __device__ float rsqrt(float x) {
  return __frsqrt_rn(x);
}

__forceinline__ __device__ float shfl_xor_sync(float x, int lane_mask) {
  return __shfl_xor(x, lane_mask, WARP_SIZE);
}

struct vec_t_float_8 {
  float d[8];
  __device__ __forceinline__ float& operator[](size_t i) { return d[i]; }
  __device__ __forceinline__ void fill(float val) {
    for (size_t i = 0; i < 8; ++i) d[i] = val;
  }
  __device__ __forceinline__ void load(const float* ptr) {
    for (size_t i = 0; i < 8; ++i) d[i] = ptr[i];
  }
  __device__ __forceinline__ void store(float* ptr) const {
    for (size_t i = 0; i < 8; ++i) ptr[i] = d[i];
  }
};

// Unified vec for __half and __hip_bfloat16 (8 elements = 16 bytes)
template <typename T>
struct vec_t_8 {
  uint4 data;
  __device__ __forceinline__ T& operator[](uint32_t i) {
    return reinterpret_cast<T*>(&data)[i];
  }
  __device__ __forceinline__ void load(const T* ptr) {
    data = *reinterpret_cast<const uint4*>(ptr);
  }
  __device__ __forceinline__ void store(T* ptr) const {
    *reinterpret_cast<uint4*>(ptr) = data;
  }
  __device__ __forceinline__ void fill(T val) {
    for (uint32_t i = 0; i < VEC_SIZE; ++i) (*this)[i] = val;
  }
  __device__ __forceinline__ void fill_zero() {
    data = uint4{0, 0, 0, 0};
  }
};

// Standalone RMSNorm: output = (input / rms(input)) * weight (Llama-style)
template <typename T>
__global__ void rmsnorm_kernel(
    T* __restrict__ output,
    const T* __restrict__ input,
    const T* __restrict__ weight,
    uint32_t hidden_size,
    float eps) {
  const uint32_t d = hidden_size;
  const uint32_t stride = hidden_size;
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * WARP_SIZE;
  const uint32_t num_threads = num_warps * WARP_SIZE;
  const uint32_t rounds = ceil_div(d, VEC_SIZE * num_threads);
  extern __shared__ float smem[];

  float sum_sq = 0.f;
  for (uint32_t i = 0; i < rounds; i++) {
    vec_t_8<T> input_vec;
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride + i * num_threads * VEC_SIZE +
                     thread_id * VEC_SIZE);
    } else {
      input_vec.fill_zero();
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      float v = castToFloat(input_vec[j]);
      sum_sq += v * v;
    }
  }
#pragma unroll
  for (uint32_t offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    sum_sq += rms_norm_rocm::shfl_xor_sync(sum_sq, offset);
  }
  smem[ty] = sum_sq;
  __syncthreads();
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      sum_sq += rms_norm_rocm::shfl_xor_sync(sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();
  float rms_rcp = rms_norm_rocm::rsqrt(smem[0] / float(d) + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t_8<T> input_vec, weight_vec, output_vec;
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride + i * num_threads * VEC_SIZE +
                     thread_id * VEC_SIZE);
      weight_vec.load(weight + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    } else {
      input_vec.fill_zero();
      weight_vec.fill_zero();
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      output_vec[j] = castFromFloat<T>(
          castToFloat(input_vec[j]) * rms_rcp * castToFloat(weight_vec[j]));
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      output_vec.store(output + bx * stride + i * num_threads * VEC_SIZE +
                        thread_id * VEC_SIZE);
    }
  }
}

// Standalone Gemma RMSNorm: output = (input / rms(input)) * (1 + weight)
template <typename T>
__global__ void gemma_rmsnorm_kernel(
    T* __restrict__ output,
    const T* __restrict__ input,
    const T* __restrict__ weight,
    uint32_t hidden_size,
    float eps) {
  const uint32_t d = hidden_size;
  const uint32_t stride = hidden_size;
  const float weight_bias = 1.f;
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * WARP_SIZE;
  const uint32_t num_threads = num_warps * WARP_SIZE;
  const uint32_t rounds = ceil_div(d, VEC_SIZE * num_threads);
  extern __shared__ float smem[];

  float sum_sq = 0.f;
  for (uint32_t i = 0; i < rounds; i++) {
    vec_t_8<T> input_vec;
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride + i * num_threads * VEC_SIZE +
                     thread_id * VEC_SIZE);
    } else {
      input_vec.fill_zero();
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      float v = castToFloat(input_vec[j]);
      sum_sq += v * v;
    }
  }
#pragma unroll
  for (uint32_t offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    sum_sq += rms_norm_rocm::shfl_xor_sync(sum_sq, offset);
  }
  smem[ty] = sum_sq;
  __syncthreads();
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      sum_sq += rms_norm_rocm::shfl_xor_sync(sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();
  float rms_rcp = rms_norm_rocm::rsqrt(smem[0] / float(d) + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t_8<T> input_vec, weight_vec, output_vec;
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride + i * num_threads * VEC_SIZE +
                     thread_id * VEC_SIZE);
      weight_vec.load(weight + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    } else {
      input_vec.fill_zero();
      weight_vec.fill_zero();
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      output_vec[j] = castFromFloat<T>(
          castToFloat(input_vec[j]) * rms_rcp *
          (weight_bias + castToFloat(weight_vec[j])));
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      output_vec.store(output + bx * stride + i * num_threads * VEC_SIZE +
                        thread_id * VEC_SIZE);
    }
  }
}

void rmsnorm_launch_config(
    uint32_t batch_size,
    uint32_t hidden_size,
    dim3& grid,
    dim3& block,
    size_t& smem) {
  TORCH_CHECK(
      hidden_size % VEC_SIZE == 0,
      "rmsnorm_rocm: hidden_size must be divisible by ",
      VEC_SIZE);
  uint32_t block_size = std::min(1024u, hidden_size / VEC_SIZE);
  uint32_t num_warps = ceil_div(block_size, WARP_SIZE);
  grid = dim3(batch_size, 1, 1);
  block = dim3(WARP_SIZE, num_warps, 1);
  smem = num_warps * sizeof(float);
}

// Llama-style: residual += input; input = (residual / rms(residual)) * weight
template <typename T>
__global__ void fused_add_rmsnorm_kernel(
    T* __restrict__ input,
    T* __restrict__ residual,
    const T* __restrict__ weight,
    uint32_t hidden_size,
    float eps) {
  const uint32_t d = hidden_size;
  const uint32_t stride_input = hidden_size;
  const uint32_t stride_residual = hidden_size;

  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * WARP_SIZE;
  const uint32_t num_threads = num_warps * WARP_SIZE;
  const uint32_t rounds = ceil_div(d, VEC_SIZE * num_threads);
  extern __shared__ float smem[];
  float* smem_x = smem + ceil_div(num_warps, 4u) * 4;

  float sum_sq = 0.f;

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t_8<T> input_vec, residual_vec;
    vec_t_float_8 x_vec;
    x_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE +
                     thread_id * VEC_SIZE);
      residual_vec.load(residual + bx * stride_residual +
                        i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    } else {
      input_vec.fill_zero();
      residual_vec.fill_zero();
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      float x = castToFloat(input_vec[j]) + castToFloat(residual_vec[j]);
      sum_sq += x * x;
      residual_vec[j] = castFromFloat<T>(x);
      x_vec[j] = x;
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      residual_vec.store(residual + bx * stride_residual +
                         i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      x_vec.store(smem_x + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
  }

#pragma unroll
  for (uint32_t offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    sum_sq += rms_norm_rocm::shfl_xor_sync(sum_sq, offset);
  }

  smem[ty] = sum_sq;
  __syncthreads();
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      sum_sq += rms_norm_rocm::shfl_xor_sync(sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();

  float rms_rcp = rms_norm_rocm::rsqrt(smem[0] / float(d) + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t_8<T> input_vec, weight_vec;
    vec_t_float_8 x_vec;
    x_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      weight_vec.load(weight + i * num_threads * VEC_SIZE +
                     thread_id * VEC_SIZE);
      x_vec.load(smem_x + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    } else {
      weight_vec.fill_zero();
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      input_vec[j] = castFromFloat<T>(
          x_vec[j] * rms_rcp * castToFloat(weight_vec[j]));
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.store(input + bx * stride_input + i * num_threads * VEC_SIZE +
                      thread_id * VEC_SIZE);
    }
  }
}


// Gemma-style: residual += input; input = (residual / rms(residual)) * (1 + weight)
template <typename T>
__global__ void gemma_fused_add_rmsnorm_kernel(
    T* __restrict__ input,
    T* __restrict__ residual,
    const T* __restrict__ weight,
    uint32_t hidden_size,
    float eps) {
  const uint32_t d = hidden_size;
  const uint32_t stride_input = hidden_size;
  const uint32_t stride_residual = hidden_size;
  const float weight_bias = 1.f;

  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * WARP_SIZE;
  const uint32_t num_threads = num_warps * WARP_SIZE;
  const uint32_t rounds = ceil_div(d, VEC_SIZE * num_threads);
  extern __shared__ float smem[];
  float* smem_x = smem + ceil_div(num_warps, 4u) * 4;

  float sum_sq = 0.f;

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t_8<T> input_vec, residual_vec;
    vec_t_float_8 x_vec;
    x_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE +
                     thread_id * VEC_SIZE);
      residual_vec.load(residual + bx * stride_residual +
                        i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    } else {
      input_vec.fill_zero();
      residual_vec.fill_zero();
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      float x = castToFloat(input_vec[j]) + castToFloat(residual_vec[j]);
      sum_sq += x * x;
      residual_vec[j] = castFromFloat<T>(x);
      x_vec[j] = x;
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      residual_vec.store(residual + bx * stride_residual +
                         i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      x_vec.store(smem_x + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
  }

#pragma unroll
  for (uint32_t offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    sum_sq += rms_norm_rocm::shfl_xor_sync(sum_sq, offset);
  }

  smem[ty] = sum_sq;
  __syncthreads();
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      sum_sq += rms_norm_rocm::shfl_xor_sync(sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();

  float rms_rcp = rms_norm_rocm::rsqrt(smem[0] / float(d) + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t_8<T> input_vec, weight_vec;
    vec_t_float_8 x_vec;
    x_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      weight_vec.load(weight + i * num_threads * VEC_SIZE +
                     thread_id * VEC_SIZE);
      x_vec.load(smem_x + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    } else {
      weight_vec.fill_zero();
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      input_vec[j] = castFromFloat<T>(
          x_vec[j] * rms_rcp *
          (weight_bias + castToFloat(weight_vec[j])));
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.store(input + bx * stride_input + i * num_threads * VEC_SIZE +
                      thread_id * VEC_SIZE);
    }
  }
}

void fused_add_rmsnorm_launch_config(
    uint32_t batch_size,
    uint32_t hidden_size,
    dim3& grid,
    dim3& block,
    size_t& smem) {
  TORCH_CHECK(
      hidden_size % VEC_SIZE == 0,
      "fused_add_rmsnorm_rocm: hidden_size must be divisible by ",
      VEC_SIZE);
  uint32_t block_size = std::min(1024u, hidden_size / VEC_SIZE);
  uint32_t num_warps = ceil_div(block_size, WARP_SIZE);
  uint32_t padded_warps = ceil_div(num_warps, 4u) * 4u;
  grid = dim3(batch_size, 1, 1);
  block = dim3(WARP_SIZE, num_warps, 1);
  smem = (padded_warps + hidden_size) * sizeof(float);
}

}  // namespace rms_norm_rocm

}  // namespace

void sgl_fused_add_rmsnorm(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight,
    double eps,
    bool enable_pdl) {
  TORCH_CHECK(input.is_contiguous() && residual.is_contiguous() &&
              weight.is_contiguous());
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 1);
  TORCH_CHECK(input.size(1) == weight.size(0));
  TORCH_CHECK(residual.sizes() == input.sizes() &&
              residual.dtype() == input.dtype());

  int64_t batch_size = input.size(0);
  int64_t hidden_size = input.size(1);
  float eps_f = static_cast<float>(eps);
  dim3 grid, block;
  size_t smem;
  rms_norm_rocm::fused_add_rmsnorm_launch_config(
      static_cast<uint32_t>(batch_size),
      static_cast<uint32_t>(hidden_size),
      grid,
      block,
      smem);

  hipStream_t stream = at::hip::getCurrentHIPStream();

  if (input.scalar_type() == at::kHalf) {
    hipLaunchKernelGGL(
        rms_norm_rocm::fused_add_rmsnorm_kernel<__half>,
        grid,
        block,
        smem,
        stream,
        reinterpret_cast<__half*>(input.data_ptr()),
        reinterpret_cast<__half*>(residual.data_ptr()),
        reinterpret_cast<const __half*>(weight.data_ptr()),
        static_cast<uint32_t>(hidden_size),
        eps_f);
  } else if (input.scalar_type() == at::kBFloat16) {
    hipLaunchKernelGGL(
        rms_norm_rocm::fused_add_rmsnorm_kernel<__hip_bfloat16>,
        grid,
        block,
        smem,
        stream,
        reinterpret_cast<__hip_bfloat16*>(input.data_ptr()),
        reinterpret_cast<__hip_bfloat16*>(residual.data_ptr()),
        reinterpret_cast<const __hip_bfloat16*>(weight.data_ptr()),
        static_cast<uint32_t>(hidden_size),
        eps_f);
  } else {
    TORCH_CHECK(
        false,
        "sgl_fused_add_rmsnorm (ROCm) only supports fp16 and bf16");
  }
}

void gemma_fused_add_rmsnorm(
    at::Tensor& input,
    at::Tensor& residual,
    at::Tensor& weight,
    double eps,
    bool enable_pdl) {
  TORCH_CHECK(input.is_contiguous() && residual.is_contiguous() &&
              weight.is_contiguous());
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 1);
  TORCH_CHECK(input.size(1) == weight.size(0));
  TORCH_CHECK(residual.sizes() == input.sizes() &&
              residual.dtype() == input.dtype());

  int64_t batch_size = input.size(0);
  int64_t hidden_size = input.size(1);
  float eps_f = static_cast<float>(eps);
  dim3 grid, block;
  size_t smem;
  rms_norm_rocm::fused_add_rmsnorm_launch_config(
      static_cast<uint32_t>(batch_size),
      static_cast<uint32_t>(hidden_size),
      grid,
      block,
      smem);

  hipStream_t stream = at::hip::getCurrentHIPStream();

  if (input.scalar_type() == at::kHalf) {
    hipLaunchKernelGGL(
        rms_norm_rocm::gemma_fused_add_rmsnorm_kernel<__half>,
        grid,
        block,
        smem,
        stream,
        reinterpret_cast<__half*>(input.data_ptr()),
        reinterpret_cast<__half*>(residual.data_ptr()),
        reinterpret_cast<const __half*>(weight.data_ptr()),
        static_cast<uint32_t>(hidden_size),
        eps_f);
  } else if (input.scalar_type() == at::kBFloat16) {
    hipLaunchKernelGGL(
        rms_norm_rocm::gemma_fused_add_rmsnorm_kernel<__hip_bfloat16>,
        grid,
        block,
        smem,
        stream,
        reinterpret_cast<__hip_bfloat16*>(input.data_ptr()),
        reinterpret_cast<__hip_bfloat16*>(residual.data_ptr()),
        reinterpret_cast<const __hip_bfloat16*>(weight.data_ptr()),
        static_cast<uint32_t>(hidden_size),
        eps_f);
  } else {
    TORCH_CHECK(
        false,
        "gemma_fused_add_rmsnorm (ROCm) only supports fp16 and bf16");
  }
}

void rmsnorm(
    at::Tensor& output,
    at::Tensor& input,
    at::Tensor& weight,
    double eps,
    bool enable_pdl) {
  (void)enable_pdl;
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous());
  TORCH_CHECK(output.is_contiguous() || output.sizes() == input.sizes());
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 1);
  TORCH_CHECK(input.size(1) == weight.size(0));
  TORCH_CHECK(output.sizes() == input.sizes() && output.dtype() == input.dtype());

  int64_t batch_size = input.size(0);
  int64_t hidden_size = input.size(1);
  float eps_f = static_cast<float>(eps);
  dim3 grid, block;
  size_t smem;
  rms_norm_rocm::rmsnorm_launch_config(
      static_cast<uint32_t>(batch_size),
      static_cast<uint32_t>(hidden_size),
      grid,
      block,
      smem);

  hipStream_t stream = at::hip::getCurrentHIPStream();

  if (input.scalar_type() == at::kHalf) {
    hipLaunchKernelGGL(
        rms_norm_rocm::rmsnorm_kernel<__half>,
        grid,
        block,
        smem,
        stream,
        reinterpret_cast<__half*>(output.data_ptr()),
        reinterpret_cast<const __half*>(input.data_ptr()),
        reinterpret_cast<const __half*>(weight.data_ptr()),
        static_cast<uint32_t>(hidden_size),
        eps_f);
  } else if (input.scalar_type() == at::kBFloat16) {
    hipLaunchKernelGGL(
        rms_norm_rocm::rmsnorm_kernel<__hip_bfloat16>,
        grid,
        block,
        smem,
        stream,
        reinterpret_cast<__hip_bfloat16*>(output.data_ptr()),
        reinterpret_cast<const __hip_bfloat16*>(input.data_ptr()),
        reinterpret_cast<const __hip_bfloat16*>(weight.data_ptr()),
        static_cast<uint32_t>(hidden_size),
        eps_f);
  } else {
    TORCH_CHECK(false, "rmsnorm (ROCm) only supports fp16 and bf16");
  }
}

void gemma_rmsnorm(
    at::Tensor& output,
    at::Tensor& input,
    at::Tensor& weight,
    double eps,
    bool enable_pdl) {
  (void)enable_pdl;
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous());
  TORCH_CHECK(output.is_contiguous() || output.sizes() == input.sizes());
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 1);
  TORCH_CHECK(input.size(1) == weight.size(0));
  TORCH_CHECK(output.sizes() == input.sizes() && output.dtype() == input.dtype());

  int64_t batch_size = input.size(0);
  int64_t hidden_size = input.size(1);
  float eps_f = static_cast<float>(eps);
  dim3 grid, block;
  size_t smem;
  rms_norm_rocm::rmsnorm_launch_config(
      static_cast<uint32_t>(batch_size),
      static_cast<uint32_t>(hidden_size),
      grid,
      block,
      smem);

  hipStream_t stream = at::hip::getCurrentHIPStream();

  if (input.scalar_type() == at::kHalf) {
    hipLaunchKernelGGL(
        rms_norm_rocm::gemma_rmsnorm_kernel<__half>,
        grid,
        block,
        smem,
        stream,
        reinterpret_cast<__half*>(output.data_ptr()),
        reinterpret_cast<const __half*>(input.data_ptr()),
        reinterpret_cast<const __half*>(weight.data_ptr()),
        static_cast<uint32_t>(hidden_size),
        eps_f);
  } else if (input.scalar_type() == at::kBFloat16) {
    hipLaunchKernelGGL(
        rms_norm_rocm::gemma_rmsnorm_kernel<__hip_bfloat16>,
        grid,
        block,
        smem,
        stream,
        reinterpret_cast<__hip_bfloat16*>(output.data_ptr()),
        reinterpret_cast<const __hip_bfloat16*>(input.data_ptr()),
        reinterpret_cast<const __hip_bfloat16*>(weight.data_ptr()),
        static_cast<uint32_t>(hidden_size),
        eps_f);
  } else {
    TORCH_CHECK(false, "gemma_rmsnorm (ROCm) only supports fp16 and bf16");
  }
}
