/* Copyright 2025 SGLang Team. All Rights Reserved.
 * HIP Gemma RMSNorm BF16 kernels (from pyhip/archive/norm).
 * Only bf16; data is assumed to be bf16.
 */
#ifndef SGL_KERNEL_GEMMA_NORM_HIP_CUH_
#define SGL_KERNEL_GEMMA_NORM_HIP_CUH_

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace gemma_norm_hip {

template <typename T1, typename T2>
__forceinline__ __device__ __host__ T1 ceil_div(T1 x, T2 y) {
  return (x + y - 1) / y;
}

__forceinline__ __device__ float rsqrt(float x) {
  return __frsqrt_rn(x);
}

__forceinline__ __device__ float shfl_xor_sync(float x, int lane_mask) {
  return __shfl_xor(x, lane_mask, 32);
}

struct vec_bf16_8 {
  uint4 data;
  __device__ __forceinline__ __bf16& operator[](uint32_t i) {
    return ((__bf16*)&data)[i];
  }
  __device__ __forceinline__ void load(const __bf16* ptr) {
    data = *reinterpret_cast<const uint4*>(ptr);
  }
  __device__ __forceinline__ void store(__bf16* ptr) const {
    *reinterpret_cast<uint4*>(ptr) = data;
  }
};

constexpr uint32_t VEC_SIZE = 8;

}  // namespace gemma_norm_hip

#endif  // SGL_KERNEL_GEMMA_NORM_HIP_CUH_
