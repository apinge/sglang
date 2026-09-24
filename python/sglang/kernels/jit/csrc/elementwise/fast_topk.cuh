// Radix-select fast top-k, adapted from sgl-kernel's AOT topk.cu (itself
// adapted from tilelang's topk_selector). Ported to the JIT layer so that
// kTopK = 512 support ships with the sglang python package instead of
// requiring an sgl-kernel wheel release.
//
// Semantics match the AOT fast_topk_v2 op: for each row b, select the
// kTopK largest scores in [row_starts[b], row_starts[b] + lengths[b]) and
// write their indices relative to row_starts[b]. Output order within a row
// is unspecified (atomic collection order), matching the AOT kernel.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

namespace sglang {

namespace fast_topk_detail {

constexpr uint32_t kThreadsPerBlock = 1024;
// Stage up to 4K threshold-bin candidates per radix round in the common path.
// Concentrated rows that exceed this capacity use the exact full-row rescan.
constexpr size_t kSmemBytes = 8 * 1024 * sizeof(uint32_t);  // 32KB

struct FastTopKParams {
  const float* __restrict__ input;         // [B, input_stride]
  const int32_t* __restrict__ row_starts;  // [B]
  int32_t* __restrict__ indices;           // [B, kTopK]
  const int32_t* __restrict__ lengths;     // [B]
  int64_t input_stride;
};

SGL_DEVICE auto convert_to_uint8(float x) -> uint8_t {
  const __half h = __float2half_rn(x);
  const uint16_t bits = __half_as_ushort(h);
  const uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

SGL_DEVICE auto convert_to_uint32(float x) -> uint32_t {
  const uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

#if defined(USE_ROCM) && defined(__gfx942__)
template <int kDppCtrl, int kRowMask, int kBankMask>
SGL_DEVICE auto dpp_add(int value) -> int {
  const auto moved = __builtin_amdgcn_update_dpp(0, value, kDppCtrl, kRowMask, kBankMask, false);
  return value + moved;
}

SGL_DEVICE auto wave64_inclusive_sum(int value) -> int {
  value = dpp_add<0x111, 0xf, 0xf>(value);  // row_shr:1
  value = dpp_add<0x112, 0xf, 0xf>(value);  // row_shr:2
  value = dpp_add<0x114, 0xf, 0xe>(value);  // row_shr:4
  value = dpp_add<0x118, 0xf, 0xc>(value);  // row_shr:8
  value = dpp_add<0x142, 0xa, 0xf>(value);  // row_bcast:15
  value = dpp_add<0x143, 0xc, 0xf>(value);  // row_bcast:31
  return value;
}
#endif

// When length <= kTopK, write the indices directly.
template <int kTopK>
SGL_DEVICE void naive_topk(const float* __restrict__ score, int32_t* __restrict__ indice, int32_t length) {
  const auto tid = threadIdx.x;
  for (int i = tid; i < kTopK; i += kThreadsPerBlock) {
    indice[i] = (i < length) ? i : -1;
  }
}

// Radix-select top-k. Assumes length > kTopK (checked by the caller).
template <int kTopK>
SGL_DEVICE void radix_select_topk(const float* __restrict__ input, int* __restrict__ index, int row_start, int length) {
  int topk = kTopK;
  constexpr auto BLOCK_SIZE = kThreadsPerBlock;
  constexpr auto RADIX = 256;
  constexpr auto SMEM_INPUT_SIZE = kSmemBytes / (2 * sizeof(int));

  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ int s_counter;
  alignas(128) __shared__ int s_threshold_bin_id;
  alignas(128) __shared__ int s_above_threshold;
  alignas(128) __shared__ int s_last_remain;
  alignas(128) __shared__ int s_prefix_bins[4];
  alignas(128) __shared__ int s_num_input[2];
#if defined(USE_ROCM) && defined(__gfx942__)
  // The radix histogram has 256 entries. One native wave64 scans four bins
  // per lane instead of involving the whole CTA in every scan round.
#endif

  auto& s_histogram = s_histogram_buf[0];
  // allocate for two rounds
  extern __shared__ int s_input_idx[][SMEM_INPUT_SIZE];

  const int tx = threadIdx.x;

  const auto run_cumsum = [&](int total_input, int next_input, bool reset_counter, int prefix_round) {
#if defined(USE_ROCM) && defined(__gfx942__)
    constexpr int WAVE_SIZE = 64;
    constexpr int BINS_PER_LANE = RADIX / WAVE_SIZE;
    const auto lane = tx % WAVE_SIZE;
    if (tx < WAVE_SIZE) {
      // Reverse the lane-to-bin mapping so a low lane owns higher bins. A
      // wave-wide inclusive prefix then becomes the required descending sum.
      const auto base = RADIX - (tx + 1) * BINS_PER_LANE;
      int values[BINS_PER_LANE];
#pragma unroll
      for (int i = 0; i < BINS_PER_LANE; ++i) {
        values[i] = s_histogram[base + i];
      }
#pragma unroll
      for (int i = BINS_PER_LANE - 2; i >= 0; --i) {
        values[i] += values[i + 1];
      }
      const auto local_total = values[0];
      const auto inclusive = wave64_inclusive_sum(local_total);
      const auto preceding = inclusive - local_total;
#pragma unroll
      for (int i = 0; i < BINS_PER_LANE; ++i) {
        const auto suffix = preceding + values[i];
        const auto above = preceding + (i + 1 < BINS_PER_LANE ? values[i + 1] : 0);
        if (suffix > topk && above <= topk) {
          s_threshold_bin_id = base + i;
          s_above_threshold = above;
          s_num_input[next_input] = 0;
          s_last_remain = topk - above;
          if (prefix_round >= 0) s_prefix_bins[prefix_round] = base + i;
          if (reset_counter) s_counter = 0;
        }
      }
    }
    __syncthreads();
#else
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      static_assert(1 << 8 == RADIX);
      if (tx < RADIX) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = s_histogram_buf[k][tx];
        if (tx < RADIX - j) {
          value += s_histogram_buf[k][tx + j];
        }
        s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
#endif
  };

  // stage 1: 8bit coarse histogram
  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();

  for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
    const auto bin = convert_to_uint8(input[idx + row_start]);
    ::atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();

  run_cumsum(length, 0, true, -1);
#if !defined(USE_ROCM) || !defined(__gfx942__)
  if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
    s_threshold_bin_id = tx;
    s_above_threshold = s_histogram[tx + 1];
    s_num_input[0] = 0;
    s_last_remain = topk - s_above_threshold;
    s_counter = 0;
  }
  __syncthreads();
#endif

  const auto threshold_bin = s_threshold_bin_id;
  const auto coarse_threshold_bin = threshold_bin;
#if defined(USE_ROCM) && defined(__gfx942__)
  topk -= s_above_threshold;
#else
  topk -= s_histogram[threshold_bin + 1];
#endif

  if (topk == 0) {
    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto bin = static_cast<int>(convert_to_uint8(input[idx + row_start]));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      }
    }
#if !defined(USE_ROCM) || !defined(__gfx942__)
    __syncthreads();
#endif
    return;
  } else {
#if !defined(USE_ROCM) || !defined(__gfx942__)
    __syncthreads();
#endif
    if (tx < RADIX + 1) {
      s_histogram[tx] = 0;
    }
    __syncthreads();

    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto raw_input = input[idx + row_start];
      const auto bin = static_cast<int>(convert_to_uint8(raw_input));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      } else if (bin == threshold_bin) {
        const auto pos = ::atomicAdd(&s_num_input[0], 1);
        // fuse the histogram computation here
        if (pos < int(SMEM_INPUT_SIZE)) {
          s_input_idx[0][pos] = idx;
          const auto bin = convert_to_uint32(raw_input);
          const auto sub_bin = (bin >> 24) & 0xFF;
          ::atomicAdd(&s_histogram[sub_bin], 1);
        }
      }
    }
    __syncthreads();
  }

  // A concentrated row can put more candidates in the coarse threshold bin
  // than fit in the fixed shared-memory staging buffer. The old path silently
  // dropped the overflow and refined an incomplete subset. Rebuild the first
  // exact histogram from the full row; later rounds continue rescanning only
  // when this overflow flag is set.
  const bool overflow = s_num_input[0] > int(SMEM_INPUT_SIZE);
  if (overflow) {
    if (tx < RADIX + 1) s_histogram[tx] = 0;
    __syncthreads();
    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto raw_input = input[idx + row_start];
      if (static_cast<int>(convert_to_uint8(raw_input)) == coarse_threshold_bin) {
        const auto sub_bin = (convert_to_uint32(raw_input) >> 24) & 0xFF;
        ::atomicAdd(&s_histogram[sub_bin], 1);
      }
    }
    __syncthreads();
    // Exact slow path: every refinement round rescans the full row under the
    // coarse bin and exact-byte prefix selected by earlier rounds. Keeping it
    // separate prevents the overflow machinery from bloating the common path.
#pragma unroll 4
    for (int round = 0; round < 4; ++round) {
      const auto r_idx = round % 2;
      const auto num_input = s_num_input[r_idx];
      run_cumsum(num_input, r_idx ^ 1, false, round);
#if !defined(USE_ROCM) || !defined(__gfx942__)
      if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
        s_threshold_bin_id = tx;
        s_above_threshold = s_histogram[tx + 1];
        s_num_input[r_idx ^ 1] = 0;
        s_last_remain = topk - s_above_threshold;
        s_prefix_bins[round] = tx;
      }
      __syncthreads();
#endif

      const auto threshold_bin = s_threshold_bin_id;
#if defined(USE_ROCM) && defined(__gfx942__)
      topk -= s_above_threshold;
#else
      topk -= s_histogram[threshold_bin + 1];
#endif

      if (topk == 0) {
        for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
          const auto raw_input = input[idx + row_start];
          if (static_cast<int>(convert_to_uint8(raw_input)) != coarse_threshold_bin) continue;
          const auto key = convert_to_uint32(raw_input);
          bool prefix_matches = true;
#pragma unroll
          for (int prefix_round = 0; prefix_round < 4; ++prefix_round) {
            if (prefix_round >= round) break;
            const auto prefix = (key >> (24 - 8 * prefix_round)) & 0xFF;
            if (prefix != static_cast<uint32_t>(s_prefix_bins[prefix_round])) {
              prefix_matches = false;
              break;
            }
          }
          if (!prefix_matches) continue;
          const auto offset = 24 - round * 8;
          const auto bin = (key >> offset) & 0xFF;
          if (bin > threshold_bin) {
            const auto pos = ::atomicAdd(&s_counter, 1);
            index[pos] = idx;
          }
        }
#if !defined(USE_ROCM) || !defined(__gfx942__)
        __syncthreads();
#endif
        return;
      }
#if !defined(USE_ROCM) || !defined(__gfx942__)
      __syncthreads();
#endif
      if (tx < RADIX + 1) {
        s_histogram[tx] = 0;
      }
      __syncthreads();
      for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
        const auto raw_input = input[idx + row_start];
        if (static_cast<int>(convert_to_uint8(raw_input)) != coarse_threshold_bin) continue;
        const auto key = convert_to_uint32(raw_input);
        bool prefix_matches = true;
#pragma unroll
        for (int prefix_round = 0; prefix_round < 4; ++prefix_round) {
          if (prefix_round >= round) break;
          const auto prefix = (key >> (24 - 8 * prefix_round)) & 0xFF;
          if (prefix != static_cast<uint32_t>(s_prefix_bins[prefix_round])) {
            prefix_matches = false;
            break;
          }
        }
        if (!prefix_matches) continue;
        const auto offset = 24 - round * 8;
        const auto bin = (key >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        } else if (bin == threshold_bin) {
          if (round == 3) {
            const auto pos = ::atomicAdd(&s_last_remain, -1);
            if (pos > 0) {
              index[kTopK - pos] = idx;
            }
          } else {
            ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
            const auto sub_bin = (key >> (offset - 8)) & 0xFF;
            ::atomicAdd(&s_histogram[sub_bin], 1);
          }
        }
      }
#if defined(USE_ROCM) && defined(__gfx942__)
      if (round < 3) __syncthreads();
#else
      __syncthreads();
#endif
    }
    return;
  }

  // Common stage-2 path: all threshold candidates fit in shared memory.
#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    const auto r_idx = round % 2;
    const auto num_input = s_num_input[r_idx];

    run_cumsum(num_input, r_idx ^ 1, false, round);
#if !defined(USE_ROCM) || !defined(__gfx942__)
    if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
      s_threshold_bin_id = tx;
      s_above_threshold = s_histogram[tx + 1];
      s_num_input[r_idx ^ 1] = 0;
      s_last_remain = topk - s_above_threshold;
      s_prefix_bins[round] = tx;
    }
    __syncthreads();
#endif

    const auto threshold_bin = s_threshold_bin_id;
#if defined(USE_ROCM) && defined(__gfx942__)
    topk -= s_above_threshold;
#else
    topk -= s_histogram[threshold_bin + 1];
#endif

    if (topk == 0) {
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(input[idx + row_start]) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        }
      }
#if !defined(USE_ROCM) || !defined(__gfx942__)
      __syncthreads();
#endif
      break;
    }
#if !defined(USE_ROCM) || !defined(__gfx942__)
    __syncthreads();
#endif
    if (tx < RADIX + 1) {
      s_histogram[tx] = 0;
    }
    __syncthreads();
    for (int i = tx; i < num_input; i += BLOCK_SIZE) {
      const auto idx = s_input_idx[r_idx][i];
      const auto raw_input = input[idx + row_start];
      const auto offset = 24 - round * 8;
      const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      } else if (bin == threshold_bin) {
        if (round == 3) {
          const auto pos = ::atomicAdd(&s_last_remain, -1);
          if (pos > 0) {
            index[kTopK - pos] = idx;
          }
        } else {
          const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
          if (pos < int(SMEM_INPUT_SIZE)) {
            s_input_idx[r_idx ^ 1][pos] = idx;
            const auto sub_bin = (convert_to_uint32(raw_input) >> (offset - 8)) & 0xFF;
            ::atomicAdd(&s_histogram[sub_bin], 1);
          }
        }
      }
    }
#if defined(USE_ROCM) && defined(__gfx942__)
    if (round < 3) __syncthreads();
#else
    __syncthreads();
#endif
  }
}

template <int kTopK, bool kUsePDL>
__global__ __launch_bounds__(fast_topk_detail::kThreadsPerBlock) void fast_topk_kernel(
    const fast_topk_detail::FastTopKParams __grid_constant__ params) {
  using namespace fast_topk_detail;
  device::PDLWaitPrimary<kUsePDL>();

  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto row_start = params.row_starts == nullptr ? 0 : params.row_starts[bid];
  const auto length = params.lengths[bid];
  const auto indice = params.indices + bid * kTopK;
  const auto score = params.input + bid * params.input_stride;
  if (length <= kTopK) {
    naive_topk<kTopK>(score, indice, length);
  } else {
    radix_select_topk<kTopK>(score, indice, row_start, length);
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

}  // namespace fast_topk_detail

/**
 * \brief Per-row top-k selection over ragged rows of a fp32 score matrix.
 *
 * Row b selects the kTopK largest values in
 * score[b, row_starts[b] : row_starts[b] + lengths[b]) and writes their
 * indices (relative to row_starts[b]) into indices[b]. Unfilled slots are
 * -1 when lengths[b] < kTopK.
 */
template <int kTopK, bool kUsePDL>
struct FastTopKKernel {
  static constexpr auto kernel = fast_topk_detail::fast_topk_kernel<kTopK, kUsePDL>;

  static void
  run(const tvm::ffi::TensorView score,
      const tvm::ffi::Optional<tvm::ffi::TensorView> row_starts,
      const tvm::ffi::TensorView indices,
      const tvm::ffi::TensorView lengths) {
    using namespace host;
    auto B = SymbolicSize{"batch"};
    auto L = SymbolicSize{"length"};
    auto S = SymbolicSize{"input_stride"};
    auto device = SymbolicDevice{};

    TensorMatcher({B, L})  // score
        .with_strides({S, 1})
        .with_dtype<fp32_t>()
        .with_device<kDLGPU>(device)
        .verify(score);
    const int32_t* row_starts_ptr = nullptr;
    if (row_starts.has_value()) {
      TensorMatcher({B})  // row_starts
          .with_dtype<int32_t>()
          .with_device<kDLGPU>(device)
          .verify(row_starts.value());
      row_starts_ptr = static_cast<const int32_t*>(row_starts.value().data_ptr());
    }
    TensorMatcher({B, kTopK})  // indices
        .with_dtype<int32_t>()
        .with_device<kDLGPU>(device)
        .verify(indices);
    TensorMatcher({B})  // lengths
        .with_dtype<int32_t>()
        .with_device<kDLGPU>(device)
        .verify(lengths);

    const auto params = fast_topk_detail::FastTopKParams{
        .input = static_cast<const float*>(score.data_ptr()),
        .row_starts = row_starts_ptr,
        .indices = static_cast<int32_t*>(indices.data_ptr()),
        .lengths = static_cast<const int32_t*>(lengths.data_ptr()),
        .input_stride = S.unwrap(),
    };

    const auto num_rows = static_cast<uint32_t>(B.unwrap());
    LaunchKernel(num_rows, fast_topk_detail::kThreadsPerBlock, device.unwrap(), fast_topk_detail::kSmemBytes)
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
