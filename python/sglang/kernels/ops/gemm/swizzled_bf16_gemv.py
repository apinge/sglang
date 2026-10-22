import torch
import triton
import triton.language as tl


@triton.jit
def _swizzled_gemv(
    X,
    W,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    STRIDE_Y_M: tl.constexpr,
    STRIDE_Y_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    acc = tl.full((BN, BK), 0, tl.float32)
    for kb in range(tl.cdiv(K, BK)):
        kk = kb * BK + ks
        x = tl.load(X + row * K + kk, kk < K, other=0).to(tl.float32)
        offset = (
            (cols[:, None] // 16) * (K * 16)
            + (kk[None, :] // 32) * 512
            + ((kk[None, :] % 32) // 8) * 128
            + (cols[:, None] % 16) * 8
            + kk[None, :] % 8
        )
        w = tl.load(W + offset, (cols[:, None] < N) & (kk[None, :] < K), other=0).to(
            tl.float32
        )
        acc += x[None, :] * w
    out = tl.sum(acc, axis=1)
    tl.store(Y + row * STRIDE_Y_M + cols * STRIDE_Y_N, out, cols < N)


def can_use_swizzled_gemv(x, weight, bias=None):
    return (
        bias is None
        and x.is_cuda
        and x.ndim == 2
        and x.shape[1] == 4096
        and x.dtype == weight.dtype == torch.bfloat16
        and x.device == weight.device
        and x.is_contiguous()
        and not x.requires_grad
        and not weight.requires_grad
        and weight.shape == (4096, 16)
        and weight.stride() == (1, 4096)
        and 0 < x.shape[0] <= 32
    )


def can_write_swizzled_gemv_output(x, weight, output):
    if not (
        output.ndim == 2
        and output.shape == (x.shape[0], weight.shape[1])
        and output.dtype == torch.bfloat16
        and output.device == x.device
        and output.stride(1) > 0
        and output.stride(0) >= output.shape[1] * output.stride(1)
    ):
        return False

    # DLPack views can have distinct Storage objects whose memory still
    # overlaps. Conservatively reject overlapping backing-storage ranges,
    # including disjoint views of the same allocation as before.
    output_storage = output.untyped_storage()
    output_start = output_storage.data_ptr()
    output_end = output_start + output_storage.nbytes()
    for source in (x, weight):
        source_storage = source.untyped_storage()
        source_start = source_storage.data_ptr()
        if output_start == source_start or (
            output_start < source_start + source_storage.nbytes()
            and source_start < output_end
        ):
            return False
    return True


def swizzled_gemv(x, w, bn=4, bk=1024, num_warps=4, output=None):
    m, k = x.shape
    n = w.shape[1]
    if output is None:
        output = torch.empty(m, n, device=x.device, dtype=torch.bfloat16)
    elif not can_write_swizzled_gemv_output(x, w, output):
        raise ValueError("invalid or overlapping swizzled GEMV output buffer")
    _swizzled_gemv[(m, triton.cdiv(n, bn))](
        x,
        w,
        output,
        m,
        n,
        k,
        bn,
        bk,
        output.stride(0),
        output.stride(1),
        num_warps=num_warps,
    )
    return output
