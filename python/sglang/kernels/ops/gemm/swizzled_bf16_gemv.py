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
    tl.store(Y + row * N + cols, out, cols < N)


def swizzled_gemv(x, w, bn=4, bk=1024, num_warps=4):
    m, k = x.shape
    n = w.shape[1]
    out = torch.empty(m, n, device=x.device, dtype=torch.bfloat16)
    _swizzled_gemv[(m, triton.cdiv(n, bn))](
        x, w, out, m, n, k, bn, bk, num_warps=num_warps
    )
    return out
