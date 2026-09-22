# Kernel source: pyhip d3510c6 (final algorithm a716010).
# Only the final down/up launch path is retained; no test/control imports.

from dataclasses import dataclass
from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import range_constexpr

HC, HS, R = 4, 2560, 320
K = HC * HS
MAX_ROWS = 32
LOG2E = 1.4426950408889634


@dataclass(frozen=True)
class Config:
    split_k: int = 16
    block_m: int = 16
    down_n: int = 64
    up_n: int = 128
    block_k: int = 64
    waves: int = 4
    skip_padding: bool = True
    preshuffle: bool = True
    fast_math: bool = True
    compensate_hidden: bool = False
    down_mode: str = "partial"
    down_waves: int = 4
    down_flip: bool = True
    down_interleave: bool = True
    down_copy_bits: int = 128
    down_block_k: int = 256
    prefetch_low: bool = True
    hidden_pad: int = 0

    def validate(self):
        if self.down_mode not in ("partial", "wave_splitk"):
            raise ValueError("down_mode must be partial or wave_splitk")
        if self.split_k <= 0 or K % (self.split_k * self.block_k):
            raise ValueError("split_k * block_k must divide 10240")
        if self.block_m not in (16, 32):
            raise ValueError("block_m must be 16 or 32")
        if self.waves not in (1, 2, 4):
            raise ValueError("waves must be 1, 2, or 4")
        if R % self.down_n or K % self.up_n:
            raise ValueError("GEMM output tiles must divide their output widths")
        down_alignment = 16 if self.down_mode == "wave_splitk" else 16 * self.waves
        if self.down_n % down_alignment or self.up_n % (16 * self.waves):
            raise ValueError("N tiles must contain each wave's MFMA tile")
        if R % self.block_k or self.block_k % 16:
            raise ValueError("block_k must be a multiple of 16 dividing 320")
        if self.hidden_pad not in (0, 4, 8, 16, 32):
            raise ValueError("hidden_pad must preserve 64-bit copy alignment")
        lds_bytes = (
            self.block_m
            * (R + self.hidden_pad)
            * 2
            * (2 if self.compensate_hidden else 1)
            + self.block_m * self.up_n * 4
        )
        if lds_bytes > 65536:
            raise ValueError("up kernel exceeds gfx942's 64 KiB LDS capacity")
        if self.down_mode == "wave_splitk":
            if self.down_waves not in (2, 4, 8) or K % (
                self.down_waves * self.down_block_k
            ):
                raise ValueError(
                    "wave split-K requires 2/4/8 waves evenly dividing K tiles"
                )
            if self.block_m * self.down_n * self.down_waves * 4 > 65536:
                raise ValueError("wave split-K exceeds gfx942's LDS capacity")
            if self.down_copy_bits not in (64, 128) or self.down_block_k % 32:
                raise ValueError(
                    "wave split-K copy requires 64/128 bits and K tiles divisible by 32"
                )


@cache
def up_launcher(rows: int, dtype: torch.dtype, config: Config):
    bm, un, bk = config.block_m, config.up_n, config.block_k
    split, waves = config.split_k, config.waves
    # FlyDSL 0.3.1 hashes scalar closure values, not fields of config objects.
    skip_padding, preshuffle, fast_math = (
        config.skip_padding,
        config.preshuffle,
        config.fast_math,
    )
    compensate_hidden = config.compensate_hidden
    prefetch_low = config.prefetch_low
    hidden_stride = R + config.hidden_pad
    down_mode = config.down_mode
    padded_rows = (rows + bm - 1) // bm * bm
    threads = 64 * waves
    elem = fx.BFloat16

    @fx.struct
    class UpShared:
        hidden: fx.Array[elem, bm * hidden_stride, 16]
        hidden_low: fx.Array[elem, bm * hidden_stride if compensate_hidden else 1, 16]
        logits: fx.Array[fx.Float32, bm * un, 16]

    @flyc.kernel
    def up_gate(X: fx.Tensor, W: fx.Tensor, P: fx.Tensor, Y: fx.Tensor):
        tid = fx.thread_idx.x
        im, jn, _ = fx.block_idx
        shared = fx.SharedAllocator().allocate(UpShared).peek()
        h = shared.hidden.view(fx.make_layout((bm, R), (hidden_stride, 1)))
        c = shared.logits.view(fx.make_layout((bm, un), (un, 1)))
        p = fx.rocdl.make_buffer_tensor(P, max_size=False)
        p4 = fx.flat_divide(p, fx.make_tile(4))
        h4 = shared.hidden.view(
            fx.make_layout((4, (R // 4, bm)), (1, (4, hidden_stride)))
        )
        if fx.const_expr(compensate_hidden):
            h_low = shared.hidden_low.view(fx.make_layout((bm, R), (hidden_stride, 1)))
            h_low4 = shared.hidden_low.view(
                fx.make_layout((4, (R // 4, bm)), (1, (4, hidden_stride)))
            )
        copy_p = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
        copy_h = fx.make_copy_atom(fx.UniversalCopy64b(), elem)
        fp = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
        fh = fx.make_rmem_tensor(fx.make_layout(4, 1), elem)
        # Each up CTA reconstructs its small low-rank input from ordered partials.
        for i in range_constexpr(bm * R // (threads * 4)):
            ix = tid + i * threads
            if fx.const_expr(down_mode == "partial"):
                acc = fx.Vector.filled(4, 0.0, fx.Float32)
                if fx.const_expr(skip_padding and rows % bm != 0):
                    if im * bm + ix // (R // 4) < rows:
                        for s in range_constexpr(split):
                            offset = (s * padded_rows + im * bm) * (R // 4) + ix
                            fx.copy(copy_p, p4[None, offset], fp)
                            acc = acc + fp.load()
                else:
                    for s in range_constexpr(split):
                        offset = (s * padded_rows + im * bm) * (R // 4) + ix
                        fx.copy(copy_p, p4[None, offset], fp)
                        acc = acc + fp.load()
                z = acc * 0.25
                if fx.const_expr(fast_math):
                    values = []
                    for j in range_constexpr(4):
                        exponent = fx.Float32(
                            fx.rocdl.exp2(fx.T.f32, fx.arith.unwrap(-z[j] * LOG2E))
                        )
                        inverse = fx.Float32(
                            fx.rocdl.rcp(fx.T.f32, fx.arith.unwrap(1.0 + exponent))
                        )
                        values.append(z[j] * inverse)
                    activated = fx.Vector.from_elements(values, fx.Float32)
                else:
                    activated = z / (1.0 + (-z * LOG2E).exp2())
            else:
                fx.copy(copy_p, p4[None, im * bm * (R // 4) + ix], fp)
                activated = fp.load()
            fh.store(activated.to(elem))
            fx.copy(copy_h, fh, h4[None, ix])
            if fx.const_expr(compensate_hidden):
                low = activated - activated.to(elem).to(fx.Float32)
                fh.store(low.to(elem))
                fx.copy(copy_h, fh, h_low4[None, ix])
        fx.gpu.barrier()

        if fx.const_expr(preshuffle):
            w_layout = fx.make_layout(
                ((16, K // 16), (8, 4, R // 32)), ((8, 16 * R), (1, 128, 512))
            )
        else:
            w_layout = fx.make_layout((K, R), (R, 1))
        w = fx.rocdl.make_buffer_tensor(
            fx.make_view(fx.get_iter(W), w_layout), max_size=False
        )
        a_tile = fx.flat_divide(h, fx.make_tile(bm, bk))[None, None, 0, None]
        if fx.const_expr(compensate_hidden):
            a_low_tile = fx.flat_divide(h_low, fx.make_tile(bm, bk))[
                None, None, 0, None
            ]
        b_tile = fx.flat_divide(w, fx.make_tile(un, bk))[None, None, jn, None]
        mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, elem))
        tiled = fx.make_tiled_mma(mma, fx.make_layout((1, waves, 1), (0, 1, 0)))
        thr = tiled.thr_slice(tid)
        copy_a = fx.make_copy_atom(fx.UniversalCopy64b(), elem)
        copy_b = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), elem)
        copy_c = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
        ca = fx.make_tiled_copy_A(copy_a, tiled).get_slice(tid)
        cb = fx.make_tiled_copy_B(copy_b, tiled).get_slice(tid)
        cc = fx.make_tiled_copy_C(copy_c, tiled).get_slice(tid)
        fa = thr.make_fragment_A(a_tile[None, None, 0])
        fb = thr.make_fragment_B(b_tile[None, None, 0])
        fc = thr.make_fragment_C(c)
        if fx.const_expr(compensate_hidden and prefetch_low):
            fa_low = thr.make_fragment_A(a_tile[None, None, 0])
        fc.fill(0)
        ga, gb = ca.partition_S(a_tile), cb.partition_S(b_tile)
        if fx.const_expr(compensate_hidden):
            ga_low = ca.partition_S(a_low_tile)
        ra, rb = ca.retile(fa), cb.retile(fb)
        if fx.const_expr(compensate_hidden and prefetch_low):
            ra_low = ca.retile(fa_low)
        for ki in range_constexpr(R // bk):
            fx.copy(copy_a, ga[None, None, None, ki], ra)
            if fx.const_expr(compensate_hidden and prefetch_low):
                fx.copy(copy_a, ga_low[None, None, None, ki], ra_low)
            fx.copy(copy_b, gb[None, None, None, ki], rb)
            fx.gemm(mma, fc, fa, fb, fc)
            if fx.const_expr(compensate_hidden):
                if fx.const_expr(prefetch_low):
                    fx.gemm(mma, fc, fa_low, fb, fc)
                else:
                    fx.copy(copy_a, ga_low[None, None, None, ki], ra)
                    fx.gemm(mma, fc, fa, fb, fc)
        fx.copy(copy_c, cc.retile(fc), cc.partition_D(c))
        fx.gpu.barrier()

        x = fx.make_view(fx.get_iter(X), fx.make_layout((rows, K), (K, 1)))
        y = fx.make_view(fx.get_iter(Y), fx.make_layout((rows, HS), (HS, 1)))
        for i in range_constexpr(bm * (un // HC) // threads):
            index = tid + i * threads
            row_local = index // (un // HC)
            col_local = index % (un // HC)
            row = im * bm + row_local
            col = jn * (un // HC) + col_local
            if row < rows:
                total = fx.Float32(0.0)
                for g in range_constexpr(HC):
                    logit = fx.memref_load(c, (row_local, col_local * HC + g))
                    if fx.const_expr(fast_math):
                        exponent = fx.Float32(
                            fx.rocdl.exp2(fx.T.f32, fx.arith.unwrap(-logit * LOG2E))
                        )
                        gate = fx.Float32(
                            fx.rocdl.rcp(fx.T.f32, fx.arith.unwrap(1.0 + exponent))
                        )
                    else:
                        gate = 1.0 / (1.0 + (-logit * LOG2E).exp2())
                    value = fx.memref_load(x, (row, g * HS + col)).to(fx.Float32)
                    total = total + gate * value
                fx.memref_store((total * 0.25).to(elem), y, (row, col))

    @flyc.jit
    def launch_up(
        X: fx.Tensor, W: fx.Tensor, P: fx.Tensor, Y: fx.Tensor, stream: fx.Stream
    ):
        up_gate(X, W, P, Y).launch(
            grid=(padded_rows // bm, K // un, 1), block=(threads, 1, 1), stream=stream
        )

    return launch_up


def default_config(rows):
    """Keep the small-row path; reuse fused down+SiLU for the 17..32 tile bucket."""
    if rows > 16:
        return Config(
            down_mode="wave_splitk",
            down_n=16,
            down_block_k=512,
            down_waves=4,
            compensate_hidden=True,
        )
    return Config(block_m=16, compensate_hidden=True)
