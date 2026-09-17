# Kernel source: pyhip d3510c6 (final algorithm a716010).
# Only the final down/up launch path is retained; no test/control imports.

from dataclasses import dataclass
from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import range_constexpr

from . import up as control

K, R, LOG2E = control.K, control.R, control.LOG2E


@dataclass(frozen=True)
class DownConfig:
    block_m: int = 16
    block_k: int = 256
    waves: int = 4
    prefetch: bool = True
    interleave: bool = True
    global_split: int = 1
    prefetch_unroll: int = 1

    def validate(self):
        if self.block_m not in (16, 32):
            raise ValueError("down block_m must be 16 or 32")
        if self.waves not in (2, 4, 8):
            raise ValueError("down waves must be 2, 4, or 8")
        if self.block_k not in (64, 128, 256, 512):
            raise ValueError("down block_k must be 64, 128, 256, or 512")
        if self.global_split not in (1, 2, 4):
            raise ValueError("global_split must be 1, 2, or 4")
        if K % (self.waves * self.block_k * self.global_split):
            raise ValueError("wave K tiles must divide K")
        if self.prefetch_unroll not in (1, 2):
            raise ValueError("prefetch_unroll must be 1 or 2")
        if self.prefetch_unroll == 2 and (
            not self.prefetch
            or K // self.waves // self.block_k // self.global_split % 2 != 1
        ):
            raise ValueError("two-beat prefetch requires an odd number of K iterations")

    @property
    def name(self):
        name = f"m{self.block_m}_k{self.block_k}_w{self.waves}_pf{int(self.prefetch)}_i{int(self.interleave)}_s{self.global_split}"
        return name if self.prefetch_unroll == 1 else name + f"_u{self.prefetch_unroll}"


def default_config(rows):
    """Measured large-T candidate; leaves the original entry's defaults alone."""
    if not 17 <= rows <= 32:
        raise ValueError("experimental large down is restricted to T=17..32")
    return DownConfig(
        block_m=32, block_k=128, global_split=4, prefetch_unroll=2 if rows <= 25 else 1
    )


@cache
def down_launcher(rows, config):
    config.validate()
    bm, bk, waves = config.block_m, config.block_k, config.waves
    prefetch, interleave = config.prefetch, config.interleave
    split = config.global_split
    prefetch_unroll = config.prefetch_unroll
    dn, iterations = 16, K // waves // bk // split
    padded_rows = (rows + bm - 1) // bm * bm

    @fx.struct
    class DownShared:
        partials: fx.Array[fx.Float32, bm * dn * waves, 16]

    @flyc.kernel
    def down_wave_splitk_pipeline(X: fx.Tensor, W: fx.Tensor, A: fx.Tensor):
        tid = fx.thread_idx.x
        lane, wave = tid % 64, tid // 64
        im, jn, sk = fx.block_idx
        x = fx.rocdl.make_buffer_tensor(
            fx.make_view(fx.get_iter(X), fx.make_layout((rows, K), (K, 1))),
            max_size=False,
        )
        # Exactly the existing W_down preshuffle layout; no new weight packing.
        w_layout = fx.make_layout(
            ((16, R // 16), (8, 4, K // 32)), ((8, 16 * K), (1, 128, 512))
        )
        w = fx.rocdl.make_buffer_tensor(
            fx.make_view(fx.get_iter(W), w_layout), max_size=False
        )
        shared = fx.SharedAllocator().allocate(DownShared).peek()
        partials = shared.partials.view(
            fx.make_layout((bm, dn, waves), (dn, 1, bm * dn))
        )
        a_tile = fx.flat_divide(w, fx.make_tile(dn, bk))[None, None, jn, None]
        b_tile = fx.flat_divide(x, fx.make_tile(bm, bk))[None, None, im, None]
        c_tile = fx.select(partials[None, None, wave], [1, 0])
        mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
        tiled = fx.make_tiled_mma(
            mma,
            fx.make_layout((1, 1, 1), (0, 0, 0)),
            (None, None, fx.make_layout((4, 4, 2), (1, 8, 4))),
        )
        thr = tiled.thr_slice(lane)
        copy_ab = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        copy_c = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
        ca = fx.make_tiled_copy_A(copy_ab, tiled).get_slice(lane)
        cb = fx.make_tiled_copy_B(copy_ab, tiled).get_slice(lane)
        cc = fx.make_tiled_copy_C(copy_c, tiled).get_slice(lane)
        fa = thr.make_fragment_A(a_tile[None, None, 0])
        fb = thr.make_fragment_B(b_tile[None, None, 0])
        fc = thr.make_fragment_C(c_tile)
        ga, gb = ca.partition_S(a_tile), cb.partition_S(b_tile)
        ra, rb = ca.retile(fa), cb.retile(fb)
        fc.fill(0)
        if fx.const_expr(prefetch):
            next_a = thr.make_fragment_A(a_tile[None, None, 0])
            next_b = thr.make_fragment_B(b_tile[None, None, 0])
            next_ra, next_rb = ca.retile(next_a), cb.retile(next_b)
            if fx.const_expr(interleave):
                first = sk * (K // split // bk) + wave
            else:
                first = sk * (K // split // bk) + wave * iterations
            fx.copy(copy_ab, ga[None, None, None, first], ra)
            fx.copy(copy_ab, gb[None, None, None, first], rb)
            # Carry prefetched fragments in SSA; the final tile is drained once.
            for ki, state in range(
                fx.Index(1),
                fx.Index(iterations),
                fx.Index(prefetch_unroll),
                init=[fa.load(), fb.load(), fc.load()],
            ):
                fa.store(state[0])
                fb.store(state[1])
                fc.store(state[2])
                if fx.const_expr(interleave):
                    kt = sk * (K // split // bk) + fx.Int32(ki) * waves + wave
                else:
                    kt = sk * (K // split // bk) + wave * iterations + fx.Int32(ki)
                fx.copy(copy_ab, ga[None, None, None, kt], next_ra)
                fx.copy(copy_ab, gb[None, None, None, kt], next_rb)
                fx.gemm(mma, fc, fa, fb, fc)
                if fx.const_expr(prefetch_unroll == 2):
                    # Refill the consumed buffer; only every other tile crosses the loop backedge.
                    second = kt + (waves if interleave else 1)
                    fx.copy(copy_ab, ga[None, None, None, second], ra)
                    fx.copy(copy_ab, gb[None, None, None, second], rb)
                    fx.gemm(mma, fc, next_a, next_b, fc)
                    carried_a, carried_b = fa.load(), fb.load()
                else:
                    carried_a, carried_b = next_a.load(), next_b.load()
                result = yield [carried_a, carried_b, fc.load()]
            fa.store(result[0])
            fb.store(result[1])
            fc.store(result[2])
            fx.gemm(mma, fc, fa, fb, fc)
        else:
            for ki, state in range(
                fx.Index(0), fx.Index(iterations), fx.Index(1), init=[fc.load()]
            ):
                fc.store(state[0])
                if fx.const_expr(interleave):
                    kt = sk * (K // split // bk) + fx.Int32(ki) * waves + wave
                else:
                    kt = sk * (K // split // bk) + wave * iterations + fx.Int32(ki)
                fx.copy(copy_ab, ga[None, None, None, kt], ra)
                fx.copy(copy_ab, gb[None, None, None, kt], rb)
                fx.gemm(mma, fc, fa, fb, fc)
                result = yield [fc.load()]
            fc.store(result)
        fx.copy(copy_c, cc.retile(fc), cc.partition_D(c_tile))
        fx.gpu.barrier()
        out = fx.make_view(
            fx.get_iter(A),
            fx.make_layout((padded_rows, R, split), (R, 1, padded_rows * R)),
        )
        for i in range_constexpr((bm * dn + waves * 64 - 1) // (waves * 64)):
            index = tid + i * waves * 64
            if index < bm * dn:
                row, col = index // dn, index % dn
                total = fx.Float32(0.0)
                for s in range_constexpr(waves):
                    total = total + fx.memref_load(partials, (row, col, s))
                if fx.const_expr(split == 1):
                    z = total * 0.25
                    exponent = fx.Float32(
                        fx.rocdl.exp2(fx.T.f32, fx.arith.unwrap(-z * LOG2E))
                    )
                    inverse = fx.Float32(
                        fx.rocdl.rcp(fx.T.f32, fx.arith.unwrap(1.0 + exponent))
                    )
                    fx.memref_store(
                        z * inverse, out, (im * bm + row, jn * dn + col, sk)
                    )
                else:
                    # Nonlinearity is applied only after all global splits are summed by up.
                    fx.memref_store(total, out, (im * bm + row, jn * dn + col, sk))

    @flyc.jit
    def launch(X: fx.Tensor, W: fx.Tensor, A: fx.Tensor, stream: fx.Stream):
        down_wave_splitk_pipeline(X, W, A).launch(
            grid=(padded_rows // bm, R // dn, split),
            block=(waves * 64, 1, 1),
            stream=stream,
        )

    return launch


@cache
def pair_launcher(rows, down_config, up_config):
    down = down_launcher(rows, down_config)
    up = control.up_launcher(rows, torch.bfloat16, up_config)

    @flyc.jit
    def launch(
        X: fx.Tensor,
        WD: fx.Tensor,
        WU: fx.Tensor,
        P: fx.Tensor,
        Y: fx.Tensor,
        stream: fx.Stream,
    ):
        if fx.const_expr(rows > 0):
            down(X, WD, P, stream)
            up(X, WU, P, Y, stream)

    return launch
