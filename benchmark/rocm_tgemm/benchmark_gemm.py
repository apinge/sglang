"""Reproduce the recorded 16-shape AITER comparison; use a free, reserved gfx942 GPU."""

import argparse
import csv
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("base", "accepted"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output already exists; preserve the previous measurement")
    os.environ["AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE"] = str(
        (args.configs_dir / ("ck_" + args.phase + ".csv")).resolve()
    )
    os.environ["AITER_CONFIG_GEMM_BF16"] = str(
        (args.configs_dir / ("tgemm_" + args.phase + ".csv")).resolve()
    )
    for name in ("AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE", "AITER_CONFIG_GEMM_BF16"):
        if not Path(os.environ[name]).is_file():
            raise FileNotFoundError(
                "Prepare the complete AITER configuration tables first"
            )

    import aiter
    import torch
    from aiter.jit.core import AITER_CONFIGS
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx
    from aiter.ops.shuffle import shuffle_weight
    from aiter.test_common import run_perftest
    from aiter.tuned_gemm import get_GEMM_A16W16_config, tgemm

    ROOT = Path(__file__).resolve().parent
    if get_gfx() != "gfx942" or get_cu_num() != 80:
        raise ValueError("Recorded configurations require gfx942 / 80 CUs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.set_grad_enabled(False)
    records = []

    def check(out, ref):
        assert out.shape == ref.shape and out.dtype == torch.bfloat16
        assert torch.isfinite(out).all()
        rel = float(
            (out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-8)
        )
        assert rel < 0.008, rel
        return rel

    shapes = []
    for row in csv.DictReader((ROOT / "configs/ck_decode_shapes.csv").open()):
        shapes.append(
            (
                "ck_decode",
                int(row["M"]),
                int(row["N"]),
                int(row["K"]),
                True,
                False,
                True,
            )
        )
    for row in csv.DictReader((ROOT / "configs/tgemm_shapes.csv").open()):
        shapes.append(
            (
                "tgemm",
                int(row["M"]),
                int(row["N"]),
                int(row["K"]),
                row["scaleAB"] == "True",
                row["bias"] == "True",
                row["bpreshuffle"] == "True",
            )
        )

    for family, m, n, k, fp8, has_bias, packed in shapes:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(n, device="cuda", dtype=torch.bfloat16) if has_bias else None
        if fp8:
            x, sx = aiter.per_token_quant_hip(x, quant_dtype=aiter.dtypes.fp8)
            w, sw = aiter.per_token_quant_hip(w, quant_dtype=aiter.dtypes.fp8)
            canonical_w = w.float() * sw
            ref = (x.float() * sx) @ canonical_w.t()
        else:
            sx = sw = None
            canonical_w = w.float()
            ref = x.float() @ canonical_w.t()
        if bias is not None:
            ref += bias.float()
        wp = shuffle_weight(w, (16, 16)) if packed else w
        if packed:
            wp.is_shuffled = True

        def fn(inp, weight, scale_a, scale_b, bias_tensor):
            if family == "ck_decode":
                return aiter.gemm_a8w8_bpreshuffle(
                    inp, weight, scale_a, scale_b, None, torch.bfloat16
                )
            return tgemm.mm(
                inp,
                weight,
                bias_tensor,
                otype=torch.bfloat16,
                scale_a=scale_a,
                scale_b=scale_b.t() if scale_b is not None else None,
            )

        kernel_args = (x, wp, sx, sw, bias)
        out = fn(*kernel_args)
        relative_l2 = check(out, ref)
        samples = []
        for _ in range(5):
            out, us = run_perftest(
                fn, *kernel_args, num_warmup=10, num_iters=101, num_rotate_args=16
            )
            check(out, ref)
            samples.append(float(us))
        graph_samples = []
        for _ in range(3):
            out, us = run_perftest(
                fn,
                *kernel_args,
                num_warmup=10,
                num_iters=101,
                num_rotate_args=16,
                testGraph=True,
            )
            check(out, ref)
            graph_samples.append(float(us))

        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            graph_out = fn(*kernel_args)
        graph_errors = []
        for seed in [43, 44]:
            torch.manual_seed(seed)
            changed = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            if fp8:
                q, qs = aiter.per_token_quant_hip(changed, quant_dtype=aiter.dtypes.fp8)
                x.copy_(q)
                sx.copy_(qs)
                reference = (x.float() * sx) @ canonical_w.t()
            else:
                x.copy_(changed)
                reference = x.float() @ canonical_w.t()
            if bias is not None:
                reference += bias.float()
            graph.replay()
            torch.cuda.synchronize()
            graph_errors.append(check(graph_out, reference))
        if family == "tgemm":
            config = get_GEMM_A16W16_config(
                m, n, k, has_bias, str(x.dtype), str(torch.bfloat16), fp8, packed
            )
        else:
            from aiter.ops.gemm_op_a8w8 import get_GEMM_config_with_quant_type

            config = get_GEMM_config_with_quant_type(
                m,
                n,
                k,
                str(w.dtype),
                AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE,
            )
        record = dict(
            family=family,
            M=m,
            N=n,
            K=k,
            dtype=str(x.dtype),
            bias=has_bias,
            packed=packed,
            us=samples,
            graph_us=graph_samples,
            relative_l2=relative_l2,
            changed_input_graph_relative_l2=graph_errors,
            config=config,
        )
        records.append(record)
        print(json.dumps(record, default=str), flush=True)
        args.output.write_text(json.dumps(records, indent=2, default=str))
        del graph, graph_out, out, ref, wp, w, x, canonical_w
        torch.cuda.empty_cache()

    print(
        "PASS",
        len(records),
        "GEMMs;",
        len(records) * 2,
        "changed-input graph replays",
        flush=True,
    )


if __name__ == "__main__":
    main()
