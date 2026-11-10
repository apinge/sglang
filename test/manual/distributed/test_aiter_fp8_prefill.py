"""Run with torchrun on gfx942: real SGLang prefill QR / decode CA routing."""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

import aiter
import torch
import torch.distributed as dist
from aiter.jit.core import get_module

from sglang.srt.distributed.parallel_state import init_model_parallel_group
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context


def phase(mode):
    return forward_context(ForwardContext(None), forward_mode=mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    rank = int(os.environ["RANK"])
    size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo")
    group = init_model_parallel_group(
        [list(range(size))],
        rank,
        "nccl",
        use_custom_allreduce=True,
        group_name="fp8_test",
    )
    manifest = json.loads(args.manifest.read_text())
    module = get_module("module_quick_all_reduce")
    binary = Path(module.__file__).resolve()
    assert str(binary) == manifest["binary"]
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == manifest["binary_sha256"]
    for name, digest in manifest["sources"].items():
        assert (
            hashlib.sha256(
                (Path(aiter.__file__).parents[1] / name).read_bytes()
            ).hexdigest()
            == digest
        )
    assert (
        group.ca_comm.__class__.__module__
        == "aiter.dist.device_communicators.custom_all_reduce"
    )
    calls = []
    original = group._all_reduce_out_place

    def record(inp, method):
        calls.append(method)
        return original(inp, method)

    group._all_reduce_out_place = record
    x = torch.full((1, 2560), rank + 1, device="cuda", dtype=torch.bfloat16)
    expected = size * (size + 1) / 2
    with phase(ForwardMode.EXTEND):
        prefill = group.all_reduce(x)
    assert calls[-1] == "qr"
    torch.testing.assert_close(
        prefill.float(), torch.full_like(prefill.float(), expected), atol=0, rtol=0.06
    )
    with phase(ForwardMode.DECODE):
        decode = group.all_reduce(x)
    assert calls[-1] == "ca"
    torch.testing.assert_close(
        decode, torch.full_like(decode, expected), atol=0, rtol=0
    )

    # Existing CA graph scope: warmup allocates zeros; capture registers buffers.
    graph = torch.cuda.CUDAGraph()
    with group.graph_capture() as capture, phase(ForwardMode.DECODE):
        warmup = group.all_reduce(x)
        assert calls[-1] == "ca" and torch.count_nonzero(warmup).item() == 0
        with torch.cuda.graph(graph, stream=capture.stream):
            graph_out = group.all_reduce(x)
    torch.cuda.synchronize()
    records = []
    for i in range(12):
        shape = (16384, 2560) if i == 5 else (i + 1, 2560)
        torch.manual_seed(100 + i + rank * 97)
        inp = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        if i == 3:
            inp.mul_(1e-8)
        reference = inp.float()
        dist.all_reduce(reference, group=group.device_group)
        with phase(ForwardMode.EXTEND):
            result = group.all_reduce(inp)
        assert calls[-1] == "qr"
        relative = (
            ((result.float() - reference).square().mean() / reference.square().mean())
            .sqrt()
            .item()
        )
        assert torch.isfinite(result).all() and relative < 0.06, relative
        records.append(
            {"shape": list(shape), "nbytes": inp.nbytes, "normalized_l2": relative}
        )
        x.fill_(i + rank + 1)
        before = len(calls)
        with phase(ForwardMode.DECODE):
            graph.replay()
        torch.cuda.synchronize()
        assert len(calls) == before, "Replay should not redispatch in Python"
        torch.testing.assert_close(
            graph_out, torch.full_like(graph_out, expected + i * size), atol=0, rtol=0
        )
        with phase(ForwardMode.DECODE):
            eager = group.all_reduce(x)
        assert calls[-1] == "ca"
        torch.testing.assert_close(eager, graph_out, atol=0, rtol=0)

    # Compile the same callable under both phases; method is a guarded control
    # value while the custom op remains opaque to Inductor.
    compiled = torch.compile(group.all_reduce, backend="inductor", fullgraph=True)
    for mode, method in [
        (ForwardMode.EXTEND, "qr"),
        (ForwardMode.DECODE, "ca"),
        (ForwardMode.EXTEND, "qr"),
    ]:
        with phase(mode):
            value = compiled(x)
        torch.cuda.synchronize()
        assert calls[-1] == method
        torch.testing.assert_close(
            value.float(),
            torch.full_like(value.float(), expected + 11 * size),
            atol=0,
            rtol=0.06 if method == "qr" else 0,
        )

    for invalid in (
        x[:, ::2],
        x.to(torch.float16),
        torch.empty(128 * 1024**2 // 2 + 8, device="cuda", dtype=torch.bfloat16),
    ):
        try:
            with phase(ForwardMode.EXTEND):
                group.all_reduce(invalid)
        except RuntimeError as exc:
            assert "rejected" in str(exc)
        else:
            raise AssertionError("Invalid prefill unexpectedly fell back")

    kernel_names = []
    if args.profile:
        # Only rank 0 needs Kineto; every rank submits the same collectives.
        from contextlib import nullcontext

        ctx = (
            torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            )
            if rank == 0
            else nullcontext()
        )
        dist.barrier()
        with ctx as prof:
            with phase(ForwardMode.EXTEND):
                group.all_reduce(x)
            with phase(ForwardMode.DECODE):
                group.all_reduce(x)
                graph.replay()
            torch.cuda.synchronize()
        if rank == 0:
            prof.export_chrome_trace(str(args.profile))
            trace = json.loads(args.profile.read_text())
            kernel_names = sorted(
                {e["name"] for e in trace["traceEvents"] if e.get("cat") == "kernel"}
            )
            assert any("CodecFP8" in n for n in kernel_names), kernel_names
            assert any(
                "ar_ll" in n
                or "cross_device_reduce" in n
                or "allreduce" in n.lower()
                and "CodecFP8" not in n
                for n in kernel_names
            ), kernel_names

    torch.cuda.synchronize()
    del graph, graph_out, compiled
    dist.barrier()
    group.destroy()
    if rank == 0:
        args.report.write_text(
            json.dumps(
                {
                    "passed": True,
                    "tp": size,
                    "binary_sha256": manifest["binary_sha256"],
                    "same_shape_routing": True,
                    "decode_graph_replays": 12,
                    "decode_eager_calls": 13,
                    "compile": "inductor fullgraph",
                    "rejections": 3,
                    "prefill_cases": records,
                    "kernels": kernel_names,
                },
                indent=2,
            )
            + "\n"
        )
        print("PASS", args.report, flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
