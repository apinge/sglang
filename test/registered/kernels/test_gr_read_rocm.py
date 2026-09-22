"""Standalone ROCm GR read integration acceptance (no full-model load).

Run with SGLANG_USE_AITER=1 and one visible gfx942 device. Tests exercise the
production adapter/launchers; references retain original weights only here.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from sglang.srt.layers.gr_read.runtime import (
    HC,
    HS,
    Bucket,
    GRReadState,
    K,
    R,
    compiled_mix_packed,
    mix_packed,
    pack_weights,
    prepare_bucket,
    unpack_weight,
)


def original_mix(x, w_down, w_up):
    # Frozen math from hyperconnection.py at SGLang 21d0d512.
    gates = F.silu(F.linear(x, w_down) / HC)
    gates = F.linear(gates, w_up)
    gates = torch.sigmoid(gates).unflatten(-1, (HC, HS))
    return (gates * x.unflatten(-1, (HC, HS))).mean(dim=-2)


def weight_pairs(args):
    if args.synthetic:
        for i in range(args.weights):
            torch.manual_seed(1729 + i)
            yield f"synthetic_{i}", torch.randn(
                R, K, device="cuda", dtype=torch.bfloat16
            ) * 0.02, torch.randn(K, R, device="cuda", dtype=torch.bfloat16) * 0.02
        return
    root = Path(args.model_path)
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    suffix = ".input_mix_weight_down.weight"
    names = sorted(n for n in index if n.endswith(suffix))
    assert len(names) >= args.weights, (len(names), args.weights)
    for name in names[: args.weights]:
        up_name = name[: -len(suffix)] + ".input_mix_weight_up.weight"
        with safe_open(root / index[name], framework="pt", device="cpu") as f:
            down = f.get_tensor(name).cuda()
        with safe_open(root / index[up_name], framework="pt", device="cpu") as f:
            up = f.get_tensor(up_name).cuda()
        yield name[: -len(suffix)], down, up


def packing_check(wd, wu, pd, pu):
    def independent(w):
        n, k = w.shape
        return (
            w.reshape(n // 16, 16, k // 32, 4, 8)
            .permute(0, 2, 3, 1, 4)
            .contiguous()
            .flatten()
        )

    up_hc = wu.reshape(HC, HS, R).permute(1, 0, 2).contiguous().reshape(K, R)
    assert torch.equal(
        pd.flatten().view(torch.uint8), independent(wd).view(torch.uint8)
    )
    assert torch.equal(
        pu.flatten().view(torch.uint8), independent(up_hc).view(torch.uint8)
    )
    assert torch.equal(unpack_weight(pd, R, K), wd)
    assert torch.equal(unpack_weight(pu, K, R), up_hc)


@torch.inference_mode()
def torch_suite(args, emit):
    old = torch.compile(original_mix)
    cases = 0
    for name, wd, wu in weight_pairs(args):
        pd, pu = pack_weights(wd, wu)
        packing_check(wd, wu, pd, pu)
        for b in args.torch_rows:
            x = torch.randn(b, K, device="cuda", dtype=torch.bfloat16)
            expected = old(x, wd, wu)
            actual = compiled_mix_packed(x, pd, pu)
            mismatch = (expected != actual).sum().item()
            error = (expected.float() - actual.float()).abs().max().item()
            emit(
                dict(
                    suite="torch", weight=name, B=b, mismatches=mismatch, max_abs=error
                )
            )
            assert torch.equal(
                expected, actual
            ), f"New/old compiled Torch differs: {name}, B={b}, count={mismatch}, max={error}"
            assert torch.equal(mix_packed(x, pd, pu), original_mix(x, wd, wu))
            cases += 1
        packing_check(wd, wu, pd, pu)
    emit(dict(suite="torch_summary", cases=cases, bitwise_equal=True))


class Guard:
    def __init__(self, shape, dtype):
        count = 1
        for s in shape:
            count *= s
        self.storage = torch.full((count + 32,), 42, dtype=dtype, device="cuda")
        self.value = self.storage[16:-16].view(shape)

    def check(self):
        assert torch.all(self.storage[:16] == 42)
        assert torch.all(self.storage[-16:] == 42)


@torch.inference_mode()
def bucket_suite(args, emit):
    # Capture each B only once. Test-only fixed weight buffers receive each
    # already-packed pair between cases; all graphs share their addresses.
    pd = torch.empty((R, K), dtype=torch.bfloat16, device="cuda")
    pu = torch.empty((K, R), dtype=torch.bfloat16, device="cuda")
    captured = {}
    stream = torch.cuda.Stream()
    replay_count = ref_count = 0
    max_scaled = 0.0
    for name, wd, wu in weight_pairs(args):
        packed_down, packed_up = pack_weights(wd, wu)
        packing_check(wd, wu, packed_down, packed_up)
        pd.copy_(packed_down)
        pu.copy_(packed_up)
        for b in args.buckets:
            if b not in captured:
                t_pad = 16 if b <= 16 else 32
                xg = Guard((b, K), torch.bfloat16)
                pg = Guard((4 * t_pad * R,), torch.float32)
                yg = Guard((b, HS), torch.bfloat16)
                prepared = prepare_bucket(b, pd, pu)
                bucket = Bucket(prepared.launch, pg.value, yg.value, yg.value.flatten())
                del prepared
                xg.value.zero_()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    bucket.launch(
                        xg.value.flatten(),
                        pd.flatten(),
                        pu.flatten(),
                        pg.value,
                        yg.value.flatten(),
                        stream,
                    )
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    bucket.launch(
                        xg.value.flatten(),
                        pd.flatten(),
                        pu.flatten(),
                        pg.value,
                        yg.value.flatten(),
                        torch.cuda.current_stream(),
                    )
                captured[b] = (graph, bucket, xg, pg, yg)
            graph, bucket, xg, pg, yg = captured[b]
            valid = torch.randn(b, K, dtype=torch.bfloat16, device="cuda")
            reference = original_mix(valid.double(), wd.double(), wu.double())
            for tail in args.tails:
                # B..0..B includes each nonempty M on the down/up sweeps.
                for m in [*range(b, -1, -1), *range(1, b + 1)]:
                    xg.value[:m].copy_(valid[:m])
                    if tail == "stale":
                        xg.value[m:].fill_(3.25)
                    elif tail == "nan":
                        xg.value[m:].fill_(float("nan"))
                    else:
                        xg.value[m:].zero_()
                    saved_x = xg.value.clone()
                    pg.value.fill_(float("nan"))
                    yg.value.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.equal(
                        saved_x.view(torch.uint8), xg.value.view(torch.uint8)
                    )
                    xg.check()
                    pg.check()
                    yg.check()
                    t_pad = 16 if b <= 16 else 32
                    p = pg.value.view(4, t_pad, R)
                    assert torch.all(p[:, b:] == 0), (name, b, m, "internal padding")
                    if m:
                        actual = yg.value[:m].double()
                        expected = reference[:m]
                        scaled = (
                            (
                                (actual - expected).abs()
                                / (0.005 + 0.01 * expected.abs())
                            )
                            .max()
                            .item()
                        )
                        max_scaled = max(max_scaled, scaled)
                        assert torch.isfinite(actual).all()
                        assert torch.allclose(
                            actual, expected, rtol=0.01, atol=0.005
                        ), (name, b, m, tail, scaled)
                        assert torch.isfinite(p[:, :m]).all()
                        ref_count += 1
                    replay_count += 1
            emit(
                dict(
                    suite="bucket",
                    weight=name,
                    B=b,
                    replays=replay_count,
                    max_scaled=max_scaled,
                )
            )
        assert torch.equal(pd, packed_down) and torch.equal(pu, packed_up)
    emit(
        dict(
            suite="bucket_summary",
            replays=replay_count,
            nonempty_fp64_checks=ref_count,
            captures=len(captured),
            max_scaled=max_scaled,
        )
    )


@torch.inference_mode()
def lifecycle_suite(args, emit):
    from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig
    from sglang.srt.model_loader.utils import set_default_torch_dtype

    # Match DefaultModelLoader, including the norm parameter's BF16 dtype.
    with torch.device("cuda"), set_default_torch_dtype(torch.bfloat16):
        module = GatedResidual(
            HyperConnectionConfig(hidden_size=HS, hc_lowrank=R, hc_per_branch_norm=True)
        )
    _, wd, wu = next(weight_pairs(args))
    module._load_gr_read_weight(module.input_mix_weight_down.weight, wd)
    module._load_gr_read_weight(module.input_mix_weight_up.weight, wu)
    module.quant_method.process_weights_after_loading(module)
    state = module._gr_read
    assert isinstance(state, GRReadState)
    pointers = state.pointers
    module.quant_method.process_weights_after_loading(module)
    assert module._gr_read_pack_count == 1 and state.pointers == pointers
    from sglang.srt.model_executor.model_runner_components.weight_updater import (
        _unsupported_derived_weight_cache_error,
    )

    assert _unsupported_derived_weight_cache_error(module) is not None
    try:
        module.cpu()
    except RuntimeError:
        pass
    else:
        raise AssertionError("Device changes must not invalidate live graph storage")
    packing_check(wd, wu, state.w_down, state.w_up)
    assert state.wd_flat.data_ptr() == module.input_mix_weight_down.weight.data_ptr()
    assert state.wu_flat.data_ptr() == module.input_mix_weight_up.weight.data_ptr()
    state.prepare([4, 32], torch.cuda.current_stream())
    state.prepare([4, 32], torch.cuda.current_stream())
    assert len(state.contexts[torch.cuda.current_stream().cuda_stream]) == 2
    x = torch.randn((4, K), device="cuda", dtype=torch.bfloat16)
    saved_x = x.clone()
    mixed, residual = module.mix(x)
    assert residual[0] is x and torch.equal(x, saved_x)
    assert torch.allclose(
        mixed.double(),
        original_mix(residual[1].double(), wd.double(), wu.double()),
        rtol=0.01,
        atol=0.005,
    )
    # Capture the same B on distinct streams and replay concurrently. Only the
    # immutable weights/compiled code are shared, never the writable P/Y.
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    graphs, inputs, outputs = [], [], []
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            state.prepare([4], stream)
            value = torch.randn((4, K), device="cuda", dtype=torch.bfloat16)
            state(value)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = state(value)
        graphs.append(graph)
        inputs.append(value)
        outputs.append(output)
    assert outputs[0].data_ptr() != outputs[1].data_ptr()
    assert (
        state.contexts[streams[0].cuda_stream][4].partial.data_ptr()
        != state.contexts[streams[1].cuda_stream][4].partial.data_ptr()
    )
    for graph, stream in zip(graphs, streams):
        with torch.cuda.stream(stream):
            graph.replay()
    torch.cuda.synchronize()
    for value, output in zip(inputs, outputs):
        assert torch.allclose(
            output.double(),
            original_mix(value.double(), wd.double(), wu.double()),
            rtol=0.01,
            atol=0.005,
        )
    assert module.mix(torch.empty((0, K), device="cuda", dtype=torch.bfloat16))[
        0
    ].shape == (0, HS)
    try:
        module._load_gr_read_weight(module.input_mix_weight_down.weight, wd)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Reload must not overwrite graph-owned packed weights")
    for x in (
        torch.empty((4, K), device="cuda"),
        torch.empty((4, K * 2), device="cuda", dtype=torch.bfloat16)[:, ::2],
    ):
        try:
            state(x)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid input was accepted")
    try:
        state(torch.empty((3, K), device="cuda", dtype=torch.bfloat16))
    except RuntimeError:
        pass
    else:
        raise AssertionError("Unprepared bucket was accepted")
    emit(dict(suite="lifecycle", pack_count=module._gr_read_pack_count, passed=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite", choices=["torch", "bucket", "lifecycle"], required=True
    )
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--weights", type=int, default=2)
    parser.add_argument("--model-path", default="/models/Qwen3.8-Flash-Next-PTPC-FP8")
    parser.add_argument("--buckets", type=int, nargs="+", default=list(range(1, 33)))
    parser.add_argument(
        "--torch-rows",
        type=int,
        nargs="+",
        default=[31, 32, 33, 64, 127, 128, 256, 1024, 8192, 16384],
    )
    parser.add_argument(
        "--tails",
        nargs="+",
        choices=["zero", "stale", "nan"],
        default=["zero", "stale", "nan"],
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(20260917)
    with open(args.output, "x") as f:

        def emit(record):
            line = json.dumps(record)
            print(line, flush=True)
            f.write(line + "\n")
            f.flush()

        emit(
            dict(
                torch=torch.__version__,
                hip=torch.version.hip,
                gpu=torch.cuda.get_device_name(),
                args=vars(args),
            )
        )
        {"torch": torch_suite, "bucket": bucket_suite, "lifecycle": lifecycle_suite}[
            args.suite
        ](args, emit)


if __name__ == "__main__":
    main()
