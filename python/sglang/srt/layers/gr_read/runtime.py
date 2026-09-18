"""Load-time packing and prepared GR read execution for gfx942.

Weights have one owner/layout for all row counts. Workspaces belong to a
module, physical row count and stream; prepare them before model execution.
"""

import logging
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

HC, HS, R = 4, 2560, 320
K = HC * HS
LAYOUT = "gr_read_bf16_hc_interleaved_preshuffle_v1"
logger = logging.getLogger(__name__)


def validate_weights(w_down, w_up):
    if w_down.shape != (R, K) or w_up.shape != (K, R):
        raise ValueError("GR read requires WD[320,10240] and WU[10240,320]")
    for weight in (w_down, w_up):
        if weight.dtype != torch.bfloat16 or not weight.is_contiguous():
            raise ValueError("GR read weights must be contiguous BF16")
    if not w_down.is_cuda or w_down.device != w_up.device or torch.version.hip is None:
        raise ValueError("GR read weights must be on the same ROCm device")
    if (
        torch.cuda.get_device_properties(w_down.device).gcnArchName.split(":")[0]
        != "gfx942"
    ):
        raise ValueError("GR read is validated on gfx942")


def pack_weights(w_down, w_up):
    """Return AITER-packed weights; never called from forward/capture."""
    from aiter.ops.shuffle import shuffle_weight

    validate_weights(w_down, w_up)
    packed_down = shuffle_weight(w_down, (16, 16))
    up_hc = w_up.reshape(HC, HS, R).permute(1, 0, 2).contiguous().reshape(K, R)
    packed_up = shuffle_weight(up_hc, (16, 16))
    return packed_down, packed_up


def unpack_weight(packed, n, k):
    return (
        packed.reshape(n // 16, k // 32, 4, 16, 8).permute(0, 3, 1, 2, 4).reshape(n, k)
    )


def mix_packed(x, packed_down, packed_up):
    """Original Torch math, with weight layout decoding inside the graph.

    Dense intermediates are call-local. In particular, do not cache these
    decoded weights: both prefill and FlyDSL own only the packed storage.
    """
    w_down = unpack_weight(packed_down, R, K)
    w_up = (
        unpack_weight(packed_up, K, R).reshape(HS, HC, R).permute(1, 0, 2).reshape(K, R)
    )
    gates = F.silu(F.linear(x, w_down) / HC)
    gates = F.linear(gates, w_up)
    gates = torch.sigmoid(gates).unflatten(-1, (HC, HS))
    return (gates * x.unflatten(-1, (HC, HS))).mean(dim=-2)


compiled_mix_packed = torch.compile(mix_packed, dynamic=True)


def selected_configs(rows):
    from . import down, small, up

    if not 1 <= rows <= 32:
        raise ValueError("FlyDSL GR read requires B=1..32")
    base = replace(up.default_config(rows), hidden_pad=4, prefetch_low=False)
    if rows <= 16:
        cfg = small.default_config(rows)
        down_cfg, up_cfg = cfg.down, cfg.up_config(base)
    else:
        down_cfg = down.default_config(rows)
        up_cfg = replace(base, down_mode="partial", down_n=64, split_k=4)
    down_cfg.validate()
    up_cfg.validate()
    return down_cfg, up_cfg


@dataclass
class Bucket:
    launch: object
    partial: torch.Tensor
    output: torch.Tensor
    output_flat: torch.Tensor


# Handles contain compiled code/argument metadata, not weights or workspaces.
# Worker Python forwards are serialized; GPU streams have independent P/Y.
_handles = {}


def prepare_bucket(rows, packed_down, packed_up):
    import flydsl.compiler as flyc

    from .down import pair_launcher

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("GR read buckets must be prepared before capture")
    down, up = selected_configs(rows)
    device = packed_down.device
    t_pad = 16 if rows <= 16 else 32
    partial = torch.empty(4 * t_pad * R, dtype=torch.float32, device=device)
    output = torch.empty((rows, HS), dtype=torch.bfloat16, device=device)
    flat = output.view(-1)
    key = (device, rows, down, up)
    launch = _handles.get(key)
    if launch is None:
        x = torch.zeros(rows * K, dtype=torch.bfloat16, device=device)
        launch = flyc.compile(
            pair_launcher(rows, down, up),
            x,
            packed_down.view(-1),
            packed_up.view(-1),
            partial,
            flat,
            torch.cuda.current_stream(device),
        )
        _handles[key] = launch
    return Bucket(launch, partial, output, flat)


class GRReadState:
    def __init__(self, w_down, w_up):
        validate_weights(w_down, w_up)
        self.w_down, self.w_up = w_down, w_up
        self.wd_flat, self.wu_flat = w_down.view(-1), w_up.view(-1)
        self.device = w_down.device
        self.pointers = (w_down.data_ptr(), w_up.data_ptr())
        self.contexts = {}
        self.streams = {}
        self.layout = LAYOUT

    def prepare(self, rows, stream):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("GR read preparation cannot run inside a graph")
        if stream.device != self.device:
            raise ValueError("GR read stream must use the weight device")
        # Keep the stream identity alive for as long as its captured P/Y.
        self.streams[stream.cuda_stream] = stream
        buckets = self.contexts.setdefault(stream.cuda_stream, {})
        for b in rows:
            if 1 <= b <= 32 and b not in buckets:
                buckets[b] = prepare_bucket(b, self.w_down, self.w_up)

    def __call__(self, x):
        if x.ndim != 2 or x.shape[1] != K or x.dtype != torch.bfloat16:
            raise ValueError("GR read requires BF16 X[B,10240]")
        if not x.is_contiguous() or x.device != self.device:
            raise ValueError("GR read X must be contiguous and on the weight device")
        if x.requires_grad and torch.is_grad_enabled():
            raise ValueError("GR read is inference-only")
        if (self.w_down.data_ptr(), self.w_up.data_ptr()) != self.pointers:
            raise RuntimeError("GR read weights moved; destroy graphs and reinitialize")
        b = x.shape[0]
        if b == 0:
            return x.new_empty((0, HS))
        if b > 32:
            return compiled_mix_packed(x, self.w_down, self.w_up)
        stream = torch.cuda.current_stream(x.device)
        bucket = self.contexts.get(stream.cuda_stream, {}).get(b)
        if bucket is None:
            raise RuntimeError(
                f"GR read B={b}, stream={stream.cuda_stream} was not prepared before forward"
            )
        x_begin, x_end = x.data_ptr(), x.data_ptr() + x.numel() * x.element_size()
        for workspace in (bucket.partial, bucket.output):
            begin = workspace.data_ptr()
            end = begin + workspace.numel() * workspace.element_size()
            if x_begin < end and begin < x_end:
                raise ValueError("GR read X must not overlap P or Y")
        bucket.launch(
            x.view(-1),
            self.wd_flat,
            self.wu_flat,
            bucket.partial,
            bucket.output_flat,
            stream,
        )
        return bucket.output


class GRReadMethod:
    """Use the same loader post-load protocol as AITER MoE methods."""

    @torch.no_grad()
    def process_weights_after_loading(self, module):
        if module._gr_read is not None:
            return
        wd = module.input_mix_weight_down.weight
        wu = module.input_mix_weight_up.weight
        packed_down, packed_up = pack_weights(wd, wu)
        # Keep parameter identities/weight_loader attributes, replace storage.
        # Original values are released; no original-weight cache is retained.
        wd.data, wu.data = packed_down, packed_up
        wd.requires_grad_(False)
        wu.requires_grad_(False)
        for weight in (wd, wu):
            weight.is_shuffled = True
            weight.gr_read_layout = LAYOUT
        module._gr_read = GRReadState(wd, wu)
        module._gr_read_pack_count += 1


def prepare_model(model, streams, rows=range(1, 33)):
    """Explicit startup/capture preparation, never called from model forward."""
    modules = [m for m in model.modules() if getattr(m, "_gr_read", None) is not None]
    states = [m._gr_read for m in modules]
    if not states:
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("GR read preparation cannot run inside a graph")
    rows = tuple(sorted({int(b) for b in rows if 1 <= b <= 32}))

    def scratch_bytes():
        return sum(
            bucket.partial.numel() * bucket.partial.element_size()
            + bucket.output.numel() * bucket.output.element_size()
            for state in states
            for buckets in state.contexts.values()
            for bucket in buckets.values()
        )

    before = scratch_bytes()
    current = torch.cuda.current_stream()
    for stream in streams:
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for state in states:
                state.prepare(rows, stream)
        current.wait_stream(stream)
    logger.info(
        "GR read prepared: modules=%d, layout=%s, B=%s, streams=%s, scratch_delta=%.2f MiB, pack_counts=%s",
        len(states),
        LAYOUT,
        rows,
        [s.cuda_stream for s in streams],
        (scratch_bytes() - before) / 2**20,
        sorted({m._gr_read_pack_count for m in modules}),
    )
