"""Model lifecycle adapter for the installed PyHIP GR read operator."""

import importlib
import logging
import time
from functools import lru_cache
from importlib import metadata

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Post-load warmup rows. Decode capture sizes are passed separately.
# Kernel selection stays inside PyHIP.
_WARMUP_ROWS = tuple(range(1, 33)) + (
    128,
    256,
    512,
    1024,
    2048,
    2560,
    3072,
    4096,
    5120,
    7680,
    10240,
)


@lru_cache(maxsize=1)
def get_pyhip_ops():
    try:
        api = importlib.import_module("pyhip.ops.gr_read")
        version = metadata.version("pyhip")
    except (ImportError, metadata.PackageNotFoundError) as exc:
        raise RuntimeError(
            "SGLANG_GR_READ_FLYDSL=1 requires an installed PyHIP package with "
            "pyhip.ops.gr_read.prepare_weights and gr_read. Install the validated "
            "PyHIP wheel in the SGLang worker's Python environment."
        ) from exc
    if not all(callable(getattr(api, name, None)) for name in ("prepare_weights", "gr_read")):
        raise RuntimeError("Installed PyHIP does not provide the required GR read API")
    logger.info("PyHIP GR read: version=%s, module=%s", version, api.__file__)
    return api.prepare_weights, api.gr_read


def supports_pyhip_gr_read(module):
    if not envs.SGLANG_GR_READ_FLYDSL.get() or torch.version.hip is None:
        return False
    if (module.hc_count, module.hidden_size, module.config.hc_lowrank) != (4, 2560, 320):
        return False
    weight = module.input_mix_weight_down.weight
    if weight.dtype != torch.bfloat16 or not weight.is_cuda:
        return False
    return torch.cuda.get_device_properties(weight.device).gcnArchName.split(":", 1)[0] == "gfx942"


class PyHIPGRReadMethod:
    """Use SGLang's post-load hook without retaining raw-layout weight copies."""

    @torch.no_grad()
    def process_weights_after_loading(self, module):
        # TODO: ShardedStateLoader and Remote KV are currently unsupported by this
        # PyHIP GR read integration. They invoke this hook before copying saved state.
        # Add layout validation and a post-copy packing lifecycle that distinguishes
        # raw weights from already-packed weights before enabling these loaders.
        if module._gr_read_weights is not None:
            return
        wd = module.input_mix_weight_down.weight
        wu = module.input_mix_weight_up.weight
        if wd.shape != (320, 10240) or wu.shape != (10240, 320):
            raise ValueError("PyHIP GR read requires the original checkpoint weight shapes")
        if wd.device != wu.device or wd.dtype != torch.bfloat16 or wu.dtype != torch.bfloat16:
            raise ValueError("PyHIP GR read weights must be BF16 on the same device")
        prepare_weights, gr_read = get_pyhip_ops()
        packed_down, packed_up = prepare_weights(wd, wu)
        # Retain Parameter identities and checkpoint shapes, with packed storage.
        # The flat tensors passed to PyHIP share this storage; no raw copy remains.
        wd.data = packed_down.view_as(wd)
        wu.data = packed_up.view_as(wu)
        wd.requires_grad_(False)
        wu.requires_grad_(False)
        module._gr_read_weights = (packed_down, packed_up)
        module._gr_read_fn = gr_read
        module._gr_read_pack_count += 1


@torch.inference_mode()
def warmup_pyhip_gr_read(model, rows=None):
    """Compile through the public API; keep no per-layer or per-stream P/Y cache."""
    modules = [m for m in model.modules() if getattr(m, "_gr_read_enabled", False)]
    if not modules:
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("PyHIP GR read warmup must precede graph capture")
    by_device = {}
    for module in modules:
        if module._gr_read_weights is None:
            raise RuntimeError("Model loader did not prepare PyHIP GR read weights")
        packed = module._gr_read_weights
        by_device.setdefault(packed[0].device, packed)
    _, gr_read = get_pyhip_ops()
    requested_rows = tuple(sorted({int(r) for r in rows if r > 0})) if rows is not None else None
    logger.info(
        "PyHIP GR read warmup begin: ts=%s, modules=%d, devices=%d",
        time.strftime("%Y-%m-%d %H:%M:%S"),
        len(modules),
        len(by_device),
    )
    started = time.perf_counter()
    try:
        for device, (packed_down, packed_up) in by_device.items():
            with torch.cuda.device(device):
                if requested_rows is None:
                    selected_rows = _WARMUP_ROWS
                else:
                    selected_rows = requested_rows
                for count in selected_rows:
                    x = torch.zeros((count, 10240), dtype=torch.bfloat16, device=device)
                    gr_read(x, packed_down, packed_up)
                torch.cuda.current_stream(device).synchronize()
                logger.info(
                    "PyHIP GR read prepared: modules=%d, device=%s, rows=%s, pack_counts=%s",
                    len(modules),
                    device,
                    selected_rows,
                    sorted({m._gr_read_pack_count for m in modules}),
                )
    finally:
        logger.info(
            "PyHIP GR read warmup end: ts=%s, elapsed=%.2f s",
            time.strftime("%Y-%m-%d %H:%M:%S"),
            time.perf_counter() - started,
        )
