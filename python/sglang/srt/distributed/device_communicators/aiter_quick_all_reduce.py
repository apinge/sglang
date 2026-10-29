# SPDX-License-Identifier: Apache-2.0
"""Opt-in native BF16/FP8 QuickReduce for eager prefill on gfx942."""

import logging
import os

import torch
import torch.distributed as dist

from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
    is_full_nvlink,
)
from sglang.srt.distributed.parallel_state import in_the_same_node_as

logger = logging.getLogger(__name__)


def aiter_fp8_requested():
    value = os.environ.get("AITER_QUICK_REDUCE_QUANTIZATION")
    if value in (None, "NONE"):
        return False
    if value != "FP8":
        raise ValueError("SGLang's AITER prefill QuickReduce requires quantization=FP8")
    return True


class AiterFp8QuickAllReduce:
    def __init__(self, group, device):
        self.disabled = True
        self._ptr = 0
        self.device = torch.device(
            f"cuda:{device}" if isinstance(device, int) else device
        )
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.group = group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        if os.environ.get("AITER_QUICK_REDUCE_CAST_BF16_TO_FP16", "0") != "0":
            raise ValueError(
                "FP8 prefill requires AITER_QUICK_REDUCE_CAST_BF16_TO_FP16=0"
            )
        if os.environ.get("ROCM_QUICK_REDUCE_QUANTIZATION", "NONE") != "NONE":
            raise ValueError(
                "Unset ROCM_QUICK_REDUCE_QUANTIZATION when enabling AITER FP8 prefill"
            )
        if dist.get_backend(group) == "nccl":
            raise ValueError("AITER FP8 handle exchange requires a CPU process group")
        if self.world_size not in (2, 4, 8):
            raise ValueError(f"AITER FP8 unsupported world size: {self.world_size}")
        arch = torch.cuda.get_device_properties(self.device).gcnArchName.split(":")[0]
        if arch != "gfx942":
            raise ValueError(f"AITER FP8 prefill is enabled only on gfx942, got {arch}")
        if not all(in_the_same_node_as(group, source_rank=0)):
            raise ValueError("AITER FP8 prefill requires a single-node group")
        # Use the same physical-device topology check as SGLang's existing QR.
        visible = os.environ.get(
            "HIP_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES")
        )
        if os.environ.get("ROCR_VISIBLE_DEVICES"):
            raise ValueError(
                "Use HIP_VISIBLE_DEVICES without ROCR_VISIBLE_DEVICES for FP8 prefill"
            )
        device_ids = (
            list(map(int, visible.split(",")))
            if visible
            else list(range(torch.cuda.device_count()))
        )
        physical_ids = [None] * self.world_size
        dist.all_gather_object(physical_ids, device_ids[self.device.index], group=group)
        if len(set(physical_ids)) != self.world_size or not is_full_nvlink(
            physical_ids, self.world_size
        ):
            raise ValueError(
                f"AITER FP8 requires distinct XGMI-connected GPUs: {physical_ids}"
            )

        import aiter

        self._ops = aiter
        self.qr_max_size = (
            int(os.environ.get("AITER_QUICK_REDUCE_MAX_SIZE_BYTES_MB", "128")) * 1024**2
        )
        # Load/JIT before any forward or CUDA graph capture.
        if not 0 < self.qr_max_size <= aiter.qr_max_size():
            raise ValueError(
                "AITER_QUICK_REDUCE_MAX_SIZE_BYTES_MB must be in [1, 2048]"
            )
        try:
            with torch.cuda.device(self.device):
                self._ptr = aiter.init_custom_qr(
                    self.rank, self.world_size, self.qr_max_size, 1
                )
                handle = torch.empty(64, dtype=torch.uint8, device="cpu")
                aiter.qr_get_handle(self._ptr, handle.data_ptr())
                handles = [None] * self.world_size
                dist.all_gather_object(handles, handle, group=group)
                aiter.qr_open_handles(self._ptr, [h.data_ptr() for h in handles])
            self.disabled = False
        except Exception:
            self.close()
            raise
        logger.info(
            "[AR] AITER FP8 prefill initialized: rank=%s tp=%s max_bytes=%s cast=False",
            self.rank,
            self.world_size,
            self.qr_max_size,
        )

    def input_rejection(self, inp):
        if self.disabled:
            return "communicator disabled"
        if inp.device != self.device:
            return f"expected device {self.device}"
        if inp.dtype != torch.bfloat16:
            return "native BF16 input required (FP16 FP8 codec is not enabled)"
        if not inp.is_contiguous():
            return "contiguous input required"
        if inp.numel() == 0 or inp.nbytes % 16 or inp.data_ptr() % 16:
            return "nonempty, 16-byte-aligned input required"
        if inp.nbytes > self.qr_max_size:
            return f"input exceeds capacity {self.qr_max_size} bytes"
        return None

    def should_quick_allreduce(self, inp):
        return self.input_rejection(inp) is None

    def quick_all_reduce(self, inp, *, out=None):
        reason = self.input_rejection(inp)
        if reason:
            raise RuntimeError(
                f"AITER FP8 prefill rejected shape={tuple(inp.shape)} dtype={inp.dtype} bytes={inp.nbytes}: {reason}"
            )
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "AITER FP8 prefill requires prefill CUDA graphs disabled"
            )
        if out is None:
            out = torch.empty_like(inp)
        elif (
            out.shape != inp.shape
            or out.dtype != inp.dtype
            or out.device != inp.device
            or not out.is_contiguous()
            or out.data_ptr() % 16
        ):
            raise ValueError(
                "FP8 output must match input shape/dtype/device and be contiguous and aligned"
            )
        self._ops.qr_all_reduce(self._ptr, inp, out, 1, False)
        return out

    def close(self):
        if self._ptr:
            with torch.cuda.device(self.device):
                torch.cuda.synchronize(self.device)
                self._ops.qr_destroy(self._ptr)
            self._ptr = 0
        self.disabled = True

    def __del__(self):
        self.close()
