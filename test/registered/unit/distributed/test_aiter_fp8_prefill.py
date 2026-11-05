import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.distributed import parallel_state
from sglang.srt.distributed.device_communicators.aiter_quick_all_reduce import (
    AiterFp8QuickAllReduce,
    aiter_fp8_requested,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    get_communication_phase,
    get_forward_context,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def coordinator():
    group = parallel_state.GroupCoordinator.__new__(parallel_state.GroupCoordinator)
    group._fp8_prefill_policy = True
    group.world_size = 2
    group.unique_name = "test_fp8"
    group.hpu_communicator = group.xpu_communicator = group.npu_communicator = None
    group.ca_comm = Mock(disabled=False)
    group.qr_comm = Mock(disabled=False)
    group.pymscclpp_comm = None
    group.torch_symm_mem_comm = None
    group.pynccl_comm = Mock()
    group._fi_workspace_hint = None
    group._fp8_logged_routes = set()
    return group


@pytest.mark.parametrize(
    "mode,method",
    [
        (ForwardMode.EXTEND, "qr"),
        (ForwardMode.SPLIT_PREFILL, "qr"),
        (ForwardMode.DECODE, "ca"),
        (ForwardMode.TARGET_VERIFY, "ca"),
    ],
)
@pytest.mark.parametrize("compiling", [False, True])
def test_same_shape_uses_phase_before_shortcuts(mode, method, compiling):
    group = coordinator()
    inp = Mock(is_cpu=False, dtype=torch.bfloat16)
    with (
        forward_context(ForwardContext(None), forward_mode=mode),
        patch.object(
            parallel_state.torch.compiler, "is_compiling", return_value=compiling
        ),
        patch.object(parallel_state, "outplace_all_reduce") as reduce,
    ):
        group.all_reduce(inp)
        reduce.assert_called_once_with(
            inp, group_name="test_fp8", outplace_all_reduce_method=method
        )
        assert group._resolve_outplace_all_reduce_method(inp) == method
    group.pynccl_comm.all_reduce.assert_not_called()


def test_nested_backend_override_and_phase_restore():
    assert get_communication_phase() is None
    with forward_context(ForwardContext("outer"), forward_mode=ForwardMode.EXTEND):
        with forward_context(ForwardContext("override")):
            assert get_communication_phase() == "prefill"
            assert get_forward_context().attn_backend == "override"
            with forward_context(
                get_forward_context(), forward_mode=ForwardMode.DECODE
            ):
                assert get_communication_phase() == "decode"
            assert get_communication_phase() == "prefill"
        assert get_forward_context().attn_backend == "outer"
    assert get_communication_phase() is None


def test_no_phase_does_not_select_fp8():
    group = coordinator()
    group.ca_comm.should_custom_ar.return_value = False
    group.qr_comm.should_quick_allreduce.return_value = True
    with patch.object(
        parallel_state, "is_in_tc_piecewise_cuda_graph", return_value=False
    ):
        assert (
            group._resolve_outplace_all_reduce_method(Mock(dtype=torch.bfloat16))
            is None
        )
    group.qr_comm.should_quick_allreduce.assert_not_called()


def test_mixed_mode_is_rejected():
    with (
        forward_context(ForwardContext(None), forward_mode=ForwardMode.MIXED),
        pytest.raises(RuntimeError, match="unsupported"),
    ):
        coordinator()._phase_all_reduce_method(Mock(dtype=torch.bfloat16))


def test_prefill_bypasses_fused_ca_but_decode_keeps_it():
    group = coordinator()
    args = (Mock(), Mock(), Mock(), 1e-6)
    with forward_context(ForwardContext(None), forward_mode=ForwardMode.EXTEND):
        assert group.fused_allreduce_rmsnorm(*args) is None
    group.ca_comm.fused_allreduce_rmsnorm.assert_not_called()
    with forward_context(ForwardContext(None), forward_mode=ForwardMode.DECODE):
        group.fused_allreduce_rmsnorm(*args)
    group.ca_comm.fused_allreduce_rmsnorm.assert_called_once_with(*args)


def test_decode_ca_rejection_does_not_fallback_to_qr():
    group = coordinator()
    group.ca_comm.should_custom_ar.return_value = False
    inp = Mock(dtype=torch.bfloat16, shape=(1, 2560), nbytes=5120)
    with pytest.raises(RuntimeError, match="decode CA rejected"):
        group._all_reduce_out_place(inp, "ca")
    group.qr_comm.quick_all_reduce.assert_not_called()


def test_prefill_cannot_use_fused_ca_quant():
    group = coordinator()
    with (
        forward_context(ForwardContext(None), forward_mode=ForwardMode.EXTEND),
        patch.object(parallel_state, "is_hip", return_value=True),
        patch.object(parallel_state, "is_gfx95_supported", return_value=True),
    ):
        assert (
            group.fused_allreduce_rmsnorm_quant_per_group(Mock(), Mock(), Mock(), 1e-6)
            is None
        )
    group.ca_comm.custom_fused_ar_rms_per_group_quant.assert_not_called()


@pytest.mark.parametrize(
    "value,expected", [(None, False), ("NONE", False), ("FP8", True)]
)
def test_env_is_opt_in(monkeypatch, value, expected):
    monkeypatch.delenv("AITER_QUICK_REDUCE_QUANTIZATION", raising=False)
    if value is not None:
        monkeypatch.setenv("AITER_QUICK_REDUCE_QUANTIZATION", value)
    assert aiter_fp8_requested() == expected


def test_env_rejects_integer_codec(monkeypatch):
    monkeypatch.setenv("AITER_QUICK_REDUCE_QUANTIZATION", "INT8")
    with pytest.raises(ValueError, match="FP8"):
        aiter_fp8_requested()


@pytest.mark.parametrize(
    "change,message",
    [
        ({"nbytes": 1048592}, "capacity"),
        ({"dtype": torch.float16}, "BF16"),
        ({"contiguous": False}, "contiguous"),
        ({"pointer": 18}, "aligned"),
        ({"nbytes": 14}, "aligned"),
        ({"numel": 0}, "nonempty"),
    ],
)
def test_input_rejections(change, message):
    qr = AiterFp8QuickAllReduce.__new__(AiterFp8QuickAllReduce)
    qr._ptr = 0
    qr.disabled = False
    qr.device = torch.device("cuda:0")
    qr.qr_max_size = 1024**2
    values = {
        "dtype": torch.bfloat16,
        "nbytes": 5120,
        "numel": 2560,
        "pointer": 16,
        "contiguous": True,
    }
    values.update(change)
    inp = SimpleNamespace(
        dtype=values["dtype"],
        device=qr.device,
        nbytes=values["nbytes"],
        numel=lambda: values["numel"],
        data_ptr=lambda: values["pointer"],
        is_contiguous=lambda: values["contiguous"],
    )
    assert message in qr.input_rejection(inp)


def test_destroy_closes_qr_before_groups():
    group = coordinator()
    group.device_group, group.cpu_group = "device", "cpu"
    group.mq_broadcaster = None
    events = []
    group.qr_comm.close.side_effect = lambda: events.append("qr")
    with patch.object(
        torch.distributed, "destroy_process_group", side_effect=events.append
    ):
        group.destroy()
    assert events == ["qr", "device", "cpu"]
    assert group.qr_comm is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
