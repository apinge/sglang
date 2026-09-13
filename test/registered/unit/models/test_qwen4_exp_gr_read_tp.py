"""GR token parallelism must not change decode or existing distributed layouts."""

from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

from sglang.srt.models import qwen4_exp as model
from sglang.srt.model_executor.forward_batch_info import ForwardMode


@pytest.fixture
def context(monkeypatch):
    monkeypatch.setenv("SGLANG_GR_READ_TP_SPLIT", "1")
    parallel = SimpleNamespace(
        tp_size=2,
        pp_size=1,
        attn_tp_size=2,
        attn_cp_size=1,
        moe_ep_size=1,
        moe_dp_size=1,
        config=SimpleNamespace(
            dp_size=1,
            dcp_size=1,
            enable_dp_attention=False,
            enable_prefill_cp=False,
            enable_prefill_context_parallel=False,
            enable_dsa_prefill_context_parallel=False,
        ),
    )
    execution = SimpleNamespace(
        graph=SimpleNamespace(
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend="disabled")
            )
        )
    )
    attention = SimpleNamespace(input_scattered=False)
    monkeypatch.setattr(model, "get_parallel", lambda: parallel)
    monkeypatch.setattr(model, "get_exec", lambda: execution)
    monkeypatch.setattr(model, "get_attn_tp_context", lambda: attention)
    monkeypatch.setattr(model, "get_is_capture_mode", lambda: False)
    monkeypatch.setattr(
        model, "get_moe_a2a_backend", lambda: SimpleNamespace(is_none=lambda: True)
    )
    return parallel, execution, attention


@pytest.mark.parametrize("mode", list(ForwardMode))
def test_only_ordinary_extend(context, mode):
    enabled = model._qwen4_exp_gr_read_tp_enabled(SimpleNamespace(forward_mode=mode))
    assert enabled == (mode == ForwardMode.EXTEND)


def test_disabled_does_not_require_parallel_initialization(monkeypatch):
    monkeypatch.setenv("SGLANG_GR_READ_TP_SPLIT", "0")

    def unexpected():
        raise AssertionError("disabled path read the parallel context")

    monkeypatch.setattr(model, "get_parallel", unexpected)
    assert not model._qwen4_exp_gr_read_tp_enabled(
        SimpleNamespace(forward_mode=ForwardMode.EXTEND)
    )


@pytest.mark.parametrize(
    "name,value",
    [
        ("tp_size", 1),
        ("pp_size", 2),
        ("attn_tp_size", 1),
        ("attn_cp_size", 2),
        ("moe_ep_size", 2),
        ("moe_dp_size", 2),
    ],
)
def test_rejects_other_parallel_layouts(context, name, value):
    setattr(context[0], name, value)
    assert not model._qwen4_exp_gr_read_tp_enabled(
        SimpleNamespace(forward_mode=ForwardMode.EXTEND)
    )


@pytest.mark.parametrize("name", ["dp_size", "dcp_size"])
def test_rejects_parallel_config(context, name):
    setattr(context[0].config, name, 2)
    assert not model._qwen4_exp_gr_read_tp_enabled(
        SimpleNamespace(forward_mode=ForwardMode.EXTEND)
    )


@pytest.mark.parametrize(
    "name",
    [
        "enable_dp_attention",
        "enable_prefill_cp",
        "enable_prefill_context_parallel",
        "enable_dsa_prefill_context_parallel",
    ],
)
def test_rejects_existing_token_parallelism(context, name):
    setattr(context[0].config, name, True)
    assert not model._qwen4_exp_gr_read_tp_enabled(
        SimpleNamespace(forward_mode=ForwardMode.EXTEND)
    )


def test_graph_and_scattered_input_fallback(context):
    context[1].graph.cuda_graph_config.prefill.backend = "full"
    batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
    assert not model._qwen4_exp_gr_read_tp_enabled(batch)
    context[1].graph.cuda_graph_config.prefill.backend = "disabled"
    context[2].input_scattered = True
    assert not model._qwen4_exp_gr_read_tp_enabled(batch)
