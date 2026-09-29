"""Exercise SGLang's adapter against an installed PyHIP package on gfx942.

HIP_VISIBLE_DEVICES=4,5 python -m pytest -q test/registered/kernels/test_gr_read_pyhip.py
"""

import importlib.metadata

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.layers import gr_read_pyhip as adapter
from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig


@pytest.fixture(autouse=True)
def gr_read_environment(monkeypatch):
    if torch.version.hip is None or not torch.cuda.is_available():
        pytest.skip("ROCm GPU required")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0] != "gfx942":
        pytest.skip("PyHIP SGLang integration is enabled on gfx942")
    pytest.importorskip("pyhip.ops.gr_read")
    monkeypatch.setenv("SGLANG_GR_READ_FLYDSL", "1")
    cached_ops = adapter.get_pyhip_ops
    cached_ops.cache_clear()
    yield
    cached_ops.cache_clear()


def make_module():
    return GatedResidual(
        HyperConnectionConfig(hidden_size=2560, hc_lowrank=320), use_combine=False
    ).to(device="cuda", dtype=torch.bfloat16)


def reference(x, wd, wu):
    x = x.double()
    hidden = F.silu(F.linear(x, wd.double()) / 4)
    gates = torch.sigmoid(F.linear(hidden, wu.double())).view(-1, 4, 2560)
    return (gates * x.view(-1, 4, 2560)).mean(1)


def test_installed_public_api():
    import pyhip.ops.gr_read as api

    assert callable(api.gr_read) and callable(api.prepare_weights)
    assert importlib.metadata.version("pyhip")
    assert "/opt/pyhip/src/" not in api.__file__


def test_post_load_packs_once_and_shares_parameter_storage(monkeypatch):
    module = make_module()
    prepare, gr_read = adapter.get_pyhip_ops()
    wd, wu = module.input_mix_weight_down.weight, module.input_mix_weight_up.weight
    expected = prepare(wd, wu)
    calls = []

    def observed_prepare(a, b):
        calls.append((a, b))
        return prepare(a, b)

    monkeypatch.setattr(adapter, "get_pyhip_ops", lambda: (observed_prepare, gr_read))
    module.quant_method.process_weights_after_loading(module)
    module.quant_method.process_weights_after_loading(module)
    assert len(calls) == module._gr_read_pack_count == 1
    assert module.input_mix_weight_down.weight is wd
    assert module.input_mix_weight_up.weight is wu
    assert wd.shape == (320, 10240) and wu.shape == (10240, 320)
    for parameter, packed, expected_packed in zip((wd, wu), module._gr_read_weights, expected):
        assert parameter.data_ptr() == packed.data_ptr()
        assert packed.shape == (320 * 10240,)
        assert torch.equal(packed, expected_packed)


@pytest.mark.parametrize("rows", (0, 1, 17, 32, 33, 513, 2049))
@torch.inference_mode()
def test_mix_uses_installed_operator_and_preserves_residuals(rows):
    torch.manual_seed(1729)
    module = make_module()
    wd = module.input_mix_weight_down.weight.clone()
    wu = module.input_mix_weight_up.weight.clone()
    module.quant_method.process_weights_after_loading(module)
    original = module._gr_read_fn
    calls = []

    def observed(x, pd, pu):
        calls.append((x, pd, pu))
        return original(x, pd, pu)

    module._gr_read_fn = observed
    x = torch.randn((rows, 10240), device="cuda", dtype=torch.bfloat16)
    actual, residuals = module.mix(x)
    assert actual.shape == (rows, 2560) and actual.dtype == torch.bfloat16
    assert residuals[0] is x
    if rows == 0:
        assert not calls and residuals[1] is x
        return
    assert len(calls) == 1 and calls[0][0] is residuals[1]
    assert calls[0][1:] == module._gr_read_weights
    torch.testing.assert_close(actual.double(), reference(residuals[1], wd, wu), rtol=0.01, atol=0.005)


@pytest.mark.parametrize("rows", (17, 33, 513))
@torch.inference_mode()
def test_mix_graph_replay_uses_current_input(rows):
    module = make_module()
    module.quant_method.process_weights_after_loading(module)
    x = torch.randn((rows, 10240), device="cuda", dtype=torch.bfloat16)
    module.mix(x)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, _ = module.mix(x)
    x.mul_(0.75).add_(0.03125)
    expected = module.mix(x)[0]
    graph.replay()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_disabled_and_unsupported_modules_do_not_import_pyhip(monkeypatch):
    def unexpected_import():
        pytest.fail("disabled or unsupported modules must not load PyHIP")

    monkeypatch.setattr(adapter, "get_pyhip_ops", unexpected_import)
    monkeypatch.setenv("SGLANG_GR_READ_FLYDSL", "0")
    disabled = make_module()
    assert not disabled._gr_read_enabled and not hasattr(disabled, "quant_method")
    monkeypatch.setenv("SGLANG_GR_READ_FLYDSL", "1")
    unsupported = GatedResidual(HyperConnectionConfig(), use_combine=False)
    assert not unsupported._gr_read_enabled and not hasattr(unsupported, "quant_method")


def test_missing_package_fails_before_loading(monkeypatch):
    native_import = adapter.importlib.import_module

    def missing(name, *args, **kwargs):
        if name == "pyhip.ops.gr_read":
            raise ModuleNotFoundError(name)
        return native_import(name, *args, **kwargs)

    monkeypatch.setattr(adapter.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="installed PyHIP"):
        make_module()


def test_unprepared_and_packed_weight_lifecycle():
    module = make_module()
    x = torch.zeros((1, 10240), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="loader did not prepare"):
        module.mix(x)
    module.quant_method.process_weights_after_loading(module)
    with pytest.raises(RuntimeError, match="fresh model"):
        module._load_gr_read_weight(module.input_mix_weight_down.weight, torch.zeros_like(module.input_mix_weight_down.weight))
    with pytest.raises(RuntimeError, match="fresh model"):
        module.to(dtype=torch.float32)
    from sglang.srt.model_executor.model_runner_components.weight_updater import (
        _unsupported_derived_weight_cache_error,
    )

    assert "fresh model" in _unsupported_derived_weight_cache_error(module)


def test_warmup_shares_code_across_layers(monkeypatch):
    modules = torch.nn.ModuleList([make_module(), make_module()])
    for module in modules:
        module.quant_method.process_weights_after_loading(module)
    calls = []
    prepare, _ = adapter.get_pyhip_ops()
    monkeypatch.setattr(adapter, "get_pyhip_ops", lambda: (prepare, lambda x, *args: calls.append(x.shape[0])))
    adapter.warmup_pyhip_gr_read(modules, rows=[0, 17, 17, 33])
    assert calls == [17, 33]
    assert all(module._gr_read_pack_count == 1 for module in modules)
