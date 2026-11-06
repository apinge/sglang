"""w13 buffers must be sized by the layer's gating, not assumed to be gate+up fused.

A non-gated MoE (e.g. NemotronH: relu2, checkpoint carries up_proj/down_proj and
no gate_proj) fuses a single projection into w13. Sizing w13 as 2*intermediate
leaves the upper half as uninitialised ``torch.empty`` that no weight loader ever
writes, which silently corrupts quantized MoE weights. The weight scales must
follow the same shard count, or the two describe different tensors.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.test.test_utils import CustomTestCase

NUM_EXPERTS = 4
HIDDEN = 256
INTERMEDIATE = 640
BLOCK_N = 128
BLOCK_K = 128


class _RecordingLayer:
    """Collects the parameters ``create_fp8_moe_weight_`` registers."""

    def __init__(self, is_gated: bool):
        self.moe_runner_config = MoeRunnerConfig(is_gated=is_gated)
        self.params = {}

    def register_parameter(self, name, param):
        self.params[name] = param


def _create_weights(is_gated: bool, block_quant: bool):
    from sglang.srt.layers.quantization import fp8 as fp8_quant

    layer = _RecordingLayer(is_gated)
    quant_config = MagicMock(
        weight_block_size=[BLOCK_N, BLOCK_K],
        activation_scheme="dynamic",
        is_checkpoint_fp8_serialized=False,
    )

    with patch.object(fp8_quant, "get_parallel") as parallel:
        parallel.return_value.tp_size = 1
        fp8_quant.Fp8MoEMethod.create_fp8_moe_weight_(
            layer=layer,
            num_experts=NUM_EXPERTS,
            hidden_size=HIDDEN,
            intermediate_size_per_partition=INTERMEDIATE,
            block_quant=block_quant,
            quant_config=quant_config,
            use_mxfp8=False,
            is_checkpoint_fp8_serialized=False,
            is_fp4_expert=False,
            params_dtype=torch.bfloat16,
        )
    return layer.params


class TestFp8MoEWeightGating(CustomTestCase):
    def test_gated_fuses_gate_and_up(self):
        params = _create_weights(is_gated=True, block_quant=True)
        self.assertEqual(params["w13_weight"].shape[1], 2 * INTERMEDIATE)
        self.assertEqual(
            params["w13_weight_scale_inv"].shape[1], 2 * (INTERMEDIATE // BLOCK_N)
        )

    def test_non_gated_w13_holds_up_only(self):
        # Regression: w13 was always sized 2*intermediate, so the upper half
        # stayed uninitialised for NemotronH.
        params = _create_weights(is_gated=False, block_quant=True)
        self.assertEqual(params["w13_weight"].shape[1], INTERMEDIATE)

    def test_non_gated_block_scale_matches_weight(self):
        params = _create_weights(is_gated=False, block_quant=True)
        weight_rows = params["w13_weight"].shape[1]
        scale_rows = params["w13_weight_scale_inv"].shape[1]
        self.assertEqual(scale_rows * BLOCK_N, weight_rows)

    def test_non_gated_per_tensor_scale_is_single(self):
        # One shard means one scale per expert; nothing to fuse afterwards.
        params = _create_weights(is_gated=False, block_quant=False)
        self.assertEqual(params["w13_weight_scale"].shape, (NUM_EXPERTS, 1))

    def test_compressed_tensors_fp8_ptpc_preserves_aiter_shuffle_marker(self):
        from compressed_tensors.quantization import QuantizationStrategy
        from sglang.srt.layers.quantization.compressed_tensors.schemes import (
            compressed_tensors_w8a8_fp8_moe as compressed_fp8_moe,
        )

        scheme = object.__new__(compressed_fp8_moe.CompressedTensorsW8A8Fp8MoE)
        scheme.static_input_scales = False
        scheme.input_quant = SimpleNamespace(strategy=QuantizationStrategy.TOKEN)
        scheme.weight_quant = SimpleNamespace(strategy=QuantizationStrategy.CHANNEL)
        scheme.use_flashinfer_trtllm = False

        def fake_shuffle(weight, _layout):
            shuffled = weight.clone()
            shuffled.is_shuffled = True
            return shuffled

        def fake_normalize(weight, weight_scale, input_scale):
            return weight, weight_scale, input_scale

        for use_fnuz in (False, True):
            for intermediate in (4, 64):
                with self.subTest(use_fnuz=use_fnuz, intermediate=intermediate):
                    normalizer = MagicMock(side_effect=fake_normalize)
                    layer = SimpleNamespace(
                        w13_weight=torch.nn.Parameter(
                            torch.randn(2, 2 * intermediate, 4)
                        ),
                        w2_weight=torch.nn.Parameter(torch.randn(2, 4, intermediate)),
                        w13_weight_scale=torch.nn.Parameter(
                            torch.ones(2, 2 * intermediate, 1)
                        ),
                        w2_weight_scale=torch.nn.Parameter(torch.ones(2, 4, 1)),
                        w13_input_scale=None,
                        w2_input_scale=None,
                    )
                    with (
                        patch.object(
                            compressed_fp8_moe,
                            "will_use_aiter_moe",
                            return_value=True,
                        ),
                        patch.object(
                            compressed_fp8_moe,
                            "is_fp8_fnuz",
                            return_value=use_fnuz,
                        ),
                        patch.object(
                            compressed_fp8_moe,
                            "normalize_e4m3fn_to_e4m3fnuz",
                            normalizer,
                        ),
                        patch.object(
                            compressed_fp8_moe,
                            "_get_aiter_ptpc_padding_size",
                            return_value=64,
                        ),
                        patch.object(
                            compressed_fp8_moe,
                            "_get_aiter_shuffle_weight",
                            return_value=fake_shuffle,
                        ),
                        patch.object(torch.cuda, "empty_cache"),
                    ):
                        scheme.process_weights_after_loading(layer)

                    self.assertEqual(normalizer.call_count, 2 if use_fnuz else 0)
                    self.assertTrue(layer.w13_weight.is_shuffled)
                    self.assertTrue(layer.w2_weight.is_shuffled)

    def test_compressed_tensors_fp8_ptpc_skips_aiter_shuffle_for_triton(self):
        from compressed_tensors.quantization import QuantizationStrategy
        from sglang.srt.layers.quantization.compressed_tensors.schemes import (
            compressed_tensors_w8a8_fp8_moe as compressed_fp8_moe,
        )

        scheme = object.__new__(compressed_fp8_moe.CompressedTensorsW8A8Fp8MoE)
        scheme.static_input_scales = False
        scheme.input_quant = SimpleNamespace(strategy=QuantizationStrategy.TOKEN)
        scheme.weight_quant = SimpleNamespace(strategy=QuantizationStrategy.CHANNEL)
        scheme.use_flashinfer_trtllm = False
        layer = SimpleNamespace(
            w13_weight=torch.nn.Parameter(torch.randn(2, 8, 4)),
            w2_weight=torch.nn.Parameter(torch.randn(2, 4, 4)),
            w13_weight_scale=torch.nn.Parameter(torch.ones(2, 8, 1)),
            w2_weight_scale=torch.nn.Parameter(torch.ones(2, 4, 1)),
            w13_input_scale=None,
            w2_input_scale=None,
        )

        with (
            patch.object(
                compressed_fp8_moe,
                "will_use_aiter_moe",
                return_value=False,
            ),
            patch.object(compressed_fp8_moe, "is_fp8_fnuz", return_value=False),
            patch.object(
                compressed_fp8_moe,
                "_get_aiter_shuffle_weight",
                side_effect=AssertionError("Triton must not request AITER shuffle"),
            ),
        ):
            scheme.process_weights_after_loading(layer)

        self.assertFalse(hasattr(layer.w13_weight, "is_shuffled"))
        self.assertFalse(hasattr(layer.w2_weight, "is_shuffled"))

    def test_compressed_tensors_fp8_ptpc_allocation_uses_effective_runner(self):
        from compressed_tensors.quantization import QuantizationStrategy
        from sglang.srt.layers.quantization.compressed_tensors.schemes import (
            compressed_tensors_w8a8_fp8_moe as compressed_fp8_moe,
        )

        scheme = object.__new__(compressed_fp8_moe.CompressedTensorsW8A8Fp8MoE)
        scheme.static_input_scales = False
        scheme.input_quant = SimpleNamespace(strategy=QuantizationStrategy.TOKEN)
        scheme.weight_quant = SimpleNamespace(strategy=QuantizationStrategy.CHANNEL)
        scheme.weight_block_size = None
        scheme.block_quant = False

        for use_aiter_moe in (False, True):
            with self.subTest(use_aiter_moe=use_aiter_moe):
                layer = torch.nn.Module()
                get_sizes = MagicMock(return_value=(8, 4, False))
                with (
                    patch.object(
                        compressed_fp8_moe,
                        "will_use_aiter_moe",
                        return_value=use_aiter_moe,
                    ),
                    patch.object(
                        compressed_fp8_moe,
                        "get_moe_weight_sizes",
                        get_sizes,
                    ),
                ):
                    scheme.create_weights(
                        layer=layer,
                        num_experts=2,
                        hidden_size=4,
                        intermediate_size_per_partition=4,
                        params_dtype=torch.bfloat16,
                    )

                get_sizes.assert_called_once_with(
                    4,
                    is_aiter_moe=False,
                    is_concat=True,
                    is_packed=False,
                )


if __name__ == "__main__":
    unittest.main()
