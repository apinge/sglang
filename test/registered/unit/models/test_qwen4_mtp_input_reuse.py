"""Qwen4 MTP prompt reuse across the current Eagle prefill boundary."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.qwen4_exp_mtp import (
    Qwen4ExpForCausalLMMTP,
    _add_embeds_to_streams,
)
from sglang.srt.speculative import eagle_worker_v2
from sglang.test.ci.ci_register import register_amd_ci, register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_amd_ci(est_time=10, stage="stage-b", runner_config="1-gpu-small-amd")


@pytest.mark.parametrize("lengths", [(3, 2), (1, 3)])
def test_prompt_embeddings_shift_once_and_repair_each_tail(monkeypatch, lengths):
    tokens = sum(lengths)
    ids = torch.arange(10, 10 + tokens)
    original = torch.arange(tokens * 2, dtype=torch.float32).reshape(tokens, 2)
    original_copy = original.clone()
    tails = torch.tensor([101, 202])
    starts = torch.tensor([0, lengths[0]])
    batch = SimpleNamespace(
        input_ids=ids.clone(),
        extend_lens=list(lengths),
        forward_mode=ForwardMode.EXTEND,
    )
    captured = []

    class ForwardReached(Exception):
        pass

    def capture(forward_batch):
        captured.append(forward_batch)
        raise ForwardReached

    def make_forward_batch(batch, *args, **kwargs):
        return SimpleNamespace(
            input_ids=batch.input_ids,
            forward_mode=batch.forward_mode,
            spec_info=batch.spec_info,
            extend_start_loc=starts,
            extend_seq_lens=torch.tensor(lengths),
            mm_input_embeds=None,
            contains_mm_inputs=lambda: False,
        )

    monkeypatch.setattr(
        eagle_worker_v2, "_eagle_prefill_tail_tokens", lambda *args: tails
    )
    monkeypatch.setattr(eagle_worker_v2.ForwardBatch, "init_new", make_forward_batch)
    worker = SimpleNamespace(
        speculative_algorithm=SimpleNamespace(is_standalone=lambda: False),
        draft_runner=SimpleNamespace(canary_manager=None, forward=capture),
        seed_dsa_topk_from_draft_extend=False,
    )
    with pytest.raises(ForwardReached):
        eagle_worker_v2.EagleDraftWorker._draft_extend_for_prefill(
            worker, batch, torch.zeros(tokens, 2), tails, original
        )

    forward_batch = captured[0]
    expected_ids = torch.cat(
        [ids[1 : lengths[0]], tails[:1], ids[lengths[0] + 1 :], tails[1:]]
    )
    expected = torch.cat(
        [
            original[1 : lengths[0]],
            torch.zeros(1, 2),
            original[lengths[0] + 1 :],
            torch.zeros(1, 2),
        ]
    )
    torch.testing.assert_close(forward_batch.input_ids, expected_ids)
    torch.testing.assert_close(forward_batch.mm_input_embeds, expected)
    torch.testing.assert_close(original, original_copy)

    def embed(token_ids):
        return torch.stack((token_ids, -token_ids), dim=-1).float()

    model = SimpleNamespace(model=SimpleNamespace(embed_tokens=embed))
    result = Qwen4ExpForCausalLMMTP._prepare_input_embeds(
        model, forward_batch.input_ids, forward_batch, None
    )
    expected[torch.tensor(lengths).cumsum(0) - 1] = embed(tails)
    torch.testing.assert_close(result, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU kernel required")
@pytest.mark.parametrize("tokens,heads,hidden", [(256, 4, 2560), (257, 3, 129)])
def test_mtp_broadcast_add_matches_reference_and_graph(tokens, heads, hidden):
    torch.manual_seed(51)
    streams = torch.randn(tokens, heads, hidden, dtype=torch.bfloat16, device="cuda")
    embeds = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda")
    initial = streams.clone()
    expected = (initial.float() + embeds[:, None].float()).to(torch.bfloat16)
    _add_embeds_to_streams(streams, embeds)
    torch.testing.assert_close(streams, expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _add_embeds_to_streams(streams, embeds)
    streams.copy_(initial)
    embeds.neg_()
    graph.replay()
    torch.cuda.synchronize()
    expected = (initial.float() + embeds[:, None].float()).to(torch.bfloat16)
    torch.testing.assert_close(streams, expected, rtol=0, atol=0)
