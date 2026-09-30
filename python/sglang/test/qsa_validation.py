"""Opt-in numerical checks for the direct PyHIP QSA integration."""

import logging
import math
import os
from pathlib import Path

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def check_attention(*, q, k, v, indices, output, query_lens, prefix_lens, scale):
    from pyhip.testing.qsa_reference import reference

    expected = reference(
        q, k, v, indices, query_lens=query_lens, prefix_lens=prefix_lens, scale=scale
    )
    torch.testing.assert_close(output.float(), expected, rtol=0.02, atol=0.02)


def _selection_valid(*, logits, lengths, actual, expected, real):
    """Judge actual selections against logits; native output only supplies the tail ABI."""
    rows, width = logits.shape
    assert actual.shape == expected.shape == (rows, 2051)
    assert actual.dtype == expected.dtype == torch.int32
    live = torch.arange(width, device=logits.device)[None, :] < lengths[:, None]
    values = logits.masked_fill(~live, -math.inf)
    scale = logits.masked_fill(~live, 0).abs().amax(1).clamp_min(1e-30)
    limit = lengths.long()[:, None] * 4
    bad = (~torch.isfinite(logits) & live).any(1)
    block_live = (
        torch.arange(512, device=logits.device)[None, :]
        < lengths.clamp_max(512)[:, None]
    )
    offsets = torch.arange(4, device=logits.device)
    tokens = actual.long()
    quartets = tokens[:, :2048].reshape(rows, 512, 4)
    canonical = (quartets[:, :, :1] // 4) * 4 + offsets
    bad |= ((quartets != canonical).any(2) & block_live).any(1) | (tokens < -1).any(1)
    complete = (tokens >= 0) & (tokens < limit)
    hits = torch.zeros((rows, width), dtype=torch.int32, device=logits.device)
    hits.scatter_add_(
        1, torch.where(complete, tokens // 4, 0).clamp_max(width - 1), complete.int()
    )
    selected = hits > 0
    gap = (
        values.masked_fill(selected, -math.inf).amax(1)
        - values.masked_fill(~selected, math.inf).amin(1)
    ) / scale
    bad |= (
        (selected & (hits != 4)).any(1)
        | (selected.sum(1) != lengths.clamp_max(512))
        | (gap > 1e-5)
        | ((lengths > 512) & ~torch.isfinite(gap))
    )
    tails = [
        torch.where(tokens >= limit, tokens, -1).sort(1).values
        for tokens in (actual, expected)
    ]
    bad |= (tails[0] != tails[1]).any(1)
    return ~(bad & real).any()


def _prefill_failure(
    *, scores, counts, actual, expected, q, keys, rows, seq_lens, extend_lens
):
    real = torch.ones(len(rows), device=q.device, dtype=torch.bool)
    actual_valid = _selection_valid(
        logits=scores, lengths=counts, actual=actual, expected=actual, real=real
    )
    native_valid = _selection_valid(
        logits=scores, lengths=counts, actual=expected, expected=expected, real=real
    )
    directory = envs.SGLANG_TORCH_PROFILER_DIR.get()
    if directory:
        path = Path(directory) / f"qsa_indexer_failure_{os.getpid()}_{int(rows[0])}.pt"
        with path.open("xb") as stream:
            torch.save(
                dict(
                    q=q.cpu(),
                    keys=keys.cpu(),
                    logits=scores.cpu(),
                    lengths=counts.cpu(),
                    actual=actual.cpu(),
                    expected=expected.cpu(),
                    rows=rows.cpu(),
                    seq_lens=seq_lens,
                    extend_lens=extend_lens,
                ),
                stream,
            )
        logger.error("QSA indexer validation inputs saved to %s", path)
    raise AssertionError(
        f"FP64 top-k boundary: pyhip_valid={bool(actual_valid)}, native_valid={bool(native_valid)}"
    )


def _prefill_selection(*, actual, expected, q, keys, seq_lens, extend_lens):
    different = (actual.sort(1).values != expected.sort(1).values).any(1)
    row_base = key_base = 0
    for length, extend in zip(seq_lens, extend_lens):
        rows = different[row_base : row_base + extend].nonzero().flatten() + row_base
        for first in range(0, rows.numel(), 64):
            chosen_rows = rows[first : first + 64]
            key_count = length // 4
            matrix = keys[key_base : key_base + key_count, 0].double().T
            scores = torch.zeros(
                (chosen_rows.numel(), key_count), dtype=torch.float64, device=q.device
            )
            for head in range(4):
                scores += (q[chosen_rows, head].double() @ matrix).relu()
            scores /= math.sqrt(128)
            counts = (
                ((chosen_rows - row_base + length - extend + 1) // 4)
                .clamp_max(key_count)
                .int()
            )
            valid = _selection_valid(
                logits=scores,
                lengths=counts,
                actual=actual[chosen_rows],
                expected=expected[chosen_rows],
                real=torch.ones(chosen_rows.numel(), device=q.device, dtype=torch.bool),
            )
            if not bool(valid):
                _prefill_failure(
                    scores=scores,
                    counts=counts,
                    actual=actual[chosen_rows],
                    expected=expected[chosen_rows],
                    q=q[chosen_rows],
                    keys=matrix.T,
                    rows=chosen_rows,
                    seq_lens=seq_lens,
                    extend_lens=extend_lens,
                )
        row_base += extend
        key_base += length // 4


class IndexerValidation:
    def __init__(self):
        self.prefill_layouts = set()
        self.prefill_checks = 0
        self.decode_checks = None

    def prefill(
        self, *, indexer, runtime, hidden, positions, logical, metadata, inputs
    ):
        q_ref, token_k, stored = indexer.project_qk(
            hidden,
            positions,
            pool=metadata.token_to_kv_pool,
            cache_loc=inputs["state_slots"],
        )
        indexer.update_key_state_and_compress(
            token_k,
            logical,
            positions,
            metadata,
            state_slots=inputs["state_slots"],
            state_stored=stored,
        )
        keys, starts, ends, lengths = metadata.get_prefill_mqa_inputs(
            indexer.layer_id, logical
        )
        row_lengths = lengths.index_select(0, metadata.token_to_batch_idx.long())
        expected = indexer.select_prefill_tokens(
            q_ref, keys, starts, ends, logical, row_lengths
        )
        ring = (
            metadata.req_pool_indices.long()[:, None] * 4
            + torch.arange(4, device=hidden.device)
        ).flatten()
        written = inputs["write_locs"][inputs["write_locs"] != 0].long()
        reference = (
            inputs["key_state"][ring],
            inputs["rope_state"][ring],
            inputs["compressed"][written],
        )
        actual, q, packed = runtime._prefill(**inputs)
        torch.testing.assert_close(q, q_ref, rtol=0, atol=0)
        torch.testing.assert_close(packed[: keys.shape[0]], keys[:, 0], rtol=0, atol=0)
        current = (
            inputs["key_state"][ring],
            inputs["rope_state"][ring],
            inputs["compressed"][written],
        )
        for value, wanted in zip(current, reference):
            torch.testing.assert_close(value, wanted, rtol=0, atol=0)
        _prefill_selection(
            actual=actual,
            expected=expected,
            q=q_ref,
            keys=keys,
            seq_lens=inputs["seq_lens"],
            extend_lens=inputs["extend_lens"],
        )
        self.prefill_layouts.add((inputs["seq_lens"], inputs["extend_lens"]))
        self.prefill_checks += 1
        logger.info(
            "PyHIP QSA indexer checked: layer=%s prefill=%s decode=%s rows=%s",
            indexer.layer_id,
            self.prefill_checks,
            0 if self.decode_checks is None else int(self.decode_checks.item()),
            len(actual),
        )
        return actual

    def decode(self, *, indexer, runtime, hidden, positions, logical, metadata, inputs):
        if self.decode_checks is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm PyHIP QSA validation before capture")
            self.decode_checks = torch.zeros(
                (), dtype=torch.int64, device=hidden.device
            )
        q_ref, token_k, stored = indexer.project_qk(
            hidden,
            positions,
            pool=metadata.token_to_kv_pool,
            cache_loc=inputs["state_slots"],
        )
        indexer.update_key_state_and_compress(
            token_k,
            logical,
            positions,
            metadata,
            state_slots=inputs["state_slots"],
            state_stored=stored,
        )
        expected = indexer.select_decode_tokens(
            q_ref,
            inputs["cache"],
            inputs["page_table"],
            inputs["lengths"],
            inputs["page_table"].shape[1] * 16,
            inputs["query_positions"],
            inputs["sequence_lengths"],
        )
        slots, locs = inputs["state_slots"], inputs["write_locs"].long()
        reference = (
            q_ref,
            inputs["key_state"][slots],
            inputs["rope_state"][slots],
            inputs["compressed"][locs],
        )
        actual, q, logits = runtime._decode_forward(**inputs)
        current = (
            q,
            inputs["key_state"][slots],
            inputs["rope_state"][slots],
            inputs["compressed"][locs],
        )
        real = slots >= 4
        valid = torch.ones((), dtype=torch.bool, device=hidden.device)
        for field, (value, wanted) in enumerate(zip(current, reference)):
            mask = real & (locs != 0) if field == 3 else real
            valid &= ~((value != wanted).flatten(1).any(1) & mask).any()
        valid &= _selection_valid(
            logits=logits,
            lengths=inputs["lengths"],
            actual=actual,
            expected=expected,
            real=real,
        )
        torch._assert_async(
            valid, "PyHIP QSA graph decode differs from native prep or selection"
        )
        self.decode_checks.add_(1)
        return actual
