"""Opt-in numerical checks for the direct PyHIP QSA integration."""

import logging
import math
import os
from pathlib import Path

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa.metadata import qsa_ring_slots_per_request

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
        self,
        *,
        indexer,
        hidden,
        positions,
        logical,
        metadata,
        state_slots,
        seq_lens,
        extend_lens,
        run,
    ):
        pool = metadata.token_to_kv_pool
        q_ref, token_k, stored = indexer.project_qk(
            hidden,
            positions,
            pool=pool,
            cache_loc=state_slots,
        )
        indexer.update_key_state_and_compress(
            token_k,
            logical,
            positions,
            metadata,
            state_slots=state_slots,
            state_stored=stored,
        )
        keys, starts, ends, lengths = metadata.get_prefill_mqa_inputs(
            indexer.layer_id, logical
        )
        row_lengths = lengths.index_select(0, metadata.token_to_batch_idx.long())
        expected = indexer.select_prefill_tokens(
            q_ref, keys, starts, ends, logical, row_lengths
        )
        ring_size = qsa_ring_slots_per_request(indexer.compress_ratio)
        ring = (
            metadata.req_pool_indices.long()[:, None] * ring_size
            + torch.arange(ring_size, device=hidden.device)
        ).flatten()
        key_state = pool.get_qsa_key_state_buffer(indexer.layer_id)
        rope_state = pool.qsa_rope_position_buffer
        compressed = pool.get_qsa_compressed_k_buffer(indexer.layer_id)
        written = metadata.write_locs[metadata.write_locs != 0].long()
        reference = (key_state[ring], rope_state[ring], compressed[written])
        q = torch.empty_like(q_ref, memory_format=torch.contiguous_format)
        actual = run(q_out=q)
        ratio, table = indexer.compress_ratio, metadata.token_slot_table
        # Keys PyHIP's logits read: a group's key is at its first token's slot // ratio.
        packed = torch.cat(
            [
                compressed[table[s, : n // ratio * ratio : ratio].long() // ratio, 0]
                for s, n in enumerate(seq_lens)
            ]
        )
        torch.testing.assert_close(q, q_ref, rtol=0, atol=0)
        torch.testing.assert_close(packed[: keys.shape[0]], keys[:, 0], rtol=0, atol=0)
        current = (key_state[ring], rope_state[ring], compressed[written])
        for value, wanted in zip(current, reference):
            torch.testing.assert_close(value, wanted, rtol=0, atol=0)
        _prefill_selection(
            actual=actual,
            expected=expected,
            q=q_ref,
            keys=keys,
            seq_lens=seq_lens,
            extend_lens=extend_lens,
        )
        self.prefill_layouts.add((seq_lens, extend_lens))
        self.prefill_checks += 1
        logger.info(
            "PyHIP QSA indexer checked: layer=%s prefill=%s decode=%s rows=%s",
            indexer.layer_id,
            self.prefill_checks,
            0 if self.decode_checks is None else int(self.decode_checks.item()),
            len(actual),
        )
        return actual

    def decode(
        self,
        *,
        indexer,
        hidden,
        positions,
        logical,
        metadata,
        state_slots,
        write_locs,
        cache,
        page_table,
        lengths,
        logical_positions,
        seq_lens,
        run,
    ):
        if self.decode_checks is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm PyHIP QSA validation before capture")
            self.decode_checks = torch.zeros(
                (), dtype=torch.int64, device=hidden.device
            )
        pool = metadata.token_to_kv_pool
        q_ref, token_k, stored = indexer.project_qk(
            hidden,
            positions,
            pool=pool,
            cache_loc=state_slots,
        )
        indexer.update_key_state_and_compress(
            token_k,
            logical,
            positions,
            metadata,
            state_slots=state_slots,
            state_stored=stored,
        )
        rows, width = page_table.shape[0], page_table.shape[1] * 16
        expected = indexer.select_decode_tokens(
            q_ref,
            cache,
            page_table,
            lengths,
            width,
            logical_positions,
            seq_lens,
        )
        key_state = pool.get_qsa_key_state_buffer(indexer.layer_id)
        rope_state = pool.qsa_rope_position_buffer
        compressed = pool.get_qsa_compressed_k_buffer(indexer.layer_id)
        slots, locs = state_slots, write_locs.long()
        reference = (q_ref, key_state[slots], rope_state[slots], compressed[locs])
        q = torch.empty_like(q_ref, memory_format=torch.contiguous_format)
        # PyHIP's top-k may read up to 512 values past the logits.
        logits = torch.empty(
            rows * width + 512, dtype=torch.float32, device=hidden.device
        )[: rows * width].view(rows, width)
        actual = run(q_out=q, logits_out=logits)
        current = (q, key_state[slots], rope_state[slots], compressed[locs])
        real = slots >= qsa_ring_slots_per_request(indexer.compress_ratio)
        valid = torch.ones((), dtype=torch.bool, device=hidden.device)
        for field, (value, wanted) in enumerate(zip(current, reference)):
            mask = real & (locs != 0) if field == 3 else real
            valid &= ~((value != wanted).flatten(1).any(1) & mask).any()
        valid &= _selection_valid(
            logits=logits,
            lengths=lengths,
            actual=actual,
            expected=expected,
            real=real,
        )
        torch._assert_async(
            valid, "PyHIP QSA graph decode differs from native prep or selection"
        )
        self.decode_checks.add_(1)
        return actual
