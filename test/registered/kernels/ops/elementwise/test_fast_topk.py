import pytest
import torch

from sglang.kernels.ops.elementwise.fast_topk import fast_topk
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")


def _check_topk_values(score, lengths, indices, topk, row_starts):
    """The returned indices must select exactly the top-k value multiset.

    Order is unspecified and tie-breaking may differ from torch.topk, so we
    compare sorted score values, not index sets.
    """
    for b in range(score.shape[0]):
        start = int(row_starts[b]) if row_starts is not None else 0
        length = int(lengths[b])
        section = score[b, start : start + length]
        row = indices[b]
        if length <= topk:
            # naive path: identity indices, then -1 fill
            assert torch.equal(
                row[:length].cpu(), torch.arange(length, dtype=torch.int32)
            )
            assert (row[length:] == -1).all()
            continue
        assert (row >= 0).all(), "long rows must fill every slot"
        assert (row < length).all(), "selected indices must stay inside the row"
        assert row.unique().numel() == topk, "selected indices must be unique"
        picked = section[row.long()]
        expected = torch.topk(section, topk).values
        assert torch.equal(
            picked.sort(descending=True).values, expected.sort(descending=True).values
        ), f"row {b}: top-{topk} value multiset mismatch"


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize(
    "batch,length",
    [
        (1, 513),
        (1, 1024),
        (1, 4096),
        (4, 8192),
        (2, 16384),
        (7, 3000),
        (33, 32768),
        (1, 65504),
        (128, 2050),
    ],
)
def test_fast_topk_long_rows(topk, batch, length):
    torch.manual_seed(0)
    score = torch.randn(batch, length, dtype=torch.float32, device="cuda")
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")

    indices = fast_topk(score, lengths, topk)
    _check_topk_values(score, lengths, indices, topk, None)


@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_short_and_mixed_rows(topk):
    torch.manual_seed(0)
    max_len = topk + 128
    batch = 8
    score = torch.randn(batch, max_len, dtype=torch.float32, device="cuda")
    # rows shorter than k (naive path), exactly k, and longer than k
    lens = [0, 1, topk // 3, topk - 1, topk, topk + 1, topk + 7, max_len]
    lengths = torch.tensor(lens[:batch], dtype=torch.int32, device="cuda")

    indices = fast_topk(score, lengths, topk)
    _check_topk_values(score, lengths, indices, topk, None)


@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_ragged_with_row_starts(topk):
    torch.manual_seed(0)
    batch, width = 16, 8192
    score = torch.randn(batch, width, dtype=torch.float32, device="cuda")
    row_starts = torch.randint(0, 2048, (batch,), dtype=torch.int32, device="cuda")
    lengths = torch.randint(1, 2048, (batch,), dtype=torch.int32, device="cuda")
    lengths = torch.minimum(lengths, width - row_starts).to(torch.int32)
    # ensure some rows are longer than k
    lengths[0] = min(width - int(row_starts[0]), topk + 100)

    indices = fast_topk(score, lengths, topk, row_starts=row_starts)
    _check_topk_values(score, lengths, indices, topk, row_starts)


@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_row_stride(topk):
    torch.manual_seed(0)
    batch, length = 8, 4096
    base = torch.randn(batch, 2 * length, dtype=torch.float32, device="cuda")
    score = base[:, :length]  # stride(0) == 2*length, stride(1) == 1
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")

    indices = fast_topk(score, lengths, topk)
    _check_topk_values(score, lengths, indices, topk, None)


@pytest.mark.parametrize("batch", [1, 4, 8])
@pytest.mark.parametrize("length", [1024, 3000, 4096, 16384])
def test_fast_topk_no_row_starts_cuda_graph(batch, length):
    """The nullptr row-start specialization must remain graph replay safe."""
    torch.manual_seed(0)
    topk = 512
    score = torch.randn(batch, length, dtype=torch.float32, device="cuda")
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")

    # Compile and initialize allocator state before capture.
    fast_topk(score, lengths, topk)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        indices = fast_topk(score, lengths, topk)

    for seed in (1, 2):
        score.copy_(
            torch.randn(
                score.shape,
                dtype=score.dtype,
                device=score.device,
                generator=torch.Generator(device="cuda").manual_seed(seed),
            )
        )
        graph.replay()
        torch.cuda.synchronize()
        _check_topk_values(score, lengths, indices, topk, None)


@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_cuda_graph_dynamic_lengths(topk):
    """Graph replay must read current lengths and cross naive/radix paths safely."""
    torch.manual_seed(0)
    batch, width = 4, 16384
    score = torch.randn(batch, width, dtype=torch.float32, device="cuda")
    lengths = torch.full((batch,), width, dtype=torch.int32, device="cuda")

    # Compile and initialize allocator state before capture.
    fast_topk(score, lengths, topk)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        indices = fast_topk(score, lengths, topk)

    replay_lengths = (
        [0, topk, topk + 1, width],
        [1, topk - 1, 4097, width // 2],
    )
    for seed, current_lengths in enumerate(replay_lengths, start=1):
        score.copy_(
            torch.randn(
                score.shape,
                dtype=score.dtype,
                device=score.device,
                generator=torch.Generator(device="cuda").manual_seed(seed),
            )
        )
        lengths.copy_(torch.tensor(current_lengths, dtype=torch.int32, device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        _check_topk_values(score, lengths, indices, topk, None)


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize(
    "fill",
    [
        "binary",  # only 0s and 1s: extreme duplication at the threshold bin
        "few_levels",  # a handful of distinct levels incl. negatives
        "constant",  # whole rows of one value
    ],
)
def test_fast_topk_duplicate_heavy(topk, fill):
    torch.manual_seed(0)
    batch, length = 16, 8192
    if fill == "binary":
        score = torch.randint(0, 2, (batch, length), dtype=torch.float32, device="cuda")
    elif fill == "few_levels":
        levels = torch.tensor([-5.0, -1.0, 0.0, 0.5, 2.0], device="cuda")
        score = levels[torch.randint(0, 5, (batch, length), device="cuda")]
    else:
        score = torch.full((batch, length), 3.25, dtype=torch.float32, device="cuda")
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")

    indices = fast_topk(score, lengths, topk)
    _check_topk_values(score, lengths, indices, topk, None)


@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_negative_and_zero(topk):
    torch.manual_seed(0)
    batch, length = 8, 16384
    score = torch.randn(batch, length, dtype=torch.float32, device="cuda") * 100
    score[:, : length // 3] = 0.0  # long zero prefix
    score[:, length // 3 : length // 2] = -1e30  # very negative block
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")

    indices = fast_topk(score, lengths, topk)
    _check_topk_values(score, lengths, indices, topk, None)


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("length", [4097, 16384, 30938, 65504])
def test_fast_topk_concentrated_threshold_overflow(topk, length):
    """A threshold bin larger than the staging buffer must remain exact."""
    batch = 2
    values = 10.0 + torch.arange(length, dtype=torch.float32, device="cuda") / length
    score = torch.stack((values, values.flip(0)))
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")

    # All values round to fp16 encodings with the same high byte, so the
    # coarse threshold bin deterministically contains the whole row. Lengths
    # above 4096 must therefore take the exact overflow path.
    fp16_high_bytes = (
        score.to(torch.float16).view(torch.int16).to(torch.int32).bitwise_and(0xFFFF)
        >> 8
    )
    assert torch.unique(fp16_high_bytes).numel() == 1

    indices = fast_topk(score, lengths, topk)
    _check_topk_values(score, lengths, indices, topk, None)


@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_overflow_with_row_starts_and_stride(topk):
    """Overflow fallback must honor nonzero starts and the physical row stride."""
    torch.manual_seed(0)
    batch, width, length = 2, 65536, 30938
    storage = torch.randn(batch, 2 * width, dtype=torch.float32, device="cuda")
    score = storage[:, :width]
    row_starts = torch.tensor([17, 257], dtype=torch.int32, device="cuda")
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")
    for row in range(batch):
        start = int(row_starts[row])
        score[row, start : start + length] = 10.0 + 0.1 * torch.randn(
            length, dtype=torch.float32, device="cuda"
        )

    indices = fast_topk(score, lengths, topk, row_starts=row_starts)
    _check_topk_values(score, lengths, indices, topk, row_starts)


@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_overflow_cuda_graph(topk):
    """The exact full-row fallback must remain safe across graph replay."""
    length = 16384
    score = torch.empty(1, length, dtype=torch.float32, device="cuda")
    lengths = torch.full((1,), length, dtype=torch.int32, device="cuda")

    def update(seed):
        score.copy_(
            10.0
            + 0.1
            * torch.randn(
                score.shape,
                dtype=score.dtype,
                device=score.device,
                generator=torch.Generator(device="cuda").manual_seed(seed),
            )
        )

    update(0)
    fast_topk(score, lengths, topk)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        indices = fast_topk(score, lengths, topk)

    for seed in (1, 2):
        update(seed)
        graph.replay()
        torch.cuda.synchronize()
        _check_topk_values(score, lengths, indices, topk, None)


def test_fast_topk_unsupported_k():
    score = torch.randn(2, 4096, dtype=torch.float32, device="cuda")
    lengths = torch.full((2,), 4096, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="topk"):
        fast_topk(score, lengths, 1024)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
