"""Does a peer copy overlap with GEMM? SDMA engine vs blit kernel.

Each of two GPUs pulls its peer's micro-batch partial sum with a cross-device
copy on a side stream while GEMMs run on the compute stream. Run once as is
(SDMA) and once with HSA_ENABLE_SDMA=0 (blit kernels on the CUs) to compare.

    HIP_VISIBLE_DEVICES=0,1 python benchmark/sdma_tp2/bench_sdma_overlap.py
    HSA_ENABLE_SDMA=0 HIP_VISIBLE_DEVICES=0,1 python benchmark/sdma_tp2/bench_sdma_overlap.py
"""

import torch

HIDDEN, TOKENS, GEMM_N = 2560, 6000, 6656
DEVICES = [torch.device("cuda", 0), torch.device("cuda", 1)]


def _make_state():
    state = {}
    for i, device in enumerate(DEVICES):
        with torch.cuda.device(device):
            state[i] = dict(
                y=torch.randn(TOKENS, HIDDEN, device=device, dtype=torch.bfloat16),
                recv=torch.empty(TOKENS, HIDDEN, device=device, dtype=torch.bfloat16),
                a=torch.randn(TOKENS, HIDDEN, device=device, dtype=torch.bfloat16),
                w=torch.randn(HIDDEN, GEMM_N, device=device, dtype=torch.bfloat16),
                comp=torch.cuda.Stream(device),
                comm=torch.cuda.Stream(device),
            )
    return state


STATE = _make_state()


def _pull(i):
    s = STATE[i]
    with torch.cuda.device(DEVICES[i]), torch.cuda.stream(s["comm"]):
        s["recv"].copy_(STATE[1 - i]["y"], non_blocking=True)


def _gemms(i, n):
    s = STATE[i]
    with torch.cuda.device(DEVICES[i]), torch.cuda.stream(s["comp"]):
        for _ in range(n):
            s["a"] @ s["w"]


def _launch_and_join(fn):
    for i, device in enumerate(DEVICES):
        with torch.cuda.device(device):
            cur = torch.cuda.current_stream()
            STATE[i]["comp"].wait_stream(cur)
            STATE[i]["comm"].wait_stream(cur)
    fn()
    for i, device in enumerate(DEVICES):
        with torch.cuda.device(device):
            cur = torch.cuda.current_stream()
            cur.wait_stream(STATE[i]["comp"])
            cur.wait_stream(STATE[i]["comm"])


def _timed(fn, iters=10):
    """Mean ms per iteration on the slower GPU, both GPUs running concurrently."""
    for _ in range(2):
        _launch_and_join(fn)
    for device in DEVICES:
        torch.cuda.synchronize(device)
    events = []
    for device in DEVICES:
        with torch.cuda.device(device):
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            events.append([start])
    for _ in range(iters):
        _launch_and_join(fn)
    for i, device in enumerate(DEVICES):
        with torch.cuda.device(device):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            events[i].append(end)
    for device in DEVICES:
        torch.cuda.synchronize(device)
    return max(start.elapsed_time(end) for start, end in events) / iters


def main():
    t_pull = _timed(lambda: [_pull(i) for i in (0, 1)])
    nbytes = STATE[0]["y"].numel() * STATE[0]["y"].element_size()
    print(
        f"peer pull {nbytes / 2**20:.1f} MiB: {t_pull:.3f} ms ({nbytes / t_pull / 1e6:.1f} GB/s)"
    )
    print(
        f"{'compute':10s} {'t_comp':>7} {'t_pull':>7} {'t_both':>7} {'serial':>7} {'overlap':>8}"
    )
    for n in (2, 6):
        t_comp = _timed(lambda: [_gemms(i, n) for i in (0, 1)])
        t_both = _timed(lambda: [(_gemms(i, n), _pull(i)) for i in (0, 1)])
        overlap = (t_comp + t_pull - t_both) / min(t_comp, t_pull)
        print(
            f"GEMM x{n:<4d} {t_comp:7.3f} {t_pull:7.3f} {t_both:7.3f} "
            f"{t_comp + t_pull:7.3f} {overlap:8.2f}"
        )


if __name__ == "__main__":
    main()
