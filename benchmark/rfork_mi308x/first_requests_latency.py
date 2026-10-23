"""Latency of the first requests a fresh server sees (lazy JIT shows up here).

Sends ROUNDS rounds of CONCURRENCY parallel requests with varied prompt lengths and
prints per-round wall time. Usage: python first_requests_latency.py <port>
"""

import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROUNDS = 3
CONCURRENCY = 16
# Prompt sizes picked to hit several prefill shapes; arbitrary.
PROMPT_WORDS = [8, 50, 200, 700, 1500, 3000]


def _request(port: int, words: int) -> None:
    body = {
        "text": " ".join(["hello"] * words) + " Summarize:",
        "sampling_params": {"temperature": 0, "max_new_tokens": 32},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(req, timeout=600).read()


def main() -> None:
    port = int(sys.argv[1])
    for r in range(ROUNDS):
        sizes = [PROMPT_WORDS[(r + i) % len(PROMPT_WORDS)] for i in range(CONCURRENCY)]
        start = time.perf_counter()
        with ThreadPoolExecutor(CONCURRENCY) as pool:
            list(pool.map(lambda w: _request(port, w), sizes))
        print(f"round {r}: {time.perf_counter() - start:.2f} s")


if __name__ == "__main__":
    main()
