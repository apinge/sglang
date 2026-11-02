#!/usr/bin/env python3
"""Send one exact-length token-ID request, following /opt/1m.md.

Example (server context must accommodate input plus output):
    python3 test_qwen38_long_request.py 1000000 350 http://127.0.0.1:7080

The default is a cold-cache test: it calls /flush_cache before /generate.
Use --no-flush-cache to retain the server's existing cache.
"""

import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


DEFAULT_PROMPTS = Path(__file__).resolve().parent / "prompts.npz"


def get_info(url, timeout):
    with urllib.request.urlopen(url + "/server_info", timeout=timeout) as response:
        info = json.load(response)
    context = int(info["context_length"])
    limit = info.get("max_req_input_len")
    if limit is None:
        limits = [
            state["max_req_input_len"]
            for state in info.get("internal_states", [])
            if state.get("max_req_input_len") is not None
        ]
        if not limits:
            raise ValueError("/server_info does not expose max_req_input_len.")
        limit = min(limits)
    return context, int(limit)


def post(url, payload, timeout):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=timeout)


def run(args):
    import numpy as np

    n, m = args.input_tokens, args.output_tokens
    if n <= 0 or m <= 0:
        raise ValueError("Input and output token counts must both be positive.")
    url = args.url.rstrip("/")
    context, limit = get_info(url, min(args.timeout, 60))
    print("context_length", context, "max_req_input_len", limit, flush=True)
    if n + m > context:
        raise ValueError(f"input + output {n + m} > context_length {context}")
    if n >= limit:
        raise ValueError(
            f"input {n} >= max_req_input_len {limit}: the KV pool/input limit is too small."
        )
    with np.load(args.prompts_file, allow_pickle=False) as archive:
        chunks = [archive["L999648_r0"], archive["L999648_r1"]]
        if any(array.ndim != 1 or array.dtype.kind not in "iu" for array in chunks):
            raise ValueError("Prompt arrays must be one-dimensional integer token IDs.")
        ids = np.concatenate(chunks)[:n]
    if len(ids) != n:
        raise ValueError(f"Prompt file has only {len(ids)} tokens; requested {n}.")

    if not args.no_flush_cache:
        with post(url + "/flush_cache", {}, min(args.timeout, 60)) as response:
            raw = response.read()
        if raw:
            try:
                status = json.loads(raw)
            except json.JSONDecodeError:
                status = None
            if status is False or (isinstance(status, dict) and status.get("success") is False):
                raise ValueError("Server did not flush the cache; wait until it is idle.")
        print("cache flushed", flush=True)

    payload = {
        "input_ids": ids.tolist(),
        "stream": True,
        "sampling_params": {"max_new_tokens": m, "temperature": 0, "ignore_eos": True},
    }
    # Serialize before timing, as in the colleague's command.
    request = urllib.request.Request(
        url + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_token = last_event = None
    final = None
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            value = line[5:].strip()
            if not value or value == b"[DONE]":
                continue
            timestamp = time.perf_counter()
            data = json.loads(value)
            if data.get("error"):
                raise ValueError(f"Server error: {data['error']}")
            meta = data.get("meta_info")
            if not isinstance(meta, dict):
                continue
            reason = meta.get("finish_reason")
            if isinstance(reason, dict) and reason.get("type") == "abort":
                raise ValueError(f"Request aborted: {reason.get('message', reason)}")
            completed = int(meta.get("completion_tokens", 0))
            if completed >= 1 and first_token is None:
                first_token = timestamp
            final, last_event = meta, timestamp
    if final is None or first_token is None:
        raise ValueError("Response did not contain generated-token metadata.")
    prompt = int(final["prompt_tokens"])
    output = int(final["completion_tokens"])
    if prompt != n or output != m:
        raise ValueError(
            f"Token counts differ: requested input/output {n}/{m}, server returned {prompt}/{output}."
        )
    tpot = f"{(last_event - first_token) / (output - 1) * 1000:.2f} ms" if output > 1 else "n/a"
    message = (
        f"prompt {prompt} cached {final.get('cached_tokens', 0)} output {output}"
        f" | TTFT {first_token - started:.2f} s, TPOT {tpot}, E2E {last_event - started:.2f} s"
    )
    if final.get("spec_verify_ct"):
        message += f" | accept {output / final['spec_verify_ct']:.2f}"
    print(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_tokens", type=int)
    parser.add_argument("output_tokens", type=int)
    parser.add_argument("url", help="Server base URL, e.g. http://127.0.0.1:7080")
    parser.add_argument("--prompts-file", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--no-flush-cache", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except urllib.error.HTTPError as error:
        raise SystemExit(f"HTTP {error.code}: {error.read(4096).decode(errors='replace')}")
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()
