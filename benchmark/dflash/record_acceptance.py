"""Record measured /generate responses without changing a workload's requests.

The workload owns prompt construction, sampling, warmup, scheduling and scoring.
This wrapper excludes its warmup responses and checks the recorded totals against
the workload's own metrics before writing acceptance.json and raw responses.
"""

import argparse
import functools
import importlib.util
import json
import os
import sys
import threading
from collections import Counter
from pathlib import Path


def _counter(meta, key):
    value = meta.get(key)
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise RuntimeError(f"Invalid {key} counter: {value!r}")
    return value


def summarize(outputs):
    metas = [out.get("meta_info", {}) or {} for out in outputs]
    pairs = []
    verifies = []
    completions = []
    for meta in metas:
        accepted = _counter(meta, "spec_num_correct_drafts")
        proposed = _counter(meta, "spec_num_proposed_drafts")
        if accepted is not None and proposed is not None:
            if accepted > proposed:
                raise RuntimeError("Accepted drafts exceed proposed drafts")
            pairs.append((accepted, proposed))
        verify = _counter(meta, "spec_verify_ct")
        completion = _counter(meta, "completion_tokens")
        if verify is not None:
            verifies.append(verify)
        if completion is None:
            raise RuntimeError("Missing completion_tokens in server response")
        completions.append(completion)

    complete = bool(outputs) and len(pairs) == len(outputs)
    verify_complete = bool(outputs) and len(verifies) == len(outputs)
    accepted_sum = sum(a for a, _ in pairs) if complete else None
    proposed_sum = sum(p for _, p in pairs) if complete else None
    verify_sum = sum(verifies) if verify_complete else None
    completion_sum = sum(completions)
    reasons = Counter()
    for meta in metas:
        reason = meta.get("finish_reason")
        reasons[
            reason.get("type", "unknown") if isinstance(reason, dict) else str(reason)
        ] += 1
    return {
        "requests": len(outputs),
        "requests_with_draft_counters": len(pairs),
        "requests_with_verify_counters": len(verifies),
        "accepted_draft_tokens": accepted_sum,
        "proposed_draft_tokens": proposed_sum,
        "spec_accept_rate": accepted_sum / proposed_sum
        if complete and proposed_sum
        else None,
        "spec_verify_ct_sum": verify_sum,
        "output_tokens": completion_sum,
        "spec_accept_length_weighted": completion_sum / verify_sum
        if verify_sum
        else None,
        "finish_reason_counts": dict(reasons),
    }


def instrument(mod, result_dir):
    """Install response capture on one of the bundled workload modules."""
    lock = threading.Lock()
    active = None
    summaries = []
    original_send = mod._send_generate

    @functools.wraps(original_send)
    def send(*args, **kwargs):
        outputs = original_send(*args, **kwargs)
        if active is not None:
            with lock:
                active.extend(outputs)
        return outputs

    mod._send_generate = send
    run_name = (
        "_run_gsm8k_requests"
        if hasattr(mod, "_run_gsm8k_requests")
        else "_run_requests"
    )
    original_run = getattr(mod, run_name)

    @functools.wraps(original_run)
    def run(*args, **kwargs):
        nonlocal active
        base_url = args[0] if args else kwargs["base_url"]
        response = mod.requests.get(base_url.rstrip("/") + "/server_info", timeout=60)
        response.raise_for_status()
        info = response.json()
        server_config = {
            key: info.get(key)
            for key in (
                "model_path",
                "speculative_draft_model_path",
                "speculative_algorithm",
                "speculative_num_draft_tokens",
                "tp_size",
                "attention_backend",
                "speculative_draft_attention_backend",
                "disable_radix_cache",
                "page_size",
                "kv_cache_dtype",
                "mamba_radix_cache_strategy",
                "max_running_requests",
                "chunked_prefill_size",
                "mem_fraction_static",
            )
        }
        if (
            kwargs.get("expect_dflash")
            and server_config["speculative_algorithm"] != "DFLASH"
        ):
            raise RuntimeError("The running server is not configured for DFLASH")
        outputs = []
        active = outputs
        try:
            metrics = original_run(*args, **kwargs)
        finally:
            active = None
        concurrency = int(kwargs["concurrency"])
        warmup = concurrency if len(kwargs["prompts"]) > concurrency else 0
        measured = outputs[warmup:]
        expected = len(kwargs["prompts"]) - warmup
        if len(measured) != expected:
            raise RuntimeError(
                "Captured response count disagrees with requested prompts"
            )
        summary = summarize(measured)
        if summary["output_tokens"] != metrics.output_tokens:
            raise RuntimeError("Captured token count disagrees with benchmark")
        if kwargs.get("expect_dflash"):
            if summary["spec_accept_rate"] is None or not summary["spec_verify_ct_sum"]:
                raise RuntimeError(
                    "Incomplete DFlash acceptance counters in server responses"
                )
        if (
            summary["spec_verify_ct_sum"] is not None
            and summary["spec_verify_ct_sum"] != metrics.spec_verify_ct_sum
        ):
            raise RuntimeError("Captured verification count disagrees with benchmark")
        summary.update(
            workload=mod.WORKLOAD,
            concurrency=concurrency,
            batch_requests=bool(kwargs["batch_requests"]),
            warmup_requests_excluded=warmup,
            max_new_tokens=kwargs["max_new_tokens"],
            sampling={key: kwargs[key] for key in ("temperature", "top_p", "top_k")},
            accuracy=getattr(metrics, "accuracy", None),
            output_toks_per_s=metrics.output_toks_per_s,
            latency_s=metrics.latency_s,
            server_version=info.get("version"),
            server_config=server_config,
        )
        result_dir.mkdir(parents=True, exist_ok=True)
        number = len(summaries) + 1
        output_path = result_dir / f"run{number}_responses.jsonl"
        with output_path.open("w", encoding="utf-8") as f:
            for output in measured:
                f.write(json.dumps(output, ensure_ascii=False) + "\n")
        summary["responses_file"] = output_path.name
        summaries.append(summary)
        (result_dir / "acceptance.json").write_text(
            json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
        )
        print("[acceptance] " + json.dumps(summary, ensure_ascii=False), flush=True)
        return metrics

    setattr(mod, run_name, run)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--benchmark", required=True)
    opts, remaining = parser.parse_known_args()
    path = Path(opts.benchmark).resolve()
    spec = importlib.util.spec_from_file_location("dflash_workload", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    instrument(mod, Path(os.environ.get("ACCEPTANCE_DIR", "results/dflash/acceptance")))
    sys.argv = [str(path), *remaining]
    mod.main()


if __name__ == "__main__":
    main()
