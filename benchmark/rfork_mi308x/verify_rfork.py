"""R-Fork correctness check for one TP size.

Starts a disk-loaded seed (also a Mooncake seed), then an RCCL client and a Mooncake
client in turn; compares every rank's per-tensor checksums (name, shape, dtype, byte
checksum of all params and buffers) against the seed and runs full GSM8K on all three.
Writes results.json, server logs and gsm8k logs under --out.

    python benchmark/rfork_mi308x/verify_rfork.py --tp 4 --seed-gpus 0,1,2,3 \
        --gpus 4,5,6,7 --out /tmp/verify_tp4
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import urllib.request
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO = BENCH_DIR.parent.parent
sys.path.insert(0, str(BENCH_DIR))
from bench_startup import CONFIGS, Server  # noqa: E402


def checksums(directory, tp):
    result = {}
    for rank in range(tp):
        rows = {}
        for line in (
            (Path(directory) / f"target_rank{rank}.tsv").read_text().splitlines()
        ):
            f = line.split("\t")
            rows[tuple(f[:2])] = (f[2], f[3], f[5])
        result[rank] = rows
    return result


def compare(seed, client, tp):
    a, b = checksums(seed, tp), checksums(client, tp)
    out = {}
    for r in range(tp):
        keys = a[r].keys() | b[r].keys()
        diff = [k for k in keys if a[r].get(k) != b[r].get(k)]
        out[r] = {"count": len(keys), "mismatches": len(diff), "examples": diff[:5]}
    return out


def gsm8k(name, port, out):
    log = out / f"gsm8k_{name}.log"
    cmd = [
        sys.executable,
        "-m",
        "sglang.test.run_eval",
        "--eval-name",
        "gsm8k",
        "--num-examples",
        "1319",
        "--max-tokens",
        "16384",
        "--port",
        str(port),
    ]
    with log.open("w") as f:
        p = subprocess.Popen(
            cmd,
            cwd=REPO,
            env=dict(os.environ, PYTHONPATH=str(REPO / "python")),
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = p.wait(timeout=3600)
        finally:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
    text = log.read_text()
    m = re.search(r"\[METRIC\] gsm8k_score=([\d.]+)", text)
    if code or not m:
        raise RuntimeError(f"GSM8K failed: {log}")
    return {"score": float(m.group(1)), "log": str(log)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, required=True)
    ap.add_argument("--seed-gpus", required=True)
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seed_gpus = [int(x) for x in a.seed_gpus.split(",")]
    gpus = [int(x) for x in a.gpus.split(",")]
    ports = json.dumps(list(range(62100, 62100 + a.tp)))
    results, active = {}, []

    def save():
        (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    def launch(name, g, port, extra):
        print(f"starting {name} on GPUs {g}", flush=True)
        s = Server(
            log_path=str(out / f"server_{name}.log"),
            gpus=g,
            port=port,
            args=extra,
            env={
                "MOONCAKE_PROTOCOL": "hip",
                "SGLANG_DEBUG_WEIGHT_CHECKSUM_DIR": str(out / f"ck_{name}"),
            },
            all_visible=True,
        )
        active.append(s)
        if s.ready_epoch is None:
            raise RuntimeError(f"{name} failed to start")
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=30)
        return s

    try:
        launch(
            "disk",
            seed_gpus,
            7090,
            ["--remote-instance-weight-loader-start-seed-via-transfer-engine"],
        )
        results["disk"] = {"gsm8k": gsm8k("disk", 7090, out)}
        save()
        for name in ["rfork_nccl", "rfork_mooncake"]:
            extra = [
                x.replace("{seed_port}", "7090").replace("{group_ports}", ports)
                for x in CONFIGS[name][1]
            ]
            c = launch(name, gpus, 7091, extra)
            results[name] = {
                "checksums": compare(out / "ck_disk", out / f"ck_{name}", a.tp)
            }
            save()
            print(
                name,
                {r: v["mismatches"] for r, v in results[name]["checksums"].items()},
                flush=True,
            )
            results[name]["gsm8k"] = gsm8k(name, 7091, out)
            save()
            c.stop()
            active.remove(c)
    finally:
        for s in reversed(active):
            s.stop()
    print(json.dumps({k: v["gsm8k"]["score"] for k, v in results.items()}), flush=True)


if __name__ == "__main__":
    main()
