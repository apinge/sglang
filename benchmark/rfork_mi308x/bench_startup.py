"""Startup-time benchmark: launch the server under each optimization on/off combination
and report per-phase timings.

Every measured instance runs on --gpus (default 4-7). R-Fork configs first start one seed
on --seed-gpus (default 0-3) and leave it running until they are done.

    # full matrix, 1 repeat
    python benchmark/rfork_mi308x/bench_startup.py --out /tmp/startup
    # pick configs and cache states
    python benchmark/rfork_mi308x/bench_startup.py --configs disk,rfork_nccl \\
        --kernel-cache warm --page-cache warm --repeat 2 --out /tmp/startup

Kernel cache: "warm" uses $HOME as is (run once beforehand to populate it, or pass
--prime); "cold" gives each run an empty HOME, so FlyDSL/comgr/triton caches start empty.
Page cache: "cold" drops only the model files' page cache (posix_fadvise) before the run;
R-Fork configs do not read the checkpoint, so they only run "warm".
"""

import argparse
import csv
import datetime
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import time

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(BENCH_DIR))
LAUNCH_SCRIPT = os.path.join(
    REPO_DIR, "scripts", "launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh"
)
sys.path.insert(0, BENCH_DIR)
from evict_model_cache import evict, resident_bytes  # noqa: E402

# name -> (extra env, extra server args, description). "{seed_port}" and
# "{group_ports}" are filled in at launch time.
CONFIGS = {
    "disk": ({}, [], "default loader (multi-thread safetensors, mmap)"),
    "disk_prefetch": (
        {},
        ["--weight-loader-prefetch-checkpoints"],
        "default + prefetch",
    ),
    "fastsafetensors": (
        {},
        [
            "--load-format",
            "fastsafetensors",
            "--model-loader-extra-config",
            '{"enable_gds": false}',
        ],
        "fastsafetensors, GDS off",
    ),
    "runai": (
        {"RUNAI_STREAMER_FS_STRATEGY": "libaio_direct"},
        ["--load-format", "runai_streamer"],
        "runai streamer, libaio_direct",
    ),
    "rfork_nccl": (
        {},
        [
            "--load-format",
            "remote_instance",
            "--remote-instance-weight-loader-backend",
            "nccl",
            "--remote-instance-weight-loader-seed-instance-ip",
            "127.0.0.1",
            "--remote-instance-weight-loader-seed-instance-service-port",
            "{seed_port}",
            "--remote-instance-weight-loader-send-weights-group-ports",
            "{group_ports}",
        ],
        "R-Fork from seed, RCCL",
    ),
    "rfork_mooncake": (
        {"MOONCAKE_PROTOCOL": "hip"},
        [
            "--load-format",
            "remote_instance",
            "--remote-instance-weight-loader-backend",
            "transfer_engine",
            "--remote-instance-weight-loader-seed-instance-ip",
            "127.0.0.1",
            "--remote-instance-weight-loader-seed-instance-service-port",
            "{seed_port}",
        ],
        "R-Fork from seed, Mooncake (hip)",
    ),
}
BASELINE = "disk"
PHASES = [
    ("wall", "launch -> ready"),
    ("import", "process start + imports"),
    ("pre_load", "args/tokenizer/dist init until weight load"),
    ("dist_init", "  of which dist init"),
    ("load", "weight load (read/transfer + post-process)"),
    ("pyhip_warmup", "PyHIP GR read warmup (JIT)"),
    ("graph", "decode CUDA graph capture"),
    ("rest", "rest until ready (KV alloc, warmup, lazy JIT)"),
]
_TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")


def _ts(line):
    m = _TS.match(line)
    return datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") if m else None


def parse_phases(log_path, start_epoch, ready_epoch):
    lines = open(log_path, errors="ignore").read().splitlines()
    stamped = [(t, l) for l in lines if (t := _ts(l)) is not None]
    first = stamped[0][0]

    def first_ts(pattern):
        return next(t for t, l in stamped if pattern in l)

    def last_float(pattern):
        vals = [float(m.group(1)) for l in lines if (m := re.search(pattern, l))]
        return max(vals) if vals else 0.0

    ready = first_ts("fired up")
    load_begin = first_ts("Load weight begin")
    load_weight = last_float(r"load_weight=([\d.]+)")
    pyhip = last_float(r"TP0\].*PyHIP GR read warmup end.*elapsed=([\d.]+)")
    graph = last_float(r"cuda_graph=\{prefill=[\d.]+, decode=([\d.]+)")
    wall = ready_epoch - start_epoch
    first_epoch = first.timestamp()
    phases = {
        "wall": wall,
        "import": first_epoch - start_epoch,
        "pre_load": (load_begin - first).total_seconds(),
        "dist_init": last_float(r"Init torch distributed ends\. elapsed=([\d.]+)"),
        "load": load_weight - pyhip,
        "pyhip_warmup": pyhip,
        "graph": graph,
    }
    phases["rest"] = (ready - load_begin).total_seconds() - load_weight - graph
    return phases


class Server:
    def __init__(self, *, log_path, gpus, port, args, env, home=None, timeout=1200):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        run_env = dict(os.environ)
        run_env.pop("HIP_VISIBLE_DEVICES", None)
        run_env.pop("CUDA_VISIBLE_DEVICES", None)
        run_env.update(env)
        run_env.update(
            LOG_FILE=log_path,
            TP_SIZE=str(len(gpus)),
            PORT=str(port),
            PYTHONPATH=os.path.join(REPO_DIR, "python"),
        )
        if home is not None:
            run_env["HOME"] = home
        # All GPUs stay visible (Mooncake HIP IPC needs the peer device); place by base id.
        cmd = ["bash", LAUNCH_SCRIPT, "--base-gpu-id", str(gpus[0]), *args]
        self.cmd = cmd
        self.start = time.time()
        self.proc = subprocess.Popen(
            cmd,
            env=run_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.ready_epoch = self._wait(timeout)

    def _wait(self, timeout):
        deadline = self.start + timeout
        while time.time() < deadline:
            if os.path.exists(self.log_path):
                text = open(self.log_path, errors="ignore").read()
                if "fired up" in text:
                    return time.time()
                if "Traceback" in text or self.proc.poll() is not None:
                    return None
            time.sleep(0.5)
        return None

    def stop(self):
        for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
            try:
                os.killpg(self.proc.pid, sig)
            except ProcessLookupError:
                break
            try:
                self.proc.wait(wait)
                break
            except subprocess.TimeoutExpired:
                continue
        # Children of the launch script share its process group; give the driver
        # time to release GPU memory before the next launch.
        time.sleep(5)


def evict_model(model_path):
    files = [
        os.path.join(model_path, f)
        for f in os.listdir(model_path)
        if os.path.isfile(os.path.join(model_path, f))
    ]
    for f in files:
        evict(f)
    return sum(resident_bytes(f) for f in files) / sum(
        os.path.getsize(f) for f in files
    )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--configs", default=",".join(CONFIGS), help=f"subset of {list(CONFIGS)}"
    )
    p.add_argument("--kernel-cache", default="warm,cold", help="warm,cold")
    p.add_argument("--page-cache", default="warm,cold", help="warm,cold")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--gpus", default="4,5,6,7")
    p.add_argument("--seed-gpus", default="0,1,2,3")
    p.add_argument("--port", type=int, default=7091)
    p.add_argument("--seed-port", type=int, default=7090)
    p.add_argument("--group-port-base", type=int, default=62000)
    p.add_argument(
        "--model-path", default="/models/Qwen3.8-Flash-Next-PTPC-FP8-PLE-BF16"
    )
    p.add_argument(
        "--prime",
        action="store_true",
        help="one unmeasured launch first to fill $HOME caches",
    )
    p.add_argument(
        "--out", required=True, help="output directory (logs, results.csv, summary.md)"
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--reparse",
        action="store_true",
        help="recompute phases from the logs of an existing --out dir, no launches",
    )
    a = p.parse_args()

    configs = a.configs.split(",")
    unknown = set(configs) - set(CONFIGS)
    if unknown:
        p.error(f"unknown configs {unknown}")
    gpus = [int(g) for g in a.gpus.split(",")]
    seed_gpus = [int(g) for g in a.seed_gpus.split(",")]
    group_ports = json.dumps(
        list(range(a.group_port_base, a.group_port_base + len(gpus)))
    )
    os.environ["MODEL_PATH"] = a.model_path
    os.makedirs(a.out, exist_ok=True)

    runs = []
    for name in configs:
        for kc in a.kernel_cache.split(","):
            for pc in a.page_cache.split(","):
                if name.startswith("rfork") and pc == "cold":
                    continue
                for r in range(a.repeat):
                    runs.append((name, kc, pc, r))
    # Seed-dependent runs last, so the seed occupies its GPUs only while needed.
    runs.sort(key=lambda x: x[0].startswith("rfork"))

    if a.reparse:
        reparse(a.out)
        return

    if a.dry_run:
        for name, kc, pc, r in runs:
            env, args, _ = CONFIGS[name]
            args = [
                x.replace("{seed_port}", str(a.seed_port)).replace(
                    "{group_ports}", group_ports
                )
                for x in args
            ]
            print(
                f"{name:16s} kernel={kc:4s} page={pc:4s} #{r}: {env} {' '.join(args)}"
            )
        return

    if a.prime:
        print("priming kernel caches ...", flush=True)
        s = Server(
            log_path=f"{a.out}/logs/prime.log", gpus=gpus, port=a.port, args=[], env={}
        )
        s.stop()

    seed = None
    results = []
    csv_path = os.path.join(a.out, "results.csv")
    for name, kc, pc, r in runs:
        env, args, _ = CONFIGS[name]
        if name.startswith("rfork") and seed is None:
            seed_args = []
            if "rfork_mooncake" in configs:
                seed_args = [
                    "--remote-instance-weight-loader-start-seed-via-transfer-engine"
                ]
            print("starting seed ...", flush=True)
            seed = Server(
                log_path=f"{a.out}/logs/seed.log",
                gpus=seed_gpus,
                port=a.seed_port,
                args=seed_args,
                env={"MOONCAKE_PROTOCOL": "hip"},
            )
            if seed.ready_epoch is None:
                print("seed failed to start, see seed.log; skipping R-Fork configs")
                seed.stop()
                break
        args = [
            x.replace("{seed_port}", str(a.seed_port)).replace(
                "{group_ports}", group_ports
            )
            for x in args
        ]
        tag = f"{name}_kc-{kc}_pc-{pc}_r{r}"
        resident = None
        if pc == "cold":
            resident = evict_model(a.model_path)
        home = (
            tempfile.mkdtemp(prefix="sglang_home_", dir=a.out) if kc == "cold" else None
        )
        print(f"[{len(results) + 1}/{len(runs)}] {tag} ...", end=" ", flush=True)
        s = Server(
            log_path=f"{a.out}/logs/{tag}.log",
            gpus=gpus,
            port=a.port,
            args=args,
            env=env,
            home=home,
        )
        row = {
            "config": name,
            "kernel_cache": kc,
            "page_cache": pc,
            "repeat": r,
            "ok": s.ready_epoch is not None,
        }
        if resident is not None:
            row["resident_after_evict"] = round(resident, 4)
        if row["ok"]:
            row.update(
                {
                    k: round(v, 2)
                    for k, v in parse_phases(s.log_path, s.start, s.ready_epoch).items()
                }
            )
            print(f"ready in {row['wall']:.1f} s", flush=True)
        else:
            print("FAILED, see log", flush=True)
        s.stop()
        results.append(row)
        with open(csv_path, "w", newline="") as f:
            keys = sorted(
                {k for x in results for k in x},
                key=lambda k: list(row).index(k) if k in row else 99,
            )
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(results)
    if seed is not None:
        seed.stop()

    write_summary(results, os.path.join(a.out, "summary.md"))
    print(open(os.path.join(a.out, "summary.md")).read())


def reparse(out):
    with open(os.path.join(out, "results.csv")) as f:
        results = list(csv.DictReader(f))
    for row in results:
        row["ok"] = row["ok"] == "True"
        row["repeat"] = int(row["repeat"])
        if not row["ok"]:
            continue
        tag = f"{row['config']}_kc-{row['kernel_cache']}_pc-{row['page_cache']}_r{row['repeat']}"
        log = os.path.join(out, "logs", f"{tag}.log")
        lines = open(log, errors="ignore").read().splitlines()
        first = next(t for l in lines if (t := _ts(l)) is not None).timestamp()
        start = first - float(row["import"])
        phases = parse_phases(log, start, start + float(row["wall"]))
        row.update({k: round(v, 2) for k, v in phases.items()})
    with open(os.path.join(out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    write_summary(results, os.path.join(out, "summary.md"))
    print(open(os.path.join(out, "summary.md")).read())


def write_summary(results, path):
    groups = {}
    for row in results:
        if row["ok"]:
            groups.setdefault(
                (row["config"], row["kernel_cache"], row["page_cache"]), []
            ).append(row)
    base = {
        (kc, pc): statistics.mean(r["wall"] for r in rows)
        for (c, kc, pc), rows in groups.items()
        if c == BASELINE
    }
    lines = [
        "| config | kernel cache | page cache | n | "
        + " | ".join(label for _, label in PHASES)
        + " | vs disk (same caches) |",
        "|" + "---|" * (len(PHASES) + 5),
    ]
    for (c, kc, pc), rows in groups.items():
        means = [statistics.mean(r[k] for r in rows) for k, _ in PHASES]
        ref = base.get((kc, "warm" if c.startswith("rfork") else pc))
        delta = f"{means[0] - ref:+.1f} s" if ref is not None and c != BASELINE else "-"
        lines.append(
            f"| {c} | {kc} | {pc} | {len(rows)} | "
            + " | ".join(f"{m:.1f}" for m in means)
            + f" | {delta} |"
        )
    failed = [r for r in results if not r["ok"]]
    if failed:
        lines.append("")
        lines.append(
            "Failed runs: "
            + ", ".join(
                f"{r['config']}/{r['kernel_cache']}/{r['page_cache']}" for r in failed
            )
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
