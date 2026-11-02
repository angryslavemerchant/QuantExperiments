"""Run one config's runs as N parallel processes on the same GPU, then merge.

    python scripts/run_parallel.py --config configs/sweep.yaml --shards 3

Codebook fitting is dominated by many small sequential GPU steps, so a big
card is mostly idle with one process; N processes overlap that idle time.
Each shard writes to <output_dir>/shard<i>/ with its own log; afterwards the
rows, per-run JSON and GSM8K generations are merged into <output_dir>/ and the
plots are redrawn. Exit code is nonzero if any shard failed (merging still
happens, so partial results are always packaged).
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def last_line(path: Path) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 4096))
            lines = [l for l in f.read().decode("utf-8", "replace").splitlines() if l.strip()]
        return lines[-1][:200] if lines else ""
    except OSError:
        return ""


def merge(out_dir: Path, shards: int, run_order: list[str]) -> int:
    from scripts.run_experiment import COLUMNS  # noqa: E402
    from shapequant.plots import make_plots

    rows: dict[str, dict] = {}
    for i in range(shards):
        sd = out_dir / f"shard{i}"
        csv_path = sd / "results.csv"
        if csv_path.exists():
            with open(csv_path, newline="") as f:
                for row in csv.DictReader(f):
                    rows[row["run_name"]] = row
        for sub in ("runs", "gsm8k"):
            if (sd / sub).is_dir():
                (out_dir / sub).mkdir(parents=True, exist_ok=True)
                for p in (sd / sub).iterdir():
                    shutil.copy2(p, out_dir / sub / p.name)
        if (sd / "environment.json").exists() and not (out_dir / "environment.json").exists():
            shutil.copy2(sd / "environment.json", out_dir / "environment.json")
    merged = out_dir / "results.csv"
    with open(merged, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for name in run_order:
            if name in rows:
                w.writerow(rows[name])
    if rows:
        make_plots(merged, out_dir)
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--poll-seconds", type=float, default=120)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    run_order = [r["name"] for r in cfg["runs"]]

    # Fetch the model once so shards don't race on the same download.
    from huggingface_hub import snapshot_download

    if not Path(cfg["model"]).is_dir():
        log(f"prefetching {cfg['model']}")
        snapshot_download(cfg["model"], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])

    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    procs = []
    for i in range(args.shards):
        (out_dir / f"shard{i}").mkdir(parents=True, exist_ok=True)
        logf = open(out_dir / f"shard{i}.log", "w")
        cmd = [sys.executable, str(ROOT / "scripts" / "run_experiment.py"), "--config", args.config,
               "--shard", str(i), "--num-shards", str(args.shards)]
        procs.append((subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=ROOT), logf))
        log(f"shard {i} started: {[n for j, n in enumerate(run_order) if j % args.shards == i]}")
        time.sleep(20)  # stagger model loads

    while any(p.poll() is None for p, _ in procs):
        time.sleep(args.poll_seconds)
        for i, (p, _) in enumerate(procs):
            state = "running" if p.poll() is None else f"exit {p.returncode}"
            log(f"shard {i} [{state}] {last_line(out_dir / f'shard{i}.log')}")

    codes = [p.returncode for p, _ in procs]
    for _, f in procs:
        f.close()
    n = merge(out_dir, args.shards, run_order)
    log(f"merged {n}/{len(run_order)} runs; shard exit codes {codes}")
    for i, c in enumerate(codes):
        if c != 0:
            log(f"--- shard {i} log tail ---")
            print("\n".join((out_dir / f"shard{i}.log").read_text(errors="replace").splitlines()[-40:]), flush=True)
    return 0 if all(c == 0 for c in codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
