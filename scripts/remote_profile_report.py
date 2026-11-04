"""Rented-GPU report, round 2 for the lane-interleaved kernel.

1. DRAM read ceiling + bench_runs.py with a launch-config sweep (tests the
   3-bpw hypothesis: one shared-memory table copy per persistent block costs
   ~blocks x 32 KB of L2 traffic per matrix on many-SM GPUs).
2. Nsight Compute: find or apt-install ncu, probe hardware-counter access, and
   if allowed profile Q4 / current / run kernels (full section set), saving
   .ncu-rep files, raw CSV metrics and a text summary. If counters are blocked,
   the probe output is saved and the step is skipped.
3. check_runs_precision.py on real Qwen2.5-7B-Instruct (both kernels now use
   power-of-two-scaled tables; rerun to confirm the layer-26 overflow is gone).

Run: python scripts/remote_profile_report.py --out artifacts/runs_kernel2/5090
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_speed_report import ensure_cupy, read_bandwidth  # noqa: E402
from prof_driver import KERNEL_REGEX  # noqa: E402

ROOT = Path(__file__).resolve().parent

PROFILES = [  # name, driver args
    ("q4_up", ["q4", "up"]),
    ("q4_down", ["q4", "down"]),
    ("cur_k1024_up", ["cur", "up", "--bits", "10"]),
    ("runs_k1024_rand_up", ["runs", "up", "--bits", "10", "--ru", "2"]),
    ("runs_k1024_rand_down", ["runs", "down", "--bits", "10", "--ru", "2"]),
    ("runs_k1024_cap2_up", ["runs", "up", "--bits", "10", "--ru", "2", "--pattern", "cap2"]),
    ("runs_k1024_rand_qo", ["runs", "qo", "--bits", "10", "--ru", "2"]),
    ("runs_k4096_rand_up", ["runs", "up", "--bits", "12", "--ru", "2"]),
    ("runs_k4096_rand_up_512x1", ["runs", "up", "--bits", "12", "--ru", "2", "--threads", "512", "--bps", "1"]),
    ("runs_k4096_rand_qo", ["runs", "qo", "--bits", "12", "--ru", "2"]),
    ("runs_k4096_rand_qo_512x1", ["runs", "qo", "--bits", "12", "--ru", "2", "--threads", "512", "--bps", "1"]),
]


def sh(cmd, timeout=1800):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, shell=isinstance(cmd, str))
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "[timeout]"


def run_to(cmd, out_file: Path, timeout=3600) -> int:
    print(f"$ {' '.join(map(str, cmd))}", flush=True)
    rc, so, se = sh(cmd, timeout)
    text = so + ("\n[stderr]\n" + se[-6000:] if rc else "")
    out_file.write_text(text)
    print(f"===== {out_file.name} (rc {rc})\n{text[-5000:]}", flush=True)
    return rc


def find_ncu(log: list) -> str | None:
    cands = [shutil.which("ncu")] + sorted(glob.glob("/usr/local/cuda*/bin/ncu")) + \
        sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))
    for c in cands:
        if c:
            log.append({"found": c})
            return c
    rc, so, se = sh("apt-cache search --names-only '^nsight-compute' 2>&1 | sort -V | tail -5", 300)
    log.append({"apt_search": so[-2000:], "rc": rc})
    if not so.strip():
        rc, so, se = sh("apt-get update -qq 2>&1 | tail -3; apt-cache search --names-only '^nsight-compute' | sort -V | tail -5",
                        600)
        log.append({"apt_update_search": so[-2000:], "rc": rc})
    pkgs = [ln.split()[0] for ln in so.splitlines() if ln.startswith("nsight-compute")]
    for pkg in reversed(pkgs):
        rc, so2, se2 = sh(f"DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends {pkg} 2>&1 | tail -5",
                          900)
        log.append({"apt_install": pkg, "rc": rc, "out": so2[-1500:]})
        cands = sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu")) + [shutil.which("ncu")]
        for c in cands:
            if c:
                return c
    return None


def profile_step(out: Path) -> None:
    pdir = out / "ncu"
    pdir.mkdir(parents=True, exist_ok=True)
    log: list = []
    ncu = find_ncu(log)
    (pdir / "ncu_setup.json").write_text(json.dumps(log, indent=1))
    if not ncu:
        print("ncu not available", flush=True)
        return
    probe = [ncu, "--metrics", "gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed",
             sys.executable, "-c", "import torch; a=torch.ones(1<<20,device='cuda'); (a*2).sum().item()"]
    rc, so, se = sh(probe, 600)
    (pdir / "probe.txt").write_text(f"rc {rc}\n{so}\n[stderr]\n{se}")
    if rc != 0 or "ERR_NVGPUCTRPERM" in so + se or "==PROF== Profiling" not in so + se:
        print("ncu counter probe failed; see ncu/probe.txt", flush=True)
        return
    for name, args in PROFILES:
        rep = pdir / name
        cmd = [ncu, "-k", KERNEL_REGEX[args[0]], "--launch-skip", "3", "--launch-count", "1", "--set", "full",
               "-f", "-o", str(rep), sys.executable, str(ROOT / "prof_driver.py"), *args]
        rc, so, se = sh(cmd, 1200)
        (pdir / f"{name}.log").write_text(f"rc {rc}\n{so[-4000:]}\n[stderr]\n{se[-4000:]}")
        if rc == 0 and Path(f"{rep}.ncu-rep").exists():
            rc2, csv, _ = sh([ncu, "--import", f"{rep}.ncu-rep", "--csv", "--page", "raw"], 600)
            (pdir / f"{name}.csv").write_text(csv)
            rc3, det, _ = sh([ncu, "--import", f"{rep}.ncu-rep", "--page", "details", "--print-details", "all"], 600)
            (pdir / f"{name}_details.txt").write_text(det)
        print(f"profiled {name}: rc {rc}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-precision", action="store_true",
                    help="skip the 7B precision check (needs a 15 GB HF download)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import torch

    setup_log: list = []
    ok = ensure_cupy(setup_log)
    (out / "setup.json").write_text(json.dumps(setup_log, indent=1))
    info = {"gpu": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(),
            "torch": torch.__version__, "cupy_ok": ok, "sms": torch.cuda.get_device_properties(0).multi_processor_count}
    if ok:
        info["read_GBps"] = read_bandwidth()
    print(json.dumps(info), flush=True)
    (out / "gpu.json").write_text(json.dumps(info, indent=1))
    if not ok:
        return 0
    run_to([sys.executable, str(ROOT / "bench_runs.py"), "--bits", "10,12", "--ru", "1,2",
            "--cfgs", "256x4,256x2,512x1,512x2,1024x1"], out / "runs_speed_cfgs.txt")
    try:
        profile_step(out)
    except Exception as e:
        (out / "profile_error.txt").write_text(repr(e))
        print("profile step failed:", repr(e), flush=True)
    if a.skip_precision:
        return 0
    run_to([sys.executable, str(ROOT / "check_runs_precision.py"), "--tokens", "64", "--capture-device", "cuda",
            "--out", str(out / "precision_7b.txt")], out / "precision_7b_log.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
