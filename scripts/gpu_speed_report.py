"""Kernel speed report on the current GPU (for a rented modern GPU).

Measures the DRAM read ceiling, then runs scripts/bench_gemv.py (current-format
kernels) and scripts/bench_formats.py (Q4 reference, format A/B prototypes)
against it, and writes everything to --out.

Run: python scripts/gpu_speed_report.py --out artifacts/kernel_speed
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch


def read_bandwidth(seconds: float = 6.0) -> float:
    a = torch.empty(256 << 20, dtype=torch.float32, device="cuda").uniform_()   # 1 GB
    best, t_end = 0.0, time.time() + seconds
    while time.time() < t_end:
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(5):
            a.sum()
        torch.cuda.synchronize()
        best = max(best, 5 * a.numel() * 4 / (time.perf_counter() - t) / 1e9)
    del a
    torch.cuda.empty_cache()
    return best


_PROBE = (
    "import cupy as cp, torch\n"
    "k = cp.RawKernel(r'extern \"C\" __global__ void f(float* x){x[threadIdx.x] += 1.f;}', 'f',\n"
    "                 options=('-arch=sm_%d%d' % torch.cuda.get_device_capability(),))\n"
    "x = torch.zeros(32, device='cuda')\n"
    "k((1,), (32,), (cp.uint64(x.data_ptr()),)); torch.cuda.synchronize(); assert float(x.sum()) == 32\n"
)


def ensure_cupy(log: list) -> bool:
    """Install CuPy (and NVIDIA's pip NVRTC if needed) until an NVRTC kernel runs."""
    steps = [None, ["cupy-cuda12x==14.2.0"], ["cuda-toolkit[nvrtc,cudart]==12.*"]]
    for pkgs in steps:
        if pkgs:
            r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], capture_output=True, text=True)
            log.append({"pip": pkgs, "rc": r.returncode, "stderr": r.stderr[-1500:]})
        r = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True)
        log.append({"probe_rc": r.returncode, "stderr": r.stderr[-1500:]})
        if r.returncode == 0:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    setup_log: list = []
    ok = ensure_cupy(setup_log)
    (out / "setup.json").write_text(json.dumps(setup_log, indent=1))
    if not ok:
        print("CuPy/NVRTC setup failed; see setup.json", flush=True)
        return 0
    root = Path(__file__).resolve().parent
    name = torch.cuda.get_device_name(0)
    bw = read_bandwidth()
    info = {"gpu": name, "capability": torch.cuda.get_device_capability(), "read_GBps": bw,
            "torch": torch.__version__}
    print(json.dumps(info), flush=True)
    (out / "gpu.json").write_text(json.dumps(info, indent=1))
    runs = {
        "formats": [sys.executable, str(root / "bench_formats.py"), "--bw", f"{bw:.1f}"],
        "current_G4M16": [sys.executable, str(root / "bench_gemv.py"), "--impl", "cuda", "--bw", f"{bw:.1f}"],
        "current_G4M16_512t": [sys.executable, str(root / "bench_gemv.py"), "--impl", "cuda", "--threads", "512",
                               "--bps", "2", "--bw", f"{bw:.1f}"],
    }
    for key, cmd in runs.items():
        r = subprocess.run(cmd, capture_output=True, text=True)
        text = r.stdout + ("\n[stderr]\n" + r.stderr[-4000:] if r.returncode else "")
        (out / f"{key}.txt").write_text(text)
        print(f"===== {key} (rc {r.returncode})\n{text}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
