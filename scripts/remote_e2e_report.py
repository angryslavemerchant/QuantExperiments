"""Rented-GPU report, round 3: end-to-end decode speed of our format.

1. DRAM read ceiling.
2. scripts/bench_e2e.py (greedy decode tg128 on Qwen2.5-7B shapes, random
   weights, one CUDA graph per token): fp16 / our Q4 reference / runs at 2.5
   and 3 bpw, for a few launch configs (RU, large-matrix config, software
   pipeline on/off). Small matrices use one block per SM (threads=0).
3. scripts/bench_runs.py per-shape sweep of the same configs (random tags).
4. llama-bench tg128 on the same box for Q4_0 and IQ2_M (bartowski GGUFs), so
   the comparison does not depend on a different host's clocks.
5. Unless --skip-precision: check_runs_precision.py on real Qwen2.5-7B-Instruct
   (the fp16-overflow fix on real activations; needs a 15 GB HF download).

Each step writes its own files under --out, so a later failure keeps earlier
results.

Run: python scripts/remote_e2e_report.py --out artifacts/e2e/5090
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_speed_report import ensure_cupy, read_bandwidth  # noqa: E402
from remote_runs_report import llama_step, run  # noqa: E402

ROOT = Path(__file__).resolve().parent
E2E_CONFIGS = [  # name, variants, extra args
    ("ru1_512x2", "q4,runs10,runs12,fp16", ["--ru", "1", "--large", "512x2"]),
    ("ru1_512x2_nopipe", "runs10,runs12", ["--ru", "1", "--large", "512x2", "--no-pipe"]),
    ("ru1_1024x1", "runs10,runs12", ["--ru", "1", "--large", "1024x1"]),
    ("ru1_1024x1_nopipe", "runs10,runs12", ["--ru", "1", "--large", "1024x1", "--no-pipe"]),
    ("ru2_512x2", "runs10,runs12", ["--ru", "2", "--large", "512x2"]),
    ("ru1_256x4_small256x4", "runs10,runs12", ["--ru", "1", "--large", "256x4", "--small", "256x4"]),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/workspace/scratch_e2e")
    ap.add_argument("--skip-precision", action="store_true")
    ap.add_argument("--skip-llama", action="store_true")
    a = ap.parse_args()
    out, work = Path(a.out), Path(a.work)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
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
    for name, variants, extra in E2E_CONFIGS:
        run([sys.executable, str(ROOT / "bench_e2e.py"), "--variants", variants, *extra,
             "--json", str(out / f"e2e_{name}.json")], out / f"e2e_{name}.txt", timeout=1800)
    run([sys.executable, str(ROOT / "bench_runs.py"), "--bits", "10,12", "--ru", "1,2", "--patterns", "random",
         "--cfgs", "512x2,512x2n,1024x1,1024x1n,256x4,0x1"], out / "runs_speed_cfgs.txt")
    if not a.skip_llama:
        try:
            llama_step(out, work, ["Q4_0", "IQ2_M"])
        except Exception as e:
            (out / "llama_error.txt").write_text(repr(e))
            print("llama step failed:", repr(e), flush=True)
    if not a.skip_precision:
        run([sys.executable, str(ROOT / "check_runs_precision.py"), "--tokens", "64", "--capture-device", "cuda",
             "--out", str(out / "precision_7b.txt")], out / "precision_7b_log.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
