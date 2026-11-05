"""Rented-GPU report, round 4: end-to-end decode speed of our format.

Round 4 adds the fused engine and programmatic dependent launch (PDL); the tests
run on the rented GPU first because PDL only overlaps kernels on sm_90+.

1. DRAM read ceiling.
2. scripts/bench_e2e.py (greedy decode tg128 on Qwen2.5-7B shapes, random
   weights, one CUDA graph per token): fp16 / our Q4 reference / runs at 2.5
   and 3 bpw, for a few launch configs (RU, large-matrix config, software
   pipeline on/off; off by default since s11). Small matrices use one block per SM (threads=0).
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
E2E_CONFIGS = [  # name, variants, extra args (s4/s5: the real stacks' per-matrix K, G=4 books)
    # defaults: fused engine (rope+attn+perm in one kernel, merged q/k/v and gate/up), PDL on sm_90+
    ("main", "q4,runs10,s4,fp16", ["--breakdown"]),
    ("nopdl", "runs10,s4", ["--pdl", "off"]),
    ("nofuse_nopdl", "q4,runs10,s4", ["--no-fuse", "--pdl", "off"]),     # = the s12 engine
    ("nofuse_pdl", "runs10,s4", ["--no-fuse", "--pdl", "on"]),
    ("small1024x1", "runs10,s4", ["--small", "1024x1"]),
    ("small512x1", "runs10,s4", ["--small", "512x1"]),
    ("small256x2", "runs10,s4", ["--small", "256x2"]),
    ("large1024x1", "runs10,s4", ["--large", "1024x1"]),
    ("s5", "runs12,s5", []),
]

# --mode profile: profile_engine (isolated / leave-one-out / torch.profiler trace), then a
# launch-config grid for tuning on this GPU (fused engine; PDL auto = on for sm_90+ only)
TUNE_CONFIGS = [(f"tune_L{lg}_S{sm}", "runs10,s4", ["--large", lg, "--small", sm])
                for lg in ("512x2", "1024x1", "256x4") for sm in ("1024x1", "0x1", "512x2")]
TUNE_CONFIGS += [
    ("tune_ru2", "runs10,s4", ["--ru", "2"]),
    ("tune_nofuse", "q4,runs10,s4", ["--no-fuse"]),
    ("tune_ref", "q4,runs12,s5,fp16", []),
]
# --mode fusion: attention split x run-kernel epilogues (blocks running them), per GPU
FUSION_CONFIGS = [("f_nofuse", "q4,runs10,s4", ["--no-fuse"])]
FUSION_CONFIGS += [(f"f_attn{ms}_noepi", "runs10,s4", ["--attn-split", str(ms)]) for ms in (1, 2, 4)]
FUSION_CONFIGS += [(f"f_attn2_epi{ne}", "runs10,s4", ["--epilogue", "--attn-split", "2", "--epi-blocks", str(ne)])
                   for ne in (4, 8, 16, 32, 64)]
FUSION_CONFIGS += [("f_attn2_noepi_L1024x1", "runs10,s4", ["--attn-split", "2", "--large", "1024x1"]),
                   ("f_attn2_noepi_ru2", "runs10,s4", ["--attn-split", "2", "--ru", "2"])]
# --mode formats: tok/s of the stage-8 format variants (s4 allocation with M column orders and
# G books varied; u1024 = sizes capped at 1024), same engine defaults as s15
FORMAT_CONFIGS = [
    ("fmt_m", "q4,runs10,s4,s4_m8,s4_m4,s4_m1", ["--breakdown"]),
    ("fmt_g", "s4_g1,s4_m4_g1,s4_m1_g1", ["--breakdown"]),
    ("fmt_u", "u1024,u1024_m4,u1024_m1,u1024_g1,u1024_m1_g1", ["--breakdown"]),
    ("fmt_m_rep", "runs10,s4,s4_m4,s4_m1,s4_m1_g1,u1024", []),     # repeat for noise
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/workspace/scratch_e2e")
    ap.add_argument("--skip-precision", action="store_true")
    ap.add_argument("--skip-llama", action="store_true")
    ap.add_argument("--mode", choices=("e2e", "profile", "fusion", "formats"), default="e2e")
    a = ap.parse_args()
    import os
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")    # s14 5090: the xet download path failed
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
    # correctness on this GPU first (PDL only really overlaps kernels on sm_90+, never locally)
    try:
        import pytest  # noqa: F401
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "pytest==9.0.2"], out / "pip_pytest.txt")
    run([sys.executable, "-m", "pytest", str(ROOT.parent / "tests" / "test_decode_e2e.py"),
         str(ROOT.parent / "tests" / "test_proto_runs.py"), "-q", "-W", "ignore", "-p", "no:cacheprovider",
         "--basetemp", str(work / "pytest")], out / "tests_gpu.txt", timeout=1800)
    if a.mode in ("profile", "fusion", "formats"):
        prof = ["--attn-split", "2"] if a.mode == "fusion" else []
        pv = "s4,s4_m1,s4_m1_g1" if a.mode == "formats" else "q4,runs10,s4"
        run([sys.executable, str(ROOT / "profile_engine.py"), "--variants", pv, "--trace", *prof,
             "--json", str(out / "profile.json")], out / "profile.txt", timeout=2400)
    configs = {"profile": TUNE_CONFIGS, "fusion": FUSION_CONFIGS, "formats": FORMAT_CONFIGS}.get(a.mode, E2E_CONFIGS)
    for name, variants, extra in configs:
        run([sys.executable, str(ROOT / "bench_e2e.py"), "--variants", variants, *extra,
             "--json", str(out / f"e2e_{name}.json")], out / f"e2e_{name}.txt", timeout=1800)
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
