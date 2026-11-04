"""Rented-GPU report for the lane-interleaved kernel (proto_runs.py).

1. DRAM read ceiling + scripts/bench_runs.py (run kernel vs Q4 reference,
   current kernel, fp16; synthetic tags; RU sweep).
2. llama.cpp token-generation speed (llama-bench, official prebuilt CUDA 12.8
   binaries) on real Qwen2.5-7B-Instruct GGUFs: f16, Q4_0, Q4_K_M and quants at
   our bit rates (Q3_K_M, IQ3_M, IQ2_M). The f16 run calibrates the non-matmul
   overhead per token against torch fp16 mv timings from step 1.
3. scripts/check_runs_precision.py on real Qwen2.5-7B-Instruct weights and
   activations (fp16-accumulation check of the run kernel).

Each step writes its own file under --out, so a later failure keeps earlier
results.

Run: python scripts/remote_runs_report.py --out artifacts/runs_kernel/5090
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_speed_report import ensure_cupy, read_bandwidth  # noqa: E402

LLAMA_TAG = "b12110"
LLAMA_URLS = [
    f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/llama-{LLAMA_TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz",
    f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/cudart-llama-{LLAMA_TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz",
]
GGUF_REPO = "bartowski/Qwen2.5-7B-Instruct-GGUF"
GGUFS = ["Q4_0", "Q4_K_M", "Q3_K_M", "IQ3_M", "IQ2_M", "f16"]


def run(cmd, out_file: Path, env=None, timeout=3600) -> int:
    print(f"$ {' '.join(map(str, cmd))}", flush=True)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
        text, rc = r.stdout + ("\n[stderr]\n" + r.stderr[-6000:] if r.returncode else ""), r.returncode
    except subprocess.TimeoutExpired:
        text, rc = "[timeout]", -1
    out_file.write_text(text)
    print(f"===== {out_file.name} (rc {rc})\n{text[-6000:]}", flush=True)
    return rc


def llama_step(out: Path, work: Path, ggufs=GGUFS) -> None:
    from huggingface_hub import hf_hub_download
    llama_dir = work / "llama"
    llama_dir.mkdir(parents=True, exist_ok=True)
    for url in LLAMA_URLS:
        dst = work / url.rsplit("/", 1)[1]
        urllib.request.urlretrieve(url, dst)
        with tarfile.open(dst) as tf:
            tf.extractall(llama_dir, filter="data")
    bench = next(llama_dir.rglob("llama-bench"))
    bench.chmod(0o755)
    libdirs = sorted({str(p.parent) for p in llama_dir.rglob("*.so*")})
    env = dict(os.environ, LD_LIBRARY_PATH=":".join(libdirs + [os.environ.get("LD_LIBRARY_PATH", "")]))
    results = {}
    for q in ggufs:
        path = hf_hub_download(GGUF_REPO, f"Qwen2.5-7B-Instruct-{q}.gguf", local_dir=str(work / "gguf"))
        f = out / f"llama_bench_{q}.json"
        rc = run([str(bench), "-m", path, "-ngl", "99", "-p", "0", "-n", "128", "-r", "5", "-fa", "1", "-o", "json"],
                 f, env=env, timeout=900)
        if rc == 0:
            try:
                rows = json.loads(f.read_text())
                results[q] = {"size_bytes": rows[0].get("model_size"), "tg_tokens_per_s": rows[0].get("avg_ts"),
                              "stddev": rows[0].get("stddev_ts")}
            except Exception as e:                                          # keep raw file
                results[q] = {"parse_error": repr(e)}
        Path(path).unlink(missing_ok=True)
        (out / "llama_summary.json").write_text(json.dumps(results, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/workspace/scratch_runs")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    import torch

    setup_log: list = []
    ok = ensure_cupy(setup_log)
    (out / "setup.json").write_text(json.dumps(setup_log, indent=1))
    info = {"gpu": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(),
            "torch": torch.__version__, "cupy_ok": ok}
    if ok:
        info["read_GBps"] = read_bandwidth()
    print(json.dumps(info), flush=True)
    (out / "gpu.json").write_text(json.dumps(info, indent=1))
    if ok:
        run([sys.executable, str(root / "bench_runs.py"), "--bits", "10,12", "--ru", "1,2,4,8"], out / "runs_speed.txt")
    try:
        llama_step(out, work)
    except Exception as e:
        (out / "llama_error.txt").write_text(repr(e))
        print("llama step failed:", repr(e), flush=True)
    if ok:
        run([sys.executable, str(root / "check_runs_precision.py"), "--tokens", "64", "--capture-device", "cuda",
             "--out", str(out / "precision_7b.txt")], out / "precision_7b_log.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
