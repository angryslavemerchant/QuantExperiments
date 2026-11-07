"""Shipping benchmark on one GPU: our runtime on a real packed file vs llama.cpp on GGUFs.

Steps (each writes its own file under --out, so a later failure keeps earlier results):
1. gpu.json: device, capability, torch version.
2. ours_<name>.json: our runtime on the packed file: decode tg (batch 1, CUDA graph, like
   `llama-bench -n 128`), prefill pp at each --pp length (like `llama-bench -p N`), and batched
   decode / fused prefill when those modules are present (shapequant.qwen35_batch,
   kernels for fused prefill) via their bench scripts.
3. llama_<quant>.json / llama_batched_<quant>.txt: llama.cpp prebuilt CUDA binaries
   (LLAMA_TAG) on each GGUF: llama-bench pp/tg (flash attention on, all layers on the GPU) and
   llama-batched-bench (npp prompt, ntg generated, npl parallel sequences).

Run: python scripts/bench_ship.py --packed /workspace/packed_out/s16_25_ft.safetensors \
        --out artifacts/qwen38_27b/ship_bench --gguf unsloth/Qwen3.8-27B-GGUF:Qwen3.8-27B-UD-IQ2_S.gguf
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LLAMA_TAG = "b12110"
LLAMA_URLS = [
    f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/llama-{LLAMA_TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz",
    f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/cudart-llama-{LLAMA_TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, out_file: Path, env=None, timeout=3600) -> int:
    log("$ " + " ".join(map(str, cmd)))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
        text, rc = r.stdout + ("\n[stderr]\n" + r.stderr[-8000:] if r.returncode else ""), r.returncode
    except subprocess.TimeoutExpired:
        text, rc = "[timeout]", -1
    out_file.write_text(text)
    log(f"{out_file.name}: rc {rc}\n{text[-3000:]}")
    return rc


def ours_step(packed: str, out: Path, pps: list[int], tg: int, max_len: int) -> None:
    import torch
    from shapequant import qwen35_runtime as q35
    res: dict = {"packed": packed}
    t0 = time.time()
    eng, info = q35.engine_from_packed(packed, max_len=max_len)
    res.update(load_seconds=time.time() - t0, weight_bytes=info["weight_bytes"],
               file_bytes=Path(packed).stat().st_size, bpw=info["summary"].get("bpw"),
               bpw_all=info["summary"].get("bpw_all"))
    eng.capture()
    s = eng.time_tokens(tg, 3)
    res["tg"] = {"n": tg, "tokens_per_s": 1.0 / s}
    log(f"ours tg{tg}: {1.0 / s:.1f} tok/s")
    (out / "ours.json").write_text(json.dumps(res, indent=1))
    q35.attach_prefill(eng)
    g = torch.Generator().manual_seed(0)
    res["pp"] = {}
    for n in pps:
        ids = torch.randint(0, eng.cfg.vocab, (n,), generator=g).tolist()
        ts = []
        for _ in range(3):
            eng.reset()
            torch.cuda.synchronize()
            a = time.perf_counter()
            q35.prefill(eng, ids)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - a)
        res["pp"][str(n)] = {"tokens_per_s": n / sorted(ts)[1], "seconds": sorted(ts)[1]}
        log(f"ours pp{n}: {n / sorted(ts)[1]:.0f} tok/s")
        (out / "ours.json").write_text(json.dumps(res, indent=1))
    res["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    (out / "ours.json").write_text(json.dumps(res, indent=1))
    del eng
    torch.cuda.empty_cache()


def llama_setup(work: Path) -> tuple[dict, Path]:
    d = work / "llama"
    d.mkdir(parents=True, exist_ok=True)
    for url in LLAMA_URLS:
        dst = work / url.rsplit("/", 1)[1]
        if not dst.exists():
            urllib.request.urlretrieve(url, dst)
        with tarfile.open(dst) as tf:
            tf.extractall(d, filter="data")
    for p in d.rglob("llama-*"):
        if p.is_file():
            p.chmod(0o755)
    libdirs = sorted({str(p.parent) for p in d.rglob("*.so*")})
    env = dict(os.environ, LD_LIBRARY_PATH=":".join(libdirs + [os.environ.get("LD_LIBRARY_PATH", "")]))
    return env, d


def llama_step(spec: str, out: Path, work: Path, env: dict, d: Path, pps: list[int], tg: int,
               npl: list[int], npp: int, ntg: int) -> None:
    from huggingface_hub import hf_hub_download
    repo, fname = spec.split(":", 1)
    q = Path(fname).stem
    path = hf_hub_download(repo, fname, local_dir=str(work / "gguf"))
    bench = next(d.rglob("llama-bench"))
    f = out / f"llama_{q}.json"
    run([str(bench), "-m", path, "-ngl", "99", "-fa", "1", "-p", ",".join(map(str, pps)), "-n", str(tg),
         "-r", "3", "-o", "json"], f, env=env, timeout=1800)
    bb = next(d.rglob("llama-batched-bench"), None)
    if bb is not None:
        ctx = max(npl) * (npp + ntg) + 256
        run([str(bb), "-m", path, "-ngl", "99", "-fa", "1", "-c", str(ctx), "-b", "2048", "-ub", "512",
             "-npp", str(npp), "-ntg", str(ntg), "-npl", ",".join(map(str, npl))],
            out / f"llama_batched_{q}.txt", env=env, timeout=3600)
    Path(path).unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/workspace/ship_bench_work")
    ap.add_argument("--gguf", action="append", default=[], help="repo:file, repeatable")
    ap.add_argument("--pp", default="512,2048")
    ap.add_argument("--tg", type=int, default=128)
    ap.add_argument("--npl", default="1,2,4,8,16,32")
    ap.add_argument("--npp", type=int, default=512)
    ap.add_argument("--ntg", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--skip-ours", action="store_true")
    ap.add_argument("--extra", action="append", default=[],
                    help="extra benchmark command (shell-split on spaces, {packed} and {out} substituted)")
    a = ap.parse_args()
    out, work = Path(a.out), Path(a.work)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    pps = [int(x) for x in a.pp.split(",")]
    npl = [int(x) for x in a.npl.split(",")]
    import torch
    info = {"gpu": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(),
            "torch": torch.__version__, "llama_tag": LLAMA_TAG}
    (out / "gpu.json").write_text(json.dumps(info, indent=1))
    log(json.dumps(info))
    if not a.skip_ours:
        try:
            ours_step(a.packed, out, pps, a.tg, a.max_len)
        except Exception as e:
            (out / "ours_error.txt").write_text(repr(e))
            log(f"ours step failed: {e!r}")
    for i, cmd in enumerate(a.extra):
        parts = [p.format(packed=a.packed, out=str(out), python=sys.executable) for p in cmd.split(" ")]
        run(parts, out / f"extra_{i}.txt", timeout=3600)
    if not a.gguf:
        log("done (no GGUFs)")
        return 0
    try:
        env, d = llama_setup(work)
    except Exception as e:
        (out / "llama_error.txt").write_text(repr(e))
        log(f"llama setup failed: {e!r}")
        return 1
    for spec in a.gguf:
        try:
            llama_step(spec, out, work, env, d, pps, a.tg, npl, a.npp, a.ntg)
        except Exception as e:
            (out / f"llama_error_{Path(spec).stem}.txt").write_text(repr(e))
            log(f"llama step {spec} failed: {e!r}")
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
