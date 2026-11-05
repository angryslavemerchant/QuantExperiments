"""End-to-end greedy decode tokens/s on Qwen2.5-7B shapes (random weights; speed only).

Comparable to `llama-bench -n 128 -p 0` (tg128): batch 1, generation from an
empty context, whole step in one CUDA graph. Variants:
  fp16    cuBLAS GEMV (needs ~16 GB)
  q4      our Q4_0-like reference kernel (4.5 bpw, same matmul efficiency as llama.cpp Q4_0 on the 5090)
  runsB   the lane-interleaved run kernel, B-bit tags per 4 weights (10 -> 2.5 bpw, 12 -> 3 bpw), G=1
  s4, s5  the real quality stacks' structure: per-matrix K from the s04 (2.5 bpw) / s05 (3 bpw)
          Fisher allocation, G=4 mixture books (tables over 48 KB read through L1); lm_head
          uses --head-bits, G=1 (the KL evals kept lm_head unquantized)
Also times the linears alone (same kernels, same order) to split GEMV vs the rest.
The lm_head uses the variant's format too (llama.cpp Q4_0 models keep it at Q6_K / Q8_0).

Run: python scripts/bench_e2e.py --variants q4,runs10,runs12 [--layers 28]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shapequant.kernels import decode_e2e as de  # noqa: E402

ALLOC = {"s4": "configs/alloc/s4_full.json", "s5": "configs/alloc/s5_full.json"}   # per-matrix K of the stacks
TYPES = ("q", "k", "v", "o", "gate", "up", "down")


def load_alloc(path):
    """{matrix name: K} (exported from the quantizer's allocation.json)."""
    return json.loads((Path(__file__).resolve().parents[1] / path).read_text())["choices"]


def breakdown(eng, reps=20):
    """us per token spent in each linear type (all layers), each type captured in its own graph."""
    ys = dict(q=eng.yq, k=eng.yk, v=eng.yv, o=eng.yo, gate=eng.yg, up=eng.yu, down=eng.yd)
    out = {}
    for n in TYPES + ("head",):
        def body():
            if n == "head":
                eng.lm_head.run(eng.xp, eng.logits)
            else:
                for L in eng.layers:
                    L[n].run(eng.xp, ys[n])
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            body()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            body()
        ts = []
        for _ in range(3):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(reps):
                g.replay()
            b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b) * 1e3 / reps)
        out[n] = sorted(ts)[1]
    return out


def cfg_pair(s):
    th, bps = (int(v) for v in s.split("x"))
    return th, bps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="q4,runs10,runs12")
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--ru", type=int, default=1)
    ap.add_argument("--large", default="512x2", help="launch config THREADSxBLOCKS_PER_SM for matrices with > 4096 rows")
    ap.add_argument("--small", default="1024x1", help="launch config for the rest (0 = one block per SM, auto size; 1024x1 best on the 5090, s13)")
    ap.add_argument("--pipe", action="store_true", help="software pipeline (slower on the 2060 and the 5090)")
    ap.add_argument("--G", type=int, default=4, help="mixture books for s4/s5")
    ap.add_argument("--head-bits", type=int, default=0, help="lm_head bits for s4/s5 (0: 10 for s4, 12 for s5)")
    ap.add_argument("--breakdown", action="store_true", help="also time each linear type alone (us/token)")
    ap.add_argument("--no-fuse", action="store_true", help="separate rope/attn/perm kernels and unmerged q/k/v, gate/up")
    ap.add_argument("--pdl", choices=("auto", "on", "off"), default="auto",
                    help="programmatic dependent launch (auto: on for sm_90+)")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    cfg = de.Cfg(layers=a.layers)
    print(f"{torch.cuda.get_device_name(0)}; Qwen2.5-7B shapes, {a.layers} layers, tg{a.tokens}, "
          f"runs: RU{a.ru} large {a.large} small {a.small} pipe {a.pipe} fuse {not a.no_fuse} pdl {a.pdl}", flush=True)
    print(f"{'variant':10s} {'weights GB':>10s} {'tok/s':>8s} {'ms/tok':>8s} {'GEMV ms':>8s} {'other ms':>9s} {'GEMV GB/s':>10s}")
    rows = []
    for v in a.variants.split(","):
        kind, bits = ("runs", int(v[4:])) if v.startswith("runs") else (v, 0)
        extra = {}
        if v in ALLOC:
            kind, bits = "runs", 10 if v == "s4" else 12
            extra = dict(alloc=load_alloc(ALLOC[v]), G=a.G, head_bits=a.head_bits or bits)
        try:
            eng = de.random_engine(cfg, kind, bits=bits or 10, ru=a.ru, large=cfg_pair(a.large),
                                   small=cfg_pair(a.small), pipe=a.pipe, fuse=not a.no_fuse,
                                   pdl=None if a.pdl == "auto" else a.pdl == "on", **extra)
        except torch.OutOfMemoryError:
            print(f"{v:10s} out of memory", flush=True)
            torch.cuda.empty_cache()
            continue
        t = eng.time_tokens(a.tokens)
        toks = eng.generate(4)
        eng.only_linears, eng.graph = True, None
        tl = eng.time_tokens(a.tokens)
        gb = eng.weight_bytes() / 1e9
        r = dict(variant=v, weights_gb=gb, tok_s=1 / t, ms_tok=t * 1e3, gemv_ms=tl * 1e3, other_ms=(t - tl) * 1e3,
                 gemv_gbps=gb / tl, sample_tokens=toks)
        rows.append(r)
        print(f"{v:10s} {gb:10.2f} {1 / t:8.1f} {t * 1e3:8.3f} {tl * 1e3:8.3f} {(t - tl) * 1e3:9.3f} {gb / tl:10.0f}",
              flush=True)
        if a.breakdown:
            r["breakdown_us"] = bd = breakdown(eng)
            print("           us/token: " + " ".join(f"{k} {x:.0f}" for k, x in bd.items()), flush=True)
        del eng
        torch.cuda.empty_cache()
    if a.json:
        Path(a.json).write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(0), args=vars(a), rows=rows), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
