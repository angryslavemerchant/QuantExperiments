"""Where the decode engine's non-GEMV time goes: each op class of one token step, captured
alone in a CUDA graph (as many launches as in a real step), at a mid-sequence position.

Run: python scripts/profile_engine.py --variant s4 [--pos 64] [--layers 28]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_e2e import ALLOC, load_alloc  # noqa: E402
from shapequant.kernels import decode_e2e as de  # noqa: E402


def graph_time(body, reps=50):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        body()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()
    ts = []
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / reps)
    return sorted(ts)[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="s4")
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--pos", type=int, default=64)
    a = ap.parse_args()
    cfg = de.Cfg(layers=a.layers)
    v = a.variant
    extra, kind, bits = {}, v, 10
    if v.startswith("runs"):
        kind, bits = "runs", int(v[4:])
    elif v in ALLOC:
        kind, bits = "runs", 10 if v == "s4" else 12
        extra = dict(alloc=load_alloc(ALLOC[v]), G=4, head_bits=bits)
    eng = de.random_engine(cfg, kind, bits=bits, large=(512, 2), small=(0, 1), **extra)
    import cupy as cp

    full = eng.time_tokens(128)
    eng.only_linears, eng.graph = True, None
    lin = eng.time_tokens(128)
    eng.only_linears = False
    eng.reset()
    eng.pos.fill_(a.pos)
    L0 = eng.layers[0]
    nl = len(eng.layers)

    def rep(f):
        def body():
            for _ in range(nl):
                f()
        return body

    def rope():
        eng._launch("rope_cache", cfg.n_heads + cfg.n_kv, cfg.head_dim // 2,
                    (de._ptr(eng.yq), de._ptr(eng.yk), de._ptr(eng.yv), de._ptr(L0.get("bq")), de._ptr(L0.get("bk")),
                     de._ptr(L0.get("bv")), de._ptr(eng.pos), cp.float32(cfg.rope_theta), de._ptr(eng.qr),
                     de._ptr(eng.kc[0]), de._ptr(eng.vc[0])))

    def attn():
        eng._launch("attn", cfg.n_heads, cfg.head_dim,
                    (de._ptr(eng.qr), de._ptr(eng.kc[0]), de._ptr(eng.vc[0]), de._ptr(eng.pos),
                     cp.float32(cfg.head_dim ** -0.5), de._ptr(eng.ao)))

    ops = {
        "addnorm1 (q/k/v input)": lambda: eng._addnorm(eng.yd, L0["ln1"], L0["q"]),
        "rope_cache": rope,
        "attn": attn,
        "perm_copy (o input)": lambda: eng._elementwise("perm_copy", (eng.ao,), L0["o"]),
        "addnorm2 (gate/up input)": lambda: eng._addnorm(eng.yo, L0["ln2"], L0["gate"]),
        "silu_mul_perm (down input)": lambda: eng._elementwise("silu_mul_perm", (eng.yg, eng.yu), L0["down"]),
        "empty kernel": lambda: eng._launch("embed", 1, 32, (de._ptr(eng.embed), de._ptr(eng.tok), cp.int32(1),
                                                              cp.int32(0), de._ptr(eng.hb[0]))),
    }
    print(f"{torch.cuda.get_device_name(0)}; {v}, {nl} layers, pos {a.pos}")
    print(f"full step {full * 1e6:8.0f} us   linears only {lin * 1e6:8.0f} us   other {(full - lin) * 1e6:6.0f} us")
    tot = 0.0
    for name, f in ops.items():
        t = graph_time(rep(f))
        if name != "empty kernel":
            tot += t
        print(f"  {name:28s} x{nl}: {t:7.0f} us  ({t / nl:5.1f} us each)")
    once = {"final addnorm + argmax + pos": lambda: (eng._addnorm(eng.yd, eng.final_norm, eng.lm_head),
                                                     torch.argmax(eng.logits, out=eng.tok.view(())),
                                                     eng.pos.add_(0))}
    for name, f in once.items():
        t = graph_time(f)
        tot += t
        print(f"  {name:28s}    : {t:7.0f} us")
    print(f"  sum of ops (isolated)            {tot:7.0f} us")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
