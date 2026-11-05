"""Where a decode token's time goes, without hardware counters (rented hosts block them).

1. isolated: each non-GEMV op class captured alone in a CUDA graph, as many launches as a step
   has, at a mid-sequence position (CUDA-event timing only).
2. leave-one-out: tg128 of the full step with one op class left out (outputs become meaningless,
   timing does not care); full - that = the op's cost in context, launch boundaries and
   overlap included.
3. trace (--trace): torch.profiler kernel durations over a few eager steps (CUPTI activity
   tracing, which usually needs no counter permission; the error is recorded if it fails).

Run: python scripts/profile_engine.py --variants runs10,s4 [--small 1024x1] [--trace] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_e2e import ALLOC, cfg_pair, load_alloc  # noqa: E402
from shapequant.kernels import decode_e2e as de  # noqa: E402

LOO = ("embed", "addnorm", "qkv", "attn", "o", "gateup", "silu", "down", "lm_head", "argmax")


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


def replay_time(eng, g, n=128):
    eng.reset()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / 1e3 / n


def leave_one_out(eng, ops, pairs=7):
    """{op: median over alternating (full, without op) tg128 pairs of the time difference}."""
    eng.skip, eng.graph = set(), None
    full = eng.capture()
    out = {}
    for op in ops:
        eng.skip, eng.graph = {op}, None
        g = eng.capture()
        d = []
        for _ in range(pairs):
            d.append(replay_time(eng, full) - replay_time(eng, g))
        out[op] = sorted(d)[pairs // 2]
    eng.skip, eng.graph = set(), None
    return out


def build(v, cfg, a):
    extra, kind, bits = {}, v, 10
    if v.startswith("runs"):
        kind, bits = "runs", int(v[4:])
    elif v in ALLOC:
        kind, bits = "runs", 10 if v == "s4" else 12
        extra = dict(alloc=load_alloc(ALLOC[v]), G=4, head_bits=bits)
    return de.random_engine(cfg, kind, bits=bits, large=cfg_pair(a.large), small=cfg_pair(a.small),
                            pdl=None if a.pdl == "auto" else a.pdl == "on", attn_split=a.attn_split, epilogue=a.epilogue, epi_blocks=a.epi_blocks, **extra)


def isolated(eng, cfg, pos):
    import cupy as cp
    eng.reset()
    eng.pos.fill_(pos)
    L0, nl = eng.layers[0], len(eng.layers)

    def attn():
        eng._attn(L0, eng.kc[0], eng.vc[0])

    ops = {
        "addnorm1": lambda: eng._addnorm(eng.yd, L0["ln1"], L0["q"]),
        "attn_rope": attn,
        "addnorm2": lambda: eng._addnorm(eng.yo, L0["ln2"], L0["gate"]),
        "silu_mul_perm": lambda: eng._elementwise("silu_mul_perm", (eng.yg, eng.yu), L0["down"]),
        "empty_kernel": lambda: eng._launch("embed", 1, 32, (de._ptr(eng.embed), de._ptr(eng.tok), cp.int32(1),
                                                              cp.int32(0), de._ptr(eng.hb[0]))),
    }
    out = {}
    for name, f in ops.items():
        def body(f=f):
            for _ in range(nl):
                f()
        out[name] = graph_time(body)
    out["final_norm+argmax"] = graph_time(lambda: (eng._addnorm(eng.yd, eng.final_norm, eng.lm_head),
                                                   torch.argmax(eng.logits, out=eng.tok.view(()))))
    return out


def trace(eng, steps=4):
    from torch.profiler import ProfilerActivity, profile
    eng.graph = None
    eng.reset()
    for _ in range(2):
        eng.step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            eng.step()
        torch.cuda.synchronize()
    rows = {}
    for e in prof.key_averages():
        t = getattr(e, "device_time_total", None)
        if t is None:
            t = getattr(e, "cuda_time_total", 0)
        if t and e.count:
            rows[e.key[:60]] = dict(us_per_token=t / steps, calls_per_token=e.count / steps)
    return dict(sorted(rows.items(), key=lambda kv: -kv[1]["us_per_token"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="runs10,s4")
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--pos", type=int, default=64)
    ap.add_argument("--large", default="512x2")
    ap.add_argument("--small", default="1024x1")
    ap.add_argument("--pdl", choices=("auto", "on", "off"), default="auto")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--attn-split", type=int, default=4)
    ap.add_argument("--epilogue", action="store_true", help="run-kernel epilogues instead of add+norm / silu kernels")
    ap.add_argument("--epi-blocks", type=int, default=64, help="blocks running each epilogue")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    cfg = de.Cfg(layers=a.layers)
    print(f"{torch.cuda.get_device_name(0)}; {a.layers} layers; large {a.large} small {a.small} pdl {a.pdl}",
          flush=True)
    res = {}
    for v in a.variants.split(","):
        eng = build(v, cfg, a)
        full = eng.time_tokens(128)
        eng.only_linears, eng.graph = True, None
        lin = eng.time_tokens(128)
        eng.only_linears = False
        r = dict(pdl=eng.pdl, full_us=full * 1e6, linears_us=lin * 1e6, other_us=(full - lin) * 1e6, loo_us={})
        print(f"\n== {v} (pdl {eng.pdl}): full {full * 1e6:.0f} us/token, linears only {lin * 1e6:.0f}, "
              f"other {(full - lin) * 1e6:.0f}", flush=True)
        print("  leave-one-out (us/token the op class adds in context):")
        for op, d in leave_one_out(eng, LOO).items():
            r["loo_us"][op] = d * 1e6
            print(f"    {op:10s} {d * 1e6:8.0f}", flush=True)
        r["isolated_us"] = iso = isolated(eng, cfg, a.pos)
        print(f"  isolated at pos {a.pos} (us per token, all layers):")
        for k, t in iso.items():
            print(f"    {k:18s} {t:8.0f}", flush=True)
        if a.trace:
            try:
                r["trace"] = tr = trace(eng)
                print("  torch.profiler (eager steps; us/token, calls/token):")
                for k, d in list(tr.items())[:20]:
                    print(f"    {d['us_per_token']:8.0f} {d['calls_per_token']:6.0f}  {k}", flush=True)
            except Exception as e:     # e.g. CUPTI blocked on the host
                r["trace_error"] = repr(e)
                print("  trace failed:", repr(e), flush=True)
        res[v] = r
        del eng
        torch.cuda.empty_cache()
    if a.json:
        Path(a.json).write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(0), args=vars(a), results=res), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
