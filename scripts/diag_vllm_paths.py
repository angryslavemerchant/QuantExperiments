"""Component check of the vLLM plugin's kernels on the real packed file, outside vLLM: every
PackedShards path (compact / not, with / without launch tuning, eager and CUDA-graph replay, bf16
input as vLLM sends it) against the reference decode, plus the q8 head and the q4 embedding.

  python scripts/diag_vllm_paths.py --packed <file> --out x.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shapequant.packed import PackedFile, q4_embed_rows, q8_rows  # noqa: E402
from shapequant.vllm_ext import core  # noqa: E402


def rel(y: torch.Tensor, ref: torch.Tensor) -> float:
    y = y.float()
    if not torch.isfinite(y).all():
        return float("nan")
    return ((y - ref).norm() / ref.norm().clamp_min(1e-12)).item()


def groups(pf: PackedFile) -> list[list[str]]:
    by_layer: dict[int, set] = {}
    for n in pf.meta:
        m = re.match(r"model\.layers\.(\d+)\.(.+)$", n)
        if m:
            by_layer.setdefault(int(m.group(1)), set()).add(m.group(2))
    out = []
    for i in (0, 3):
        kinds = by_layer.get(i, set())
        used = set()
        for pieces in core.STACKED.values():
            if all(p in kinds for p in pieces):
                out.append([f"model.layers.{i}.{p}" for p in pieces])
                used |= set(pieces)
        out += [[f"model.layers.{i}.{k}"] for k in sorted(kinds - used)]
    return out


def graph_run(fn, x):
    fn(x)                       # compile / warm
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(x)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = fn(x)
    g.replay()
    torch.cuda.synchronize()
    return y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ms", default="1,2,3,5,8,17,32,33,64,100,512,2048")
    ap.add_argument("--bmax", type=int, default=64)
    ap.add_argument("--xs", type=int, default=1)
    a = ap.parse_args()
    Ms = [int(v) for v in a.ms.split(",")]
    pf = PackedFile(a.packed)
    dev = torch.device("cuda")
    rec = {"linear": [], "head": None, "embed": None}
    torch.manual_seed(0)
    bad = 0
    for names in groups(pf):
        W = torch.cat([pf.decode(n, dev) for n in names])
        for compact, tl in ((True, True), (True, False), (False, True)):
            with core.neutral_defaults():
                ps = core.PackedShards.from_packed(pf, names, dev=dev, compact=compact, bmax=a.bmax, xs=bool(a.xs))
                if tl:
                    ps.tune_launch()
                ps.tune()
            h = core.register(ps)
            row = {"names": names, "compact": compact, "tune_launch": tl, "cross": ps.cross, "err": {}}
            for M in Ms:
                x = torch.randn(M, ps.C, device=dev, dtype=torch.bfloat16)
                ref = x.float() @ W.t()
                e = rel(torch.ops.shapequant.linear(x, h), ref)
                row["err"][str(M)] = round(e, 5)
                if M <= 64:
                    row["err"][f"{M}g"] = round(rel(graph_run(lambda t: torch.ops.shapequant.linear(t, h), x), ref), 5)
            worst = max((v if v == v else 9.0) for v in row["err"].values())
            bad += worst > 0.02
            print(f"{names[0].split('layers.')[1]:<40} x{len(names)} compact={compact} tl={tl} cross={ps.cross} "
                  f"worst={worst:.4f} {row['err']}", flush=True)
            rec["linear"].append(row)
            del ps
            torch.cuda.empty_cache()
    # head
    q = pf.aux_tensor("lm_head.q").to(dev).contiguous()
    sc = pf.aux_tensor("lm_head.scale").to(dev).contiguous()
    V, C = q.shape
    R = -(-V // 64) * 64
    head = core.Q8Head(q, sc, R=R)
    hh = core.register(head)
    he = {}
    for M in (1, 4, 31, 40, 300):
        x = torch.randn(M, C, device=dev, dtype=torch.bfloat16)
        ref = torch.cat([x.float() @ q8_rows(q, sc, s, min(V, s + 16384)).t() for s in range(0, V, 16384)], 1)
        y = torch.ops.shapequant.linear(x, hh)
        he[str(M)] = round(rel(y[:, :V], ref), 5)
        he[f"{M}pad_max"] = float(y[:, V:].abs().max()) if R > V else 0.0
        if M <= 32:
            he[f"{M}g"] = round(rel(graph_run(lambda t: torch.ops.shapequant.linear(t, hh), x)[:, :V], ref), 5)
    print("head", V, C, R, he, flush=True)
    rec["head"] = he
    # embedding
    eq = pf.aux_tensor("embed.q").to(dev).contiguous()
    es = pf.aux_tensor("embed.scale").to(dev).contiguous()
    emb = core.Q4Embedding(eq, es, torch.bfloat16)
    he2 = core.register(emb)
    ids = torch.randint(0, eq.shape[0], (7, 33), device=dev)
    ref = q4_embed_rows(eq, es, ids.flatten()).view(7, 33, -1)
    y = torch.ops.shapequant.embed(ids, he2)
    ee = {"err": round(rel(y, ref), 5), "row_rms": float(ref.pow(2).mean().sqrt()),
          "graph_err": round(rel(graph_run(lambda t: torch.ops.shapequant.embed(t, he2), ids), ref), 5)}
    # embed -> head of the same token: sanity that the aux tensors are a real model's (not zeros)
    tok = torch.tensor([9707, 1879, 13], device=dev)
    ee["self_logit_rank"] = [int((torch.ops.shapequant.linear(torch.ops.shapequant.embed(tok, he2), hh)[i, :V]
                                  > torch.ops.shapequant.linear(torch.ops.shapequant.embed(tok, he2), hh)[i, t]).sum())
                             for i, t in enumerate(tok.tolist())]
    print("embed", ee, flush=True)
    rec["embed"] = ee
    bad += not (ee["err"] < 0.01 and he["1"] < 0.01 and he["40"] < 0.01)
    Path(a.out).write_text(json.dumps(rec, indent=1))
    print("BAD" if bad else "ALL OK", bad, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
