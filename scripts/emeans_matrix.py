"""Matrix-level check: e-means-style online codebook fit ("ema") vs our Lloyd k-means, on real weights.

Each matrix is row-normalized and cut into d=4 chunks, as shapequant.codebook.compress does; the codebook
is fitted by each method and every chunk of the matrix is snapped to it. Reported: relative MSE on the
whole matrix (what the codebook actually encodes) and on the fit subsample (Lloyd's own objective).

Run: python scripts/emeans_matrix.py --model Qwen/Qwen3.5-4B --out artifacts/emeans/matrix.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from shapequant import codebook as cb


def load(model: str, names: list[str]) -> dict[str, torch.Tensor]:
    root = Path(snapshot_download(model, allow_patterns=["*.json", "*.safetensors"]))
    idx = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for n in names:
        with safe_open(root / idx[n], "pt") as f:
            out[n] = f.get_tensor(n).float()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--layers", default="2,15,27")
    ap.add_argument("--Ks", default="256,1024,4096")
    ap.add_argument("--out", default="artifacts/emeans/matrix.json")
    args = ap.parse_args()
    pre = "model.language_model.layers"
    kinds = ["mlp.down_proj", "mlp.up_proj", "linear_attn.in_proj_qkv", "linear_attn.out_proj"]
    names = [f"{pre}.{l}.{k}.weight" for l in map(int, args.layers.split(",")) for k in kinds]
    W = load(args.model, names)
    methods = {
        "lloyd25_sub1M": dict(fit="lloyd", iters=25, sub=1_000_000),     # shipped recipe
        "lloyd100_sub1M": dict(fit="lloyd", iters=100, sub=1_000_000),
        "lloyd25_full": dict(fit="lloyd", iters=25, sub=None),
        "ema_1x": dict(fit="ema", iters=25, compute=1.0),
        "ema_4x": dict(fit="ema", iters=25, compute=4.0),
    }
    res = []
    for n, w in W.items():
        w = w.cuda()
        rms = w.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-12)
        x, _ = cb.chunk_rows(w / rms, 4)
        tot = float(x.pow(2).sum())
        for K in map(int, args.Ks.split(",")):
            for mname, m in methods.items():
                g = torch.Generator().manual_seed(0)
                cb.FIT = m["fit"]
                cb.EMA_COMPUTE = m.get("compute", 1.0)
                cb.EMA_REF_CHUNKS = 1_000_000
                t0 = time.time()
                sub = None
                if m["fit"] == "lloyd" and m["sub"] and x.shape[0] > m["sub"]:
                    sub = torch.randperm(x.shape[0], generator=g)[: m["sub"]].cuda()
                c = cb.kmeans(x if sub is None else x[sub], K, m["iters"], g, 65536).to(torch.float16)
                torch.cuda.synchronize()
                dt = time.time() - t0
                _, e = cb.assign(x, c, 65536)
                rel = float(e.sum()) / tot
                rel_fit = None
                if sub is not None:
                    _, ef = cb.assign(x[sub], c, 65536)
                    rel_fit = float(ef.sum()) / float(x[sub].pow(2).sum())
                r = {"matrix": n.removeprefix(pre + "."), "K": K, "method": mname, "rel_mse": rel,
                     "rel_mse_fitset": rel_fit, "chunks": x.shape[0], "seconds": dt}
                res.append(r)
                print(json.dumps(r), flush=True)
        cb.FIT = "lloyd"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    # summary: mean ratio vs the shipped recipe per K
    base = {(r["matrix"], r["K"]): r["rel_mse"] for r in res if r["method"] == "lloyd25_sub1M"}
    for K in map(int, args.Ks.split(",")):
        for mname in methods:
            rs = [r["rel_mse"] / base[(r["matrix"], K)] for r in res if r["K"] == K and r["method"] == mname]
            print(f"K={K:5d} {mname:16s} mean MSE vs shipped: {100 * (sum(rs) / len(rs) - 1):+.2f}%")


if __name__ == "__main__":
    main()
