"""Quality cost of bank-conflict-free encoding (see scripts/bench_bankfree.py).

Constraint: in every 16 consecutive chunks of a row (one half-warp's loads),
the codeword classes (tag mod 16 = shared-memory bank pair) are all distinct
('perm', 2 wavefronts per lookup) or each used at most twice ('cap2').
Exact per-group assignment (Hungarian over the 16 chunks x 16 classes, each
chunk taking its best codeword inside the class it gets).

Codebook arrangements (which codewords share a class):
  kmeans  : k-means index order as is (arbitrary)
  spread  : greedy max-min colouring, so near codewords land in different classes
            and every class covers the space
  refit   : spread, then 3 rounds of constrained encode + weighted centroid update

Same data, weights and metric as screen_neural_decoder.py (pooled weighted
relative error, no masks, G = 1). Baseline: the free unconstrained encode.

Run: python scripts/screen_bankfree.py [--layers 8,20] [--K 1024,4096]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shapequant import codebook as cb  # noqa: E402
from shapequant.calibrate import column_importance, load_calibration_sequences  # noqa: E402
from shapequant.masks import _prepare, _row_scale  # noqa: E402
from shapequant.pipeline import compressible_linears  # noqa: E402

NC = 16  # classes = bank pairs


def class_best(x, wt, c, batch=32768):
    """Per chunk and class: best weighted error and the codeword index achieving it."""
    K = c.shape[0]
    n = x.shape[0]
    be = torch.empty(n, NC, device=x.device)
    bi = torch.empty(n, NC, dtype=torch.long, device=x.device)
    c = c.float()
    for s in range(0, n, batch):
        xs, ws = x[s:s + batch], wt[s:s + batch]
        # sum_j w_j (x_j - c_kj)^2 = sum w x^2 - 2 sum w x c + sum w c^2
        e = (ws * xs * xs).sum(1, keepdim=True) - 2 * (ws * xs) @ c.T + ws @ (c * c).T     # [b, K]
        e = e.view(-1, K // NC, NC)                                                         # k = hi*16 + cls
        v, hi = e.min(1)
        be[s:s + batch] = v.clamp_min(0)
        bi[s:s + batch] = hi * NC + torch.arange(NC, device=x.device)
    return be, bi


def constrained_encode(x, wt, c, cap: int):
    be, bi = class_best(x, wt, c)
    n = x.shape[0]
    assert n % NC == 0
    cost = be.view(-1, NC, NC).cpu().numpy()
    choice = np.empty((cost.shape[0], NC), dtype=np.int64)
    for gi in range(cost.shape[0]):
        m = cost[gi] if cap == 1 else np.repeat(cost[gi], cap, axis=1)
        r, col = linear_sum_assignment(m)
        choice[gi, r] = col // cap
    cls = torch.from_numpy(choice).to(x.device).view(-1)
    idx = bi.gather(1, cls[:, None]).squeeze(1)
    err = be.gather(1, cls[:, None]).squeeze(1)
    return idx, err


def spread_classes(c: torch.Tensor, seed: int) -> torch.Tensor:
    """Permutation of codebook rows so that class k mod 16 sets are spread out."""
    K = c.shape[0]
    cap = K // NC
    D = torch.cdist(c.float(), c.float())
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(K, generator=g).to(c.device)
    cls = torch.full((K,), -1, dtype=torch.long, device=c.device)
    count = torch.zeros(NC, dtype=torch.long, device=c.device)
    for k in order.tolist():
        a = cls >= 0
        mins = torch.full((NC,), float("inf"), device=c.device)
        if a.any():
            mins = mins.scatter_reduce(0, cls[a], D[k, a], reduce="amin")
        mins[count >= cap] = -1.0
        best = int(torch.argmax(mins))
        cls[k] = best
        count[best] += 1
    # build new order: position hi*16 + cls
    newc = torch.empty_like(c)
    slot = torch.zeros(NC, dtype=torch.long)
    for k in range(K):
        cl = int(cls[k])
        newc[int(slot[cl]) * NC + cl] = c[k]
        slot[cl] += 1
    return newc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--layers", default="8,20")
    ap.add_argument("--K", default="1024,4096")
    ap.add_argument("--out", default="artifacts/prescreen/bankfree_0p5b.csv")
    ap.add_argument("--grouping", choices=("consecutive", "stride2"), default="consecutive",
                    help="which 16 chunks share a lookup step: cuda_gemv (consecutive) or proto_runs (stride2)")
    ap.add_argument("--fast", action="store_true", help="only the refit arrangement")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
    seqs = load_calibration_sequences(tok, {"dataset": "HuggingFaceFW/fineweb-edu", "subset": "sample-10BT",
                                            "split": "train", "num_sequences": 64, "context": 512})
    imp = column_importance(model, seqs, batch_size=8)
    layers = {int(v) for v in a.layers.split(",")}
    mats = [(n, m.weight.detach().float(), imp[n].to(dev))
            for n, m in compressible_linears(model) if int(n.split(".")[2]) in layers]
    del model
    torch.cuda.empty_cache()
    data = []
    for n, w, h in mats:
        R, C = w.shape
        rs = _row_scale(w)
        x, wt, *_ = _prepare(w, h, rs, torch.arange(C, device=dev)[None], torch.zeros(R, dtype=torch.long, device=dev),
                             4, None, 4)
        assert (C // 4) % NC == 0, n
        if a.grouping == "stride2":
            # proto_runs.py lookup groups: within each 32-chunk block, chunks 2l+e (l = 0..15) for e = 0, 1
            nch = C // 4
            assert nch % 32 == 0, n
            c = torch.arange(nch, device=dev).view(-1, 16, 2).transpose(1, 2).reshape(-1)
            idx = (torch.arange(R, device=dev)[:, None] * nch + c[None]).reshape(-1)
            x, wt = x[idx], wt[idx]
        data.append((n, x.contiguous(), wt.contiguous(), float((x.pow(2) * wt).sum())))
    del mats
    total = sum(d[3] for d in data)
    print(f"{len(data)} matrices from layers {sorted(layers)}", flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    def record(K, arrangement, constraint, errs, t0, base=None):
        rel = sum(errs) / total
        rows.append({"K": K, "arrangement": arrangement, "constraint": constraint, "pooled_wrel": rel,
                     "vs_free": (rel / base - 1) if base else 0.0, "seconds": time.time() - t0})
        tail = f"  {100 * (rel / base - 1):+6.1f}% vs free" if base else ""
        print(f"K={K:5d} {arrangement:8s} {constraint:5s} pooled wrel {rel:.5f}{tail}  {time.time() - t0:.0f}s", flush=True)
        with open(out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        return rel

    for K in [int(v) for v in a.K.split(",")]:
        t0 = time.time()
        books, errs = [], []
        for i, (n, x, wt, _) in enumerate(data):
            g = torch.Generator().manual_seed(i)
            sel = torch.randperm(x.shape[0], generator=g)[:1_000_000].to(dev)
            c = cb.kmeans(x[sel], K, 25, g, 65536, wt[sel])
            books.append(c)
            errs.append(float(cb.assign(x, c, 65536, wt)[1].sum()))
        base = record(K, "kmeans", "free", errs, t0)
        spread = [spread_classes(c, i) for i, c in enumerate(books)]
        for arr, bks in (() if a.fast else (("kmeans", books), ("spread", spread))):
            for cap, cname in ((1, "perm"), (2, "cap2")):
                t0 = time.time()
                errs = [float(constrained_encode(x, wt, c, cap)[1].sum()) for (n, x, wt, _), c in zip(data, bks)]
                record(K, arr, cname, errs, t0, base)
        for cap, cname in ((1, "perm"), (2, "cap2")):
            t0 = time.time()
            errs = []
            for (n, x, wt, _), c0 in zip(data, spread):
                c = c0.clone().float()
                for _ in range(3):
                    idx, _e = constrained_encode(x, wt, c, cap)
                    W = torch.zeros_like(c).index_add_(0, idx, wt)
                    S = torch.zeros_like(c).index_add_(0, idx, wt * x)
                    used = W > 0
                    c = torch.where(used, S / W.clamp_min(1e-12), c)
                errs.append(float(constrained_encode(x, wt, c, cap)[1].sum()))
            record(K, "refit", cname, errs, t0, base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
