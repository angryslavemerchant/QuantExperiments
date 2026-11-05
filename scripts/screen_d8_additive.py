"""Longer vectors with additive books: d=8, two books of Ka entries each
(codeword = A[i] + B[j]), against the free d=4 codebook at the same bits.

bits per weight = (log2 Ka + log2 Kb) / 8 for d=8, log2 K / 4 for d=4:
  2.00 bpw: d4 K256      vs  d8 256+256   (AQLM 2x8 regime)
  2.25 bpw: d4 K512      vs  d8 512+512
  2.50 bpw: d4 K1024     vs  d8 1024+1024 (tables 2 x 1024 x 8 fp16 = 32 KB, fits smem)
plus the d4 32+32 additive point (format A, lost on 7B) for reference.

Encoding: beam search (top-`beam` A entries by their own distance, then the
exact best B for each), then coordinate refinement. Books: greedy residual
k-means start, then alternating closed-form updates (per-code weighted mean of
the target given fixed tags). Same normalized chunks and importance weights as
screen_neural_decoder.py (pooled weighted relative error, no masks, G = 1).

Run: python scripts/screen_d8_additive.py [--layers 8,20]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shapequant import codebook as cb  # noqa: E402
from shapequant.calibrate import column_importance, load_calibration_sequences  # noqa: E402
from shapequant.masks import _prepare, _row_scale, rvq_encode, rvq_fit  # noqa: E402
from shapequant.pipeline import compressible_linears  # noqa: E402


def _wdist(x, wt, c, c2):
    """[n, K] weighted squared distance minus the constant sum(wt x^2)."""
    return wt @ c2.T - 2.0 * ((wt * x) @ c.T)


def beam_encode(x, wt, A, B, beam=16, batch=4096, refine=2):
    A = A.float()
    B = B.float()
    A2, B2 = A * A, B * B
    n = x.shape[0]
    ia = torch.empty(n, dtype=torch.long, device=x.device)
    ib = torch.empty(n, dtype=torch.long, device=x.device)
    for s in range(0, n, batch):
        xb, wb = x[s:s + batch], wt[s:s + batch]
        top = _wdist(xb, wb, A, A2).topk(beam, dim=1, largest=False).indices      # [b, beam]
        r = xb[:, None, :] - A[top]                                               # [b, beam, d]
        wr = wb[:, None, :]
        d = (wr * r * r).sum(-1)[..., None] + torch.einsum("bjd,kd->bjk", wr.expand_as(r), B2) \
            - 2.0 * torch.einsum("bjd,kd->bjk", wr * r, B)                        # [b, beam, Kb]
        v, j = d.min(-1)
        k = v.argmin(1)
        rows = torch.arange(xb.shape[0], device=x.device)
        ia[s:s + batch] = top[rows, k]
        ib[s:s + batch] = j[rows, k]
    for _ in range(refine):
        ia, _ = cb.assign(x - B[ib], A, 65536, wt)
        ib, _ = cb.assign(x - A[ia], B, 65536, wt)
    err = ((x - A[ia] - B[ib]).pow(2) * wt).sum(1)
    return ia, ib, err


def ls_update(book, tags, target, wt):
    """Per-code, per-coordinate weighted mean of the target (exact given tags)."""
    K, d = book.shape
    num = torch.zeros(K, d, dtype=torch.float64, device=target.device)
    den = torch.zeros(K, d, dtype=torch.float64, device=target.device)
    num.index_add_(0, tags, (wt * target).double())
    den.index_add_(0, tags, wt.double())
    out = book.float().clone()
    ok = den > 1e-12
    out[ok] = (num[ok] / den[ok]).float()
    return out


def fit_additive(x, wt, Ka, Kb, gen, rounds=6, beam=16):
    A = cb.kmeans(x, Ka, 25, gen, 65536, wt).float()
    ia, _ = cb.assign(x, A, 65536, wt)
    B = cb.kmeans(x - A[ia], Kb, 25, gen, 65536, wt).float()
    hist = []
    for _ in range(rounds):
        ia, ib, err = beam_encode(x, wt, A, B, beam)
        hist.append(float(err.sum()))
        A = ls_update(A, ia, x - B[ib], wt)
        B = ls_update(B, ib, x - A[ia], wt)
    ia, ib, err = beam_encode(x, wt, A.half().float(), B.half().float(), beam)
    hist.append(float(err.sum()))
    return float(err.sum()), hist


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--layers", default="8,20")
    ap.add_argument("--beam", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--out", default="artifacts/prescreen/d8_additive_0p5b.csv")
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
    data = {4: [], 8: []}
    for n, w, h in mats:
        R, C = w.shape
        rs = _row_scale(w)
        perms = torch.arange(C, device=dev)[None]
        rm = torch.zeros(R, dtype=torch.long, device=dev)
        for d in (4, 8):
            x, wt, *_ = _prepare(w, h, rs, perms, rm, d, None, 4)
            data[d].append((n, x.contiguous(), wt.contiguous(), float((x.pow(2) * wt).sum())))
    del mats
    total = sum(t[3] for t in data[4])
    print(f"{len(data[4])} matrices from layers {sorted(layers)}", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    def record(bpw, method, errs, t0, extra=""):
        rel = sum(errs) / total
        rows.append({"bpw": bpw, "method": method, "pooled_wrel": rel, "seconds": round(time.time() - t0, 1),
                     "note": extra})
        print(f"{bpw:.2f} bpw  {method:24s} pooled wrel {rel:.5f}  {time.time() - t0:5.0f}s  {extra}", flush=True)
        with open(out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)

    for lk in (8, 9, 10):
        bpw = lk / 4
        t0 = time.time()
        errs = []
        for i, (n, x, wt, _) in enumerate(data[4]):
            g = torch.Generator().manual_seed(i)
            sel = torch.randperm(x.shape[0], generator=g)[:1_000_000].to(dev)
            c = cb.kmeans(x[sel], 1 << lk, 25, g, 65536, wt[sel]).half()
            errs.append(float(cb.assign(x, c, 65536, wt)[1].sum()))
        record(bpw, f"d4 free K{1 << lk}", errs, t0)

        t0 = time.time()
        errs, gains = [], []
        Ka = 1 << lk
        for i, (n, x, wt, _) in enumerate(data[8]):
            g = torch.Generator().manual_seed(i)
            e, hist = fit_additive(x, wt, Ka, Ka, g, rounds=a.rounds, beam=a.beam)
            errs.append(e)
            gains.append(hist[0] / max(hist[-1], 1e-12))
        record(bpw, f"d8 additive {Ka}+{Ka}", errs, t0,
               f"alternation gain x{sum(gains) / len(gains):.3f}")

    t0 = time.time()
    errs = []
    for i, (n, x, wt, _) in enumerate(data[4]):
        g = torch.Generator().manual_seed(i)
        sel = torch.randperm(x.shape[0], generator=g)[:1_000_000].to(dev)
        books = rvq_fit(x[sel], wt[sel], (32, 32), 25, g, 65536)
        errs.append(float(rvq_encode(x, wt, books, 65536)[1].sum()))
    record(2.5, "d4 additive 32+32", errs, t0, "format A (lost on 7B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
