"""Sign-split codebooks: each group stores its d sign bits plus an index into a
book of magnitude vectors; decoded = sign(x) * m.

With signs fixed to sign(x), the weighted error is sum wt (|x| - m)^2, so the
magnitude book is simply a weighted k-means (or additive fit) on |x|. The
effective codebook is the magnitude book mirrored into every orthant
(K_mag * 2^d codewords) from a table 2^d times smaller.

Matched-bit comparisons (same chunks/weights as screen_d8_additive.py):
  2.00 bpw: d8 8 signs + K256        vs d4 K256, d8 additive 256+256
  2.25 bpw: d8 8 signs + K1024       vs d4 K512, d8 additive 512+512
  2.50 bpw: d4 4 signs + K64         vs d4 K1024
            d8 8 signs + K4096       vs d8 additive 1024+1024
            d8 8 signs + 64+64 add.  (tiny tables)
  3.00 bpw: d4 4 signs + K256        vs d4 K4096

Run: python scripts/screen_signsplit.py [--layers 8,20]
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
from shapequant.masks import _prepare, _row_scale  # noqa: E402
from shapequant.pipeline import compressible_linears  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from screen_d8_additive import fit_additive  # noqa: E402


def kmeans_err(x, wt, K, i):
    g = torch.Generator().manual_seed(i)
    sel = torch.randperm(x.shape[0], generator=g)[:1_000_000].to(x.device)
    c = cb.kmeans(x[sel], K, 25, g, 65536, wt[sel]).half()
    return float(cb.assign(x, c, 65536, wt)[1].sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--layers", default="8,20")
    ap.add_argument("--out", default="artifacts/prescreen/signsplit_0p5b.csv")
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
            data[d].append((x.contiguous(), wt.contiguous()))
    del mats
    total = sum(float((x.pow(2) * wt).sum()) for x, wt in data[4])
    print(f"{len(data[4])} matrices from layers {sorted(layers)}", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    def record(bpw, method, errs, t0, table_kb):
        rel = sum(errs) / total
        rows.append({"bpw": bpw, "method": method, "pooled_wrel": rel, "table_kb_fp16": table_kb,
                     "seconds": round(time.time() - t0, 1)})
        print(f"{bpw:.2f} bpw  {method:28s} pooled wrel {rel:.5f}  table {table_kb:7.2f} KB  "
              f"{time.time() - t0:5.0f}s", flush=True)
        with open(out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)

    def free(d, K, bpw):
        t0 = time.time()
        record(bpw, f"d{d} free K{K}", [kmeans_err(x, wt, K, i) for i, (x, wt) in enumerate(data[d])],
               t0, K * d * 2 / 1024)

    def signs(d, K, bpw):
        t0 = time.time()
        record(bpw, f"d{d} {d} signs + K{K}",
               [kmeans_err(x.abs(), wt, K, i) for i, (x, wt) in enumerate(data[d])], t0, K * d * 2 / 1024)

    def signs_add(d, K, bpw):
        t0 = time.time()
        errs = []
        for i, (x, wt) in enumerate(data[d]):
            e, _ = fit_additive(x.abs(), wt, K, K, torch.Generator().manual_seed(i))
            errs.append(e)
        record(bpw, f"d{d} {d} signs + add {K}+{K}", errs, t0, 2 * K * d * 2 / 1024)

    # 2.5 bpw, d4 (references from screen_d8_additive: d4 K1024 0.03487, d8 add 1024+1024 0.03056)
    signs(4, 64, 2.5)
    free(4, 1024, 2.5)
    # 2.5 bpw, d8
    signs(8, 4096, 2.5)
    signs_add(8, 64, 2.5)
    # 2.0 / 2.25 bpw, d8 (refs: d4 K256 0.06967, d8 add 256+256 0.06138; d4 K512 0.04939, add 512+512 0.04350)
    signs(8, 256, 2.0)
    signs(8, 1024, 2.25)
    # 3.0 bpw, d4
    signs(4, 256, 3.0)
    free(4, 4096, 3.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
