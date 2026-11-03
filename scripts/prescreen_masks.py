"""Local pre-screen of Stage 3/4 variants on real matrices (no rental).

    python scripts/prescreen_masks.py [--model Qwen/Qwen2.5-0.5B] [--d 4 --K 256]

Metric: importance-weighted relative reconstruction error
sum h_c (w - w^)^2 / sum h_c w^2, the diagonal proxy for layer-output error,
pooled over the selected matrices (each matrix weighted by its output energy).
Bits per weight are reported per term. Mask/order storage (M x C x 16 bits per
matrix) is listed separately because it shrinks with matrix size: 16*M/R bpw.
A proxy screen: rankings should transfer to 7B, magnitudes may not.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shapequant.calibrate import column_importance, load_calibration_sequences  # noqa: E402
from shapequant import codebook as cb  # noqa: E402
from shapequant.masks import compress_masked  # noqa: E402
from shapequant.pipeline import compressible_linears  # noqa: E402


def variants(K: int):
    return [
        # round 3: residual (two-stage) codebooks, bits-matched by tag bits
        ("b20_single_d4_k256", dict(d=4, K=256)),                       # 2.0 tag bpw
        ("b20_rvq_d8_256x2", dict(d=8, K=(256, 256))),                  # 2.0
        ("b20_rvq_d6_64x2", dict(d=6, K=(64, 64))),                     # 2.0
        ("b20_rvq_d8_256x2_masks16", dict(d=8, K=(256, 256), M=16, init="identity+random")),
        ("b25_single_d4_k1024", dict(d=4, K=1024)),                     # 2.5
        ("b25_rvq_d8_1024x2", dict(d=8, K=(1024, 1024))),               # 2.5
        ("b25_rvq_d8_1024x2_masks16", dict(d=8, K=(1024, 1024), M=16, init="identity+random")),
    ] + [
        ("base_seed1", dict(seed_offset=1)),                 # k-means noise floor
        ("base_seed2", dict(seed_offset=2)),
        ("idrand16", dict(M=16, init="identity+random")),   # free seeded orders
        ("idrand64", dict(M=64, init="identity+random", rounds=3)),
        ("rand16", dict(M=16, init="random")),
        ("scales_idrand16", dict(B=32, scale_bits=4, M=16, init="identity+random")),
    ] + [
        ("base", dict()),
        ("base_K2x", dict(K=2 * K)),                         # bits control for local scales
        ("sort", dict(init="sort")),                         # Stage 3 shuffle
        ("spec_masks16", dict(M=16, init="random+sort")),    # Stage 4 as specified
        ("learned4", dict(M=4, init="cluster", update=True)),
        ("learned16", dict(M=16, init="cluster", update=True)),
        ("learned64", dict(M=64, init="cluster", update=True)),
        ("scales", dict(B=32, scale_bits=4)),
        ("scales_sort", dict(B=32, scale_bits=4, init="sort")),
        ("scales_learned16", dict(B=32, scale_bits=4, M=16, init="cluster", update=True)),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--layers", default="2,8,14,20")
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--only", default="", help="comma-separated variant names")
    ap.add_argument("--out", default="artifacts/prescreen/masks_0p5b.csv")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev).eval()
    ccfg = {"dataset": "HuggingFaceFW/fineweb-edu", "subset": "sample-10BT", "split": "train",
            "num_sequences": 64, "context": 512}
    seqs = load_calibration_sequences(tok, ccfg)
    imp = column_importance(model, seqs, batch_size=8)
    layers = {int(x) for x in args.layers.split(",")}
    mats = [(n, m.weight.detach().float(), imp[n].to(dev))
            for n, m in compressible_linears(model) if int(n.split(".")[2]) in layers]
    del model
    torch.cuda.empty_cache()
    print(f"{len(mats)} matrices from layers {sorted(layers)}; d={args.d} K={args.K}", flush=True)

    want = set(args.only.split(",")) if args.only else None
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for vname, kw in variants(args.K):
        if want and vname not in want:
            continue
        kw = dict(kw)
        K = kw.pop("K", args.K)
        d = kw.pop("d", args.d)
        seed_offset = kw.pop("seed_offset", 0)
        t0 = time.time()
        num = den = 0.0
        bits: dict[str, float] = {}
        n_weights = 0
        per = []
        for name, w, h in mats:
            fit = compress_masked(w, h, d, K, seed=cb.layer_seed(seed_offset, name), **kw)
            e = ((w - fit.decoded).pow(2) * h[None, :]).sum().item()
            s = (w.pow(2) * h[None, :]).sum().item()
            num += e
            den += s
            per.append(e / s)
            for k, v in fit.bits.items():
                bits[k] = bits.get(k, 0) + v
            n_weights += w.numel()
        bpw = {k: v / n_weights for k, v in bits.items()}
        core = sum(v for k, v in bpw.items() if k != "orders")
        row = {"variant": vname, "d": d, "K": str(K), "pooled_wrel": num / den,
               "bpw_tags": bpw["tags"],
               "mean_wrel": sum(per) / len(per), "bpw_core": core, "bpw_orders": bpw["orders"],
               "bpw_block_scales": bpw["block_scales"], "seconds": time.time() - t0}
        rows.append(row)
        print(f"{vname:28s} tags {bpw['tags']:.3f}  pooled wrel {row['pooled_wrel']:.5f}  mean {row['mean_wrel']:.5f}  "
              f"bpw core {core:.3f} (+orders {bpw['orders']:.3f})  {row['seconds']:.0f}s", flush=True)
        with open(out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    base = next((r for r in rows if r["variant"] in ("base", "b20_single_d4_k256")), None)
    if base:
        print("\nrelative to base (pooled):")
        for r in rows:
            print(f"  {r['variant']:28s} {100 * (r['pooled_wrel'] / base['pooled_wrel'] - 1):+6.1f}%  "
                  f"(+{r['bpw_core'] - base['bpw_core']:.3f} bpw core, +{r['bpw_orders']:.3f} orders)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
