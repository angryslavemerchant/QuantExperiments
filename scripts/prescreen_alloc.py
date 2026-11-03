"""Local pre-screen: per-matrix bit allocation (mixed codebook sizes) on 0.5B.

    python scripts/prescreen_alloc.py [--model Qwen/Qwen2.5-0.5B]

Every block linear is fitted once per candidate codebook size (d=4,
K in {256, 1024, 4096}, importance-weighted act_row, no masks). An allocation
picks one K per matrix under a total-bit budget (real bits: tags + codebook +
row scales). Allocations are scored end to end: mean KL vs the original on
WikiText-2 windows. Methods:

- uniform     : every matrix the same K (the current recipe)
- fisher      : minimize sum_i 0.5 * sum F (w - w^)^2, with F = E[g^2] the
                empirical diagonal Fisher of next-token loss on calibration
                text (as in SqueezeLLM / HAWQ-style sensitivity)
- outerr      : minimize the imatrix-weighted layer-output error sum h (w - w^)^2
                (no downstream sensitivity: every layer's output counts equally)
- heuristic   : GGUF-style: attn_v, ffn_down and the first/last two blocks are
                upgraded first at each level

Optimization: Lagrangian (per matrix argmin err + lambda * bits, lambda by
bisection), then greedy fill of leftover budget by best marginal gain per bit.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shapequant import codebook as cb  # noqa: E402
from shapequant.calibrate import column_importance, load_calibration_sequences  # noqa: E402
from shapequant.evaluate import Scorer, load_wikitext_windows  # noqa: E402
from shapequant.masks import compress_masked  # noqa: E402
from shapequant.pipeline import compressible_linears  # noqa: E402

KS = (256, 1024, 4096)


def fisher_diag(model, seqs: torch.Tensor, names: list[str]) -> dict[str, torch.Tensor]:
    """Empirical diagonal Fisher E[g^2] of next-token CE, per block-linear weight (CPU fp32)."""
    mods = dict(compressible_linears(model))
    for p in model.parameters():
        p.requires_grad_(False)
    acc = {}
    for n in names:
        mods[n].weight.requires_grad_(True)
        # accumulate on the CPU: a GPU fp32 copy of every weight does not fit
        # next to the model on a 6 GB card (it spills into shared memory and crawls)
        acc[n] = torch.zeros(mods[n].weight.shape, dtype=torch.float32)
    dev = next(model.parameters()).device
    for i in range(seqs.shape[0]):
        ids = seqs[i : i + 1].to(dev)
        logits = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
        loss = F.cross_entropy(logits, ids[0, 1:])
        loss.backward()
        for n in names:
            g = mods[n].weight.grad
            acc[n].add_(g.float().pow(2).cpu())
            mods[n].weight.grad = None
        if (i + 1) % 8 == 0:
            print(f"  fisher {i + 1}/{seqs.shape[0]}", flush=True)
    for n in names:
        mods[n].weight.requires_grad_(False)
    return {n: a / seqs.shape[0] for n, a in acc.items()}


def lagrangian(err: list[list[float]], bits: list[list[int]], budget: int) -> list[int]:
    """Per-matrix choice index minimizing sum err under sum bits <= budget."""
    def pick(lam):
        return [min(range(len(KS)), key=lambda k: e[k] + lam * b[k]) for e, b in zip(err, bits)]

    def total(ch):
        return sum(b[k] for b, k in zip(bits, ch))

    lo, hi = 0.0, 1.0
    while total(pick(hi)) > budget:
        hi *= 10
    for _ in range(80):  # bisect in log space
        mid = math.sqrt(lo * hi) if lo > 0 else hi / 1e6
        if total(pick(mid)) > budget:
            lo = mid
        else:
            hi = mid
    ch = pick(hi)
    # greedy fill of leftover budget: best error reduction per extra bit
    while True:
        spare = budget - total(ch)
        best, gain = None, 0.0
        for i, k in enumerate(ch):
            for k2 in range(k + 1, len(KS)):
                db = bits[i][k2] - bits[i][k]
                if db <= spare:
                    g = (err[i][k] - err[i][k2]) / db
                    if g > gain:
                        best, gain = (i, k2), g
        if best is None:
            return ch
        ch[best[0]] = best[1]


def heuristic(names: list[str], bits: list[list[int]], budget: int, n_layers: int) -> list[int]:
    def important(n):
        layer = int(n.split(".")[2])
        return ("v_proj" in n or "down_proj" in n or layer < 2 or layer >= n_layers - 2)

    ch = [0] * len(names)
    used = sum(b[0] for b in bits)
    for level in range(1, len(KS)):
        for want in (True, False):
            for i, n in enumerate(names):
                if important(n) == want and ch[i] == level - 1:
                    db = bits[i][level] - bits[i][level - 1]
                    if used + db <= budget:
                        ch[i] = level
                        used += db
    return ch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--fisher-seqs", type=int, default=64)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--fit-chunks", type=int, default=200_000)
    ap.add_argument("--out", default="artifacts/prescreen/alloc_0p5b.csv")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda"
    t00 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev).eval()
    lin = compressible_linears(model)
    names = [n for n, _ in lin]
    n_layers = model.config.num_hidden_layers
    ccfg = {"dataset": "HuggingFaceFW/fineweb-edu", "subset": "sample-10BT", "split": "train",
            "num_sequences": 64, "context": 512}
    seqs = load_calibration_sequences(tok, ccfg)
    imp = column_importance(model, seqs, batch_size=8)
    print(f"[{time.time() - t00:.0f}s] importance done", flush=True)
    fisher = fisher_diag(model, seqs[: args.fisher_seqs], names)
    print(f"[{time.time() - t00:.0f}s] importance + Fisher done", flush=True)

    originals = {n: m.weight.detach().clone() for n, m in lin}
    decoded: dict[tuple[str, int], torch.Tensor] = {}
    err_f, err_o, bits = [], [], []
    for n, m in lin:
        w = originals[n].float()
        h = imp[n].to(dev)
        fz = fisher[n].to(dev)
        ef, eo, bb = [], [], []
        for K in KS:
            fit = compress_masked(w, h, 4, K, seed=cb.layer_seed(0, n), iters=args.iters,
                                  max_fit_chunks=args.fit_chunks)
            dw = (w - fit.decoded).pow(2)
            ef.append(0.5 * float((fz * dw).sum()))
            eo.append(float((dw * h[None, :]).sum()))
            bb.append(fit.bits["tags"] + fit.bits["codebook"] + fit.bits["row_scales"])
            decoded[(n, K)] = fit.decoded.to(torch.bfloat16).cpu()
        err_f.append(ef)
        if len(err_f) % 14 == 0:
            print(f"  fitted {len(err_f)}/{len(names)} matrices ({time.time() - t00:.0f}s)", flush=True)
        err_o.append(eo)
        bits.append(bb)
    n_weights = sum(originals[n].numel() for n in names)
    print(f"[{time.time() - t00:.0f}s] fits done ({len(names)} matrices x {len(KS)} K)", flush=True)

    windows = load_wikitext_windows(tok, {"dataset": "Salesforce/wikitext", "subset": "wikitext-2-raw-v1",
                                          "split": "test", "context": 2048, "num_windows": args.windows})
    scorer = Scorer(model, windows, logit_chunk=256)
    mods = dict(lin)

    def evaluate(choice):
        with torch.no_grad():
            for n, k in zip(names, choice):
                mods[n].weight.copy_(decoded[(n, KS[k])].to(dev))
        r = scorer.score(model)
        with torch.no_grad():
            for n in names:
                mods[n].weight.copy_(originals[n])
        return r

    uni_bits = [sum(b[k] for b in bits) for k in range(len(KS))]
    budgets = {"b2.25": (uni_bits[0] + uni_bits[1]) // 2, "b2.5": uni_bits[1],
               "b2.75": (uni_bits[1] + uni_bits[2]) // 2}
    plans = [(f"uniform_K{K}", f"u{K}", [k] * len(names)) for k, K in enumerate(KS)]
    for bname, B in budgets.items():
        plans.append((f"fisher_{bname}", bname, lagrangian(err_f, bits, B)))
        plans.append((f"outerr_{bname}", bname, lagrangian(err_o, bits, B)))
        plans.append((f"heuristic_{bname}", bname, heuristic(names, bits, B, n_layers)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for pname, bname, ch in plans:
        r = evaluate(ch)
        tb = sum(b[k] for b, k in zip(bits, ch))
        hist = [ch.count(k) for k in range(len(KS))]
        by_type = {}
        for n, k in zip(names, ch):
            t = n.rsplit(".", 1)[-1]
            by_type.setdefault(t, []).append(KS[k])
        mix = " ".join(f"{t}:{sum(math.log2(v) for v in vs) / len(vs):.1f}" for t, vs in by_type.items())
        row = {"plan": pname, "budget": bname, "bpw": tb / n_weights, "mean_kl": r["mean_kl"],
               "p99_kl": r["p99_kl"], "top1": r["top1_agree"], "ppl": r["perplexity"],
               "fisher_err": sum(e[k] for e, k in zip(err_f, ch)),
               "n256": hist[0], "n1024": hist[1], "n4096": hist[2], "mean_log2K_by_type": mix}
        rows.append(row)
        print(f"{pname:20s} bpw {row['bpw']:.3f}  KL {row['mean_kl']:.4f}  p99 {row['p99_kl']:.3f}  "
              f"top1 {row['top1']:.2f}  ppl {row['ppl']:.2f}  K-hist {hist}  {mix}", flush=True)
        with open(out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    print(f"[{time.time() - t00:.0f}s] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
