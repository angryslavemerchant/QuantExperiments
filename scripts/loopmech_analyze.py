"""Loop mechanism study, step 1c: analyse the replay (scripts/loopmech_replay.py output).

  python scripts/loopmech_analyze.py --traces artifacts/loopmech/traces.jsonl --replay ~/loopmech/replay \
      --out artifacts/loopmech/analysis.json

1. p(</think>) behaviour per trace: max, how often the exit is "on the table" (p > 1e-2), and p at lines where the
   model announces it is done ("final", "Let's write", "Now produce", ...), finishing vs runaway traces.
2. A "ready to close" direction per recorded layer: mean residual at the closing point of finishing traces minus
   mean at their other decision points. Scored by leave-one-prompt-out AUC; then runaway traces are projected on it.
Saves directions (unit vectors, fp32) to <out>.pt for step 2.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re

FINAL_RE = re.compile(r"\b(final|let'?s (write|produce|answer|craft|compose|output)|now (write|produce|answer|craft|"
                      r"compose|output|respond)|ready to (write|answer)|that'?s it|done\.?$|good\.?$)", re.I)


def auc(pos, neg):
    import torch
    pos, neg = torch.as_tensor(pos, dtype=torch.float64), torch.as_tensor(neg, dtype=torch.float64)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = torch.cat([pos, neg])
    r = allv.argsort().argsort().double() + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--model", default=os.path.expanduser("~/models/q27-sq"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    end_id = tok.convert_tokens_to_ids("</think>")
    traces = {(t["prompt"], t["i"]): t for t in map(json.loads, open(a.traces))}

    per_trace, final_pts = [], collections.defaultdict(list)
    R = collections.defaultdict(list)   # layer -> list of [n, d] fp16
    lab = []                            # per stored resid row: (prompt, i, runaway, is_close, rel_pos, abs_tok, p_end)
    for f in sorted(glob.glob(os.path.join(os.path.expanduser(a.replay), "*.pt"))):
        d = torch.load(f)
        m = d["meta"]
        t = traces[(m["prompt"], m["i"])]
        gen, npr = t["token_ids"], m["n_prompt"]
        pos = d["pos"].tolist()
        pe = d["p_end"]
        # line text ending at each decision point
        prev = 0
        lines = []
        for p in pos:
            j = p - npr
            lines.append(tok.decode(gen[prev:j + 1]).strip().splitlines()[-1:] or [""])
            prev = j + 1
        lines = [x[0] for x in lines]
        is_close = (d["next_id"] == end_id)
        per_trace.append({"prompt": m["prompt"], "i": m["i"], "runaway": m["runaway"], "think_len": m["think_len"],
                          "n_points": len(pos), "p_end_max": float(pe.max()),
                          "n_p_gt_1e-2": int((pe > 1e-2).sum()), "n_p_gt_1e-1": int((pe > 1e-1).sum()),
                          "closed": bool(is_close.any())})
        for k, (ln, p) in enumerate(zip(lines, pe.tolist())):
            if FINAL_RE.search(ln) and not is_close[k]:
                top = [tok.decode([int(x)]) for x in d["top_ids"][k][:3]]
                final_pts[(m["prompt"], m["runaway"])].append((p, ln[-80:], top))
        idx = d["resid_idx"].tolist()
        for li, x in d["resid"].items():
            R[li].append(x)
        for k in idx:
            lab.append((m["prompt"], m["i"], m["runaway"], bool(is_close[k]), k / max(1, len(pos) - 1),
                        pos[k] - npr, float(pe[k])))

    # ---- 1. exit availability
    print("== p(</think>) at line breaks (per trace) ==")
    g = collections.defaultdict(list)
    for r in per_trace:
        g[(r["prompt"], r["runaway"])].append(r)
    summary = {}
    for k in sorted(g):
        v = g[k]
        mx = sorted(x["p_end_max"] for x in v)
        s = {"n": len(v), "median_max_p_end": mx[len(mx) // 2],
             "frac_traces_ever_p_gt_1e-1": sum(x["n_p_gt_1e-1"] > 0 for x in v) / len(v),
             "mean_points_p_gt_1e-2": sum(x["n_p_gt_1e-2"] for x in v) / len(v)}
        summary[f"{k[0]}|{'runaway' if k[1] else 'finished'}"] = s
        print(f"{k[0]:14s} {'RUNAWAY ' if k[1] else 'finished'} n={s['n']:3d}  median max p_end "
              f"{s['median_max_p_end']:.1e}  ever>0.1: {s['frac_traces_ever_p_gt_1e-1']:.0%}  "
              f"pts>0.01 per trace: {s['mean_points_p_gt_1e-2']:.1f}")

    print("\n== lines that announce the end ('final', 'Let's write', ...) but were not followed by </think> ==")
    fin = {}
    for k in sorted(final_pts):
        v = final_pts[k]
        ps = sorted(x[0] for x in v)
        fin[f"{k[0]}|{'runaway' if k[1] else 'finished'}"] = {"n": len(v), "median_p_end": ps[len(ps) // 2]}
        print(f"{k[0]:14s} {'RUNAWAY ' if k[1] else 'finished'} {len(v):4d} such lines, median p_end "
              f"{ps[len(ps) // 2]:.1e}, p>0.1 in {sum(p > 0.1 for p in ps)}")
        for p, ln, top in sorted(v, key=lambda x: -x[0])[:2]:
            print(f"      p={p:.2e} top={top} | {ln!r}")

    # ---- 2. ready-to-close direction per layer
    print("\n== 'ready to close' direction: closing point vs other points of finishing coding traces ==")
    prompts = sorted({x[0] for x in lab})
    close = torch.tensor([x[3] for x in lab])
    run = torch.tensor([x[2] for x in lab])
    is_g = torch.tensor([x[0] == "gsm8k" for x in lab])
    pr = [x[0] for x in lab]
    dirs, layer_stats = {}, {}
    for li in sorted(R):
        X = torch.cat(R[li], 0).float()
        assert X.shape[0] == len(lab)
        fin_mask = ~run
        aucs = []
        for hold in prompts:
            if hold == "gsm8k":
                continue
            ho = torch.tensor([p == hold for p in pr])
            tr = fin_mask & ~ho
            dvec = X[tr & close].mean(0) - X[tr & ~close].mean(0)
            dvec = dvec / dvec.norm()
            s = X @ dvec
            te = fin_mask & ho
            aucs.append(auc(s[te & close], s[te & ~close]))
        dvec = X[fin_mask & close].mean(0) - X[fin_mask & ~close].mean(0)
        dvec = dvec / dvec.norm()
        dirs[li] = dvec
        s = X @ dvec
        thr = s[fin_mask & close].quantile(0.1)  # 90% of real closing points score above this
        run_pts = run & ~is_g
        # within runaway traces: what fraction of points reach a closing-like score, and when
        frac_run = float((s[run_pts] > thr).float().mean())
        frac_fin_nonclose = float((s[fin_mask & ~close & ~is_g] > thr).float().mean())
        # correlation of the projection with p_end over all points (log scale)
        lp = torch.tensor([x[6] for x in lab]).clamp_min(1e-12).log()
        corr = float(torch.corrcoef(torch.stack([s, lp]))[0, 1])
        layer_stats[li] = {"lopo_auc": [round(x, 3) for x in aucs], "mean_lopo_auc": sum(aucs) / len(aucs),
                           "runaway_pts_above_close_p10": frac_run,
                           "finished_nonclose_pts_above_close_p10": frac_fin_nonclose, "corr_logp_end": corr}
        print(f"L{li:2d}  LOPO AUC {sum(aucs) / len(aucs):.3f} {['%.2f' % x for x in aucs]}  corr(proj, log p_end) "
              f"{corr:+.2f}  pts at closing level: runaway {frac_run:.1%} vs finished-nonclose {frac_fin_nonclose:.1%}")

    out = {"per_prompt": summary, "final_lines": fin, "layers": layer_stats, "per_trace": per_trace}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    torch.save({"dirs": dirs, "note": "unit diff-of-means close vs non-close, finishing traces"},
               os.path.splitext(a.out)[0] + "_dirs.pt")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
