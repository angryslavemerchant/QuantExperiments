"""Neural-decoder codebooks: can a tiny network computed from the tag bits
replace a free k-means codebook?

Codebook(theta) = f_theta(bits(t)) for every tag t in [0, 2^b), bits as +-1.
f = linear skip (b -> 4) + one hidden layer (b -> H -> 4, SiLU). H = 0 is the
pure linear decoder (sum of one 2-entry book per bit). Fitted per matrix by
alternating: weighted nearest-codeword assignment (tags are re-chosen freely),
then Adam on the weighted squared error with assignments fixed (via per-codeword
sufficient statistics, so each step is O(2^b)).

Baselines at the same tag bits: free k-means codebook, additive 2-book (exact
joint search), on the same normalized chunks and importance weights as the
masks prescreen (pooled weighted relative error, no masks, G = 1).

Decode cost per 4-weight group: b*H + 4*H + 4*b multiply-adds (+ H activations).

Run: python scripts/screen_neural_decoder.py [--layers 8,20] [--bits 10,12]
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


class Decoder(torch.nn.Module):
    def __init__(self, b: int, H: int, d: int = 4):
        super().__init__()
        self.skip = torch.nn.Linear(b, d)
        self.H = H
        if H:
            self.l1 = torch.nn.Linear(b, H)
            self.l2 = torch.nn.Linear(H, d)

    def forward(self, z):
        y = self.skip(z)
        if self.H:
            y = y + self.l2(torch.nn.functional.silu(self.l1(z)))
        return y


def bit_inputs(b: int, dev) -> torch.Tensor:
    t = torch.arange(1 << b, device=dev)
    return (((t[:, None] >> torch.arange(b, device=dev)) & 1).float() * 2 - 1)


def fit_neural(x, wt, b, H, gen_seed, rounds=40, steps=150, fit_n=300_000, batch=65536):
    dev = x.device
    torch.manual_seed(gen_seed)
    g = torch.Generator(device="cpu").manual_seed(gen_seed)
    sel = torch.randperm(x.shape[0], generator=g)[:fit_n].to(dev)
    xf, wf = x[sel], wt[sel]
    z = bit_inputs(b, dev)
    net = Decoder(b, H).to(dev)
    # scale the initial codebook to the data spread
    with torch.no_grad():
        std = xf.std(0)
        c0 = net(z)
        net.skip.weight.mul_((std / c0.std(0).clamp_min(1e-6))[:, None])
        net.skip.bias.copy_(xf.mean(0))
        if H:
            net.l2.weight.mul_(0.5 * (std / c0.std(0).clamp_min(1e-6))[:, None])
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    K = 1 << b
    for r in range(rounds):
        with torch.no_grad():
            C = net(z)
            idx, _ = cb.assign(xf, C, batch, wf)
            W = torch.zeros(K, 4, device=dev).index_add_(0, idx, wf)
            S = torch.zeros(K, 4, device=dev).index_add_(0, idx, wf * xf)
            norm = W.sum()
        for _ in range(steps):
            C = net(z)
            loss = ((W * C * C) - 2 * S * C).sum() / norm
            opt.zero_grad()
            loss.backward()
            opt.step()
        if r == rounds // 2:
            for gr in opt.param_groups:
                gr["lr"] = 1e-3
    with torch.no_grad():
        C = net(z)
    _, err = cb.assign(x, C, batch, wt)
    used = int(torch.unique(cb.assign(xf, C, batch, wf)[0]).numel())
    return float(err.sum()), used


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--layers", default="8,20")
    ap.add_argument("--bits", default="10,12")
    ap.add_argument("--H", default="0,2,4,8,16,32,64,128")
    ap.add_argument("--out", default="artifacts/prescreen/neural_decoder_0p5b.csv")
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
        perms = torch.arange(C, device=dev)[None]
        rm = torch.zeros(R, dtype=torch.long, device=dev)
        x, wt, *_ = _prepare(w, h, rs, perms, rm, 4, None, 4)
        data.append((n, x.contiguous(), wt.contiguous(), float((x.pow(2) * wt).sum())))
    del mats
    print(f"{len(data)} matrices from layers {sorted(layers)}", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    def record(bits, method, H, macs, errs, used, t0):
        rel = sum(errs) / sum(d[3] for d in data)
        rows.append({"bits": bits, "method": method, "H": H, "macs_per_group": macs, "pooled_wrel": rel,
                     "mean_used_codes": used, "seconds": time.time() - t0})
        print(f"b={bits:2d} {method:22s} H={str(H):>4s} MACs/grp {str(macs):>5s}  pooled wrel {rel:.5f}  "
              f"used codes ~{used}  {time.time() - t0:.0f}s", flush=True)
        with open(out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)

    for bits in [int(v) for v in a.bits.split(",")]:
        K = 1 << bits
        # free k-means
        t0 = time.time()
        errs = []
        for i, (n, x, wt, _) in enumerate(data):
            g = torch.Generator().manual_seed(i)
            sel = torch.randperm(x.shape[0], generator=g)[:1_000_000].to(dev)
            c = cb.kmeans(x[sel], K, 25, g, 65536, wt[sel])
            errs.append(float(cb.assign(x, c, 65536, wt)[1].sum()))
        record(bits, "free k-means", "-", "lookup", errs, K, t0)
        # additive 2 books
        t0 = time.time()
        errs = []
        ka = 1 << (bits // 2)
        for i, (n, x, wt, _) in enumerate(data):
            g = torch.Generator().manual_seed(i)
            sel = torch.randperm(x.shape[0], generator=g)[:1_000_000].to(dev)
            books = rvq_fit(x[sel], wt[sel], (ka, ka), 25, g, 65536)
            errs.append(float(rvq_encode(x, wt, books, 65536)[1].sum()))
        record(bits, f"additive {ka}+{ka}", "-", "2 lookups", errs, K, t0)
        for H in [int(v) for v in a.H.split(",")]:
            t0 = time.time()
            errs, used = [], []
            for i, (n, x, wt, _) in enumerate(data):
                e, u = fit_neural(x, wt, bits, H, i)
                errs.append(e)
                used.append(u)
            macs = bits * H + 4 * H + 4 * bits
            record(bits, "neural decoder", str(H), str(macs), errs, int(sum(used) / len(used)), t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
