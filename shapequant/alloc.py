"""Per-matrix bit allocation: which codebook size K each block linear gets.

Screened on 0.5B (scripts/prescreen_alloc.py): at a matched total-bit budget,
allocating by Fisher-weighted error beat uniform K by ~13% KL at 2.5 bpw, and
a GGUF-style rule (attn_v, ffn_down, first/last two blocks upgraded first) was
best at ~2.8 bpw. Both are offered here.

Procedure (run-level, before the per-matrix fits):
1. Fisher: F = E[g^2], empirical diagonal Fisher of next-token loss on the
   calibration text, per block-linear weight.
2. Proxy fits: every matrix fitted once per candidate K with the plain
   importance-weighted codebook (light k-means). Damage estimate
   e[i][K] = 0.5 * sum F (w - w^_K)^2; cost b[i][K] = real bits.
3. Choose: Lagrangian over per-matrix choices to meet the budget, then fill
   the leftover greedily (fisher); or the GGUF-style priority order (heuristic).
The final fits (full recipe) then use the chosen K per matrix.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from . import codebook as cb
from . import masks as mk
from .pipeline import compressible_linears


def fisher_diag(model, seqs: torch.Tensor, log=print, names: list[str] | None = None) -> dict[str, torch.Tensor]:
    """E[g^2] of next-token CE per block-linear weight (float32, on the model's device
    when it fits, else CPU). names: only these matrices (the others get no gradient; each
    weight's gradient is the same either way), so a big model can be done in groups."""
    mods = dict(compressible_linears(model))
    if names is not None:
        mods = {n: mods[n] for n in names}
    for p in model.parameters():
        p.requires_grad_(False)
    dev = next(model.parameters()).device
    if dev.type == "cuda":
        # memory freed earlier (e.g. the full input Hessians, ~44 GB at 7B, moved to the CPU) stays
        # in PyTorch's cache and would read as used, sending the accumulators to the CPU: one 26 GB
        # device-to-host copy + single-core add per sequence (~25 s each instead of ~1 s)
        torch.cuda.empty_cache()
    free = torch.cuda.mem_get_info()[0] if dev.type == "cuda" else 0
    need = sum(m.weight.numel() for m in mods.values()) * 4
    # peak ~1.6x need: fp32 accumulators + bf16 grads + activations of one sequence
    acc_dev = dev if free > 2.0 * need else torch.device("cpu")
    log(f"  fisher accumulators on {acc_dev} (free {free / 1e9:.0f} GB, need {need / 1e9:.0f} GB)")
    acc = {}
    for n, m in mods.items():
        m.weight.requires_grad_(True)
        acc[n] = torch.zeros(m.weight.shape, dtype=torch.float32, device=acc_dev)
    t0 = time.time()
    try:
        for i in range(seqs.shape[0]):
            ids = seqs[i : i + 1].to(dev)
            logits = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
            F.cross_entropy(logits, ids[0, 1:]).backward()
            del logits
            for n, m in mods.items():
                acc[n].add_(m.weight.grad.float().pow(2).to(acc_dev))
                m.weight.grad = None
            if (i + 1) % 16 == 0:
                log(f"  fisher {i + 1}/{seqs.shape[0]} ({time.time() - t0:.0f}s)")
    finally:
        for m in mods.values():
            m.weight.requires_grad_(False)
            m.weight.grad = None
    return {n: a.div_(seqs.shape[0]) for n, a in acc.items()}


def proxy_table(originals: dict, importance: dict, fisher: dict, names: list[str], d: int,
                Ks: tuple[int, ...], seed: int, iters: int, max_fit_chunks: int, batch: int,
                device, log=print) -> tuple[list[list[float]], list[list[float]]]:
    """err[i][k], bits[i][k] for every matrix and candidate K (light plain fits)."""
    err, bits = [], []
    t0 = time.time()
    for j, n in enumerate(names):
        w = originals[n].to(device, torch.float32)
        h = importance[n].to(device, torch.float32)
        fz = fisher[n].to(device)
        e, b = [], []
        for K in Ks:
            fit = mk.compress_masked(w, h, d, K, seed=cb.layer_seed(seed, n), iters=iters,
                                     max_fit_chunks=max_fit_chunks, batch=batch)
            e.append(0.5 * float((fz * (w - fit.decoded).pow(2)).sum()))
            b.append(float(fit.bits["tags"] + fit.bits["codebook"] + fit.bits["row_scales"]))
            del fit
        err.append(e)
        bits.append(b)
        if (j + 1) % 28 == 0:
            log(f"  proxy fits {j + 1}/{len(names)} ({time.time() - t0:.0f}s)")
    return err, bits


def lagrangian(err: list[list[float]], bits: list[list[float]], budget: float) -> list[int]:
    """Per-matrix choice index minimizing sum err subject to sum bits <= budget."""
    n_opt = len(err[0])

    def pick(lam):
        return [min(range(n_opt), key=lambda k: e[k] + lam * b[k]) for e, b in zip(err, bits)]

    def total(ch):
        return sum(b[k] for b, k in zip(bits, ch))

    if total([0] * len(err)) > budget:
        raise ValueError("budget below the smallest option for every matrix")
    lo, hi = 0.0, 1.0
    while total(pick(hi)) > budget:
        hi *= 10
    for _ in range(100):
        mid = math.sqrt(lo * hi) if lo > 0 else hi / 1e8
        if total(pick(mid)) > budget:
            lo = mid
        else:
            hi = mid
    ch = pick(hi)
    while True:  # greedy fill of the leftover budget
        spare = budget - total(ch)
        best, gain = None, 0.0
        for i, k in enumerate(ch):
            for k2 in range(k + 1, n_opt):
                db = bits[i][k2] - bits[i][k]
                if 0 < db <= spare:
                    g = (err[i][k] - err[i][k2]) / db
                    if g > gain:
                        best, gain = (i, k2), g
        if best is None:
            return ch
        ch[best[0]] = best[1]


def heuristic(names: list[str], bits: list[list[float]], budget: float, n_layers: int,
              base: int | None = None) -> list[int]:
    """GGUF-style mixing around a base level: important matrices (attn_v,
    ffn_down, first/last two blocks) go one level up; to pay for it, the least
    important go one level down (MLP gate/up in the middle blocks first, then
    attention q/o/k); then any leftover is spent upgrading important, then other
    matrices. `base` defaults to the highest uniform level within the budget."""
    n_opt = len(bits[0])

    def total(ch):
        return sum(b[k] for b, k in zip(bits, ch))

    if base is None:
        base = max(k for k in range(n_opt) if sum(b[k] for b in bits) <= budget)

    def important(n):
        layer = int(n.split(".")[2])
        return "v_proj" in n or "down_proj" in n or layer < 2 or layer >= n_layers - 2

    def expendability(i):  # higher = downgraded first
        n = names[i]
        layer = int(n.split(".")[2])
        centrality = 1.0 - abs(layer - (n_layers - 1) / 2) / ((n_layers - 1) / 2)
        kind = 2 if ("gate_proj" in n or "up_proj" in n) else 1 if ("q_proj" in n or "o_proj" in n) else 0
        return (kind, centrality)

    ch = [min(base + 1, n_opt - 1) if important(n) else base for n in names]
    for i in sorted((i for i, n in enumerate(names) if not important(n)), key=expendability, reverse=True):
        if total(ch) <= budget:
            break
        if ch[i] > 0:
            ch[i] -= 1
    for i in sorted((i for i, n in enumerate(names) if important(n)), key=lambda i: names[i]):
        if total(ch) <= budget:
            break
        ch[i] = base  # could not afford every important upgrade
    for want in (True, False):  # spend what is left
        for i, n in enumerate(names):
            if important(n) == want and ch[i] < n_opt - 1:
                db = bits[i][ch[i] + 1] - bits[i][ch[i]]
                if total(ch) + db <= budget:
                    ch[i] += 1
    return ch


def budget_bits(bits: list[list[float]], Ks: tuple[int, ...], spec) -> float:
    """'uniform:1024' -> total bits of every matrix at K=1024 (proxy accounting)."""
    kind, val = str(spec).split(":")
    if kind != "uniform":
        raise ValueError(f"budget spec {spec!r}")
    key = tuple(int(v) for v in val.split("x")) if "x" in val else int(val)
    k = Ks.index(key)
    return sum(b[k] for b in bits)
