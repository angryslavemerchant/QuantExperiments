"""Head-only (and embedding-only) quantization KL: cache final hidden states once, then score
block-32 symmetric formats of lm_head against the fp16 head. Embedding formats rerun the model with
a quantized embed_tokens (only when --embed is given).

    python scripts/head_kl.py --model Qwen/Qwen2.5-7B-Instruct --windows 32 --out artifacts/head_kl/7b.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shapequant.evaluate import load_wikitext_windows  # noqa: E402


def quant(w: torch.Tensor, bits: int, search: bool = False) -> torch.Tensor:
    """Block-32 symmetric quant with an fp16 scale; search tries scale factors (make_qx_quants-like)."""
    R, C = w.shape
    b = w.view(R, C // 32, 32)
    qmax = 2 ** (bits - 1) - 1
    best = berr = None
    for f in [1.0] + ([0.8, 0.85, 0.9, 0.95, 1.05, 1.1] if search else []):
        s = (b.abs().amax(-1, keepdim=True) / qmax * f).half().float().clamp_min(1e-10)
        d = (b / s).round().clamp(-qmax - 1, qmax) * s
        e = (d - b).pow(2).sum(-1, keepdim=True)
        if best is None:
            best, berr = d, e
        else:
            sel = e < berr
            best, berr = torch.where(sel, d, best), torch.where(sel, e, berr)
    return best.view(R, C)


HEAD_FORMATS = [("q8", 8, False), ("q6", 6, False), ("q6 search", 6, True), ("q5 search", 5, True),
                ("q4 (runtime)", 4, False), ("q4 search", 4, True)]
EMBED_FORMATS = [("q6", 6, False), ("q4", 4, False), ("q4 search", 4, True), ("q3 search", 3, True)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--embed", action="store_true", help="also score embedding-only formats")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16).cuda().eval()
    W = load_wikitext_windows(tok, {"dataset": "Salesforce/wikitext", "subset": "wikitext-2-raw-v1",
                                    "split": "test", "context": a.context, "num_windows": a.windows})
    body = m.model

    def hidden() -> list[torch.Tensor]:
        with torch.no_grad():
            return [body(input_ids=W[i:i + 1].cuda(), use_cache=False).last_hidden_state[0].float()
                    for i in range(W.shape[0])]

    H0 = m.lm_head.weight.detach().float()
    Hs = hidden()

    def score(Hq: torch.Tensor, hs: list[torch.Tensor]) -> tuple[float, float]:
        kl, ag, n = 0.0, 0, 0
        with torch.no_grad():
            for h, hq in zip(Hs, hs):
                for s in range(0, h.shape[0], 128):
                    rl = F.log_softmax(h[s:s + 128] @ H0.T, -1)
                    lq = F.log_softmax(hq[s:s + 128] @ Hq.T, -1)
                    kl += float((rl.exp() * (rl - lq)).sum())
                    ag += int((rl.argmax(-1) == lq.argmax(-1)).sum())
                    n += rl.shape[0]
        return kl / n, 100 * ag / n

    res = {"model": a.model, "windows": a.windows, "context": a.context, "head": {}, "embed": {}}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    def dump() -> None:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)

    for name, bits, search in HEAD_FORMATS:
        k, t = score(quant(H0, bits, search), Hs)
        res["head"][name] = {"kl": k, "top1": t}
        print(f"head {name}: KL {k:.5f} top1 {t:.2f}%", flush=True)
        dump()
    if a.embed:
        E = body.embed_tokens.weight
        E0 = E.detach().clone()
        for name, bits, search in EMBED_FORMATS:
            E.data.copy_(quant(E0.float(), bits, search).half())
            k, t = score(H0, hidden())
            res["embed"][name] = {"kl": k, "top1": t}
            print(f"embed {name}: KL {k:.5f} top1 {t:.2f}%", flush=True)
            dump()
        E.data.copy_(E0)


if __name__ == "__main__":
    main()
