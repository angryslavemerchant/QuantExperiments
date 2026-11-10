"""Reference (bf16) per-token scores for Qwen3.8-27B on a 24 GB card: the 54 GB of weights are streamed
through the GPU one decoder layer at a time while every sequence's hidden states stay on the GPU, so the
weights are read once per run, however many texts are scored.

Windows like scripts/ppl_llamacpp_style.py: each text is cut into ctx-token windows (Qwen: no BOS), the
second half of each window is scored. Writes, per text, the per-window NLL (comparable with
ppl_llamacpp_style.py --max-chunks) and per-token NLL + the reference top-k log-probs (for KL / loop
analysis).

  python scripts/bf16_stream_ref.py --texts artifacts/overnight/domains/*.txt --max-chunks 24 \
      --out-dir artifacts/overnight/ref_bf16
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch


class Streamed(torch.nn.Module):
    """Wraps one decoder layer built on the meta device: loads its weights for the call, runs the batch in
    sub-batches, frees the weights."""

    def __init__(self, inner, prefix, index, files, sub):
        super().__init__()
        self.inner, self.prefix, self.index, self.files, self.sub = inner, prefix, index, files, sub
        self.attention_type = getattr(inner, "attention_type", None)
        self.layer_type = getattr(inner, "layer_type", None)

    def _load(self, dev):
        from safetensors import safe_open
        sd = {}
        by_file = {}
        for name, f in self.index.items():
            if name.startswith(self.prefix):
                by_file.setdefault(f, []).append(name)
        for f, names in by_file.items():
            with safe_open(self.files / f, "pt", device=str(dev)) as g:
                for n in names:
                    sd[n[len(self.prefix):]] = g.get_tensor(n)
        self.inner.to_empty(device=dev)
        missing, unexpected = self.inner.load_state_dict(sd, strict=False)
        bad = [m for m in missing if "rotary" not in m]
        if bad or unexpected:
            raise RuntimeError(f"{self.prefix}: missing {bad[:5]} unexpected {unexpected[:5]}")
        # fp32 buffers / params the HF init would have created (e.g. A_log stays as stored)

    def forward(self, hidden_states, *args, **kwargs):
        dev = hidden_states.device
        self._load(dev)
        B = hidden_states.shape[0]
        outs = []
        for s in range(0, B, self.sub):
            def cut(v):
                if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == B and B > 1:
                    return v[s:s + self.sub]
                if isinstance(v, tuple) and v and isinstance(v[0], torch.Tensor):
                    return tuple(cut(x) for x in v)
                return v
            o = self.inner(hidden_states[s:s + self.sub], *[cut(a) for a in args],
                           **{k: cut(v) for k, v in kwargs.items()})
            outs.append(o[0] if isinstance(o, tuple) else o)
        self.inner.to("meta")
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return torch.cat(outs, 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--texts", nargs="+", required=True)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--max-chunks", type=int, default=24)
    ap.add_argument("--sub", type=int, default=4, help="sequences per layer call")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--layers", type=int, default=0, help="debug: only the first N decoder layers")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    snap = Path(snapshot_download(a.repo, allow_patterns=["*.json", "*.safetensors", "merges.txt", "*.jinja"]))
    index = json.load(open(snap / "model.safetensors.index.json"))["weight_map"]
    cfg = AutoConfig.from_pretrained(snap).text_config
    cfg._attn_implementation = "sdpa"
    tok = AutoTokenizer.from_pretrained(snap)
    dev = torch.device(a.device)
    if a.layers:
        cfg.num_hidden_layers = a.layers
        cfg.layer_types = cfg.layer_types[:a.layers]

    # windows
    seqs, owner = [], []
    for t in a.texts:
        ids = tok(open(t, encoding="utf-8").read(), add_special_tokens=False)["input_ids"]
        n = min(len(ids) // a.ctx, a.max_chunks)
        for c in range(n):
            seqs.append(ids[c * a.ctx:(c + 1) * a.ctx])
            owner.append(t)
    print(f"{len(seqs)} windows of {a.ctx} from {len(a.texts)} texts", flush=True)

    torch.set_default_dtype(torch.bfloat16)
    with torch.device("meta"):
        model = M.Qwen3_5TextModel(cfg)
    torch.set_default_dtype(torch.float32)
    pre = "model.language_model."
    for i, layer in enumerate(model.layers):
        model.layers[i] = Streamed(layer, f"{pre}layers.{i}.", index, snap, a.sub)
    # small modules: embed, norm, rotary to GPU with real weights
    def load_small(mod, prefix):
        sd = {}
        for name, f in index.items():
            if name.startswith(prefix):
                with safe_open(snap / f, "pt", device=str(dev)) as g:
                    sd[name[len(prefix):]] = g.get_tensor(name)
        mod.to_empty(device=dev)
        mod.load_state_dict(sd, strict=True)
    load_small(model.embed_tokens, pre + "embed_tokens.")
    load_small(model.norm, pre + "norm.")
    with torch.device(dev):
        model.rotary_emb = M.Qwen3_5TextRotaryEmbedding(cfg)
    model.eval()

    t0 = time.time()
    ids = torch.tensor(seqs, device=dev)
    with torch.inference_mode():
        h = model(input_ids=ids, use_cache=False).last_hidden_state      # [N, T, H] (norm applied)
        print(f"layers done in {time.time() - t0:.0f}s", flush=True)
        model.embed_tokens.to("meta")
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        head = None
        with safe_open(snap / index["lm_head.weight"], "pt", device=str(dev)) as g:
            head = g.get_tensor("lm_head.weight")
        first = a.ctx // 2
        res = {}
        for s in range(len(seqs)):
            x = h[s, first:a.ctx - 1]                                     # predicts positions first+1..ctx-1
            logits = (x @ head.T).float()
            lp = torch.log_softmax(logits, -1)
            tgt = ids[s, first + 1:a.ctx]
            nll = -lp.gather(1, tgt[:, None])[:, 0]
            top = lp.topk(a.topk, -1)
            r = res.setdefault(owner[s], {"nll_windows": [], "tok_nll": [], "top_ids": [], "top_lp": []})
            r["nll_windows"].append(nll.mean().item())
            r["tok_nll"].append([round(v, 4) for v in nll.tolist()])
            r["top_ids"].append(top.indices.tolist())
            r["top_lp"].append([[round(v, 3) for v in row] for row in top.values.tolist()])
    os.makedirs(a.out_dir, exist_ok=True)
    for t, r in res.items():
        nll = sum(r["nll_windows"]) / len(r["nll_windows"])
        out = {"text": t, "ctx": a.ctx, "chunks": len(r["nll_windows"]), "nll": nll, "ppl": math.exp(nll), **r}
        stem = Path(t).stem
        json.dump(out, open(os.path.join(a.out_dir, stem + ".json"), "w"))
        print(f"{stem:14s} bf16 PPL {math.exp(nll):8.4f} (NLL {nll:.4f}, {len(r['nll_windows'])} windows)", flush=True)
    print(f"total {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
