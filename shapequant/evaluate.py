"""Evaluation: WikiText-2 distribution metrics against a cached reference, and GSM8K.

Reference caching: storing full float32 next-token distributions for ~100k
positions x 151,936 vocab would be ~60 GB. Instead we cache the reference
model's final hidden states (exact bf16 tensors, ~6 MB per 2,048-token window
for the 1.5B model). The lm_head (and final norm, which is already applied to
the cached states) is never compressed, so reference log-probabilities rebuilt
from the cache are identical to the original model's.
"""

from __future__ import annotations

import json
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- WikiText

def load_wikitext_windows(tokenizer, cfg: dict) -> torch.Tensor:
    from datasets import load_dataset

    ds = load_dataset(cfg["dataset"], cfg["subset"], split=cfg["split"])
    text = "\n\n".join(ds["text"])
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    ctx, n = cfg["context"], cfg["num_windows"]
    if ids.numel() < ctx * n:
        raise ValueError(f"need {ctx * n} tokens, dataset has {ids.numel()}")
    return ids[: ctx * n].reshape(n, ctx)


def _decoder(model):
    return model.model


def block_quant(w: torch.Tensor, bits: int, block: int = 32) -> torch.Tensor:
    """Symmetric per-block quant with an fp16 scale (amax / qmax), dequantized to float32. bits 4
    and 8 match the runtime's q4/q8 head kernels (kernels.proto_formats.q4_pack / q8_pack)."""
    R, C = w.shape
    qmax = 2 ** (bits - 1) - 1
    out = torch.empty(R, C, dtype=torch.float32, device=w.device)
    for s in range(0, R, 8192):                                   # bounded fp32 temporaries
        b = w[s:s + 8192].float().view(-1, C // block, block)
        sc = (b.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax).half().float()
        sc = torch.where(sc == 0, torch.ones_like(sc), sc)       # all-zero block: fp16 underflow
        out[s:s + 8192] = ((b / sc).round().clamp(-qmax - 1, qmax) * sc).view(-1, C)
    return out


class Scorer:
    """Holds the evaluation windows, the cached reference hidden states, and a
    float32 copy of the (never-compressed) lm_head weight."""

    def __init__(self, model, windows: torch.Tensor, logit_chunk: int):
        self.windows = windows
        self.logit_chunk = logit_chunk
        self.device = next(model.parameters()).device
        self.head_w = model.lm_head.weight.detach().to(torch.float32)
        self.head_b = (model.lm_head.bias.detach().to(torch.float32)
                       if model.lm_head.bias is not None else None)
        self.ref_hidden = [self._hidden(model, w) for w in windows]

    @torch.no_grad()
    def _hidden(self, model, window: torch.Tensor) -> torch.Tensor:
        ids = window[None].to(self.device)
        return _decoder(model)(input_ids=ids).last_hidden_state[0]  # [T, H] bf16

    def _logprobs(self, h: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(F.linear(h.to(torch.float32), self.head_w, self.head_b), dim=-1)

    @torch.no_grad()
    def score_deploy(self, model, variants: list[dict]) -> dict:
        """KL vs the original with the deployed head/embedding formats on top of whatever the
        model's block linears currently are. variants: [{name, head: bits|None, embed: bits|None}];
        instead of bits, head_w / embed_w give the weights directly (e.g. a GGUF file's own
        output / token_embd). Returns {"deploy_<name>_kl", "deploy_<name>_top1"} per variant."""
        out: dict = {}
        emb = _decoder(model).embed_tokens.weight
        e0 = emb.detach().clone()
        try:
            for v in variants:
                hb, eb = v.get("head"), v.get("embed")
                if v.get("head_w") is not None:
                    head = v["head_w"].to(self.head_w.device, torch.float32)
                else:
                    head = block_quant(self.head_w, hb) if hb else self.head_w
                if v.get("embed_w") is not None:
                    emb.data.copy_(v["embed_w"].to(e0.device, e0.dtype))
                elif eb:
                    emb.data.copy_(block_quant(e0, eb).to(e0.dtype))
                kl, agree, n = 0.0, 0, 0
                for w, ref_h in zip(self.windows, self.ref_hidden):
                    h = self._hidden(model, w)
                    T = w.numel() - 1
                    for s in range(0, T, self.logit_chunk):
                        e = min(s + self.logit_chunk, T)
                        ref_lp = self._logprobs(ref_h[s:e])
                        lp = F.log_softmax(F.linear(h[s:e].float(), head, self.head_b), dim=-1)
                        kl += float((ref_lp.exp() * (ref_lp - lp)).sum(-1).clamp_min(0).sum())
                        agree += int((ref_lp.argmax(-1) == lp.argmax(-1)).sum())
                        n += e - s
                emb.data.copy_(e0)
                del head
                out[f"deploy_{v['name']}_kl"] = kl / n
                out[f"deploy_{v['name']}_top1"] = 100.0 * agree / n
        finally:
            emb.data.copy_(e0)
        return out

    @torch.no_grad()
    def score(self, model) -> dict:
        kls, agree, nll = [], 0, 0.0
        count = 0
        for w, ref_h in zip(self.windows, self.ref_hidden):
            h = self._hidden(model, w)
            targets = w[1:].to(self.device)
            T = w.numel() - 1  # last position has no next-token target
            for s in range(0, T, self.logit_chunk):
                e = min(s + self.logit_chunk, T)
                ref_lp = self._logprobs(ref_h[s:e])
                lp = self._logprobs(h[s:e])
                kl = (ref_lp.exp() * (ref_lp - lp)).sum(-1)
                kls.append(kl.clamp_min(0).double().cpu())
                agree += int((ref_lp.argmax(-1) == lp.argmax(-1)).sum())
                nll += float(-lp.gather(-1, targets[s:e, None]).sum())
                count += e - s
        kl_all = torch.cat(kls)
        return {
            "mean_kl": float(kl_all.mean()),
            "p99_kl": float(torch.quantile(kl_all, 0.99)),
            "top1_agree": 100.0 * agree / count,
            "perplexity": float(torch.exp(torch.tensor(nll / count, dtype=torch.float64))),
            "eval_tokens": count,
        }


# ---------------------------------------------------------------- GSM8K

GSM8K_TEMPLATE = (
    "Solve the following math problem. Reason step by step, then give the final "
    "answer on its own line in the form \"#### <number>\".\n\nProblem: {question}"
)

_NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"


def _normalize_number(s: str) -> Decimal | None:
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return Decimal(s).normalize()
    except InvalidOperation:
        return None


def _fmt(x: Decimal | None) -> str | None:
    return None if x is None else format(x, "f")


def extract_answer(text: str) -> Decimal | None:
    """Number after the last '####'; otherwise the last number in the text."""
    marked = re.findall(r"####\s*(" + _NUM + ")", text)
    if marked:
        return _normalize_number(marked[-1])
    nums = re.findall(_NUM, text)
    return _normalize_number(nums[-1]) if nums else None


def gold_answer(answer_field: str) -> Decimal | None:
    return _normalize_number(answer_field.split("####")[-1])


def load_gsm8k(cfg: dict) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(cfg["dataset"], cfg["subset"], split=cfg["split"])
    n = cfg["num_problems"]
    return [{"question": ds[i]["question"], "answer": ds[i]["answer"]} for i in range(n)]


def build_prompts(tokenizer, problems: list[dict]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": GSM8K_TEMPLATE.format(question=p["question"])}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in problems
    ]


@torch.no_grad()
def run_gsm8k(model, tokenizer, problems: list[dict], cfg: dict, out_path: Path | None) -> dict:
    from transformers import GenerationConfig

    device = next(model.parameters()).device
    prompts = build_prompts(tokenizer, problems)
    eos = model.generation_config.eos_token_id
    gen_cfg = GenerationConfig(
        do_sample=False,
        max_new_tokens=cfg["max_new_tokens"],
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        eos_token_id=eos,
    )
    tokenizer.padding_side = "left"
    records, correct = [], 0
    t0 = time.time()
    bs = cfg["batch_size"]
    for s in range(0, len(prompts), bs):
        batch = tokenizer(prompts[s : s + bs], return_tensors="pt", padding=True,
                          add_special_tokens=False).to(device)
        out = model.generate(**batch, generation_config=gen_cfg)
        texts = tokenizer.batch_decode(out[:, batch.input_ids.shape[1]:], skip_special_tokens=True)
        for j, text in enumerate(texts):
            p = problems[s + j]
            pred, gold = extract_answer(text), gold_answer(p["answer"])
            ok = pred is not None and gold is not None and pred == gold
            correct += ok
            records.append({"i": s + j, "pred": _fmt(pred), "gold": _fmt(gold), "correct": ok, "output": text})
    if out_path is not None:
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
    return {"gsm8k_acc": 100.0 * correct / len(problems), "gsm8k_seconds": time.time() - t0}
