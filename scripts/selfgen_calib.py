"""Self-generated calibration text: the original model's own responses (chat template, thinking on) to
chat, math and coding prompts, written as full rendered conversations (one "text" per line).

LookAtMyQuant 2 found calibrating on the model's own long worked solutions far better than generic text
for what it measured; this produces the analogous data for shapequant's mixture calibration
(calibration.sources with {jsonl: ...}). Prompts come from TRAIN splits, past the documents the
mixture's direct sources use (skip), so the calibration never sees evaluation text.

Run (WSL, vLLM venv): python scripts/selfgen_calib.py --model Qwen/Qwen3.5-4B --out artifacts/selfgen/qwen35_4b.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from itertools import islice
from pathlib import Path


def prompts(n_chat: int, n_math: int, n_code: int, seed: int) -> list[tuple[str, str]]:
    from datasets import load_dataset
    out = []
    uc = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    for r in islice(uc, 100_000, 100_000 + n_chat):            # past the docs a direct ultrachat source uses
        out.append(("chat", r["messages"][0]["content"]))
    subjects = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
                "prealgebra", "precalculus"]
    per = -(-n_math // len(subjects))
    for s in subjects:
        ds = load_dataset("EleutherAI/hendrycks_math", s, split="train", streaming=True)
        k0 = 500 if s == "algebra" else 0                       # past the docs the direct MATH (algebra) source uses
        for r in islice(ds, k0, k0 + per):
            out.append(("math", r["problem"] + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."))
    mc = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train", streaming=True)
    for r in islice(mc, n_code):
        out.append(("code", r["problem"]))
    random.Random(seed).shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--out", default="artifacts/selfgen/qwen35_4b.jsonl")
    ap.add_argument("--chat", type=int, default=600)
    ap.add_argument("--math", type=int, default=600)
    ap.add_argument("--code", type=int, default=600)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from vllm import LLM, SamplingParams

    ps = prompts(args.chat, args.math, args.code, args.seed)
    llm = LLM(args.model, max_model_len=args.max_tokens + 2048, gpu_memory_utilization=0.85, seed=args.seed,
              limit_mm_per_prompt={"image": 0, "video": 0}, max_num_seqs=128)
    tok = llm.get_tokenizer()
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
             for _, p in ps]
    sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=args.max_tokens, seed=args.seed)
    outs = llm.generate(texts, sp)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ntok = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for (kind, _), t, o in zip(ps, texts, outs):
            c = o.outputs[0]
            ntok += len(c.token_ids)
            f.write(json.dumps({"kind": kind, "finish": c.finish_reason, "text": t + c.text}, ensure_ascii=False) + "\n")
    print(f"wrote {len(outs)} conversations, {ntok} generated tokens -> {args.out}")


if __name__ == "__main__":
    main()
