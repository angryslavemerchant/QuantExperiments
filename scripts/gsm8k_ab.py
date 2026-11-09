"""GSM8K (test, 5-shot, greedy) with the INT8 prefill GEMMs on vs off, in one in-process vLLM engine.

Self-contained (no lm_eval): the first 5 train problems as the few-shot prefix ("Question: ...
Answer: <reasoning> #### <n>"), completions cut at the next "Question:", the last number in the
completion compared with the reference. Prefix caching is off, so every prompt (~900 tokens with
the prefix) is prefilled, batched into steps of up to --chunk tokens: the INT8 path (M >= 1024)
runs for nearly all of it. Absolute scores differ from lm_eval's flexible-extract numbers; the point
is the on/off difference on the same prompts.

  python scripts/gsm8k_ab.py --model ~/models/q27-sq --out artifacts/gsm8k_ab.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def last_number(s: str) -> str | None:
    s = s.split("Question:")[0]
    if "####" in s:
        s = s.split("####")[-1]
    m = NUM.findall(s)
    return m[-1].replace(",", "").rstrip(".") if m else None


def same(a: str | None, b: str) -> bool:
    if a is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return a == b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.80)
    ap.add_argument("--modes", default="int8,bf16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    from shapequant.vllm_ext import core
    train = load_dataset("openai/gsm8k", "main", split="train")
    test = load_dataset("openai/gsm8k", "main", split="test")
    if a.limit:
        test = test.select(range(a.limit))
    shots = "".join(f"Question: {r['question']}\nAnswer: {r['answer']}\n\n" for r in train.select(range(5)))
    prompts = [shots + f"Question: {r['question']}\nAnswer:" for r in test]
    refs = [r["answer"].split("####")[-1].strip().replace(",", "") for r in test]
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=a.gpu_mem,
              max_num_seqs=64, max_num_batched_tokens=a.chunk, enable_prefix_caching=False, seed=0)
    sp = SamplingParams(max_tokens=512, temperature=0, stop=["Question:", "\n\n\n"])
    res = {"model": a.model, "n": len(prompts), "modes": {}}
    outs = {}
    for mode in a.modes.split(","):
        core._INT8_PREFILL = mode == "int8"
        t0 = time.time()
        gen = llm.generate(prompts, sp, use_tqdm=False)
        texts = [g.outputs[0].text for g in gen]
        acc = sum(same(last_number(t), r) for t, r in zip(texts, refs)) / len(refs)
        outs[mode] = texts
        res["modes"][mode] = {"acc": acc, "seconds": round(time.time() - t0, 1)}
        print(f"{mode}: GSM8K 5-shot {100 * acc:.2f}% ({len(refs)} problems, {time.time() - t0:.0f}s)", flush=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    if len(outs) == 2:
        m0, m1 = list(outs)
        a0 = [same(last_number(t), r) for t, r in zip(outs[m0], refs)]
        a1 = [same(last_number(t), r) for t, r in zip(outs[m1], refs)]
        res["identical_text"] = sum(x == y for x, y in zip(outs[m0], outs[m1]))
        res["only_" + m0] = sum(x and not y for x, y in zip(a0, a1))
        res["only_" + m1] = sum(y and not x for x, y in zip(a0, a1))
        print(f"identical completions {res['identical_text']}/{len(refs)}; right only with {m0}: {res['only_' + m0]}, "
              f"only with {m1}: {res['only_' + m1]}", flush=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
