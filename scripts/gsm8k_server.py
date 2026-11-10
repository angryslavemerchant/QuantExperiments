"""GSM8K (test, 5-shot, greedy) through any OpenAI-compatible /v1/completions server (vLLM, llama-server),
so different runtimes and quantizations are scored on identical prompts.

Same protocol as scripts/gsm8k_ab.py: the first 5 train problems as the few-shot prefix
("Question: ... Answer: <reasoning> #### <n>"), raw completion (no chat template), temperature 0,
max 512 tokens, cut at the next "Question:", the last number compared with the reference.

  python scripts/gsm8k_server.py --endpoint http://127.0.0.1:8000 --api-key "$SQ_API_KEY" \
      --name sq-2.5bpw --conc 2 --out artifacts/gsm8k/server_sq.json

Chat-tuned models may end the turn right after "Answer:" (the Q4_K_M GGUF of Qwen3.8 does ~2/3 of the time in
llama.cpp: empty completions). For a cross-runtime comparison use, on both sides,
  --stops "Question:" --ban-ids 248046,248044
(no end-of-turn tokens, cut only at the next question).

Per-problem completions are saved too (--out), so two runs can be compared problem by problem.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

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
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--api-key", default="none")
    ap.add_argument("--model", help="default: the server's first /v1/models entry")
    ap.add_argument("--name", required=True)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--stops", default="Question:|\n\n\n", help="'|'-separated stop strings")
    ap.add_argument("--ban-ids", default="", help="comma-separated token ids given logit_bias -100, e.g. the "
                    "end tokens (Qwen3.8: 248046,248044), so a chat model cannot end the turn right after 'Answer:'")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from datasets import load_dataset
    base = a.endpoint.rstrip("/")
    hdr = {"Content-Type": "application/json", "Authorization": f"Bearer {a.api_key}"}
    model = a.model or json.load(urllib.request.urlopen(urllib.request.Request(
        f"{base}/v1/models", headers=hdr)))["data"][0]["id"]
    train = load_dataset("openai/gsm8k", "main", split="train")
    test = load_dataset("openai/gsm8k", "main", split="test")
    if a.limit:
        test = test.select(range(a.limit))
    shots = "".join(f"Question: {r['question']}\nAnswer: {r['answer']}\n\n" for r in train.select(range(5)))
    prompts = [shots + f"Question: {r['question']}\nAnswer:" for r in test]
    refs = [r["answer"].split("####")[-1].strip().replace(",", "") for r in test]

    def run(p: str) -> str:
        body = {"model": model, "prompt": p, "max_tokens": 512, "temperature": 0, "seed": 0,
                "stop": a.stops.split("|")}
        if a.ban_ids:
            body["logit_bias"] = {i.strip(): -100 for i in a.ban_ids.split(",")}
        for attempt in range(3):
            try:
                req = urllib.request.Request(f"{base}/v1/completions", data=json.dumps(body).encode(), headers=hdr)
                return json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["text"]
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(5)
        return ""

    t0 = time.time()
    texts: list[str] = []
    with ThreadPoolExecutor(a.conc) as ex:
        for i, t in enumerate(ex.map(run, prompts)):
            texts.append(t)
            if (i + 1) % 100 == 0:
                acc = sum(same(last_number(x), r) for x, r in zip(texts, refs)) / len(texts)
                print(f"[{a.name}] {i + 1}/{len(prompts)}: {100 * acc:.2f}% so far, {time.time() - t0:.0f}s", flush=True)
    ok = [same(last_number(t), r) for t, r in zip(texts, refs)]
    res = {"name": a.name, "model": model, "endpoint": base, "stops": a.stops.split("|"), "ban_ids": a.ban_ids,
           "n": len(refs), "acc": sum(ok) / len(refs),
           "seconds": round(time.time() - t0, 1), "correct": ok, "completions": texts}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[{a.name}] GSM8K 5-shot {100 * res['acc']:.2f}% ({len(refs)} problems, {res['seconds']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
