"""Side-by-side answers from several OpenAI-compatible endpoints on a fixed mixed prompt set
(coding, math, reasoning, writing, instruction following, knowledge, multilingual), for human
comparison. Thinking on, Qwen sampling (0.6 / 0.95 / 20), one fixed seed per prompt; every answer is
saved as it arrives.

  python scripts/quality_prompts.py --target ours=http://127.0.0.1:8000 --target q4km=http://127.0.0.1:8100 \
      --out artifacts/x/quality.json
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import time
import urllib.request
from pathlib import Path

PROMPTS = [
    ("coding", "Write a Python function `merge_intervals(intervals)` that merges overlapping closed intervals "
               "given as a list of [start, end] pairs and returns them sorted. Include 4 assert-based tests."),
    ("coding", "This JavaScript is supposed to debounce a function but calls it too often. Find the bug and fix it:\n"
               "```js\nfunction debounce(fn, ms) {\n  return (...args) => {\n    let t;\n    clearTimeout(t);\n"
               "    t = setTimeout(() => fn(...args), ms);\n  };\n}\n```"),
    ("coding", "Write a SQL query: table orders(id, customer_id, total, created_at). For each customer, return their "
               "second-largest order total (customers with fewer than two orders excluded)."),
    ("math", "A bag has 5 red and 7 blue marbles. Three are drawn without replacement. What is the probability that "
             "exactly two are red? Give the exact fraction."),
    ("math", "Find all real x with sqrt(x + 7) - sqrt(x) = 1."),
    ("math", "How many trailing zeros does 125! have? Explain briefly."),
    ("reasoning", "Alice is taller than Bob. Carol is shorter than Bob but taller than Dave. Eve is taller than "
                  "Alice. Who is the second shortest? Explain."),
    ("reasoning", "A train leaves at 9:40 and arrives at 13:15 the same day, stopping 3 times for 6 minutes each. "
                  "What was its average moving speed if the route is 252 km?"),
    ("writing", "Write a 120-word product description for a waterproof hiking backpack, upbeat but not cheesy."),
    ("writing", "Write a polite email declining a meeting invitation next Tuesday because of a conflict, proposing "
                "two alternative times."),
    ("writing", "Write a short poem (8 lines, ABAB rhyme) about debugging at 3am."),
    ("instructions", "List exactly five countries in Africa whose names start with the letter M, as a JSON array of "
                     "strings and nothing else."),
    ("instructions", "Summarize the plot of Romeo and Juliet in exactly three sentences, without using the letter 'z'."),
    ("instructions", "Answer in all lowercase and in under 40 words: why is the sky blue?"),
    ("knowledge", "Explain the difference between TCP and UDP, with one realistic use case for each."),
    ("knowledge", "What causes the seasons on Earth? Address the common misconception about distance to the Sun."),
    ("knowledge", "Explain what a KV cache is in transformer inference and why it grows with context length."),
    ("multilingual", "Translate into French and Spanish: 'The meeting has been moved to Thursday afternoon; please "
                     "bring the updated budget.'"),
    ("multilingual", "用三句话解释什么是机器学习。"),
    ("summary", "Summarize the following in two bullet points:\nThe city council voted 7-2 on Monday to approve a "
                "pilot program that converts three downtown parking lots into weekend markets. Supporters said it "
                "would boost small businesses; opponents worried about traffic. The pilot runs six months, after "
                "which the council will review sales data and resident complaints before deciding on renewal."),
]


def ask(endpoint: str, model: str, api_key: str | None, prompt: str, seed: int, max_tokens: int, timeout: float) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.6, "top_p": 0.95,
            "top_k": 20, "max_tokens": max_tokens, "seed": seed, "chat_template_kwargs": {"enable_thinking": True}}
    hdr = {"Content-Type": "application/json"}
    if api_key:
        hdr["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(endpoint.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(), headers=hdr)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        res = json.loads(r.read())
    ch = res["choices"][0]
    msg = ch.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    if not reasoning and "</think>" in content:
        reasoning, content = content.split("</think>", 1)
    return {"answer": content.strip(), "reasoning_chars": len(reasoning), "finish": ch.get("finish_reason"),
            "completion_tokens": (res.get("usage") or {}).get("completion_tokens"), "seconds": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", action="append", required=True, help="name=endpoint (repeatable)")
    ap.add_argument("--model", default="default")
    ap.add_argument("--api-key")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    targets = dict(t.split("=", 1) for t in a.target)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rec = {"targets": targets, "prompts": [{"category": c, "prompt": p} for c, p in PROMPTS], "answers": {}}

    def job(name, i):
        try:
            return name, i, ask(targets[name], a.model, a.api_key, PROMPTS[i][1], 100 + i, a.max_tokens, a.timeout)
        except Exception as e:  # noqa: BLE001
            return name, i, {"error": repr(e)[:300]}

    with cf.ThreadPoolExecutor(a.conc) as ex:
        futs = [ex.submit(job, n, i) for n in targets for i in range(len(PROMPTS))]
        for f in cf.as_completed(futs):
            name, i, r = f.result()
            rec["answers"].setdefault(name, {})[str(i)] = r
            print(f"[{name}] #{i} {PROMPTS[i][0]}: {r.get('completion_tokens')} tokens {r.get('seconds')}s "
                  f"{r.get('error', '')}", flush=True)
            out.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
