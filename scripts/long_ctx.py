"""Long-context retrieval (needle in a haystack) against an OpenAI-compatible chat server.

For each context length and needle depth: wikitext-2 filler (real prose) sized with the model's
tokenizer to the target token count, one "passphrase" sentence inserted at the depth fraction,
then the question at the end; thinking off, greedy, short answer. Scored by exact passphrase
match; wall time per request is ~prefill (answers are a few tokens). Writes <out>.json after every
request.

  python scripts/long_ctx.py --endpoint http://127.0.0.1:8000 --model ours --tokenizer Qwen/Qwen3.8-27B \
      --lengths 8000,32000,128000 --depths 0.1,0.5,0.9 --out artifacts/x/longctx_ours.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
from pathlib import Path

CITIES = ["Lisbon", "Osaka", "Quito", "Tallinn", "Windhoek", "Hobart", "Reykjavik", "Cusco", "Tbilisi", "Ghent"]
WORDS = ["amber", "falcon", "quartz", "tundra", "velvet", "cobalt", "saffron", "glacier", "ember", "lantern"]


def filler_text() -> str:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:60000]")
    return "".join(t for t in ds["text"] if t.strip())


def build(tok, base_ids: list[int], n_tokens: int, depth: float, rng: random.Random) -> tuple[str, str, str]:
    city, pw = rng.choice(CITIES), f"{rng.choice(WORDS)}-{rng.randint(1000, 9999)}"
    needle = f" The secret passphrase for {city} is {pw}. "
    question = (f"\n\nAbove is a long document. Somewhere in it is a sentence giving the secret passphrase for "
                f"{city}. What is the secret passphrase for {city}? Reply with only the passphrase.")
    budget = n_tokens - len(tok(needle)["input_ids"]) - len(tok(question)["input_ids"]) - 32
    if budget > len(base_ids):
        raise ValueError(f"filler has {len(base_ids)} tokens, need {budget}")
    start = rng.randint(0, len(base_ids) - budget)
    ids = base_ids[start:start + budget]
    cut = int(len(ids) * depth)
    text = tok.decode(ids[:cut]) + needle + tok.decode(ids[cut:]) + question
    return text, pw, city


def ask(endpoint: str, model: str, api_key: str | None, text: str, timeout: float) -> tuple[str, float, dict]:
    body = {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": 24, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    hdr = {"Content-Type": "application/json"}
    if api_key:
        hdr["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(endpoint.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(), headers=hdr)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        res = json.loads(r.read())
    msg = res["choices"][0]["message"]
    return (msg.get("content") or "") + (msg.get("reasoning_content") or ""), time.time() - t0, res.get("usage") or {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", default="default")
    ap.add_argument("--api-key")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--lengths", default="8000,16000,32000,64000,128000")
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--trials", type=int, default=1, help="needles per (length, depth)")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--name", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    base_ids = tok(filler_text())["input_ids"]
    rng = random.Random(0)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rec = {"name": a.name, "endpoint": a.endpoint, "model": a.model, "filler_tokens": len(base_ids), "runs": []}
    for n in map(int, a.lengths.split(",")):
        for d in map(float, a.depths.split(",")):
            for k in range(a.trials):
                text, pw, city = build(tok, base_ids, n, d, rng)
                r = {"tokens": n, "depth": d, "trial": k, "passphrase": pw}
                try:
                    ans, secs, usage = ask(a.endpoint, a.model, a.api_key, text, a.timeout)
                    r.update(answer=ans.strip()[:200], correct=pw.lower() in ans.lower(), seconds=round(secs, 2),
                             prompt_tokens=usage.get("prompt_tokens"))
                except Exception as e:  # noqa: BLE001 - record and continue (e.g. context too long)
                    r.update(error=repr(e)[:300], correct=False)
                rec["runs"].append(r)
                print(f"[{a.name}] {n:7d} tok depth {d:.1f}: {'OK ' if r['correct'] else 'MISS'} "
                      f"{r.get('seconds', '-')}s  {r.get('answer', r.get('error', ''))[:60]!r}", flush=True)
                out.write_text(json.dumps(rec, indent=1))
    by_len: dict = {}
    for r in rec["runs"]:
        by_len.setdefault(r["tokens"], []).append(r)
    rec["summary"] = {str(n): {"acc": sum(x["correct"] for x in v) / len(v),
                               "mean_s": (sum(x.get("seconds", 0) for x in v if "seconds" in x) /
                                          max(1, sum("seconds" in x for x in v)))} for n, v in by_len.items()}
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
