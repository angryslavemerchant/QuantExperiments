"""Decode speed through an OpenAI-compatible server at a few concurrency levels (thinking on,
fixed prompts and seeds), plus speculative-decoding acceptance scraped from vLLM's /metrics.

  python scripts/spec_bench.py --endpoint http://127.0.0.1:8000 --model ours --name dspark --out x.json
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import time
import urllib.request
from pathlib import Path

PROMPTS = [
    "Explain step by step how to compute the determinant of a 3x3 matrix, with a worked example.",
    "Write a Python function that returns the n-th Fibonacci number in O(log n) time, and explain it.",
    "A shop sells pens at 3 for $2 and notebooks at $4 each. Sam buys 12 pens and some notebooks for $36. How many notebooks?",
    "Summarize the causes of the French Revolution in a structured way.",
    "Write a short story (about 300 words) about a lighthouse keeper who finds a message in a bottle.",
    "Compare quicksort and mergesort: complexity, stability, memory, and when to use each.",
    "Prove that the square root of 2 is irrational.",
    "Describe how HTTPS establishes a secure connection, step by step.",
]


def metrics(endpoint: str) -> dict:
    try:
        txt = urllib.request.urlopen(endpoint.rstrip("/") + "/metrics", timeout=30).read().decode()
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for key in ("spec_decode_num_accepted_tokens", "spec_decode_num_draft_tokens", "spec_decode_num_drafts",
                "generation_tokens"):
        vals = [float(v) for v in re.findall(rf"^vllm:{key}(?:_total)?(?:\{{[^}}]*\}})? ([0-9.e+]+)$", txt, re.M)]
        if vals:
            out[key] = sum(vals)
    return out


def one(endpoint: str, model: str, prompt: str, seed: int, max_tokens: int) -> tuple[int, float]:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": seed, "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": True}}
    req = urllib.request.Request(endpoint.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    res = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    return res["usage"]["completion_tokens"], time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", default="default")
    ap.add_argument("--name", default="")
    ap.add_argument("--conc", default="1,4")
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rec = {"name": a.name, "endpoint": a.endpoint, "runs": []}
    one(a.endpoint, a.model, PROMPTS[0], 0, 32)                       # warm-up
    for c in map(int, a.conc.split(",")):
        m0 = metrics(a.endpoint)
        n = max(c, 2) if c > 1 else 3
        t0 = time.time()
        with cf.ThreadPoolExecutor(c) as ex:
            res = list(ex.map(lambda i: one(a.endpoint, a.model, PROMPTS[i % len(PROMPTS)], 7 + i, a.max_tokens),
                              range(n)))
        wall = time.time() - t0
        m1 = metrics(a.endpoint)
        toks = sum(r[0] for r in res)
        r = {"conc": c, "requests": n, "tokens": toks, "wall_s": round(wall, 2), "agg_tok_s": round(toks / wall, 1),
             "per_seq_tok_s": round(sum(r[0] / r[1] for r in res) / len(res), 1)}
        d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
        if d.get("spec_decode_num_drafts"):
            r["accepted_per_draft"] = round(d.get("spec_decode_num_accepted_tokens", 0) / d["spec_decode_num_drafts"], 2)
            r["acceptance_rate"] = round(d.get("spec_decode_num_accepted_tokens", 0) /
                                         max(1, d.get("spec_decode_num_draft_tokens", 0)), 3)
        rec["runs"].append(r)
        print(f"[{a.name}] conc {c}: {r['agg_tok_s']} tok/s aggregate, {r['per_seq_tok_s']} per sequence"
              + (f", {r['accepted_per_draft']} accepted / draft (rate {r['acceptance_rate']})" if "accepted_per_draft" in r else ""),
              flush=True)
        Path(a.out).write_text(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
