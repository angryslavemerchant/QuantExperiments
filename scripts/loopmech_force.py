"""Loop mechanism study, step 2a: is the opening of the thinking causal for runaways?

The runaway traces open their thinking with "We need ..." (59% runaway) and the finishing ones with "The user ..."
(7%). Here the first words of the thinking are forced, on the same prompts with the same sampling settings, and the
runaway rate is compared. Also prints the model's own first-token distribution per prompt.

  python scripts/loopmech_force.py --out artifacts/loopmech/force.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from loop_probe import PROMPTS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default="bit_reverse,regex_log,csv_merge,hexdump_parse")
    ap.add_argument("--prefixes", default="We need|The user")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=12288)
    ap.add_argument("--conc", type=int, default=8)
    a = ap.parse_args()
    key = open(os.path.expanduser("~/.sq_api_key")).read().strip()
    hdr = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    model = "qwen3.8-27b-sq"

    def post(path, body, timeout=7200):
        return json.load(urllib.request.urlopen(urllib.request.Request(
            f"{a.endpoint}{path}", json.dumps(body).encode(), hdr), timeout=timeout))

    names = a.prompts.split(",")
    prefixes = a.prefixes.split("|")
    base = {}
    for name in names:
        ids = post("/tokenize", {"model": model, "messages": [{"role": "user", "content": PROMPTS[name]}],
                                 "add_generation_prompt": True})["tokens"]
        base[name] = ids
        r = post("/v1/completions", {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 1.0,
                                     "logprobs": 10})
        lp = r["choices"][0]["logprobs"]["top_logprobs"][0]
        top = sorted(lp.items(), key=lambda x: -x[1])[:6]
        print(f"{name}: first thinking token " + ", ".join(f"{t!r} {math.exp(v):.2f}" for t, v in top), flush=True)
    pre_ids = {p: post("/tokenize", {"model": model, "prompt": p, "add_special_tokens": False})["tokens"]
               for p in prefixes}

    jobs = [(n, p, i) for n in names for p in prefixes for i in range(a.n)]
    done = set()
    if os.path.exists(a.out):
        done = {(r["prompt"], r["prefix"], r["i"]) for r in map(json.loads, open(a.out))}
    jobs = [j for j in jobs if j not in done]
    lock = threading.Lock()
    t0 = time.time()

    def run(job):
        name, p, i = job
        body = {"model": model, "prompt": base[name] + pre_ids[p], "max_tokens": a.max_tokens,
                "temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": 7000 + i,
                "skip_special_tokens": False}
        try:
            r = post("/v1/completions", body)
        except Exception as e:
            print(f"{name}/{p}#{i} error {e}", flush=True)
            return
        ch = r["choices"][0]
        text = ch["text"]
        rec = {"prompt": name, "prefix": p, "i": i, "finish": ch["finish_reason"],
               "n_gen": r["usage"]["completion_tokens"], "runaway": ch["finish_reason"] == "length",
               "closed_think": "</think>" in text,
               "think_tokens": None, "text": p + text}
        with lock:
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
        print(f"[{time.time() - t0:6.0f}s] {name}/{p!r}#{i}: {rec['n_gen']} tok, {rec['finish']}", flush=True)

    with ThreadPoolExecutor(a.conc) as ex:
        list(ex.map(run, jobs))

    tab = collections.defaultdict(lambda: [0, 0])
    for r in map(json.loads, open(a.out)):
        tab[(r["prompt"], r["prefix"])][0] += 1
        tab[(r["prompt"], r["prefix"])][1] += r["runaway"]
    for k, (n, ra) in sorted(tab.items()):
        print(f"{k[0]:14s} forced {k[1]!r:12s} n={n:2d} runaway {ra:2d} ({ra / n:.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
