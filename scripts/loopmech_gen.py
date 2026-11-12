"""Loop mechanism study, step 1a: generate thinking traces (looping and finishing) with exact token ids.

Prompts: the loop-prone hex prompts (scripts/loop_probe.py), ordinary coding prompts, GSM8K problems. Thinking on,
server defaults (temperature 1.0, top_p 0.95, top_k 20, reasoning effort = template default). A trace is a runaway
when it hits max_tokens.

  python scripts/loopmech_gen.py --out artifacts/loopmech/traces.jsonl
"""

from __future__ import annotations

import argparse
import json
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
    ap.add_argument("--max-tokens", type=int, default=12288)
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--plan", default="hex16to32:64,hexdump_parse:16,csv_merge:12,regex_log:12,bit_reverse:12,gsm8k:48")
    a = ap.parse_args()
    key = open(os.path.expanduser("~/.sq_api_key")).read().strip()
    hdr = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    jobs = []
    for item in a.plan.split(","):
        name, n = item.split(":")
        if name == "gsm8k":
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from datasets import load_dataset
            ds = load_dataset("openai/gsm8k", "main", split="test")
            jobs += [("gsm8k", ds[i]["question"], i) for i in range(int(n))]
        else:
            jobs += [(name, PROMPTS[name], i) for i in range(int(n))]
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["prompt"], r["i"]))
    jobs = [j for j in jobs if (j[0], j[2]) not in done]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    lock = threading.Lock()
    t_start = time.time()

    def run(job):
        name, text, i = job
        body = {"model": "qwen3.8-27b-sq", "messages": [{"role": "user", "content": text}],
                "max_tokens": a.max_tokens, "seed": 5000 + i, "return_token_ids": True}
        t0 = time.time()
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(
                f"{a.endpoint}/v1/chat/completions", json.dumps(body).encode(), hdr), timeout=7200))
        except Exception as e:  # keep going
            print(f"{name}#{i} error {e}", flush=True)
            return
        ch = r["choices"][0]
        rec = {"prompt": name, "i": i, "finish": ch["finish_reason"], "n_gen": len(ch["token_ids"]),
               "runaway": ch["finish_reason"] == "length", "prompt_token_ids": r["prompt_token_ids"],
               "token_ids": ch["token_ids"], "reasoning": ch["message"].get("reasoning") or "",
               "content": ch["message"].get("content") or "", "seconds": round(time.time() - t0, 1)}
        with lock:
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
        print(f"[{time.time() - t_start:6.0f}s] {name}#{i}: {rec['n_gen']} tok, {rec['finish']}", flush=True)

    with ThreadPoolExecutor(a.conc) as ex:
        list(ex.map(run, jobs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
