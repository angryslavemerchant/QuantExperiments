"""Functional + speed checks for shapequant.vllm_ext.think_stop on a running server (chat API, as clients use it).

  python scripts/think_stop_check.py            # needs the server started with --logits-processors ...ThinkStop
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from loop_probe import PROMPTS  # noqa: E402

KEY = open(os.path.expanduser("~/.sq_api_key")).read().strip()
H = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}


def chat(msg, max_tokens=8000, xargs=None, think=True, seed=1, stream=False):
    body = {"model": "qwen3.8-27b-sq", "messages": [{"role": "user", "content": msg}], "max_tokens": max_tokens,
            "seed": seed}
    if xargs:
        body["vllm_xargs"] = xargs
    if not think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", json.dumps(body).encode(), H)
    if not stream:
        return json.load(urllib.request.urlopen(req, timeout=3600))
    first = last = None
    usage = None
    for line in urllib.request.urlopen(req, timeout=3600):
        line = line.decode().strip()
        if not line.startswith("data: ") or line.endswith("[DONE]"):
            continue
        j = json.loads(line[6:])
        usage = j.get("usage") or usage
        if j.get("choices") and (j["choices"][0]["delta"].get("content") or j["choices"][0]["delta"].get("reasoning")):
            now = time.time()
            first = first or now
            last = now
    return (usage["completion_tokens"] - 1) / (last - first), usage["completion_tokens"]


def main() -> int:
    ok = True
    # 1. lowered gate on a loop-prone prompt: thinking should close, answer present, no </think> leaking
    for seed in (1, 2):
        r = chat(PROMPTS["bit_reverse"], max_tokens=12000, xargs={"think_stop_gate": 1000}, seed=seed)
        c = r["choices"][0]
        reas, cont = c["message"].get("reasoning") or "", c["message"].get("content") or ""
        n = r["usage"]["completion_tokens"]
        good = c["finish_reason"] == "stop" and "```" in cont and "</think>" not in cont
        ok &= good
        print(f"[gate 1000] bit_reverse seed {seed}: finish {c['finish_reason']}, {n} tokens, reasoning "
              f"{len(reas)} chars, answer {len(cont)} chars, code {'yes' if '```' in cont else 'NO'}, "
              f"last thinking line {reas.strip().splitlines()[-1][:80]!r} -> {'OK' if good else 'FAIL'}")
    # 2. thinking off, aggressive gate: must be untouched (no </think> in the answer, normal length)
    r = chat("Write a markdown table of the 12 months with their number of days and one holiday each, then a "
             "short paragraph about leap years.", max_tokens=3000, xargs={"think_stop_gate": 1}, think=False)
    cont = r["choices"][0]["message"].get("content") or ""
    good = "</think>" not in cont and cont.count("|") > 20 and r["choices"][0]["finish_reason"] == "stop"
    ok &= good
    print(f"[thinking off, gate 1] {r['usage']['completion_tokens']} tokens, table rows "
          f"{sum(l.startswith('|') for l in cont.splitlines())}, '</think>' in answer: {'</think>' in cont} "
          f"-> {'OK' if good else 'FAIL'}")
    # 3. speed: processor active below its gate (default) vs declined (gate 0), single stream
    msg = "Explain how a hash map works, with a small Python example. Be thorough."
    for label, xa in (("processor on (below gate)", None), ("processor off (gate 0)", {"think_stop_gate": 0})):
        sp = [chat(msg, max_tokens=1500, xargs=xa, seed=s, stream=True)[0] for s in (11, 12, 13)]
        print(f"[speed] {label}: " + ", ".join(f"{x:.1f}" for x in sp) + " tok/s")
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
