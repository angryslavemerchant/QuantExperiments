"""Repetition / looping detector for generated text, usable as a library or on saved eval outputs.

loop_score(text) -> dict with
  rep_line:  max count of one identical non-trivial line (>15 chars)
  rep_gram:  max count of one identical 8-word sequence
  rep_norm:  max count of one line after replacing every number with 0 (enumeration runaway)
  tail_rep:  how many times the last 200 chars' period repeats back-to-back at the end (degenerate loop at
             max_tokens)
is_loop(text): rep_line >= 3 or rep_gram >= 4 or rep_norm >= 10 or tail_rep >= 3.

Code answers legitimately repeat lines (e.g. `return result`, test asserts), so for code a stricter rule
is used: is_loop(text, code=True) requires rep_line >= 6 or rep_gram >= 6 or rep_norm >= 15 or tail_rep >= 3.

  python scripts/loop_detect.py artifacts/overnight/code/*.raw.json   # code_eval / gsm8k_server outputs
"""

from __future__ import annotations

import collections
import json
import re
import sys

_DIGITS = re.compile(r"\d+")


def tail_period_repeats(t: str, max_period: int = 400) -> int:
    t = t.rstrip()
    best = 1
    for p in range(8, max_period):
        if len(t) < 3 * p:
            break
        unit = t[-p:]
        k = 1
        while len(t) >= (k + 1) * p and t[-(k + 1) * p: -k * p] == unit:
            k += 1
        best = max(best, k)
    return best


def loop_score(t: str) -> dict:
    lines = [ln.strip() for ln in t.split("\n") if len(ln.strip()) > 15]
    cnt = collections.Counter(lines)
    w = t.split()
    grams = collections.Counter(tuple(w[i:i + 8]) for i in range(max(0, len(w) - 7)))
    # "enumeration runaway": the same line with only its numbers changing (n = 309, 319, 329, ...)
    norm = collections.Counter(_DIGITS.sub("0", ln) for ln in lines)
    return {"rep_line": max(cnt.values()) if cnt else 0, "rep_gram": max(grams.values()) if grams else 0,
            "rep_norm": max(norm.values()) if norm else 0, "tail_rep": tail_period_repeats(t)}


def is_loop(t: str, code: bool = False) -> bool:
    s = loop_score(t)
    if code:
        return s["rep_line"] >= 6 or s["rep_gram"] >= 6 or s["rep_norm"] >= 15 or s["tail_rep"] >= 3
    return s["rep_line"] >= 3 or s["rep_gram"] >= 4 or s["rep_norm"] >= 10 or s["tail_rep"] >= 3


def texts_of(d: dict) -> list[str]:
    if "responses" in d:
        return [(r.get("reasoning") or "") + "\n" + (r.get("content") or "") for r in d["responses"]]
    return d["completions"]


def main() -> int:
    for f in sys.argv[1:]:
        d = json.load(open(f, encoding="utf-8"))
        texts = texts_of(d)
        code = "responses" in d
        loops = [i for i, t in enumerate(texts) if is_loop(t, code=code)]
        length = [i for i, r in enumerate(d.get("responses", [])) if r.get("finish") == "length"]
        print(f"{d.get('name', f)}: {len(loops)}/{len(texts)} loops, {len(length)} hit max_tokens, "
              f"loops among max_tokens: {len(set(loops) & set(length))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
