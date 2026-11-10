"""Re-score saved scripts/gsm8k_server.py runs with the FIRST "#### <n>" answer instead of the last number.

Needed when end-of-turn tokens were banned (--ban-ids): a chat model that would have stopped after
"#### 18" keeps going (repeats its work, "Final Answer: \\boxed{18}", ...), and the last-number rule then
picks up a number from that runoff. Falls back to the last number when there is no "####".

  python scripts/gsm8k_rescore.py artifacts/gsm8k/server_sq_v2.json artifacts/gsm8k/server_q4km_v2.json
"""

from __future__ import annotations

import json
import re
import sys

NUM = re.compile(r"-?\d[\d,]*\.?\d*")
HASH = re.compile(r"####\s*\$?\s*(-?\d[\d,]*\.?\d*)")


def first_answer(t: str) -> str | None:
    t = t.split("Question:")[0]
    m = HASH.search(t)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    n = NUM.findall(t)
    return n[-1].replace(",", "").rstrip(".") if n else None


def same(a: str | None, b: str) -> bool:
    if a is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return a == b


def main() -> int:
    from datasets import load_dataset
    refs = [r["answer"].split("####")[-1].strip().replace(",", "")
            for r in load_dataset("openai/gsm8k", "main", split="test")]
    runs = {}
    for f in sys.argv[1:]:
        d = json.load(open(f, encoding="utf-8"))
        ok = [same(first_answer(t), r) for t, r in zip(d["completions"], refs)]
        runs[d["name"]] = ok
        print(f"{d['name']:22s} last-number {100 * d['acc']:6.2f}%   first-#### {100 * sum(ok) / len(ok):6.2f}%")
    names = list(runs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = runs[names[i]], runs[names[j]]
            print(f"{names[i]} vs {names[j]}: both right {sum(x and y for x, y in zip(a, b))}, "
                  f"only {names[i]} {sum(x and not y for x, y in zip(a, b))}, only {names[j]} {sum(y and not x for x, y in zip(a, b))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
