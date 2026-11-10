"""Side-by-side report of lm-eval battery runs (scripts/eval_session battery step) for two or more models,
with MGSM re-scored by one answer parser for every model (lm-eval's "The answer is N" regex misses chat
answers like "Janet makes **$18** every day": strict-match is 0 for chat models).

MGSM answer: the last bold span holding a number (**18**, **$18**, **45 miles**), else the last number of
the answer; '70.000' / '195 000' read as thousands, '50,00' as a decimal comma.

  python scripts/battery_report.py artifacts/overnight/battery/sq artifacts/overnight/battery/q4
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import defaultdict

SEP = "[   .,]"
NUMRE = re.compile(r"-?\d+(?:" + SEP + r"\d+)*")


def norm_num(n: str) -> str | None:
    """'70.000' / '195 000' -> thousands; '50,00' / '7,5' -> decimal comma; '1,234.5' -> 1234.5."""
    n = n.replace(" ", " ").replace(" ", " ").strip()
    if re.fullmatch(r"-?\d{1,3}([ .,]\d{3})+", n):
        return re.sub(r"[ .,]", "", n)
    n = n.replace(" ", "")
    if re.fullmatch(r"-?\d+,\d{1,2}", n):
        return n.replace(",", ".")
    return n.replace(",", "").rstrip(".")


def numbers(text: str) -> list[str]:
    text = re.sub(r"(\d)\\,(\d)", r"\1\2", text)           # LaTeX thin space: 325\,000
    return [x for x in (norm_num(m) for m in NUMRE.findall(text)) if x]


def mgsm_answer(text: str) -> str | None:
    text = text.split("</think>")[-1]
    bold = text.split("**")[1::2]                 # the text inside **...** pairs
    for cand in reversed(bold):
        n = numbers(cand)
        if n:
            return n[0] if len(n) == 1 else n[-1]
    n = numbers(text)
    return n[-1] if n else None


def same(a, b) -> bool:
    try:
        return a is not None and abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


def load(root: str) -> dict:
    out = {}
    for f in glob.glob(os.path.join(root, "**", "results_*.json"), recursive=True):
        r = json.load(open(f, encoding="utf-8"))["results"]
        for task, m in r.items():
            for key in ("prompt_level_strict_acc,none", "exact_match,custom-extract", "exact_match,get-answer",
                        "exact_match,flexible-extract", "inst_level_strict_acc,none"):
                if key in m:
                    out.setdefault(task, {})[key.split(",")[0] + ("" if "flexible" not in key else "_flex")] = m[key]
    # MGSM re-scored
    for f in glob.glob(os.path.join(root, "**", "samples_mgsm_native_cot_*.jsonl"), recursive=True):
        task = re.search(r"samples_(mgsm_native_cot_\w+?)_\d{4}-", os.path.basename(f)).group(1)
        rows = [json.loads(line) for line in open(f, encoding="utf-8")]
        ok = [same(mgsm_answer(r["resps"][0][0]), r["target"]) for r in rows]
        out.setdefault(task, {})["rescored"] = sum(ok) / len(ok)
    return out


def main() -> int:
    roots = sys.argv[1:]
    data = {os.path.basename(r.rstrip("/")): load(r) for r in roots}
    names = list(data)
    tasks = sorted(set().union(*[set(d) for d in data.values()]))
    print(f"{'task':34s} {'metric':22s} " + " ".join(f"{n:>9s}" for n in names))
    for t in tasks:
        metrics = sorted(set().union(*[set(data[n].get(t, {})) for n in names]))
        for m in metrics:
            if t.startswith("mgsm") and m != "rescored":
                continue
            vals = [data[n].get(t, {}).get(m) for n in names]
            print(f"{t:34s} {m:22s} " + " ".join(f"{100 * v:9.1f}" if v is not None else f"{'-':>9s}" for v in vals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
