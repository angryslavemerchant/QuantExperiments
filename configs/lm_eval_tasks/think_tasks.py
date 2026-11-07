"""Scoring for the thinking-mode tasks (math500_think, gpqa_diamond_think).

Both ask for the final answer in \\boxed{...}; lm-eval strips the <think> block first
(think_end_token) so only the final response is scored. Each also reports `no_answer`
(no \\boxed{} found: usually a response cut at max_gen_toks), to tell truncation apart
from wrong answers.
"""

from __future__ import annotations

import random
import re


def last_boxed(text: str) -> str | None:
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return None


# ---------------------------------------------------------------- MATH-500

def _strip_equiv(a: str, b: str) -> bool:
    """Hendrycks MATH string normalisation (pure Python; no sympy, no timeouts)."""
    try:
        from lm_eval.tasks.hendrycks_math.utils import is_equiv
        return bool(is_equiv(a, b))
    except Exception:
        return a.strip() == b.strip()


def math_results(doc: dict, results: list[str]) -> dict:
    """math_verify (symbolic; the headline) and exact (normalised string match, a check on it)."""
    resp = results[0]
    box = last_boxed(resp)
    mv = ex = 0
    if box is not None:
        ex = int(_strip_equiv(box, doc["answer"]))
        try:
            from math_verify import parse, verify
            mv = int(bool(verify(parse("\\boxed{" + doc["answer"] + "}"), parse("\\boxed{" + box + "}"))))
        except Exception:
            mv = 0
        mv = max(mv, ex)          # anything the string match accepts counts for math_verify too
    return {"math_verify": mv, "exact": ex, "no_answer": int(box is None)}


# ---------------------------------------------------------------- GPQA diamond

def _clean(t):
    t = (t or " ").strip().replace(" [title]", ". ")
    return re.sub(r"\s+", " ", t)


def gpqa_docs(dataset):
    """Shuffle the four options per question with a per-question seed (reproducible across runs
    and models, independent of global RNG state)."""
    def f(doc, idx):
        opts = [_clean(doc["Incorrect Answer 1"]), _clean(doc["Incorrect Answer 2"]),
                _clean(doc["Incorrect Answer 3"]), _clean(doc["Correct Answer"])]
        random.Random(1000 + idx).shuffle(opts)
        gold = opts.index(_clean(doc["Correct Answer"]))
        return {"question": _clean(doc["Question"]), "A": opts[0], "B": opts[1], "C": opts[2], "D": opts[3],
                "gold": "ABCD"[gold]}
    return dataset.map(f, with_indices=True)


def gpqa_results(doc: dict, results: list[str]) -> dict:
    resp = results[0]
    box = last_boxed(resp)
    letter = None
    if box is not None:
        m = re.search(r"[ABCD]", box.replace("\\text", ""))
        letter = m.group(0) if m else None
    if letter is None:
        m = re.findall(r"answer is:?\s*\(?\**([ABCD])\b", resp)
        letter = m[-1] if m else None
    return {"exact_match": int(letter == doc["gold"]), "no_answer": int(letter is None)}
