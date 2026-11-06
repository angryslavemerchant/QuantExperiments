"""Standard LLM benchmarks (EleutherAI lm-evaluation-harness) on the live model.

Runs on whatever weights the model currently holds, so every row of an experiment
(original, our stacks after fine-tune, GGUF / VPTQ references) goes through the same
protocol: block linears from the method, embeddings / lm_head / norms at 16 bits.
The packed runtime reproduces these weights' logits (scripts/check_runtime.py), so
these scores stand for the runtime too.

Config (eval.lm_eval):
    batch_size: 16
    tasks:
      - {task: mmlu, fewshot: 5}
      - {task: arc_challenge, fewshot: 0}
      - {task: hellaswag, fewshot: 0, limit: 2000}
      - {task: winogrande, fewshot: 0}
      - {task: gsm8k, fewshot: 5, limit: 400}

Each task is evaluated with its own few-shot count and example limit (lm-eval takes
one num_fewshot per call). The headline metric per task is reported as lme_<task>
(acc_norm where the task defines it for multiple choice, else acc; gsm8k: flexible-extract
exact match); the full lm-eval results go to the row's JSON file.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

HEADLINE = {
    "mmlu": ("acc,none",),
    "arc_challenge": ("acc_norm,none", "acc,none"),
    "arc_easy": ("acc_norm,none", "acc,none"),
    "hellaswag": ("acc_norm,none", "acc,none"),
    "winogrande": ("acc,none",),
    "piqa": ("acc_norm,none", "acc,none"),
    "gsm8k": ("exact_match,flexible-extract", "exact_match,strict-match"),
}


def headline(task: str, res: dict) -> tuple[str, float | None]:
    r = res.get("results", {}).get(task, {})
    for key in HEADLINE.get(task, ("acc_norm,none", "acc,none", "exact_match,strict-match")):
        if key in r and isinstance(r[key], (int, float)):
            return key, float(r[key])
    return "", None


def run_lm_eval(model, tok, lcfg: dict, out_path: Path, log=print) -> dict:
    """Evaluate every configured task on `model` (an HF causal LM in eval mode, on its
    device). Returns {"lme_<task>": score, ...}; full results written to out_path."""
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=lcfg.get("batch_size", 16),
              max_length=lcfg.get("max_length", 4096))
    flat, full = {}, {}
    for spec in lcfg["tasks"]:
        task = spec["task"]
        t0 = time.time()
        res = simple_evaluate(model=lm, tasks=[task], num_fewshot=spec.get("fewshot"), limit=spec.get("limit"),
                              bootstrap_iters=0, log_samples=False, random_seed=0, numpy_random_seed=0,
                              torch_random_seed=0, fewshot_random_seed=0,
                              gen_kwargs=spec.get("gen_kwargs"))
        key, score = headline(task, res)
        flat[f"lme_{task}"] = score
        full[task] = {"metric": key, "score": score, "seconds": time.time() - t0, "fewshot": spec.get("fewshot"),
                      "limit": spec.get("limit"), "results": res.get("results", {}),
                      "groups": res.get("groups", {}), "n_samples": res.get("n-samples", {})}
        log(f"    lm-eval {task} ({spec.get('fewshot')}-shot{', limit ' + str(spec['limit']) if spec.get('limit') else ''}): "
            f"{key} {score if score is None else round(100 * score, 2)} ({time.time() - t0:.0f}s)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(full, indent=2, default=str))
    return flat
