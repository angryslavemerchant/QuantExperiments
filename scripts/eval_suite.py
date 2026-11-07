"""Standard benchmark suite on an HF checkpoint with lm-eval-harness's vLLM backend.

One vLLM engine per checkpoint; each task runs with its own few-shot count, limit and
prompt mode, and its result is written as soon as it finishes (a later failure or timeout
keeps the earlier tasks):

  base-style prompts (no chat template), greedy:
    mmlu 5-shot, arc_challenge 0, hellaswag 0, winogrande 0, gsm8k 5-shot (all 1319),
    mmlu_pro 5-shot CoT (limit per subject, --mmlu-pro-limit)
  chat template, thinking on (Qwen's sampling: T 0.6, top-p 0.95, top-k 20; 32k tokens):
    ifeval, math500_think, gpqa_diamond_think (configs/lm_eval_tasks; GPQA is gated on the
    Hub and is skipped with a note when the token lacks access)

Runs inside an environment with vllm + lm_eval (scripts/eval_box.sh builds one).

  python scripts/eval_suite.py --ckpt /workspace/eval_ckpt/ours --name ours --out artifacts/eval27b/ours
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "configs" / "lm_eval_tasks"

THINK_GEN = {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_gen_toks": 32768}

HEADLINE = {
    "mmlu": "acc,none", "arc_challenge": "acc_norm,none", "hellaswag": "acc_norm,none",
    "winogrande": "acc,none", "gsm8k": "exact_match,flexible-extract",
    "mmlu_pro": "exact_match,custom-extract", "ifeval": "prompt_level_strict_acc,none",
    "math500_think": "math_verify,none", "gpqa_diamond_think": "exact_match,none",
}


def plan(a) -> list[dict]:
    base = [
        {"task": "mmlu", "fewshot": 5},
        {"task": "arc_challenge", "fewshot": 0},
        {"task": "hellaswag", "fewshot": 0},
        {"task": "winogrande", "fewshot": 0},
        {"task": "gsm8k", "fewshot": 5, "log": True},
        {"task": "mmlu_pro", "fewshot": 5, "limit": a.mmlu_pro_limit, "log": True},
    ]
    think = [
        {"task": "ifeval", "fewshot": 0, "chat": True, "gen": THINK_GEN, "log": True},
        {"task": "math500_think", "fewshot": 0, "chat": True, "log": True},
        {"task": "gpqa_diamond_think", "fewshot": 0, "chat": True, "log": True},
    ]
    tasks = base + think
    if a.tasks:
        keep = set(a.tasks.split(","))
        tasks = [t for t in tasks if t["task"] in keep]
    return tasks


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def headline(task: str, res: dict):
    r = res.get("results", {}).get(task, {})
    key = HEADLINE.get(task)
    if key in r:
        return key, r[key], r.get(key.replace(",", "_stderr,"))
    for k, v in r.items():
        if isinstance(v, (int, float)) and "stderr" not in k and k != "alias":
            return k, v, r.get(k.replace(",", "_stderr,"))
    return None, None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", help="comma list (default: the whole suite)")
    ap.add_argument("--mmlu-pro-limit", type=int, default=100, help="questions per MMLU-Pro subject")
    ap.add_argument("--max-model-len", type=int, default=36864)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-num-seqs", type=int, default=256,
                    help="concurrent sequences (vLLM default 1024 exceeds the DeltaNet state slots of a 27B on 96 GB)")
    ap.add_argument("--limit", type=int, help="override every task's limit (smoke tests)")
    ap.add_argument("--max-gen", type=int, help="cap max_gen_toks everywhere (smoke tests)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import lm_eval
    import vllm
    from lm_eval import simple_evaluate
    from lm_eval.models.vllm_causallms import VLLM
    from lm_eval.tasks import TaskManager

    env = {"lm_eval": lm_eval.__version__ if hasattr(lm_eval, "__version__") else None,
           "vllm": vllm.__version__, "ckpt": a.ckpt, "name": a.name}
    try:
        import torch
        env.update(torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
    except Exception:
        pass
    (out / "env.json").write_text(json.dumps(env, indent=2))
    log(json.dumps(env))

    t0 = time.time()
    kw = dict(pretrained=a.ckpt, dtype="bfloat16", max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem,
              batch_size="auto", seed=0, max_gen_toks=2048, enable_prefix_caching=True,
              max_num_seqs=a.max_num_seqs)  # hybrid models: one DeltaNet state slot per running sequence
    try:   # text only: no vision memory profiling (multimodal base repos)
        lm = VLLM(**kw, limit_mm_per_prompt={"image": 0, "video": 0})
    except Exception as e:
        log(f"engine with limit_mm_per_prompt failed ({type(e).__name__}: {str(e)[:300]}); retrying without")
        lm = VLLM(**kw)
    log(f"engine ready in {time.time() - t0:.0f}s")
    tm = TaskManager(include_path=str(TASK_DIR))

    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"name": a.name, "tasks": {}}
    for spec in plan(a):
        task = spec["task"]
        if task in summary["tasks"] and summary["tasks"][task].get("score") is not None:
            log(f"{task}: already done, skipping")
            continue
        chat = spec.get("chat", False)
        lm.enable_thinking = True if chat else None
        lm.think_end_token = "</think>" if chat else None
        gen = dict(spec.get("gen") or {})
        if a.max_gen:
            gen["max_gen_toks"] = a.max_gen
        limit = a.limit if a.limit is not None else spec.get("limit")
        t1 = time.time()
        log(f"{task}: {spec.get('fewshot')}-shot{' chat+thinking' if chat else ''}"
            f"{f', limit {limit}' if limit else ''}")
        try:
            res = simple_evaluate(model=lm, tasks=[task], num_fewshot=spec.get("fewshot"), limit=limit,
                                  task_manager=tm, apply_chat_template=chat, fewshot_as_multiturn=chat,
                                  gen_kwargs=gen or None, log_samples=bool(spec.get("log")),
                                  bootstrap_iters=1000,
                                  random_seed=0, numpy_random_seed=0, torch_random_seed=0, fewshot_random_seed=0)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            gated = "gated" in msg.lower() or "403" in msg
            log(f"{task}: FAILED {msg[:400]}")
            summary["tasks"][task] = {"score": None, "error": msg[:2000],
                                      **({"note": "dataset gated for this token"} if gated else {}),
                                      "traceback": traceback.format_exc()[-4000:]}
            summary_path.write_text(json.dumps(summary, indent=2))
            continue
        key, score, se = headline(task, res)
        rec = {"metric": key, "score": score, "stderr": se, "seconds": time.time() - t1,
               "fewshot": spec.get("fewshot"), "limit": limit, "chat_thinking": chat,
               "gen_kwargs": gen or None, "n_samples": res.get("n-samples", {}),
               "results": res.get("results", {}), "groups": res.get("groups", {})}
        samples = (res.get("samples") or {})
        if samples:
            with gzip.open(out / f"samples_{task}.jsonl.gz", "wt", encoding="utf-8") as f:
                for sub, rows in samples.items():
                    for r in rows:
                        f.write(json.dumps({"task": sub, "doc_id": r.get("doc_id"), "target": r.get("target"),
                                            "resps": r.get("filtered_resps") or r.get("resps"),
                                            "metrics": {k: r.get(k) for k in r.get("metrics", [])}},
                                           default=str) + "\n")
        summary["tasks"][task] = rec
        summary_path.write_text(json.dumps(summary, indent=2, default=str))
        log(f"{task}: {key} = {score if score is None else round(100 * score, 2)}"
            f"{'' if se is None else f' +- {100 * se:.2f}'} ({time.time() - t1:.0f}s)")
    log(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
