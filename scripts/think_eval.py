"""Thinking-mode GPQA-diamond and MATH-500 straight on vLLM, saved chunk by chunk.

Same prompts, sampling and scoring as configs/lm_eval_tasks/{gpqa_diamond,math500}_think.yaml
(think_tasks.py scorers), without lm-eval: each chunk of questions is generated and scored, then
appended to <out>/samples_<task>.jsonl and summarized in <out>/summary.json, so a cutoff keeps
every finished chunk; a rerun skips questions already in the samples file.

  python scripts/think_eval.py --model DIR --name ours --out artifacts/think27b/ours --tasks gpqa,math500
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("think_tasks", ROOT / "configs" / "lm_eval_tasks" / "think_tasks.py")
tt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tt)

GPQA_PROMPT = ("{question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}\n\n"
               "Please reason step by step, and put the letter of your final answer within \\boxed{{}}.")
MATH_PROMPT = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_task(task: str) -> list[dict]:
    from datasets import load_dataset
    if task == "gpqa":
        ds = tt.gpqa_docs(load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train"))
        return [{"id": i, "prompt": GPQA_PROMPT.format(**d), "doc": {"gold": d["gold"]}} for i, d in enumerate(ds)]
    if task == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return [{"id": i, "prompt": MATH_PROMPT.format(problem=d["problem"]), "doc": {"answer": d["answer"]}}
                for i, d in enumerate(ds)]
    raise ValueError(task)


def score(task: str, doc: dict, text: str) -> dict:
    final = text.split("</think>")[-1] if "</think>" in text else ""   # no </think>: cut off while thinking
    r = tt.gpqa_results(doc, [final]) if task == "gpqa" else tt.math_results(doc, [final])
    r["think_closed"] = int("</think>" in text)
    return r


def summarize(rows: list[dict], metric: str) -> dict:
    n = len(rows)
    out = {"n": n}
    for k in rows[0]["scores"]:
        p = sum(r["scores"][k] for r in rows) / n
        out[k] = p
        out[k + "_stderr"] = math.sqrt(p * (1 - p) / (n - 1)) if n > 1 else None
    out["mean_gen_tokens"] = sum(r["gen_tokens"] for r in rows) / n
    out["max_gen_tokens_hit"] = sum(r["finish"] == "length" for r in rows) / n
    out["headline"] = metric
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="gpqa,math500")
    ap.add_argument("--chunk", type=int, default=250,
                    help="questions per chunk (results saved after each; each chunk ends in a long-generation tail)")
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--max-model-len", type=int, default=36864)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--limit", type=int, help="first N questions per task (smoke)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem,
              max_num_seqs=a.max_num_seqs, enable_prefix_caching=True, seed=0)
    log(f"loaded {a.model} in {time.time() - t0:.0f}s on {torch.cuda.get_device_name(0)}")
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"name": a.name, "model": a.model, "tasks": {}}

    for task in a.tasks.split(","):
        metric = "exact_match" if task == "gpqa" else "math_verify"
        items = load_task(task)
        if a.limit:
            items = items[:a.limit]
        sp_path = out / f"samples_{task}.jsonl"
        done = {}
        if sp_path.exists():
            for line in sp_path.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                done[r["id"]] = r
        todo = [it for it in items if it["id"] not in done]
        log(f"{task}: {len(items)} questions, {len(done)} already done")
        for c0 in range(0, len(todo), a.chunk):
            chunk = todo[c0:c0 + a.chunk]
            t1 = time.time()
            # one seed per question: reruns of a question give the same sample
            params = [SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=a.max_tokens,
                                     seed=1000 + it["id"]) for it in chunk]
            msgs = [[{"role": "user", "content": it["prompt"]}] for it in chunk]
            res = llm.chat(msgs, params, use_tqdm=False, chat_template_kwargs={"enable_thinking": True})
            with open(sp_path, "a", encoding="utf-8") as f:
                for it, r in zip(chunk, res):
                    o = r.outputs[0]
                    row = {"id": it["id"], "scores": score(task, it["doc"], o.text), "gen_tokens": len(o.token_ids),
                           "finish": o.finish_reason, "target": it["doc"], "text": o.text}
                    done[it["id"]] = row
                    f.write(json.dumps(row) + "\n")
            rows = [done[i] for i in sorted(done)]
            summary["tasks"][task] = summarize(rows, metric)
            summary["tasks"][task]["complete"] = len(rows) == len(items)
            summary_path.write_text(json.dumps(summary, indent=2))
            s = summary["tasks"][task]
            log(f"{task}: {len(rows)}/{len(items)} done, {metric} {100 * s[metric]:.1f} +- {100 * (s[metric + '_stderr'] or 0):.1f}, "
                f"no_answer {100 * s['no_answer']:.1f}%, mean {s['mean_gen_tokens']:.0f} tokens (chunk {time.time() - t1:.0f}s)")
    log(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
