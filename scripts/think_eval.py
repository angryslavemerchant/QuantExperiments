"""Thinking-mode GPQA-diamond and MATH-500 straight on vLLM, saved chunk by chunk.

Same prompts, sampling and scoring as configs/lm_eval_tasks/{gpqa_diamond,math500}_think.yaml
(think_tasks.py scorers), without lm-eval. Default (--chunk 0): every question is queued on the
vLLM engine at once and each one is scored and appended to <out>/samples_<task>.jsonl the moment
its samples finish (summary.json refreshed every --summary-every), so a cutoff loses only the
questions still generating. --chunk N instead generates N questions per blocking llm.chat call
(nothing of a chunk is saved until all of it finishes: 100 questions x 8 samples of 32k-token
reasoning outlasted a 3.4 h window, s37). A rerun skips questions already in the samples file.

  python scripts/think_eval.py --model DIR --name ours --out artifacts/think27b/ours --tasks gpqa,math500
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("think_tasks", ROOT / "configs" / "lm_eval_tasks" / "think_tasks.py")
tt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tt)

GPQA_PROMPT = ("{question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}\n\n"
               "Please reason step by step, and put the letter of your final answer within \\boxed{{}}.")
MATH_PROMPT = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
# --style lighteval: lighteval 0.13.0's wording (tasks/tasks/gpqa.py gpqa_instruct_prompt,
# tasks/tasks/math_500.py MATH_QUERY_TEMPLATE)
LE_GPQA_PROMPT = ("Answer the following multiple choice question. The last line of your response should be of the "
                  "following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by "
                  "step before answering.\n\n{question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}")
LE_MATH_PROMPT = ('Solve the following problem. The final line of your response MUST be of the following format: '
                  '"ANSWER: $ANSWER" (without quotes) where $ANSWER is the final answer. Think step by step before '
                  'answering.\n\n{problem}')


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_task(task: str, style: str = "boxed") -> list[dict]:
    from datasets import load_dataset
    le = style == "lighteval"
    if task == "gpqa":
        ds = tt.gpqa_docs(load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train"))
        tpl = LE_GPQA_PROMPT if le else GPQA_PROMPT
        return [{"id": i, "prompt": tpl.format(**d), "doc": {"gold": d["gold"]}} for i, d in enumerate(ds)]
    if task == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        tpl = LE_MATH_PROMPT if le else MATH_PROMPT
        return [{"id": i, "prompt": tpl.format(problem=d["problem"]), "doc": {"answer": d["answer"]}}
                for i, d in enumerate(ds)]
    raise ValueError(task)


def _answer_line(final: str, tag: str) -> str | None:
    """Text after the last 'Answer:' / 'ANSWER:' marker (lighteval-style responses)."""
    hits = re.findall(rf"{tag}\s*:\s*(.+)", final, flags=re.IGNORECASE)
    return hits[-1].strip().strip("*$ .") if hits else None


def score(task: str, doc: dict, text: str) -> dict:
    """Boxed answers first; else (lighteval-style prompts) the 'Answer: X' / 'ANSWER: x' line."""
    final = text.split("</think>")[-1] if "</think>" in text else ""   # no </think>: cut off while thinking
    if task == "gpqa":
        r = tt.gpqa_results(doc, [final])
        if r["no_answer"]:
            line = _answer_line(final, "answer")
            m = re.match(r"\(?([ABCD])\b", line) if line else None
            if m:
                r = {"exact_match": int(m.group(1) == doc["gold"]), "no_answer": 0}
    else:
        r = tt.math_results(doc, [final])
        if r["no_answer"]:
            line = _answer_line(final, "answer")
            if line:
                r = tt.math_results(doc, ["\\boxed{" + line + "}"])
    r["think_closed"] = int("</think>" in text)
    return r


def summarize(rows: list[dict], metric: str) -> dict:
    """Per-question scores are means over that question's samples; stderr over questions."""
    n = len(rows)
    out = {"n": n, "samples_per_question": rows[0].get("n_samples", 1)}
    for k in rows[0]["scores"]:
        vals = [r["scores"][k] for r in rows]
        p = sum(vals) / n
        out[k] = p
        out[k + "_stderr"] = (math.sqrt(sum((v - p) ** 2 for v in vals) / (n - 1) / n) if n > 1 else None)
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
    ap.add_argument("--chunk", type=int, default=0,
                    help="0: stream (save each question as it finishes); N: blocking chunks of N questions")
    ap.add_argument("--summary-every", type=int, default=10, help="stream mode: refresh summary.json every N questions")
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--max-model-len", type=int, default=36864)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--limit", type=int, help="first N questions per task (smoke)")
    ap.add_argument("--n", type=int, default=1, help="samples per question (pass@1 averaged over them)")
    ap.add_argument("--style", choices=["boxed", "lighteval"], default="boxed", help="prompt wording")
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
        items = load_task(task, a.style)
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

        def make_row(it, r):
            per = [score(task, it["doc"], o.text) for o in r.outputs]
            k = len(per)
            return {"id": it["id"], "n_samples": k,
                    "scores": {m: sum(p[m] for p in per) / k for m in per[0]},
                    "gen_tokens": sum(len(o.token_ids) for o in r.outputs) / k,
                    "finish": "length" if any(o.finish_reason == "length" for o in r.outputs) else "stop",
                    "target": it["doc"], "samples": [{"scores": p, "finish": o.finish_reason,
                                                      "gen_tokens": len(o.token_ids), "text": o.text}
                                                     for p, o in zip(per, r.outputs)]}

        def write_summary():
            rows = [done[i] for i in sorted(done)]
            if not rows:
                return
            summary["tasks"][task] = summarize(rows, metric)
            summary["tasks"][task]["complete"] = len(rows) == len(items)
            summary_path.write_text(json.dumps(summary, indent=2))
            s = summary["tasks"][task]
            log(f"{task}: {len(rows)}/{len(items)} done, {metric} {100 * s[metric]:.1f} +- "
                f"{100 * (s[metric + '_stderr'] or 0):.1f}, no_answer {100 * s['no_answer']:.1f}%, "
                f"mean {s['mean_gen_tokens']:.0f} tokens")

        if a.chunk <= 0 and todo:
            # streaming: all questions queued; each saved as soon as its n samples finish
            eng = llm.llm_engine
            tok = llm.get_tokenizer()
            by_id = {}
            for it in todo:
                prompt = tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=True)
                sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=a.max_tokens, n=a.n,
                                    seed=1000 + it["id"])
                eng.add_request(f"{task}-{it['id']}", prompt, sp)
                by_id[f"{task}-{it['id']}"] = it
            t1, last, fresh = time.time(), time.time(), 0
            with open(sp_path, "a", encoding="utf-8") as f:
                while eng.has_unfinished_requests():
                    for r in eng.step():
                        if not getattr(r, "finished", False) or r.request_id not in by_id:
                            continue
                        it = by_id.pop(r.request_id)
                        row = make_row(it, r)
                        done[it["id"]] = row
                        f.write(json.dumps(row) + "\n")
                        f.flush()
                        fresh += 1
                        if fresh % a.summary_every == 0:
                            write_summary()
                    if time.time() - last > 300:
                        last = time.time()
                        log(f"{task}: {fresh} finished this run, {len(by_id)} generating/queued ({time.time() - t1:.0f}s)")
            write_summary()
            continue
        for c0 in range(0, len(todo), a.chunk):
            chunk = todo[c0:c0 + a.chunk]
            t1 = time.time()
            # one seed per question: reruns of a question give the same sample
            params = [SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=a.max_tokens, n=a.n,
                                     seed=1000 + it["id"]) for it in chunk]
            msgs = [[{"role": "user", "content": it["prompt"]}] for it in chunk]
            res = llm.chat(msgs, params, use_tqdm=False, chat_template_kwargs={"enable_thinking": True})
            with open(sp_path, "a", encoding="utf-8") as f:
                for it, r in zip(chunk, res):
                    per = [score(task, it["doc"], o.text) for o in r.outputs]
                    k = len(per)
                    row = {"id": it["id"], "n_samples": k,
                           "scores": {m: sum(p[m] for p in per) / k for m in per[0]},
                           "gen_tokens": sum(len(o.token_ids) for o in r.outputs) / k,
                           "finish": "length" if any(o.finish_reason == "length" for o in r.outputs) else "stop",
                           "target": it["doc"], "samples": [{"scores": p, "finish": o.finish_reason,
                                                             "gen_tokens": len(o.token_ids), "text": o.text}
                                                            for p, o in zip(per, r.outputs)]}
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
