"""Loop mechanism study, step 3: runaway rate vs quality for anti-loop interventions, thinking on.

Two-phase generation per request so an intervention touches only the thinking:
  phase 1  thinking, intervention active after --gate thinking tokens, stops at </think> (or --max-think = runaway)
  phase 2  answer from prompt + thinking + </think>, unmodified model
Conditions:
  base   no intervention
  steer  + alpha * gap * d at layer --layer ("ready to close" direction, scripts/loopmech_analyze.py)
  bias   logit bonus on </think> right after a line break, ramping slope*(t - gate)/1000 (budget-style control)
Sets: loop-prone prompts (scripts/loop_probe.py), a seeded HumanEval+ subset (scored with evalplus in
~/eval-venv afterwards, see --score), a seeded MMLU-Pro subset.

  python scripts/loopmech_eval.py --out artifacts/loopmech/eval --conds base,steer,bias
  ~/eval-venv/bin/python scripts/loopmech_eval.py --out artifacts/loopmech/eval --score
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
sys.path.insert(0, os.path.dirname(__file__))

HE_PROMPT = ("Please provide a self-contained Python script that solves the following problem in a markdown code "
             "block:\n```\n{p}\n```")
MMLU_PROMPT = ("The following is a multiple choice question. Think step by step and then finish your answer with "
               "\"the answer is (X)\" where X is the correct letter choice.\n\nQuestion: {q}\nOptions:\n{opts}")
LETTERS = "ABCDEFGHIJ"
END_ID = 248069
BIAS = {"gate": 3000, "slope": 6.0, "cap": 30.0, "nl": None}


def make_bias_proc():
    """vLLM model-runner-V2 logits processor: + min(cap, slope * (n_out - gate) / 1000) on </think> when the
    last token ends a line, for requests with extra_args.think_bias."""
    import torch
    from vllm.v1.worker.gpu.sample.logits_processor import LogitsProcessor

    class ThinkBias(LogitsProcessor):
        def __init__(self, vllm_config, req_states):
            self.rs = req_states
            self.active = torch.zeros(req_states.max_num_reqs, dtype=torch.bool, device=req_states.device)
            self.nl = torch.zeros(req_states.vocab_size, dtype=torch.bool, device=req_states.device)
            self.nl[torch.tensor(sorted(i for i in BIAS["nl"] if i < req_states.vocab_size))] = True

        def add_request(self, req_idx, sampling_params) -> bool:
            on = bool((sampling_params.extra_args or {}).get("think_bias"))
            self.active[req_idx] = on
            return on

        def apply(self, logits, ctx):
            slots = ctx.expanded_idx_mapping.long()
            n_out = ctx.pos.long() + 1 - self.rs.prompt_len.gpu[slots].long()   # tokens generated so far
            on = self.active[slots] & self.nl[ctx.input_ids.long()] & (n_out >= BIAS["gate"])
            bonus = ((n_out - BIAS["gate"]).float() * (BIAS["slope"] / 1000)).clamp(0, BIAS["cap"])
            logits[:, END_ID] += bonus * on.float()
            return logits

    return ThinkBias


def load_sets(a):
    from loop_probe import PROMPTS
    rng = random.Random(1234)
    items = []
    for name in a.loop_prompts.split(","):
        items += [{"set": "loop", "id": f"{name}#{i}", "seed": 11000 + i, "prompt": PROMPTS[name]}
                  for i in range(a.loop_n)]
    he = [json.loads(l) for l in open(os.path.expanduser("~/.cache/evalplus/HumanEvalPlus-v0.1.10.jsonl"))]
    for p in rng.sample(he, a.he_n):
        items.append({"set": "humaneval", "id": p["task_id"], "seed": 12000,
                      "prompt": HE_PROMPT.format(p=p["prompt"].strip())})
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    for k in rng.sample(range(len(ds)), a.mmlu_n):
        r = ds[k]
        opts = "\n".join(f"{LETTERS[j]}. {o}" for j, o in enumerate(r["options"]))
        items.append({"set": "mmlu_pro", "id": str(r["question_id"]), "seed": 13000, "gold": r["answer"],
                      "category": r["category"], "prompt": MMLU_PROMPT.format(q=r["question"], opts=opts)})
    return items


def generate(a, items, conds):
    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from shapequant.vllm_ext import core  # noqa: F401
    from loopmech_steer import gap_for

    from transformers import AutoTokenizer
    _t = AutoTokenizer.from_pretrained(a.model)
    BIAS["nl"] = {i for i in range(len(_t)) if _t.decode([i]).endswith("\n")}
    BIAS["gate"] = a.gate
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_think + a.max_answer + 1024,
              gpu_memory_utilization=0.88, max_num_seqs=48, enable_prefix_caching=True, seed=0,
              enforce_eager=True, logits_processors=[make_bias_proc()])
    tok = llm.get_tokenizer()
    assert tok.convert_tokens_to_ids("</think>") == END_ID
    model = llm.apply_model(lambda m: m)[0]
    layers = next(m for n, m in model.named_modules() if n.endswith("layers") and isinstance(m, torch.nn.ModuleList))
    dev = next(model.parameters()).device
    d = torch.load(a.dirs)["dirs"][a.layer].float()
    gap, _ = gap_for(a.layer, a.replay, d)
    st = {"vec": None, "gate_abs": None}

    def hook(mod, args, kwargs, out):
        if st["vec"] is None:
            return out
        hs, res = out
        pos = kwargs["positions"]
        if pos.ndim == 2:
            pos = pos[0]
        m = (pos >= st["gate_abs"]).to(hs.dtype).unsqueeze(-1)
        return hs + m * st["vec"], res

    layers[a.layer].register_forward_hook(hook, with_kwargs=True)
    pids = [tok.encode(tok.apply_chat_template([{"role": "user", "content": it["prompt"]}],
                                               add_generation_prompt=True, tokenize=False),
                       add_special_tokens=False) for it in items]
    assert tok.decode(pids[0][-3:]).endswith("<think>\n"), tok.decode(pids[0][-3:])
    os.makedirs(a.out, exist_ok=True)
    for cond in conds:
        path = os.path.join(a.out, f"{cond}.jsonl")
        if os.path.exists(path):
            print(f"{cond}: exists, skipped", flush=True)
            continue
        # steering gate in absolute positions: prompts differ in length, use the longest prompt + gate
        st["gate_abs"] = max(len(p) for p in pids) + a.gate
        st["vec"] = (a.alpha * gap * d).to(dev, torch.bfloat16) if cond == "steer" else None
        BIAS["gate"] = a.gate
        sp1 = [SamplingParams(temperature=1.0, top_p=0.95, top_k=20, max_tokens=a.max_think, seed=it["seed"],
                              stop_token_ids=[END_ID], skip_special_tokens=False,
                              extra_args={"think_bias": cond == "bias"}) for it in items]
        o1 = llm.generate([TokensPrompt(prompt_token_ids=p) for p in pids], sp1, use_tqdm=False)
        st["vec"] = None
        think = []
        for o in o1:
            ids = list(o.outputs[0].token_ids)
            closed = bool(ids) and ids[-1] == END_ID
            think.append((ids[:-1] if closed else ids, closed))
        idx2 = [k for k, (_, c) in enumerate(think) if c]
        sp2 = [SamplingParams(temperature=1.0, top_p=0.95, top_k=20, max_tokens=a.max_answer,
                              seed=items[k]["seed"], skip_special_tokens=True) for k in idx2]
        o2 = llm.generate([TokensPrompt(prompt_token_ids=pids[k] + think[k][0] + [END_ID]) for k in idx2], sp2,
                          use_tqdm=False)
        ans = {k: o.outputs[0] for k, o in zip(idx2, o2)}
        with open(path, "w") as f:
            for k, it in enumerate(items):
                ids, closed = think[k]
                a2 = ans.get(k)
                f.write(json.dumps({**{x: it[x] for x in it if x != "prompt"}, "cond": cond,
                                    "think_tokens": len(ids), "runaway": not closed,
                                    "think_tail": tok.decode(ids[-200:]),
                                    "answer": a2.text if a2 else "",
                                    "answer_finish": a2.finish_reason if a2 else None}) + "\n")
        summarize(a.out, [cond])


def mmlu_pred(text):
    m = re.findall(r"answer is \(?\s*([A-J])\s*\)?", text)
    if not m:
        m = re.findall(r"\*\*([A-J])[\.\)]", text) or re.findall(r"\b([A-J])\)", text[-200:])
    return m[-1] if m else None


def summarize(out, conds):
    he_res = {}
    for cond in conds:
        rp = os.path.join(out, f"{cond}_he_eval.json")
        if os.path.exists(rp):
            he_res[cond] = json.load(open(rp))
    print(f"{'cond':6s} {'set':10s} {'n':>4s} {'runaway':>8s} {'med think':>9s} {'p90 think':>9s} {'score':>12s}")
    for cond in conds:
        path = os.path.join(out, f"{cond}.jsonl")
        if not os.path.exists(path):
            continue
        rs = [json.loads(l) for l in open(path)]
        for s in ("loop", "humaneval", "mmlu_pro"):
            v = [r for r in rs if r["set"] == s]
            if not v:
                continue
            tt = sorted(r["think_tokens"] for r in v)
            score = ""
            if s == "mmlu_pro":
                ok = sum(mmlu_pred(r["answer"]) == r["gold"] for r in v)
                score = f"{ok}/{len(v)}"
            elif s == "humaneval" and cond in he_res:
                ok = sum(he_res[cond].get(r["id"], (0, 0))[1] for r in v)
                okb = sum(he_res[cond].get(r["id"], (0, 0))[0] for r in v)
                score = f"{okb}/{ok}/{len(v)}"
            print(f"{cond:6s} {s:10s} {len(v):4d} {sum(r['runaway'] for r in v):8d} {tt[len(tt) // 2]:9d} "
                  f"{tt[int(len(tt) * 0.9)]:9d} {score:>12s}", flush=True)
        if any(r["set"] == "loop" for r in rs):
            by = {}
            for r in rs:
                if r["set"] == "loop":
                    p = r["id"].split("#")[0]
                    by.setdefault(p, []).append(r["runaway"])
            print("        loop by prompt: " + ", ".join(f"{p} {sum(v)}/{len(v)}" for p, v in by.items()))


def score_humaneval(a, conds):
    """Run in ~/eval-venv: writes evalplus samples (all 164 tasks, empty for unselected) and scores them."""
    from code_eval import extract
    he = [json.loads(l)["task_id"] for l in open(os.path.expanduser("~/.cache/evalplus/HumanEvalPlus-v0.1.10.jsonl"))]
    for cond in conds:
        path = os.path.join(a.out, f"{cond}.jsonl")
        if not os.path.exists(path):
            continue
        rs = {r["id"]: r for r in map(json.loads, open(path)) if r["set"] == "humaneval"}
        sp = os.path.join(a.out, f"{cond}_he_samples.jsonl")
        with open(sp, "w") as f:
            for t in he:
                sol = extract(rs[t]["answer"]) if t in rs and rs[t]["answer"] else "pass"
                f.write(json.dumps({"task_id": t, "solution": sol}) + "\n")
        resp = sp.replace(".jsonl", "_eval_results.json")
        if os.path.exists(resp):
            os.remove(resp)
        subprocess.run([sys.executable, "-m", "evalplus.evaluate", "--dataset", "humaneval", "--samples", sp], check=True, stdout=subprocess.DEVNULL)
        ev = json.load(open(resp))["eval"]
        res = {}
        for t in rs:
            e = ev[t][0]
            res[t] = (int(e["base_status"] == "pass"), int(e["base_status"] == "pass" and e["plus_status"] == "pass"))
        json.dump(res, open(os.path.join(a.out, f"{cond}_he_eval.json"), "w"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/models/q27-sq"))
    ap.add_argument("--dirs", default="artifacts/loopmech/analysis_dirs.pt")
    ap.add_argument("--replay", default=os.path.expanduser("~/loopmech/replay"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--conds", default="base,steer,bias")
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--gate", type=int, default=3000)
    ap.add_argument("--max-think", type=int, default=12288)
    ap.add_argument("--max-answer", type=int, default=4096)
    ap.add_argument("--loop-prompts", default="csv_merge,bit_reverse,regex_log,hexdump_parse,hex16to32")
    ap.add_argument("--loop-n", type=int, default=6)
    ap.add_argument("--he-n", type=int, default=40)
    ap.add_argument("--mmlu-n", type=int, default=40)
    ap.add_argument("--score", action="store_true", help="score HumanEval (run with ~/eval-venv python)")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    conds = a.conds.split(",")
    if a.score:
        score_humaneval(a, conds)
        summarize(a.out, conds)
        return 0
    if a.summary:
        summarize(a.out, conds)
        return 0
    generate(a, load_sets(a), conds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
