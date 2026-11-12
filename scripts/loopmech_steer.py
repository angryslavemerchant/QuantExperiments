"""Loop mechanism study, step 2b: causal test of the "ready to close" direction (scripts/loopmech_analyze.py).

Adds alpha * gap * d to the residual stream after one layer, only at positions >= --gate (≈ thinking tokens, the
prompts are ~100 tokens), where d is the unit close-vs-other direction and gap is the mean projection difference
between closing points and other points of finishing traces. Generates in-process (vLLM, eager, hooks) on
loop-prone prompts and GSM8K, and records runaway rate, thinking length and the answer.

  python scripts/loopmech_steer.py --layer 31 --alphas 0,0.5,1 --out artifacts/loopmech/steer_L31.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
sys.path.insert(0, os.path.dirname(__file__))
from loop_probe import PROMPTS  # noqa: E402


def gap_for(layer, replay, d):
    import torch
    s_close, s_other = [], []
    for f in sorted(glob.glob(os.path.join(replay, "*.pt"))):
        x = torch.load(f)
        if x["meta"]["runaway"]:
            continue
        idx = x["resid_idx"]
        close = (x["next_id"][idx] == 248069)
        s = x["resid"][layer].float() @ d
        s_close += s[close].tolist()
        s_other += s[~close].tolist()
    import statistics as st
    return st.mean(s_close) - st.mean(s_other), st.pstdev(s_other)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/models/q27-sq"))
    ap.add_argument("--dirs", default="artifacts/loopmech/analysis_dirs.pt")
    ap.add_argument("--replay", default=os.path.expanduser("~/loopmech/replay"))
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--alphas", default="0,0.5,1")
    ap.add_argument("--gate", type=int, default=3000)
    ap.add_argument("--plan", default="csv_merge:6,bit_reverse:6,regex_log:6,gsm8k:16")
    ap.add_argument("--max-tokens", type=int, default=12288)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from vllm import LLM, SamplingParams
    from shapequant.vllm_ext import core  # noqa: F401

    d = torch.load(a.dirs)["dirs"][a.layer].float()
    gap, sd = gap_for(a.layer, a.replay, d)
    print(f"L{a.layer}: close-vs-other projection gap {gap:.2f} (sd of other points {sd:.2f})", flush=True)

    jobs = []
    for item in a.plan.split(","):
        name, n = item.split(":")
        if name == "gsm8k":
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from datasets import load_dataset
            ds = load_dataset("openai/gsm8k", "main", split="test")
            jobs += [("gsm8k", ds[100 + i]["question"], i, ds[100 + i]["answer"].split("####")[-1].strip())
                     for i in range(int(n))]
        else:
            jobs += [(name, PROMPTS[name], i, None) for i in range(int(n))]

    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_tokens + 512, gpu_memory_utilization=0.88,
              max_num_seqs=32, enable_prefix_caching=True, seed=0, enforce_eager=True)
    tok = llm.get_tokenizer()
    model = llm.apply_model(lambda m: m)[0]
    layers = next(m for n, m in model.named_modules() if n.endswith("layers") and isinstance(m, torch.nn.ModuleList))
    dev = next(model.parameters()).device
    st = {"vec": None}

    def hook(mod, args, kwargs, out):
        if st["vec"] is None:
            return out
        hs, res = out
        pos = kwargs["positions"]
        if pos.ndim == 2:
            pos = pos[0]
        m = (pos >= a.gate).to(hs.dtype).unsqueeze(-1)
        return hs + m * st["vec"], res

    layers[a.layer].register_forward_hook(hook, with_kwargs=True)

    done = set()
    if os.path.exists(a.out):
        done = {(r["alpha"], r["prompt"], r["i"]) for r in map(json.loads, open(a.out))}
    msgs = [[{"role": "user", "content": text}] for (_, text, _, _) in jobs]
    for alpha in [float(x) for x in a.alphas.split(",")]:
        todo = [k for k, j in enumerate(jobs) if (alpha, j[0], j[2]) not in done]
        if not todo:
            continue
        st["vec"] = None if alpha == 0 else (alpha * gap * d).to(dev, torch.bfloat16)
        sps = [SamplingParams(temperature=1.0, top_p=0.95, top_k=20, max_tokens=a.max_tokens, seed=9000 + jobs[k][2],
                              skip_special_tokens=False)
               for k in todo]
        outs = llm.chat([msgs[k] for k in todo], sps, use_tqdm=False)
        with open(a.out, "a") as f:
            for k, o in zip(todo, outs):
                name, _, i, gold = jobs[k]
                c = o.outputs[0]
                text = c.text
                think, _, ans = text.partition("</think>")
                rec = {"layer": a.layer, "alpha": alpha, "gate": a.gate, "prompt": name, "i": i,
                       "finish": c.finish_reason, "n_gen": len(c.token_ids), "runaway": c.finish_reason == "length",
                       "closed_think": "</think>" in text, "think_tokens": len(tok.encode(think)),
                       "answer": ans.strip()[-3000:], "gold": gold}
                f.write(json.dumps(rec) + "\n")
        rs = [json.loads(l) for l in open(a.out)]
        rs = [r for r in rs if r["alpha"] == alpha]
        for p in sorted({r["prompt"] for r in rs}):
            v = [r for r in rs if r["prompt"] == p]
            print(f"alpha {alpha}: {p:12s} runaway {sum(r['runaway'] for r in v)}/{len(v)}  "
                  f"median think {sorted(r['think_tokens'] for r in v)[len(v) // 2]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
