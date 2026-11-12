"""Loop mechanism study, step 1b: replay generated thinking traces through the shapequant model (in-process vLLM,
prefill only) and record, at every line break inside the thinking ("decision points" - where the model can close
its thinking or start another paragraph):
  - next-token probabilities of </think> and of backtrack openers (But / Wait / Actually / Hmm / ...),
  - the residual stream at a set of layers.

  python scripts/loopmech_replay.py --traces artifacts/loopmech/traces.jsonl --out ~/loopmech/replay
Writes one .pt per trace: {"meta", "pos", "next_id", "p_end", "p_back", "top", "resid": {layer: [n, d] fp16}}.
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

BACKTRACK = ["But", "Wait", "Actually", "Hmm", "Alternatively", "However", "Hold", "Let", "Hmm,", "Oh", "Also",
             "Another", "Maybe", "Or", "Double", "Now", "Hm"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/models/q27-sq"))
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="every4", help="'every4' or comma list")
    ap.add_argument("--max-points", type=int, default=160, help="decision points kept per trace (resid only)")
    ap.add_argument("--gpu-mem", type=float, default=0.80)
    a = ap.parse_args()
    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from shapequant.vllm_ext import core  # noqa: F401

    traces = [json.loads(l) for l in open(a.traces)]
    max_len = max(len(t["prompt_token_ids"]) + len(t["token_ids"]) for t in traces) + 8
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=max_len, gpu_memory_utilization=a.gpu_mem,
              max_num_seqs=1, max_num_batched_tokens=4096, enable_prefix_caching=False, seed=0,
              enforce_eager=True)
    tok = llm.get_tokenizer()
    end_id = tok.convert_tokens_to_ids("</think>")
    back_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in BACKTRACK})
    model = llm.apply_model(lambda m: m)[0]
    layers = next(m for n, m in model.named_modules() if n.endswith("layers") and isinstance(m, torch.nn.ModuleList))
    L = len(layers)
    keep = list(range(3, L, 4)) if a.layers == "every4" else [int(x) for x in a.layers.split(",")]
    norm = next(m for n, m in model.named_modules() if n.endswith("model.norm") or n.endswith("language_model.model.norm"))
    print(f"{L} layers, recording {keep}; </think>={end_id}; backtrack ids {len(back_ids)}", flush=True)

    cap: dict = {"sel": None, "resid": {}, "final": None}

    def layer_hook(idx):
        def h(mod, inp, out):
            if cap["sel"] is None:
                return
            hs, res = (out if isinstance(out, tuple) else (out, None))
            x = hs if res is None else hs + res          # residual stream after this layer
            cap["resid"].setdefault(idx, []).append(x.detach().to(torch.float16).cpu())  # copy now: vLLM adds in place
        return h

    def norm_hook(mod, inp, out):
        if cap["sel"] is None:
            return
        x = out[0] if isinstance(out, tuple) else out
        cap["final"] = (cap["final"] or []) + [x.detach().clone()]

    for i in keep:
        layers[i].register_forward_hook(layer_hook(i))
    norm.register_forward_hook(norm_hook)
    sp = SamplingParams(max_tokens=1, temperature=0)
    os.makedirs(os.path.expanduser(a.out), exist_ok=True)
    for k, t in enumerate(traces):
        path = os.path.join(os.path.expanduser(a.out), f"{t['prompt']}_{t['i']:03d}.pt")
        if os.path.exists(path):
            continue
        ids = t["prompt_token_ids"] + t["token_ids"]
        n_prompt = len(t["prompt_token_ids"])
        gen = t["token_ids"]
        stop = gen.index(end_id) if end_id in gen else len(gen)     # thinking span = gen[:stop]
        # decision points: positions p (absolute) inside thinking whose token text ends with a newline;
        # the prediction made there is for token p+1 (a new paragraph or </think>)
        pts = []
        for j in range(stop):
            if "\n" in tok.decode([gen[j]])[-1:] and j + 1 <= stop:
                pts.append(n_prompt + j)
        if not pts:
            continue
        cap.update(sel=True, resid={}, final=None)
        llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)
        cap["sel"] = None
        pos = torch.tensor(pts)
        final = torch.cat(cap["final"], 0)
        with torch.no_grad():
            logits = model.compute_logits(final[pos.to(final.device)]).float()
        lp = torch.log_softmax(logits, -1)
        p_end = lp[:, end_id].exp().cpu()
        p_back = lp[:, back_ids].exp().sum(-1).cpu()
        top = lp.topk(5, -1)
        nxt = torch.tensor([ids[p + 1] if p + 1 < len(ids) else -1 for p in pts])
        # keep resid for a subset: all points near the end + the closing point, evenly spaced elsewhere
        idx = list(range(len(pts)))
        if len(idx) > a.max_points:
            tail = idx[-a.max_points // 4:]
            head = idx[:-a.max_points // 4]
            step = len(head) / (a.max_points - len(tail))
            idx = sorted({head[int(s * step)] for s in range(a.max_points - len(tail))} | set(tail))
        sel = pos[idx]
        resid = {}
        for li, chunks in cap["resid"].items():
            x = torch.cat(chunks, 0)
            assert x.shape[0] == len(ids), (li, x.shape, len(ids))
            resid[li] = x[sel].clone()
        torch.save({"meta": {k2: t[k2] for k2 in ("prompt", "i", "finish", "n_gen", "runaway")} |
                    {"n_prompt": n_prompt, "think_len": stop},
                    "pos": pos, "next_id": nxt, "p_end": p_end, "p_back": p_back,
                    "top_ids": top.indices.cpu(), "top_lp": top.values.cpu(), "resid_idx": torch.tensor(idx),
                    "resid": resid}, path)
        print(f"[{k + 1}/{len(traces)}] {t['prompt']}#{t['i']} {'RUNAWAY' if t['runaway'] else 'done'} "
              f"think {stop} tok, {len(pts)} points, mean p_end {p_end.mean():.2e}", flush=True)
        del cap["resid"], cap["final"], final, logits, lp
        cap.update(resid={}, final=None)
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
