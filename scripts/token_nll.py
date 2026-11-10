"""Per-token NLL of a text under the shapequant model (in-process vLLM, prompt_logprobs), same windows as
scripts/ppl_llamacpp_style.py, saved for analysis (where in a document the loss sits).

  python scripts/token_nll.py --model ~/models/q27-sq --text artifacts/overnight/domains/chat.txt \
      --max-chunks 8 --out artifacts/overnight/tok_chat.json
Env SHAPEQUANT_BF16_LAYERS / --model with a swap checkpoint: same for layer-swap variants.
"""

from __future__ import annotations

import argparse
import json
import math
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True, help="one or more files, comma-separated")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--max-chunks", type=int, default=8)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--full", action="store_true", help="score every position (default: second half)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from shapequant.vllm_ext import core  # noqa: F401
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.ctx + 16, gpu_memory_utilization=a.gpu_mem,
              max_num_seqs=1, max_num_batched_tokens=a.ctx, enable_prefix_caching=False, seed=0,
              enforce_eager=True)
    tok = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0)   # target + top-1
    res = {"model": a.model, "env_bf16_layers": os.environ.get("SHAPEQUANT_BF16_LAYERS", ""), "texts": {}}
    for path in a.text.split(","):
        ids_all = tok.encode(open(path, encoding="utf-8").read(), add_special_tokens=False)
        n = min(len(ids_all) // a.ctx, a.max_chunks)
        first = 0 if a.full else a.ctx // 2
        wins = []
        for c in range(n):
            ids = ids_all[c * a.ctx:(c + 1) * a.ctx]
            lp = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)[0].prompt_logprobs
            rng = range(max(1, first + 1), a.ctx)
            nll = [round(-lp[j][ids[j]].logprob, 4) for j in rng]
            top = [max(lp[j].items(), key=lambda kv: kv[1].logprob) for j in rng]
            wins.append({"ids": ids, "first": max(1, first + 1), "nll": nll,
                         "top1": [t for t, _ in top], "top1_lp": [round(v.logprob, 4) for _, v in top]})
        tot = [x for w in wins for x in w["nll"]]
        m = sum(tot) / len(tot)
        res["texts"][path] = {"nll": m, "ppl": math.exp(m), "windows": wins}
        print(f"{path}: PPL {math.exp(m):.4f} over {len(tot)} tokens", flush=True)
    json.dump(res, open(a.out, "w"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
