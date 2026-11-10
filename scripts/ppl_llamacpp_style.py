"""Wikitext-2 perplexity scored exactly the way llama.cpp's `llama-perplexity` scores it, so a vLLM model
(here the shapequant checkpoint) and a GGUF are compared on identical text and identical windows.

llama-perplexity: tokenize the whole file, cut into n_ctx-token chunks (the tail is dropped), and in
each chunk score only the second half (positions n_ctx/2 .. n_ctx-1, each predicted from all earlier
tokens in its chunk); PPL = exp(mean NLL). Qwen GGUFs add no BOS, so none is added here either.

  python scripts/ppl_llamacpp_style.py --write-text wiki.test.raw          # the file both sides use
  python scripts/ppl_llamacpp_style.py --model ~/models/q27-sq --text wiki.test.raw --ctx 2048 --out x.json
  llama-perplexity -m model.gguf -f wiki.test.raw -c 2048 -b 2048 -ngl 99 -fa on
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-text", help="write wikitext-2-raw-v1 test (llama.cpp's wiki.test.raw) here and exit")
    ap.add_argument("--model")
    ap.add_argument("--text")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--gpu-mem", type=float, default=0.80)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.write_text:
        from datasets import load_dataset
        rows = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
        with open(a.write_text, "w", encoding="utf-8", newline="\n") as f:
            f.write("".join(rows))
        print(f"wrote {a.write_text}: {sum(map(len, rows))} chars")
        return 0
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from shapequant.vllm_ext import core  # noqa: F401  (registers the shapequant model)
    text = open(a.text, encoding="utf-8").read()
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.ctx + 16, gpu_memory_utilization=a.gpu_mem,
              max_num_seqs=1, max_num_batched_tokens=a.ctx, enable_prefix_caching=False, seed=0)
    toks = llm.get_tokenizer().encode(text, add_special_tokens=False)
    n_chunk = len(toks) // a.ctx
    first = a.ctx // 2
    print(f"{len(toks)} tokens, {n_chunk} chunks of {a.ctx}, scoring positions {first + 1}..{a.ctx - 1}", flush=True)
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)
    nll, n = 0.0, 0
    t0 = time.time()
    per_chunk = []
    for c in range(n_chunk):
        ids = toks[c * a.ctx:(c + 1) * a.ctx]
        out = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)[0]
        lp = out.prompt_logprobs            # [None, {tok: Logprob}, ...] aligned with ids
        s = 0.0
        for j in range(first + 1, a.ctx):
            s -= lp[j][ids[j]].logprob
        k = a.ctx - first - 1
        nll += s
        n += k
        per_chunk.append(s / k)
        if (c + 1) % 20 == 0 or c + 1 == n_chunk:
            print(f"[{c + 1}/{n_chunk}] PPL {math.exp(nll / n):.4f}  ({time.time() - t0:.0f}s)", flush=True)
    res = {"model": a.model, "ctx": a.ctx, "tokens": len(toks), "chunks": n_chunk, "scored": n,
           "nll": nll / n, "ppl": math.exp(nll / n), "per_chunk_nll": per_chunk}
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    print(f"PPL {res['ppl']:.4f} (NLL {res['nll']:.4f}) over {n} tokens", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
