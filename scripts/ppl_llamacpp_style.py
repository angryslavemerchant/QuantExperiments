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


def score_text(llm, path, ctx, max_chunks, out_path, model):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    text = open(path, encoding="utf-8").read()
    toks = llm.get_tokenizer().encode(text, add_special_tokens=False)
    n_chunk = len(toks) // ctx
    if max_chunks:
        n_chunk = min(n_chunk, max_chunks)
    first = ctx // 2
    print(f"{path}: {len(toks)} tokens, {n_chunk} chunks of {ctx}, scoring positions {first + 1}..{ctx - 1}", flush=True)
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)
    nll, n = 0.0, 0
    t0 = time.time()
    per_chunk = []
    for c in range(n_chunk):
        ids = toks[c * ctx:(c + 1) * ctx]
        o = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)[0]
        lp = o.prompt_logprobs            # [None, {tok: Logprob}, ...] aligned with ids
        s = 0.0
        for j in range(first + 1, ctx):
            s -= lp[j][ids[j]].logprob
        k = ctx - first - 1
        nll += s
        n += k
        per_chunk.append(s / k)
    res = {"model": model, "text": path, "ctx": ctx, "tokens": len(toks), "chunks": n_chunk, "scored": n,
           "nll": nll / n, "ppl": math.exp(nll / n), "per_chunk_nll": per_chunk}
    if out_path:
        with open(out_path, "w") as f:
            json.dump(res, f, indent=1)
    print(f"{path}: PPL {res['ppl']:.4f} (NLL {res['nll']:.4f}) over {n} tokens, {time.time() - t0:.0f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-text", help="write wikitext-2-raw-v1 test (llama.cpp's wiki.test.raw) here and exit")
    ap.add_argument("--model")
    ap.add_argument("--text", help="text file; several comma-separated with --out-dir (one model load)")
    ap.add_argument("--out-dir", help="with several --text files: writes <out-dir>/<text stem>.json each")
    ap.add_argument("--max-chunks", type=int, default=0, help="score at most this many chunks per text")
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
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.ctx + 16, gpu_memory_utilization=a.gpu_mem,
              max_num_seqs=1, max_num_batched_tokens=a.ctx, enable_prefix_caching=False, seed=0)
    texts = a.text.split(",")
    for tf in texts:
        out = a.out
        if a.out_dir:
            os.makedirs(a.out_dir, exist_ok=True)
            out = os.path.join(a.out_dir, os.path.splitext(os.path.basename(tf))[0] + ".json")
        score_text(llm, tf, a.ctx, a.max_chunks, out, a.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
