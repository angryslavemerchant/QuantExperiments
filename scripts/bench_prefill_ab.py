"""End-to-end vLLM prefill time, bf16 vs INT8 prefill GEMMs (vllm_ext core._INT8_PREFILL), toggled
inside one in-process engine and alternated, so the GPU's run-to-run speed drift hits both sides.

  python scripts/bench_prefill_ab.py --model ~/models/q27-sq --lengths 8192,32768 --chunk 4096 --rounds 3
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", default="8192,32768")
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-model-len", type=int, default=65536)
    ap.add_argument("--gpu-mem", type=float, default=0.88)
    ap.add_argument("--ab", choices=["int8", "fused"], default="int8",
                    help="int8: bf16 vs INT8 GEMMs; fused: INT8 with the old (torch.mm + vLLM quant) vs fused prep")
    ap.add_argument("--out")
    a = ap.parse_args()
    import torch
    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    from shapequant.vllm_ext import core
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem,
              max_num_seqs=1, max_num_batched_tokens=a.chunk, enable_prefix_caching=False)
    tok = llm.get_tokenizer()
    text = "\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train[:60000]")["text"] if t.strip())
    ids = tok(text)["input_ids"]
    sp = SamplingParams(max_tokens=1, temperature=0)
    res = {"chunk": a.chunk, "ab": a.ab, "runs": {}}
    names = ("bf16", "int8") if a.ab == "int8" else ("old", "fused")

    def set_mode(mode):
        if a.ab == "int8":
            core._INT8_PREFILL = mode == "int8"
        else:
            core._INT8_PREFILL, core._INT8_FUSED = True, mode == "fused"

    for L in map(int, a.lengths.split(",")):
        t = {n: [] for n in names}
        for mode in names:                                 # warm both paths at this length
            set_mode(mode)
            llm.generate([{"prompt_token_ids": ids[L:2 * L]}], sp, use_tqdm=False)
        for r in range(a.rounds):
            for mode in (names if r % 2 == 0 else names[::-1]):
                set_mode(mode)
                torch.cuda.synchronize()
                t0 = time.time()
                llm.generate([{"prompt_token_ids": ids[:L]}], sp, use_tqdm=False)
                torch.cuda.synchronize()
                t[mode].append(time.time() - t0)
        n0, n1 = names
        b, i = statistics.median(t[n0]), statistics.median(t[n1])
        print(f"{L:6d} tokens, chunk {a.chunk}: {n0} {L / b:6.0f} tok/s ({b:.1f}s)  {n1} {L / i:6.0f} tok/s ({i:.1f}s)  "
              f"-> {b / i:.2f}x   [{n0} {', '.join(f'{v:.1f}' for v in t[n0])} | {n1} {', '.join(f'{v:.1f}' for v in t[n1])}]",
              flush=True)
        res["runs"][L] = t
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
