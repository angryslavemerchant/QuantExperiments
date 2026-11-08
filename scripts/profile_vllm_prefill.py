"""Where prefill time goes in vLLM (one long prompt, max_tokens=1), kernels grouped by component.

In-process vLLM engine (VLLM_ENABLE_V1_MULTIPROCESSING=0) so torch.profiler sees the GPU kernels;
for each prompt length, one untimed warmup prefill of the same length, then a profiled one.
Prompt = wikitext-2 test text cut to the length (real tokens, as long_ctx.py).

  python scripts/profile_vllm_prefill.py --model ~/models/q27-sq --lengths 8192,65536 --out x.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

# (component, substrings of the kernel name, lowercase), first match wins
RULES = [
    ("sq dequant", ("dequant", "tile_rows", "q8_tile", "slab")),
    ("sq batched/gemv", ("runs_b", "gemv", "q8_head", "embed4", "q4_embed")),
    ("attention", ("flash", "fmha", "attn", "paged", "unified_attention", "prefill_attention", "cascade")),
    ("deltanet", ("chunk_", "gated_delta", "delta_rule", "fused_recurrent", "solve_tril", "wy_", "cumsum",
                  "fla", "l2norm", "causal_conv", "conv1d", "gdn")),
    ("gemm (cuBLAS/cutlass)", ("gemm", "cutlass", "sm80_xmma", "ampere_", "cublas", "s16816", "h16816", "matmul")),
    ("norm/act/rope", ("rms", "norm", "silu", "gelu", "act_and_mul", "rotary", "rope", "sigmoid", "softplus")),
    ("copy/cast/elementwise", ("copy", "cast", "elementwise", "fill", "cat", "index", "reduce", "vectorized", "unrolled")),
]


def group(name: str) -> str:
    n = name.lower()
    for comp, keys in RULES:
        if any(k in n for k in keys):
            return comp
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", default="8192,65536")
    ap.add_argument("--max-model-len", type=int, default=131072)
    ap.add_argument("--gpu-mem", type=float, default=0.93)
    ap.add_argument("--chunk", type=int, default=4096, help="--max-num-batched-tokens (prefill chunk)")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem,
              max_num_seqs=1, max_num_batched_tokens=a.chunk, enable_prefix_caching=False)
    tok = llm.get_tokenizer()
    text = "\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train[:60000]")["text"] if t.strip())
    ids = tok(text)["input_ids"]
    sp = SamplingParams(max_tokens=1, temperature=0)
    res = {"model": a.model, "chunk": a.chunk, "runs": []}
    for L in map(int, a.lengths.split(",")):
        prompt = {"prompt_token_ids": ids[:L]}
        llm.generate([{"prompt_token_ids": ids[L:2 * L]}], sp, use_tqdm=False)       # warmup, same length
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            llm.generate([prompt], sp, use_tqdm=False)
            torch.cuda.synchronize()
        wall = time.time() - t0
        comp, kern, cnt = defaultdict(float), defaultdict(float), defaultdict(int)
        for e in prof.key_averages():
            us = getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)
            if us <= 0 or e.key.startswith("ProfilerStep"):
                continue
            comp[group(e.key)] += us
            kern[e.key] += us
            cnt[e.key] += e.count
        gpu = sum(comp.values())
        print(f"\n=== {L} tokens: wall {wall:.1f}s ({L / wall:.0f} tok/s), GPU kernel time {gpu / 1e6:.1f}s")
        for c, us in sorted(comp.items(), key=lambda x: -x[1]):
            print(f"  {c:24s} {us / 1e6:7.2f}s  {100 * us / gpu:5.1f}%")
        print("  top kernels:")
        for k, us in sorted(kern.items(), key=lambda x: -x[1])[:a.top]:
            print(f"    {us / 1e6:7.2f}s  x{cnt[k]:<6d} [{group(k)}] {k[:100]}")
        res["runs"].append({"tokens": L, "wall_s": wall, "gpu_s": gpu / 1e6,
                            "components_s": {c: us / 1e6 for c, us in comp.items()},
                            "top_kernels_s": {k: us / 1e6 for k, us in sorted(kern.items(), key=lambda x: -x[1])[:40]}})
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
