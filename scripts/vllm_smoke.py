"""Check and time a model under vLLM (one model per process; run inside the vLLM env).

  run:     python scripts/vllm_smoke.py run --model DIR --name ours --out artifacts/vllm/ours.json
  compare: python scripts/vllm_smoke.py compare artifacts/vllm/ours.json artifacts/vllm/ref.json

run records:
  - wikitext-2 test NLL over `--windows` windows of `--window` tokens (prompt logprobs of the
    real tokens): the shapequant plugin against the same weights dequantized to bf16 (the ref)
    should agree to kernel rounding;
  - greedy continuations of a few chat prompts (token ids);
  - throughput: `c` requests of `--prompt-len` random tokens -> `--gen` tokens each (ignore_eos),
    all submitted at once, for each c in --conc: aggregate output tok/s and wall time (c = 1 is
    single-stream decode).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

CHAT = ["Explain why the sky is blue in two sentences.",
        "Write a Python function that returns the n-th Fibonacci number iteratively.",
        "What is 17 * 23? Show your work briefly.",
        "List three differences between TCP and UDP.",
        "Translate 'The weather is nice today' into French and German.",
        "Summarize the plot of Romeo and Juliet in one paragraph.",
        "Give a short proof that the square root of 2 is irrational.",
        "What are the main causes of inflation?"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wikitext_windows(tok, n: int, L: int) -> list[list[int]]:
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]))["input_ids"]
    step = max(L, (len(ids) - L) // n)
    return [ids[i * step:i * step + L] for i in range(n) if i * step + L <= len(ids)]


def run(a) -> int:
    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rec = {"name": a.name, "model": a.model, "gpu": torch.cuda.get_device_name(0)}
    t0 = time.time()
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem,
              max_num_seqs=a.max_num_seqs, enable_prefix_caching=False, seed=0,
              enforce_eager=a.eager)
    rec["load_s"] = round(time.time() - t0, 1)
    rec["torch_mem_GB"] = round(torch.cuda.memory_allocated() / 1e9, 2) if torch.cuda.is_initialized() else None
    log(f"loaded in {rec['load_s']}s")
    tok = llm.get_tokenizer()

    # quality: wikitext NLL
    wins = wikitext_windows(tok, a.windows, a.window)
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
    t1 = time.time()
    res = llm.generate([TokensPrompt(prompt_token_ids=w) for w in wins], sp, use_tqdm=False)
    lps = []
    for w, r in zip(wins, res):
        row = []
        for j, d in enumerate(r.prompt_logprobs[1:], start=1):
            row.append(d[w[j]].logprob)
        lps.append(row)
    n = sum(len(r) for r in lps)
    rec["wikitext"] = {"windows": len(wins), "tokens": n, "nll": -sum(map(sum, lps)) / n,
                       "seconds": round(time.time() - t1, 1)}
    rec["wikitext"]["ppl"] = math.exp(rec["wikitext"]["nll"])
    rec["token_logprobs"] = lps
    log(f"wikitext nll {rec['wikitext']['nll']:.4f} ppl {rec['wikitext']['ppl']:.3f} ({n} tokens)")

    # greedy chat continuations
    msgs = [[{"role": "user", "content": c}] for c in CHAT]
    gp = SamplingParams(max_tokens=a.chat_tokens, temperature=0.0)
    res = llm.chat(msgs, gp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
    rec["chat"] = [{"prompt": c, "ids": list(r.outputs[0].token_ids), "text": r.outputs[0].text}
                   for c, r in zip(CHAT, res)]

    # throughput
    g = torch.Generator().manual_seed(0)
    V = min(tok.vocab_size, 150000)
    tp = SamplingParams(max_tokens=a.gen, temperature=0.0, ignore_eos=True)
    rec["throughput"] = []
    warm = [TokensPrompt(prompt_token_ids=torch.randint(1000, V, (a.prompt_len,), generator=g).tolist())]
    llm.generate(warm, SamplingParams(max_tokens=8, ignore_eos=True), use_tqdm=False)
    for c in a.conc:
        ps = [TokensPrompt(prompt_token_ids=torch.randint(1000, V, (a.prompt_len,), generator=g).tolist())
              for _ in range(c)]
        torch.cuda.synchronize()
        t1 = time.time()
        res = llm.generate(ps, tp, use_tqdm=False)
        dt = time.time() - t1
        ntok = sum(len(r.outputs[0].token_ids) for r in res)
        row = {"conc": c, "seconds": round(dt, 3), "out_tokens": ntok, "out_tok_s": round(ntok / dt, 1),
               "per_seq_tok_s": round(ntok / dt / c, 1)}
        rec["throughput"].append(row)
        log(f"conc {c}: {row['out_tok_s']} tok/s aggregate, {row['per_seq_tok_s']} per sequence ({dt:.1f}s)")
    out.write_text(json.dumps(rec))
    log(f"wrote {out}")
    return 0


def compare(a) -> int:
    x, y = json.loads(Path(a.a).read_text()), json.loads(Path(a.b).read_text())
    diffs = [abs(p - q) for rx, ry in zip(x["token_logprobs"], y["token_logprobs"]) for p, q in zip(rx, ry)]
    same = sum(cx["ids"] == cy["ids"] for cx, cy in zip(x["chat"], y["chat"]))
    prefix = []
    for cx, cy in zip(x["chat"], y["chat"]):
        k = 0
        while k < min(len(cx["ids"]), len(cy["ids"])) and cx["ids"][k] == cy["ids"][k]:
            k += 1
        prefix.append(k)
    rep = {"a": x["name"], "b": y["name"],
           "nll": [x["wikitext"]["nll"], y["wikitext"]["nll"]],
           "token_logprob_absdiff_mean": sum(diffs) / len(diffs), "token_logprob_absdiff_max": max(diffs),
           "chat_identical": f"{same}/{len(x['chat'])}", "chat_common_prefix_tokens": prefix,
           "throughput": {r["conc"]: [r["out_tok_s"], s["out_tok_s"]] for r, s in zip(x["throughput"], y["throughput"])}}
    print(json.dumps(rep, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--name", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--max-model-len", type=int, default=8192)
    r.add_argument("--gpu-mem", type=float, default=0.85)
    r.add_argument("--max-num-seqs", type=int, default=256)
    r.add_argument("--eager", action="store_true", help="no torch.compile / CUDA graphs")
    r.add_argument("--windows", type=int, default=16)
    r.add_argument("--window", type=int, default=1024)
    r.add_argument("--chat-tokens", type=int, default=128)
    r.add_argument("--prompt-len", type=int, default=256)
    r.add_argument("--gen", type=int, default=256)
    r.add_argument("--conc", type=lambda s: [int(v) for v in s.split(",")], default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--out")
    a = ap.parse_args()
    return run(a) if a.cmd == "run" else compare(a)


if __name__ == "__main__":
    raise SystemExit(main())
