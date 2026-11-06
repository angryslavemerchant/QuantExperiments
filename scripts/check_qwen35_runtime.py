"""Real-weight check of the Qwen3.5 / 3.8 runtime (shapequant.qwen35_runtime): the engine built
from a self-contained packed file vs the HF model with the same weights (packed matrices decoded
into it, the file's 8-bit head and 4-bit embedding), on a few chat prompts. Per prompt: batched
prefill, greedy-decode N tokens (CUDA graph), teacher-force prompt + engine tokens through HF,
compare logits at every generated position. Then decode / prefill speed.

    python scripts/check_qwen35_runtime.py --packed artifacts/.../packed/run_ft.safetensors \\
        --out artifacts/.../runtime_check.json [--ref-device cuda]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shapequant import packed, qwen35_runtime as q35  # noqa: E402
from shapequant.evaluate import block_quant  # noqa: E402

PROMPTS = [
    "Explain why the sky is blue in two sentences.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "What is 17 * 23? Show your work.",
    "Translate to French: The weather is lovely today and I want to go for a walk.",
]


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--new-tokens", type=int, default=48)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--ref-device", default="cpu", help="where the fp32 HF reference runs")
    ap.add_argument("--tg", type=int, default=128)
    ap.add_argument("--pp", type=int, default=512)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    pf = packed.PackedFile(args.packed)
    if pf.aux is None:
        raise SystemExit("needs a self-contained packed file (aux section)")
    base = pf.summary["base_model"]
    tok = AutoTokenizer.from_pretrained(base)
    t0 = time.time()
    eng, info = q35.engine_from_packed(args.packed, max_len=args.max_len)
    q35.attach_prefill(eng)
    load_s = time.time() - t0
    torch.cuda.synchronize()
    eng_mem = torch.cuda.max_memory_allocated() / 1e9

    ref = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).to(args.ref_device).eval()
    packed.apply_to_model(ref, args.packed, device=args.ref_device)
    e0 = ref.model.embed_tokens.weight.detach().clone()
    ref.lm_head.weight = torch.nn.Parameter(block_quant(ref.lm_head.weight.detach(), 8).half().float(),
                                            requires_grad=False)             # untie, the file's head
    ref.model.embed_tokens.weight.data = block_quant(e0, 4).half().float()   # the file's embedding
    del e0

    eng.capture()
    rows = []
    for p in PROMPTS:
        text = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
        ids = [int(t) for t in tok(text, add_special_tokens=False)["input_ids"]]
        eng.reset()
        lg = [q35.prefill(eng, ids)]
        gen = [int(eng.tok)]
        while len(gen) < args.new_tokens:
            eng.graph.replay()
            lg.append(eng.logits.float().clone())
            gen.append(int(eng.tok))
        got = torch.stack(lg).cpu()
        r = ref(torch.tensor([ids + gen[:-1]], device=args.ref_device)).logits[0, len(ids) - 1:].float().cpu()
        rel = float((got - r).norm() / r.norm())
        agree = float((r.argmax(-1) == torch.tensor(gen)).float().mean())
        rows.append({"prompt": p, "prompt_tokens": len(ids), "rel_logit_err": rel, "greedy_agree": agree,
                     "text": tok.decode(gen, skip_special_tokens=True)})
        print(f"[{len(ids)} tok] rel err {rel:.2e}  greedy agree {agree:.3f}  | {rows[-1]['text'][:90]!r}", flush=True)
    del ref

    res = {"packed": str(args.packed), "summary": pf.summary, "load_s": load_s, "engine_weight_bytes": info["weight_bytes"],
           "embed_bytes": eng.embed.nbytes(), "engine_peak_gb": eng_mem, "prompts": rows,
           "max_rel_logit_err": max(r["rel_logit_err"] for r in rows),
           "min_greedy_agree": min(r["greedy_agree"] for r in rows)}
    if args.tg:
        s = eng.time_tokens(args.tg, 3)
        res.update(tg=args.tg, tg_tok_s=1 / s)
    if args.pp:
        ids = torch.randint(0, eng.cfg.vocab, (min(args.pp, args.max_len),), device="cuda")
        ts = []
        for _ in range(4):
            eng.reset()
            torch.cuda.synchronize()
            t = time.time()
            q35.prefill(eng, ids)
            torch.cuda.synchronize()
            ts.append(time.time() - t)
        res.update(pp=ids.numel(), pp_tok_s=ids.numel() / sorted(ts[1:])[1])
    print(f"tg {res.get('tg_tok_s', 0):.1f} tok/s, pp {res.get('pp_tok_s', 0):.0f} tok/s; "
          f"weights {info['weight_bytes'] / 1e9:.3f} GB (+ embed {res['embed_bytes'] / 1e9:.3f} GB)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    ok = res["max_rel_logit_err"] < 2e-2
    print(f"max rel logit err {res['max_rel_logit_err']:.2e}, min greedy agree {res['min_greedy_agree']:.3f}: "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
