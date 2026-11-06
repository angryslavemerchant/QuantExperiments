"""Real-weight runtime check: the decode engine built from a packed file vs the HF model
with the same weights (packed matrices decoded into it, the engine's lm_head weight),
on a few chat prompts. Per prompt: greedy-decode N tokens with the engine (CUDA graph),
teacher-force prompt + engine tokens through HF, compare logits at every generated
position; report relative logit error, greedy agreement and engine tokens/s.

    python scripts/check_runtime.py --packed artifacts/.../packed/run_ft.safetensors \
        --out artifacts/runtime/check.json [--head q4] [--ref-device cpu]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shapequant import packed, runtime  # noqa: E402

PROMPTS = [
    "Explain why the sky is blue in two sentences.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "What is 17 * 23? Show your work.",
    "Translate to French: The weather is lovely today and I want to go for a walk.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--head", default="q8", choices=("q4", "q8", "fp16"))
    ap.add_argument("--new-tokens", type=int, default=48)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--ref-device", default="cpu", help="where the fp32 HF reference runs")
    ap.add_argument("--tg", type=int, default=128, help="tokens for the tok/s timing (0: skip)")
    ap.add_argument("--pp", type=int, default=512, help="prompt tokens for the prefill timing (0: no prefill)")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    pf = packed.PackedFile(args.packed)
    base = pf.summary["base_model"]
    tok = AutoTokenizer.from_pretrained(base)
    t0 = time.time()
    if pf.aux is not None:          # self-contained file: the deployed path, nothing but the file
        args.head = "q8"
        eng, info = runtime.engine_from_packed(args.packed, max_len=args.max_len, prefill=args.pp > 0)
        load_s = time.time() - t0
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
        eng, info = runtime.engine_from_packed(args.packed, model=model, head=args.head, max_len=args.max_len,
                                               prefill=args.pp > 0)
        load_s = time.time() - t0
    torch.cuda.synchronize()
    eng_mem = torch.cuda.max_memory_allocated() / 1e9

    # reference: same weights, fp32, teacher-forced
    ref = model.float().to(args.ref_device).eval()
    packed.apply_to_model(ref, args.packed, device=args.ref_device)
    with torch.no_grad():
        # untie first (small Qwens share embed and lm_head): the engine's head is its own copy
        hw = runtime.head_reference_weight(ref, args.head).half().float().to(args.ref_device)
        ref.lm_head.weight = torch.nn.Parameter(hw, requires_grad=False)
        # the engine keeps every other tensor in fp16
        for blk in ref.model.layers:
            for mod in (blk.input_layernorm, blk.post_attention_layernorm):
                mod.weight.copy_(mod.weight.half().float())
            for lin in (blk.self_attn.q_proj, blk.self_attn.k_proj, blk.self_attn.v_proj):
                lin.bias.copy_(lin.bias.half().float())
        ref.model.norm.weight.copy_(ref.model.norm.weight.half().float())
        if pf.aux is not None:      # the file's 4-bit embedding
            from shapequant.evaluate import block_quant
            ref.model.embed_tokens.weight.copy_(block_quant(ref.model.embed_tokens.weight, 4).half().float())
        else:
            ref.model.embed_tokens.weight.copy_(ref.model.embed_tokens.weight.half().float())

    rows = []
    for p in PROMPTS:
        text = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                       tokenize=False)
        ids = [int(t) for t in tok(text, add_special_tokens=False)["input_ids"]]
        gen = runtime.generate(eng, ids, args.new_tokens, graph=True)
        # engine logits along its own greedy path (eager, for the comparison)
        lg = runtime.feed(eng, ids + gen[:-1], keep_logits=True)[len(ids) - 1:]
        got = torch.stack(lg).cpu()
        with torch.no_grad():
            r = ref(torch.tensor([ids + gen[:-1]], device=args.ref_device)).logits[0, len(ids) - 1:].float().cpu()
        rel = float((got - r).norm() / r.norm())
        agree = float((r.argmax(-1) == torch.tensor(gen)).float().mean())
        top2 = r.topk(2, -1).values
        margin = (top2[:, 0] - top2[:, 1])
        dis = (r.argmax(-1) != torch.tensor(gen))
        rows.append({"prompt": p, "prompt_tokens": len(ids), "rel_logit_err": rel, "greedy_agree": agree,
                     "disagree_ref_margins": margin[dis].tolist(),
                     "text": tok.decode(gen, skip_special_tokens=True)})
        print(f"[{len(ids)} tok] rel err {rel:.2e}  greedy agree {agree:.3f}  | {rows[-1]['text'][:90]!r}")

    res = {"packed": str(args.packed), "summary": pf.summary, "head": args.head, "load_s": load_s,
           "self_contained": info.get("self_contained", False),
           "engine_weight_bytes": info["weight_bytes"], "embed_bytes": info["embed_bytes"],
           "engine_peak_gb": eng_mem, "prompts": rows,
           "max_rel_logit_err": max(r["rel_logit_err"] for r in rows),
           "min_greedy_agree": min(r["greedy_agree"] for r in rows)}
    if args.tg or args.pp:
        b = runtime.bench(eng, pp=min(args.pp, args.max_len), tg=args.tg)
        res.update(b)
        print(f"bench: {b}; weights {info['weight_bytes'] / 1e9:.3f} GB (+ embed {info['embed_bytes'] / 1e9:.3f} GB)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    ok = res["max_rel_logit_err"] < 2e-2
    print(f"max rel logit err {res['max_rel_logit_err']:.2e}, min greedy agree {res['min_greedy_agree']:.3f}: "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
