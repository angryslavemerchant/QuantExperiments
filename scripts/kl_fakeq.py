"""KL of simulated INT8 prefill variants (SHAPEQUANT_FAKEQ, see vllm_ext/core.py) against the
unmodified model, over full-vocabulary next-token distributions.

In-process vLLM (VLLM_ENABLE_V1_MULTIPROCESSING=0); the model's compute_logits is wrapped to keep
the logits of every prompt position (prompt_logprobs forces them). Pass 1 (FAKEQ off) stores the
reference log-probs on the CPU in fp16; each variant then runs the same windows and is compared
on the GPU, row chunk by row chunk. The fake-quant switches are module globals read at call time,
so all variants run in one engine.

  python scripts/kl_fakeq.py --model ~/models/q27-sq --windows 8 --out artifacts/fakeq/kl.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
sys.path.insert(0, str(Path(__file__).resolve().parent))

# name -> (FAKEQ, HAD block, skip list)
VARIANTS = {
    "off": ("", 0, []),          # noise floor: the reference config run again
    "w8": ("w8", 0, []),
    "w8a8": ("w8a8", 0, []),
    "w8a8_had128": ("w8a8", 128, []),
    "w8a8_had1024": ("w8a8", 1024, []),
    "a8_had128": ("a8", 128, []),
    "int8": ("", 0, [], True),     # the real W8A8 kernel path (SHAPEQUANT_INT8_PREFILL)
}
ROWS = 256


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--gpu-mem", type=float, default=0.80)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from vllm import LLM, SamplingParams
    from vllm_smoke import wikitext_windows
    from shapequant.vllm_ext import core

    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.window + 16, gpu_memory_utilization=a.gpu_mem,
              max_num_seqs=1, max_num_batched_tokens=max(2048, a.window + 16), enable_prefix_caching=False, seed=0)
    model = llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
    wins = wikitext_windows(llm.get_tokenizer(), a.windows, a.window)
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)

    cap: list[torch.Tensor] = []
    orig = model.compute_logits

    def compute_logits(*args, **kw):
        out = orig(*args, **kw)
        if out is not None and out.shape[0] > 1:          # prefill rows only
            cap.append(out)
        return out

    model.compute_logits = compute_logits

    def logprob_rows(win):
        """[L, V] logits of one window -> list of (log-softmax fp32 row chunk) generators' input."""
        cap.clear()
        llm.generate([{"prompt_token_ids": win}], sp, use_tqdm=False)
        lg = torch.cat(cap)[: len(win) - 1]               # position i predicts token i + 1
        cap.clear()
        return lg

    def set_variant(fq, had, skip, int8=False):
        core._FAKEQ, core._FAKEQ_HAD, core._FAKEQ_SKIP = fq, had, skip
        core._INT8_PREFILL, core._INT8_MIN_M = int8, 256

    set_variant("", 0, [])
    ref, ref_nll = [], 0.0
    for win in wins:
        lg = logprob_rows(win)
        tgt = torch.tensor(win[1:], device=lg.device)
        rows = []
        for r0 in range(0, lg.shape[0], ROWS):
            lp = lg[r0:r0 + ROWS].float().log_softmax(-1)
            ref_nll -= lp.gather(1, tgt[r0:r0 + ROWS, None]).sum().item()
            rows.append(lp.half().cpu())
        ref.append(torch.cat(rows))
        del lg
    ntok = sum(r.shape[0] for r in ref)
    res = {"model": a.model, "tokens": ntok, "ref_nll": ref_nll / ntok, "variants": {}}
    print(f"reference: {ntok} tokens, NLL {ref_nll / ntok:.4f}", flush=True)

    for name in a.variants.split(","):
        set_variant(*VARIANTS[name])
        kl = nll = agree = 0.0
        kmax = 0.0
        for win, rp in zip(wins, ref):
            lg = logprob_rows(win)
            tgt = torch.tensor(win[1:], device=lg.device)
            for r0 in range(0, lg.shape[0], ROWS):
                lq = lg[r0:r0 + ROWS].float().log_softmax(-1)
                lp = rp[r0:r0 + ROWS].to(lg.device).float()
                k = (lp.exp() * (lp - lq)).sum(-1)       # KL(ref || variant) per position
                kl += k.sum().item()
                kmax = max(kmax, k.max().item())
                nll -= lq.gather(1, tgt[r0:r0 + ROWS, None]).sum().item()
                agree += (lq.argmax(-1) == lp.argmax(-1)).sum().item()
            del lg
        r = {"kl": kl / ntok, "kl_max": kmax, "nll": nll / ntok, "top1_agree": agree / ntok}
        res["variants"][name] = r
        print(f"{name:14s} KL {r['kl']:.5f} (max {kmax:.3f})  NLL {r['nll']:.4f}  top-1 agree {100 * r['top1_agree']:.2f}%",
              flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    set_variant("", 0, [])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
