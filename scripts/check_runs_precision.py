"""Numerical check of the lane-interleaved kernel (proto_runs.py) on real
Qwen2.5-7B-Instruct weights and activations.

The run kernel sums 16 products (4 chunks) in half2 before converting to fp32,
so large activation outliers could cost precision or overflow. Per matrix:
quantize W with a quick K=1024 k-means (M=16 column orders, real row scales),
then compare y for real captured inputs against the exact fp32 product of the
dequantized weights, for: the run kernel, the current kernel (cuda_gemv, fp32
accumulation per chunk), and torch fp16 mv of the dequantized weights (what an
ordinary fp16 inference would do).

Run: python scripts/check_runs_precision.py [--tokens 64] [--layers all]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shapequant import codebook as cb  # noqa: E402
from shapequant import masks as mk  # noqa: E402
from shapequant.calibrate import load_calibration_sequences  # noqa: E402
from shapequant.kernels import cuda_gemv, proto_runs  # noqa: E402
from shapequant.kernels.gemv import PackedLinear  # noqa: E402
from shapequant.masks import _prepare, _row_scale  # noqa: E402
from shapequant.packed import pack_bits  # noqa: E402

NAMES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--capture-device", default="cpu", help="device for the activation-capture forward pass")
    ap.add_argument("--out", default="artifacts/prescreen/runs_precision_7b.txt")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    nl = model.config.num_hidden_layers
    layers = list(range(nl)) if a.layers == "all" else [int(v) for v in a.layers.split(",")]
    try:
        seq = load_calibration_sequences(tok, {"dataset": "HuggingFaceFW/fineweb-edu", "subset": "sample-10BT",
                                               "split": "train", "num_sequences": 1, "context": a.tokens})[0]
    except Exception:                                                   # offline: cached wikitext test split
        from datasets import load_dataset
        txt = "\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"][:400])
        seq = tok(txt)["input_ids"][:a.tokens]
    caps: dict = {}
    hooks = []
    for li in layers:
        blk = model.model.layers[li]
        for n in NAMES:
            mod = blk.get_submodule(n)
            hooks.append(mod.register_forward_hook(
                lambda m, inp, out, key=(li, n): caps.__setitem__(key, inp[0][0].float().cpu().clone())))
    model.to(a.capture_device)
    with torch.no_grad():
        model(torch.as_tensor(seq)[None].to(a.capture_device))
    for h in hooks:
        h.remove()
    weights = {(li, n): model.model.layers[li].get_submodule(n).weight.detach().cpu().clone()
               for li in layers for n in NAMES}
    del model
    torch.cuda.empty_cache()
    print(f"captured {len(caps)} inputs ({a.tokens} tokens)", flush=True)

    dev = "cuda"
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{'layer':>5s} {'matrix':18s} {'max|x|':>9s} {'max|y|':>9s}   "
             f"{'runs relL2':>10s} {'runs relmax':>11s}   {'cur relL2':>10s} {'cur relmax':>10s}   "
             f"{'fp16mv relL2':>12s} {'fp16mv relmax':>13s} {'nonfinite':>9s}"]
    print(lines[0], flush=True)
    worst = {"runs": 0.0, "cur": 0.0, "fp16": 0.0}
    for li in layers:
        for n in NAMES:
            if weights[(li, n)].shape[1] % 256:
                continue                                                # run kernel needs C % 256 == 0
            w = weights[(li, n)].to(dev).float()
            X = caps[(li, n)].to(dev)                                   # [T, C]
            R, C = w.shape
            g = torch.Generator().manual_seed(li * 7 + NAMES.index(n))
            perms = mk.seeded_perms(C, 16, g).to(dev)
            rm = torch.randint(0, 16, (R,), generator=g).to(dev)
            rs = _row_scale(w)
            h = X.pow(2).mean(0) + 1e-8
            x, wt, *_ = _prepare(w, h, rs, perms, rm, 4, None, 4)
            sel = torch.randperm(x.shape[0], generator=g)[:300_000].to(dev)
            c = cb.kmeans(x[sel], 1024, 10, g, 65536, wt[sel])
            tags = cb.assign(x, c, 65536, wt)[0].view(R, C // 4).cpu()
            lin = PackedLinear(pack_bits(tags, 10).to(dev), c.half(), rs.half(), None, rm, perms, 1024, 4)
            rl = proto_runs.RunsLinear(lin, tags, 4)
            Wd = lin.dense().float()
            xp_all = None
            err = {k: [0.0, 0.0] for k in ("runs", "cur", "fp16")}
            nonfinite = 0
            ymax = 0.0
            for t in range(X.shape[0]):
                xh = X[t].half()
                ref = Wd @ xh.float()
                ymax = max(ymax, float(ref.abs().max()))
                xp_all = xh[lin.perms_l].contiguous()
                outs = {"runs": proto_runs.gemv(rl, xh, xp=xp_all).float(),
                        "cur": cuda_gemv.gemv(lin, xh).float(),
                        "fp16": torch.mv(Wd.half(), xh).float()}
                for k, y in outs.items():
                    if not torch.isfinite(y).all():
                        nonfinite += 1
                        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
                    d = y - ref
                    err[k][0] = max(err[k][0], float(d.norm() / ref.norm().clamp_min(1e-30)))
                    err[k][1] = max(err[k][1], float(d.abs().max() / ref.abs().max().clamp_min(1e-30)))
            for k in worst:
                worst[k] = max(worst[k], err[k][0])
            line = (f"{li:5d} {n:18s} {float(X.abs().max()):9.1f} {ymax:9.1f}   "
                    f"{err['runs'][0]:10.2e} {err['runs'][1]:11.2e}   {err['cur'][0]:10.2e} {err['cur'][1]:10.2e}   "
                    f"{err['fp16'][0]:12.2e} {err['fp16'][1]:13.2e} {nonfinite:9d}")
            print(line, flush=True)
            lines.append(line)
            del w, X, lin, rl, Wd
            torch.cuda.empty_cache()
    summ = (f"\nworst relative L2 error over all matrices and tokens: runs {worst['runs']:.2e}, "
            f"cur {worst['cur']:.2e}, fp16 mv {worst['fp16']:.2e}")
    print(summ)
    lines.append(summ)
    out.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
