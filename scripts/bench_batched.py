"""Batched decode throughput of the Qwen3.5 / 3.8 runtime (shapequant.qwen35_batch), in the
llama-batched-bench setting: npl sequences each holding a context of `ctx` tokens, decoding
`tg` more tokens together; reports aggregate tok/s (B * tg / time) and per-sequence tok/s for
each batch size, as JSON.

Synthetic engine (default): the model's shapes from its HF config (transformers AutoConfig, no
weights downloaded), random codes at the shipped format (d = 4 runs, G books per matrix, M = 1,
K from --bits or a per-matrix allocation file, 8-bit head, 4-bit embedding), built directly on
the GPU (speed does not depend on the values). Or --packed: a real packed file
(qwen35_runtime.engine_from_packed).

    python scripts/bench_batched.py --model Qwen/Qwen3.8-27B --alloc configs/alloc/qwen38_27b_s14_25.json \\
        --ctx 512 --tg 64 --batch 1 2 4 8 16 32 --out artifacts/bench/batched_27b.json
    python scripts/bench_batched.py --packed model.safetensors --ctx 512 --out res.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shapequant import qwen35_batch as qb, qwen35_runtime as q35  # noqa: E402
from shapequant.kernels import decode_e2e as de  # noqa: E402


def synthetic_engine(model: str, max_len: int, bits: int = 10, G: int = 4, alloc: dict | None = None,
                     layers: int | None = None, seed: int = 0, dev="cuda") -> q35.Qwen35Engine:
    """Random-code Qwen3.5 engine with `model`'s shapes. alloc: {"model.layers.i.<module>": K}
    (missing -> 2**bits). layers: keep only the first n layers (quick runs)."""
    from transformers import AutoConfig
    hc = AutoConfig.from_pretrained(model)
    hc = getattr(hc, "text_config", hc)
    d = hc.to_dict()
    if layers:
        d["num_hidden_layers"] = layers
        d["layer_types"] = list(d["layer_types"])[:layers]
    cfg = q35._cfg(d, max_len)
    g = torch.Generator(device=dev).manual_seed(seed)
    H, I = cfg.hidden, cfg.inter
    conv_dim = 2 * cfg.hk * cfg.dk + cfg.hv * cfg.dv

    def lin(name, R, C):
        b = alloc[name].bit_length() - 1 if alloc and name in alloc else bits
        th, bps = (0, 1) if R <= 4096 else (512, 1)          # engine_from_packed's launch configs
        return de.RunsLin.random(R, C, b, torch.arange(C)[None], g, dev=dev, G=G, threads=th, blocks_per_sm=bps)

    def vec(n, s=0.1, base=1.0):
        return (torch.randn(n, device=dev, generator=g) * s + base).float()

    Ls = []
    for i, typ in enumerate(cfg.layer_types):
        p = f"model.layers.{i}."
        L = {"ln1": vec(H), "ln2": vec(H),
             "gate": lin(p + "mlp.gate_proj", I, H), "up": lin(p + "mlp.up_proj", I, H),
             "down": lin(p + "mlp.down_proj", H, I)}
        if typ == "linear_attention":
            L.update(qkv=lin(p + "linear_attn.in_proj_qkv", conv_dim, H),
                     z=lin(p + "linear_attn.in_proj_z", cfg.hv * cfg.dv, H),
                     out=lin(p + "linear_attn.out_proj", H, cfg.hv * cfg.dv),
                     wa=vec(cfg.hv * H, 0.02, 0.0).view(cfg.hv, H), wb=vec(cfg.hv * H, 0.02, 0.0).view(cfg.hv, H),
                     convw=vec(conv_dim * cfg.conv_k, 0.3, 0.0).view(conv_dim, cfg.conv_k),
                     dt_bias=vec(cfg.hv, 0.5, 0.0), A=vec(cfg.hv, 0.5, 2.0).abs(), gnorm=vec(cfg.dv))
        else:
            L.update(q=lin(p + "self_attn.q_proj", 2 * cfg.n_heads * cfg.head_dim, H),
                     k=lin(p + "self_attn.k_proj", cfg.n_kv * cfg.head_dim, H),
                     v=lin(p + "self_attn.v_proj", cfg.n_kv * cfg.head_dim, H),
                     o=lin(p + "self_attn.o_proj", H, cfg.n_heads * cfg.head_dim),
                     qn=vec(cfg.head_dim), kn=vec(cfg.head_dim))
        Ls.append(L)
    V = cfg.vocab
    head = de.Q8Lin(torch.randint(-127, 127, (V, H), dtype=torch.int8, device=dev, generator=g),
                    torch.full((V, H // 32), 1e-3, dtype=torch.float16, device=dev), V, H)
    emb = de.Q4Embed(torch.randint(0, 255, (V, H // 2), dtype=torch.uint8, device=dev, generator=g),
                     torch.full((V, H // 32), 1e-2, dtype=torch.float16, device=dev))
    return q35.Qwen35Engine(cfg, Ls, emb, vec(H), head, dev, fused=True)


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="HF model id / path for the config (synthetic engine)")
    ap.add_argument("--packed", help="real packed file instead of a synthetic engine")
    ap.add_argument("--bits", type=int, default=10, help="synthetic: tag bits (K = 2**bits) where --alloc has none")
    ap.add_argument("--alloc", help="synthetic: per-matrix K, JSON {name: K} or {'choices': {...}}")
    ap.add_argument("--G", type=int, default=4, help="synthetic: books per matrix")
    ap.add_argument("--layers", type=int, default=None, help="synthetic: first n layers only (quick runs)")
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--ctx", type=int, default=512, help="context per sequence before the timed steps (npp)")
    ap.add_argument("--tg", type=int, default=64, help="timed decode steps (ntg)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--head-loop-max", type=int, default=1, help="B up to this: batch-1 head kernel per sequence")
    ap.add_argument("--warps", type=int, default=8)
    ap.add_argument("--bps", type=int, default=2)
    ap.add_argument("--max-split", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if bool(a.model) == bool(a.packed):
        raise SystemExit("give exactly one of --model / --packed")
    max_len = a.ctx + a.tg + 1
    t0 = time.time()
    if a.packed:
        eng, info = q35.engine_from_packed(a.packed, max_len=max_len, fused=True)
        src = {"packed": a.packed, "summary": info["summary"]}
    else:
        alloc = None
        if a.alloc:
            alloc = json.loads(Path(a.alloc).read_text())
            alloc = alloc.get("choices", alloc)
        eng = synthetic_engine(a.model, max_len, a.bits, a.G, alloc, a.layers)
        src = {"model": a.model, "bits": a.bits, "G": a.G, "alloc": a.alloc, "layers": a.layers}
    torch.cuda.synchronize()
    res = {"gpu": torch.cuda.get_device_name(), "source": src, "ctx": a.ctx, "tg": a.tg,
           "weight_bytes": eng.weight_bytes(), "build_s": round(time.time() - t0, 1),
           "launch": {"warps": a.warps, "bps": a.bps, "max_split": a.max_split, "head_loop_max": a.head_loop_max},
           "runs": []}
    print(f"engine built in {res['build_s']} s, weights {res['weight_bytes'] / 1e9:.2f} GB", flush=True)
    for B in a.batch:
        beng = qb.Qwen35BatchEngine(eng, B, head_loop_max=a.head_loop_max, warps=a.warps, bps=a.bps,
                                    max_split=a.max_split)
        torch.cuda.reset_peak_memory_stats()
        t = beng.time_steps(a.tg, a.ctx, a.reps)
        r = {"B": B, "step_ms": t * 1e3, "agg_tok_s": B / t, "seq_tok_s": 1 / t,
             "state_gb": beng.state_bytes() / 1e9, "peak_gb": torch.cuda.max_memory_allocated() / 1e9}
        res["runs"].append(r)
        print(f"B={B:2d}: {r['step_ms']:.2f} ms/step, {r['agg_tok_s']:.1f} tok/s aggregate, "
              f"{r['seq_tok_s']:.1f} tok/s per sequence (states {r['state_gb']:.2f} GB, peak {r['peak_gb']:.2f} GB)",
              flush=True)
        del beng
        torch.cuda.empty_cache()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
