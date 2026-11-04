"""Cross-check shapequant.vptq_ref against VPTQ's own code (runs on the GPU box).

Installs the published `vptq` wheel without dependencies and uses its
`ops.quant_gemm` (CUDA kernels when they load, an independent implementation;
otherwise its pure-PyTorch dequant) on the same checkpoint tensors:

1. layer check: quant_gemm(x) vs F.linear(x, our decoded weight) on a few
   block linears of every repo, with 1 token (gemv kernel) and 16 tokens
   (dequant kernel + matmul);
2. full model (repos given with --full): every block linear of Qwen2.5-7B-Instruct
   runs VPTQ's quant_gemm forward with the checkpoint's own tensors, scored with
   the harness Scorer (same wikitext windows, KL vs the 16-bit model) and next to
   it the same model with our decoded weights. Embedding, lm_head, norms and
   biases stay at the original values in both, as in the harness.

The vptq package root is imported as a bare path, so its model loader
(transformers-version bound) and tooling imports are never executed; the two
tooling modules that `vptq.utils.pack` imports at the top are stubbed when absent.

Never fails the job: every problem is recorded in the JSON and the exit code is 0.
Run: python scripts/check_vptq.py --config CFG --out JSON [--full REPO ...] REPO [REPO ...]
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import subprocess
import sys
import time
import traceback
import types
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shapequant.evaluate import Scorer, load_wikitext_windows  # noqa: E402
from shapequant.pipeline import compressible_linears, restore_originals, snapshot_originals  # noqa: E402
from shapequant.vptq_ref import VPTQSource, decode_linear, _PARTS  # noqa: E402

VPTQ_WHEEL = "vptq==0.0.5.post1"
LAYERS = ["model.layers.0.self_attn.k_proj", "model.layers.0.self_attn.v_proj",
          "model.layers.14.self_attn.q_proj", "model.layers.14.self_attn.o_proj",
          "model.layers.14.mlp.gate_proj", "model.layers.14.mlp.down_proj",
          "model.layers.27.mlp.up_proj"]


def log(msg: str) -> None:
    print(f"[check_vptq {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _installed(top: str) -> bool:
    if top in sys.modules:
        return True
    try:
        return importlib.util.find_spec(top) is not None
    except (ValueError, ImportError):
        return False


def load_vptq_ops(report: dict):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "-q", VPTQ_WHEEL],
                       capture_output=True, text=True)
    report["pip"] = {"rc": r.returncode, "stderr": r.stderr[-2000:]}
    spec = importlib.util.find_spec("vptq")
    if spec is None:
        raise RuntimeError("vptq not importable after pip install")
    pkg = types.ModuleType("vptq")
    pkg.__path__ = [list(spec.submodule_search_locations)[0]]
    sys.modules["vptq"] = pkg
    stubbed = []
    for top in ("accelerate", "sentence_transformers"):
        if not _installed(top):
            sys.modules[top] = types.ModuleType(top)
            stubbed.append(top)
    if "sentence_transformers" in stubbed:
        sub = types.ModuleType("sentence_transformers.SentenceTransformer")
        sub.SentenceTransformer = object
        sys.modules["sentence_transformers.SentenceTransformer"] = sub
    report["stubbed"] = stubbed
    qg = importlib.import_module("vptq.ops.quant_gemm")
    report["cuda_kernels"] = bool(qg.__dict__.get("__cuda_ops_installed", False))
    return qg


def their_forward(qg, t: dict, c: dict, bias):
    G, v = c["group_num"], c["vector_lens"][1]
    K, Kr = c["num_centroids"][1], c["num_res_centroids"][1]

    def fwd(x):
        y = qg.quant_gemm(
            x, bias=None, indices=t["indices"], centroids=t["centroids.weight"],
            outlier_indices=None, outlier_centroids=None, residual_indices=None,
            residual_centroids=t.get("res_centroids.weight") if Kr > 0 else None,
            perm=None, weight_scale=t.get("weight_scale"), weight_bias=t.get("weight_bias"),
            vector_len=v, outlier_vector_len=-1, num_codebooks=G, num_centroids=K,
            num_outlier_centroids=-1, num_res_centroids=Kr, is_indice_packed=True,
            group_size=c["group_size"], outlier_size=0, in_features=c["in_features"],
            out_features=c["out_features"], padding=(-c["out_features"]) % v, outlier_padding=0)
        return y if bias is None else y + bias
    return fwd


def tensors(src: VPTQSource, name: str, dev) -> dict:
    return {p: src.f.get_tensor(f"{name}.{p}").to(dev) for p in _PARTS if f"{name}.{p}" in src.keys}


def layer_check(qg, repo: str, dev) -> dict:
    src = VPTQSource(repo)
    g = torch.Generator(device="cpu").manual_seed(0)
    rows = []
    for name in LAYERS:
        c = src.layers[name]
        t = tensors(src, name, dev)
        ours = decode_linear(t, c)
        fwd = their_forward(qg, t, c, None)
        row = {"layer": name}
        for n_tok in (1, 16):
            x = torch.randn(1, n_tok, c["in_features"], generator=g).to(dev, torch.bfloat16)
            theirs = fwd(x).float()
            mine = F.linear(x.float(), ours)
            row[f"rel_err_{n_tok}tok"] = float((theirs - mine).pow(2).sum() / mine.pow(2).sum())
        rows.append(row)
        log(f"{repo.split('/')[-1]:44s} {name[13:]:22s} rel err 1tok {row['rel_err_1tok']:.2e}"
            f"  16tok {row['rel_err_16tok']:.2e}")
    return {"layers": rows, "max_rel_err": max(max(r["rel_err_1tok"], r["rel_err_16tok"]) for r in rows)}


def full_model(qg, repo: str, model, scorer, originals, dev) -> dict:
    src = VPTQSource(repo)
    mods = compressible_linears(model)
    res = {}
    # ours: decoded dense weights (the harness path)
    with torch.no_grad():
        for n, m in mods:
            w, _, _ = src.matrix(n, tuple(m.weight.shape), device=dev)
            m.weight.copy_(w.to(m.weight.dtype))
    res["ours"] = scorer.score(model)
    log(f"{repo.split('/')[-1]} ours  : {res['ours']}")
    restore_originals(model, originals)
    # theirs: VPTQ quant_gemm forward in every block linear
    saved = {}
    try:
        for n, m in mods:
            t = tensors(src, n, dev)
            saved[n] = m.forward
            m.forward = their_forward(qg, t, src.layers[n], m.bias)
        res["theirs"] = scorer.score(model)
        log(f"{repo.split('/')[-1]} theirs: {res['theirs']}")
    finally:
        for n, m in mods:
            if n in saved:
                m.forward = saved[n]
        torch.cuda.empty_cache()
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="experiment config (model + eval windows)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--full", nargs="*", default=[], help="repos to score as a whole model")
    ap.add_argument("repos", nargs="+")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report: dict = {"wheel": VPTQ_WHEEL, "torch": torch.__version__, "repos": {}, "full": {}}

    def save():
        out.write_text(json.dumps(report, indent=1))

    try:
        qg = load_vptq_ops(report)
    except Exception:
        report["error"] = traceback.format_exc()
        save()
        log(report["error"])
        return 0
    log(f"vptq loaded; CUDA kernels: {report['cuda_kernels']}; stubbed {report['stubbed']}")
    dev = "cuda"
    for repo in a.repos:
        try:
            report["repos"][repo] = layer_check(qg, repo, dev)
        except Exception:
            report["repos"][repo] = {"error": traceback.format_exc()}
            log(report["repos"][repo]["error"])
        save()
    # bf16 activations and bf16 accumulation: agreement is ~1e-4 relative; a layout error is ~1.
    report["layer_verdict"] = ("match" if all(r.get("max_rel_err", 1.0) < 1e-2 for r in report["repos"].values())
                               else "MISMATCH or error")
    log(f"layer verdict: {report['layer_verdict']}")
    save()
    if a.full:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            cfg = yaml.safe_load(open(a.config))
            tok = AutoTokenizer.from_pretrained(cfg["model"])
            model = AutoModelForCausalLM.from_pretrained(
                cfg["model"], dtype=getattr(torch, cfg["reference_dtype"])).to(dev).eval()
            originals = snapshot_originals(model)
            windows = load_wikitext_windows(tok, cfg["eval"]["wikitext"])
            scorer = Scorer(model, windows, cfg["eval"]["logit_chunk"])
            log(f"full-model check on {windows.shape[0]} x {windows.shape[1]} wikitext tokens")
        except Exception:
            report["full_error"] = traceback.format_exc()
            save()
            log(report["full_error"])
            return 0
        for repo in a.full:
            try:
                report["full"][repo] = full_model(qg, repo, model, scorer, originals, dev)
            except Exception:
                report["full"][repo] = {"error": traceback.format_exc()}
                log(report["full"][repo]["error"])
            restore_originals(model, originals)
            save()
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
