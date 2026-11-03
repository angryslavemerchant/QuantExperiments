"""Cross-check shapequant.vptq_ref against VPTQ's own code (runs on the GPU box).

Installs the published `vptq` wheel without dependencies and calls its
`ops.quant_gemm` on the same checkpoint tensors: with the CUDA kernels when they
load (an independent implementation), otherwise its pure-PyTorch dequant. The
outputs are compared with F.linear(x, our decoded weight). The vptq package
root is imported as a bare path, so its model loader (transformers-version
bound) and tooling imports are never executed; the two tooling modules that
`vptq.utils.pack` imports at the top are stubbed.

Never fails the job: every problem is recorded in the JSON and the exit code is 0.
Run: python scripts/check_vptq.py --out artifacts/.../vptq_check.json REPO [REPO ...]
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import subprocess
import sys
import traceback
import types
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shapequant.vptq_ref import VPTQSource, decode_linear, _PARTS  # noqa: E402

VPTQ_WHEEL = "vptq==0.0.5.post1"
LAYERS = ["model.layers.0.self_attn.k_proj", "model.layers.0.self_attn.v_proj",
          "model.layers.14.self_attn.q_proj", "model.layers.14.self_attn.o_proj",
          "model.layers.14.mlp.gate_proj", "model.layers.14.mlp.down_proj",
          "model.layers.27.mlp.up_proj"]


def load_vptq_ops(report: dict):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "-q", VPTQ_WHEEL],
                       capture_output=True, text=True)
    report["pip"] = {"rc": r.returncode, "stderr": r.stderr[-2000:]}
    spec = importlib.util.find_spec("vptq")
    if spec is None:
        raise RuntimeError("vptq not importable after pip install")
    root = list(spec.submodule_search_locations)[0]
    pkg = types.ModuleType("vptq")
    pkg.__path__ = [root]
    sys.modules["vptq"] = pkg
    for name in ("accelerate", "sentence_transformers", "sentence_transformers.SentenceTransformer"):
        if importlib.util.find_spec(name.split(".")[0]) is None:
            m = types.ModuleType(name)
            m.SentenceTransformer = object
            sys.modules[name] = m
    qg = importlib.import_module("vptq.ops.quant_gemm")
    report["cuda_kernels"] = bool(getattr(qg, "_" + "_cuda_ops_installed", False)
                                  or qg.__dict__.get("__cuda_ops_installed", False))
    return qg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("repos", nargs="+")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report: dict = {"wheel": VPTQ_WHEEL, "torch": torch.__version__, "repos": {}}
    try:
        qg = load_vptq_ops(report)
    except Exception:
        report["error"] = traceback.format_exc()
        out.write_text(json.dumps(report, indent=1))
        print(report["error"])
        return 0
    print(f"vptq loaded; CUDA kernels: {report['cuda_kernels']}", flush=True)
    dev = "cuda"
    g = torch.Generator(device="cpu").manual_seed(0)
    for repo in a.repos:
        rows = []
        try:
            src = VPTQSource(repo)
            for name in LAYERS:
                c = src.layers[name]
                t = {p: src.f.get_tensor(f"{name}.{p}").to(dev) for p in _PARTS
                     if f"{name}.{p}" in src.keys}
                ours = decode_linear(t, c)
                G, v = c["group_num"], c["vector_lens"][1]
                K, Kr = c["num_centroids"][1], c["num_res_centroids"][1]
                row = {"layer": name}
                for n_tok in (1, 16):  # 1 token -> gemv kernel; 16 -> dequant + linear
                    x = torch.randn(1, n_tok, c["in_features"], generator=g).to(dev, torch.bfloat16)
                    theirs = qg.quant_gemm(
                        x, bias=None, indices=t["indices"], centroids=t["centroids.weight"],
                        outlier_indices=None, outlier_centroids=None,
                        residual_indices=None,
                        residual_centroids=t.get("res_centroids.weight") if Kr > 0 else None,
                        perm=None, weight_scale=t.get("weight_scale"), weight_bias=t.get("weight_bias"),
                        vector_len=v, outlier_vector_len=-1, num_codebooks=G, num_centroids=K,
                        num_outlier_centroids=-1, num_res_centroids=Kr,
                        is_indice_packed=True, group_size=c["group_size"], outlier_size=0,
                        in_features=c["in_features"], out_features=c["out_features"],
                        padding=(-c["out_features"]) % v, outlier_padding=0).float()
                    mine = F.linear(x.float(), ours)
                    rel = float((theirs - mine).pow(2).sum() / mine.pow(2).sum())
                    row[f"rel_err_{n_tok}tok"] = rel
                rows.append(row)
                print(f"{repo.split('/')[-1]:44s} {name[13:]:22s} "
                      f"rel err 1tok {row['rel_err_1tok']:.2e}  16tok {row['rel_err_16tok']:.2e}", flush=True)
            report["repos"][repo] = {"layers": rows,
                                     "max_rel_err": max(max(r["rel_err_1tok"], r["rel_err_16tok"]) for r in rows)}
        except Exception:
            report["repos"][repo] = {"layers": rows, "error": traceback.format_exc()}
            print(report["repos"][repo]["error"], flush=True)
        out.write_text(json.dumps(report, indent=1))
    # bf16 activations and bf16 kernel accumulation: agreement is ~1e-4 relative;
    # a layout error is ~1.
    report["verdict"] = ("match" if all(r.get("max_rel_err", 1.0) < 1e-2 for r in report["repos"].values())
                         else "MISMATCH or error")
    out.write_text(json.dumps(report, indent=1))
    print("verdict:", report["verdict"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
