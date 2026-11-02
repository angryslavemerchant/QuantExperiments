"""Rerun an experiment stage from a YAML config.

    python scripts/run_experiment.py --config configs/stage1_baselines.yaml

Writes under the config's output_dir:
    results.csv          one row per run (appended after each run; finished runs are skipped on rerun)
    runs/<name>.json     full metrics for the run
    gsm8k/<name>.jsonl   per-problem generations (if GSM8K enabled)
    stage0_report.json   sanity-check verdicts (if stage0_checks configured)
    kl_vs_bits.png, gsm8k_vs_bits.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shapequant import codebook as cb  # noqa: E402
from shapequant import memory  # noqa: E402
from shapequant.evaluate import GSM8K_TEMPLATE, Scorer, load_gsm8k, load_wikitext_windows, run_gsm8k  # noqa: E402
from shapequant.pipeline import apply_run, compressible_linears, snapshot_originals  # noqa: E402
from shapequant.plots import make_plots  # noqa: E402

COLUMNS = [
    "stage", "run_name", "method", "model", "d", "K", "shuffle", "masks", "M", "finetune", "start",
    "seed", "bits_per_weight_rounded", "bits_per_weight_ideal", "total_mb", "pct_of_original",
    "mean_kl", "p99_kl", "top1_agree", "perplexity", "gsm8k_acc", "codebook_usage",
    "recon_mse", "recon_rel_mse", "compress_seconds", "eval_tokens",
]

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def done_runs(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="") as f:
        return {row["run_name"] for row in csv.DictReader(f)}


def append_row(csv_path: Path, row: dict) -> None:
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in COLUMNS})


def stage0_checks(cfg: dict, model, rows: dict[str, dict], out_dir: Path) -> dict:
    s0 = cfg["stage0_checks"]
    report: dict = {}

    orig = rows["original"]
    report["original_self_kl_zero"] = {
        "pass": orig["mean_kl"] == 0.0 and orig["top1_agree"] == 100.0,
        "mean_kl": orig["mean_kl"], "top1_agree": orig["top1_agree"],
    }

    q4 = rows[s0["q4_run"]]
    ppl_ratio = q4["perplexity"] / orig["perplexity"]
    report["q4_near_lossless"] = {
        "pass": q4["mean_kl"] < s0["q4_max_mean_kl"] and ppl_ratio < s0["q4_max_ppl_ratio"],
        "mean_kl": q4["mean_kl"], "ppl": q4["perplexity"], "orig_ppl": orig["perplexity"],
        "ppl_ratio": ppl_ratio,
    }

    # Decode exactness on one matrix, compressed from its original weights.
    name = s0["decode_matrix"]
    d, K = s0["d"], s0["K"]
    mod = dict(compressible_linears(model))[name]
    w = mod.weight.detach().to(torch.float32)  # model is restored to original before checks
    t = time.time()
    cm, stats = cb.compress(w, d, K, iters=cfg["kmeans"]["iters"],
                            max_fit_chunks=cfg["kmeans"]["max_fit_chunks"],
                            seed=cb.layer_seed(cfg["seed"], name),
                            batch=cfg["kmeans"]["assign_batch"])
    a, b = cb.decode(cm), cb.decode(cm)
    report["decode_exact"] = {
        "pass": bool(torch.equal(a, b)), "matrix": name, "shape": list(w.shape), "d": d, "K": K,
        "codebook_usage": stats["codebook_usage"],
        "normalized_chunk_mse": stats["normalized_chunk_mse"],
        "rel_mse_vs_original": float(((a - w) ** 2).sum() / (w ** 2).sum()),
        "seconds": time.time() - t,
    }

    # Memory formula vs. hand arithmetic, term by term.
    R, C = w.shape
    n_tags = R * math.ceil(C / d)
    tag_bits = n_tags * math.ceil(math.log2(K))
    cb_bits = K * d * 16
    scale_bits = R * 16
    hand = tag_bits + cb_bits + scale_bits
    formula, _ = memory.codebook_bits(R, C, d, K)
    report["memory_formula"] = {
        "pass": formula == hand, "R": R, "C": C, "tags_bits": tag_bits, "codebook_bits": cb_bits,
        "row_scale_bits": scale_bits, "hand_total_bits": hand, "formula_total_bits": formula,
        "bits_per_weight": formula / (R * C),
    }
    report["all_pass"] = all(v["pass"] for v in report.values() if isinstance(v, dict))
    (out_dir / "stage0_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_dir = Path(cfg["output_dir"])
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    (out_dir / "gsm8k").mkdir(exist_ok=True)
    csv_path = out_dir / "results.csv"
    seed = cfg["seed"]
    seed_everything(seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers

    env = {
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "config": cfg, "gsm8k_template": GSM8K_TEMPLATE,
    }
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2, default=str))
    log(f"env: {env['python']} torch {env['torch']} transformers {env['transformers']} gpu {env['gpu']}")

    dtype = getattr(torch, cfg["reference_dtype"])
    log(f"loading {cfg['model']} ({cfg['reference_dtype']})")
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=dtype).to(args.device).eval()
    originals = snapshot_originals(model)

    ecfg = cfg["eval"]
    windows = load_wikitext_windows(tok, ecfg["wikitext"])
    log(f"wikitext: {windows.shape[0]} windows x {windows.shape[1]} tokens; caching reference")
    scorer = Scorer(model, windows, ecfg["logit_chunk"])
    gcfg = ecfg.get("gsm8k", {})
    problems = load_gsm8k(gcfg) if gcfg.get("enabled") else None

    finished = done_runs(csv_path)
    rows: dict[str, dict] = {}
    for run in cfg["runs"]:
        name = run["name"]
        if name in finished:
            log(f"skip {name} (already in results.csv)")
            rows[name] = json.loads((out_dir / "runs" / f"{name}.json").read_text())
            continue
        log(f"=== run {name}: {run}")
        seed_everything(seed)
        stats = apply_run(model, originals, run, cfg["kmeans"], seed, log=log)
        log(f"  compressed in {stats['compress_seconds']:.1f}s; "
            f"{stats['bits_per_weight_rounded']:.3f} bpw; scoring wikitext")
        metrics = scorer.score(model)
        log(f"  mean KL {metrics['mean_kl']:.5f}  p99 {metrics['p99_kl']:.4f}  "
            f"top1 {metrics['top1_agree']:.2f}%  ppl {metrics['perplexity']:.3f}")
        if problems is not None:
            g = run_gsm8k(model, tok, problems, gcfg, out_dir / "gsm8k" / f"{name}.jsonl")
            log(f"  GSM8K {g['gsm8k_acc']:.1f}% ({g['gsm8k_seconds']:.0f}s)")
            metrics.update(g)
        row = {
            "stage": run.get("stage", cfg.get("stage")), "run_name": name, "method": run["method"],
            "model": cfg["model"], "d": run.get("d"), "K": run.get("K"),
            "shuffle": run.get("shuffle", False), "masks": run.get("masks", False), "M": run.get("M"),
            "finetune": run.get("finetune", False),
            "start": run.get("start", "q4") if run["method"] == "codebook" else "fp16",
            "seed": seed, **stats, **metrics,
        }
        rows[name] = row
        (out_dir / "runs" / f"{name}.json").write_text(json.dumps(row, indent=2))
        append_row(csv_path, row)
        make_plots(csv_path, out_dir)

    if cfg.get("stage0_checks"):
        from shapequant.pipeline import restore_originals
        restore_originals(model, originals)
        report = stage0_checks(cfg, model, rows, out_dir)
        for k, v in report.items():
            log(f"stage0 {k}: {v if not isinstance(v, dict) else ('PASS' if v['pass'] else 'FAIL')}")
        if not report["all_pass"]:
            log("STAGE 0 FAILED")
            return 2
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
