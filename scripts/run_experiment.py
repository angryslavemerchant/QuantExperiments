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
from shapequant.pipeline import apply_run, compressible_linears, restore_originals, snapshot_originals  # noqa: E402
from shapequant.plots import make_plots  # noqa: E402

COLUMNS = [
    "stage", "run_name", "method", "source", "model", "d", "K", "shuffle", "masks", "M", "finetune", "weighting", "start",
    "seed", "bits_per_weight_rounded", "bits_per_weight_ideal", "total_mb", "pct_of_original",
    "mean_kl", "p99_kl", "top1_agree", "perplexity", "gsm8k_acc", "codebook_usage",
    "recon_mse", "recon_rel_mse", "compress_seconds", "eval_tokens",
    "G", "alloc", "mean_log2K", "mask_nonidentity_frac", "ft_kl_first", "ft_kl_last",
    "rotate", "ldlq", "out_rel_err_fullH",
    "lme_mmlu", "lme_arc_challenge", "lme_hellaswag", "lme_winogrande", "lme_gsm8k",
]

T0 = time.time()


LOG_FILE: Path | None = None   # out_dir/run.log once known (recovered with the artifacts)


def log(msg: str) -> None:
    mem = ""
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        mem = f" [gpu {torch.cuda.memory_allocated() / 1e9:.1f}/{torch.cuda.max_memory_allocated() / 1e9:.1f} GB]"
    line = f"[{time.time() - T0:8.1f}s]{mem} {msg}"
    print(line, flush=True)
    if LOG_FILE is not None:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")


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


def export_packed(fits: dict, path: Path, cfg: dict, row_name: str, log, model=None) -> dict:
    """Write the run's compressed matrices in the packed format (shapequant.packed); with `model`,
    also the aux section (4-bit embedding, 8-bit head, the other tensors, the config)."""
    from shapequant import packed
    items = {n: packed.pack_matrix(f, **f.stats["pack"]) for n, f in fits.items()}
    aux = packed.aux_from_model(model, list(items)) if model is not None else None
    summary = packed.save(path, items, base_model=cfg["model"], extra={"run": row_name}, aux=aux)
    log(f"  exported {summary['matrices']} matrices to {path}: {summary['bpw']:.4f} bpw "
        + (f"(all tensors {summary['bpw_all']:.4f}) " if aux else "")
        + f"({path.stat().st_size / 1e6:.1f} MB)")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--shard", type=int, default=0,
                    help="run only runs[i] with i %% num_shards == shard (see run_parallel.py)")
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_dir = Path(cfg["output_dir"])
    if args.num_shards > 1:
        if cfg.get("stage0_checks"):
            raise SystemExit("stage0_checks cannot be sharded")
        cfg["runs"] = [r for i, r in enumerate(cfg["runs"]) if i % args.num_shards == args.shard]
        out_dir = out_dir / f"shard{args.shard}"
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    (out_dir / "gsm8k").mkdir(exist_ok=True)
    global LOG_FILE
    LOG_FILE = out_dir / "run.log"
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
    lcfg = ecfg.get("lm_eval", {})
    problems = load_gsm8k(gcfg) if gcfg.get("enabled") else None

    importance = None
    seqs = None
    if any(run.get("weighting", "none") in ("act", "act_row") or run["method"] in ("masked", "stack")
           for run in cfg["runs"]):
        from shapequant.calibrate import column_importance, load_calibration_sequences
        ccfg = cfg["calibration"]
        seqs = load_calibration_sequences(tok, ccfg, seed)
        log(f"calibration: {seqs.shape[0]} x {seqs.shape[1]} tokens from {ccfg['dataset']}")
        importance = column_importance(model, seqs, ccfg.get("batch_size", 4))

    finished = done_runs(csv_path)
    hessians = None
    if any((r.get("ldlq") or r.get("rotate")) and r["name"] not in finished for r in cfg["runs"]):
        from shapequant.calibrate import input_hessians
        hessians = input_hessians(model, seqs, ccfg.get("batch_size", 4), max_gpu_bytes=ccfg.get("hessian_group_bytes"),
                                  log=log)
        log(f"calibration: full input Hessians for {len(set(map(id, hessians.values())))} distinct inputs")
    for run in cfg["runs"]:
        if run.get("alloc_from"):
            # Per-matrix K from a saved allocation (configs/alloc/*.json, {"choices": {name: K}}),
            # optionally mapped to another format's sizes (alloc_map: {"<K>": K' or [Ka, Kb]}),
            # e.g. a d=4 Fisher allocation reused for d=8 additive books at the same tag bits.
            if run.get("alloc"):
                raise ValueError(f"{run['name']}: alloc_from and alloc are exclusive")
            src = json.loads(Path(run["alloc_from"]).read_text())["choices"]
            amap = {str(k): v for k, v in run.get("alloc_map", {}).items()}
            names = {n for n, _ in compressible_linears(model)}
            if set(src) != names:
                raise ValueError(f"{run['name']}: {run['alloc_from']} does not cover this model's matrices")
            run["_K_by_name"] = {n: amap.get(str(k), k) for n, k in src.items()}
            hist: dict[str, int] = {}
            for v in run["_K_by_name"].values():
                hist[str(v)] = hist.get(str(v), 0) + 1
            log(f"  {run['name']}: allocation from {run['alloc_from']}: K histogram {hist}")
    for r in cfg["runs"]:
        # a previously solved allocation (configs/alloc/*.json, exported from allocation.json):
        # same per-matrix K without re-running Fisher + proxy fits
        if r.get("alloc_file"):
            d = json.loads(Path(r["alloc_file"]).read_text())
            r["_K_by_name"] = {n: int(k) for n, k in d["choices"].items()}
            r.setdefault("alloc", f"file:{Path(r['alloc_file']).stem}")
            log(f"  {r['name']}: allocation from {r['alloc_file']} ({len(r['_K_by_name'])} matrices)")
    alloc_runs = [r for r in cfg["runs"] if r.get("alloc") and not r.get("alloc_file") and r["name"] not in finished]
    if alloc_runs:
        # Per-matrix codebook sizes: Fisher + light proxy fits once, then each
        # run's allocation is solved against its own budget.
        from shapequant import alloc as al
        acfg = cfg["allocation"]
        Ks = tuple(tuple(k) if isinstance(k, (list, tuple)) else k for k in acfg["Ks"])
        names = [n for n, _ in compressible_linears(model)]
        fseqs = seqs[: acfg["fisher_sequences"]].reshape(-1, acfg["fisher_seq_len"])
        log(f"allocation: Fisher on {fseqs.shape[0]} x {fseqs.shape[1]} calibration tokens")
        # Fisher accumulators for every weight (fp32, ~100 GB at 27B) need not fit at once: layer
        # groups of at most a third of the free GPU memory, each Fisher pass followed by its proxy
        # fits (identical result, the fits only read their own matrix's Fisher)
        from shapequant.calibrate import layer_groups
        torch.cuda.empty_cache()
        budget = int(acfg.get("fisher_group_bytes") or torch.cuda.mem_get_info()[0] // 3)
        per_layer: dict[int, int] = {}
        for n in names:
            li = int(n.split(".")[2])
            per_layer[li] = per_layer.get(li, 0) + originals[n].numel() * 4
        groups = layer_groups(sorted(per_layer.items()), budget)
        if len(groups) > 1:
            log(f"allocation: Fisher + proxy fits in {len(groups)} layer groups")
        err, bits = [], []
        for g in groups:
            gset = set(g)
            gnames = [n for n in names if int(n.split(".")[2]) in gset]
            fisher = al.fisher_diag(model, fseqs, log=log, names=gnames)
            restore_originals(model, originals)
            log(f"allocation: proxy fits d={acfg['d']} K in {Ks} (layers {g[0]}-{g[-1]})")
            e, b = al.proxy_table(originals, importance, fisher, gnames, acfg["d"], Ks, seed,
                                  acfg["proxy_iters"], acfg["proxy_fit_chunks"],
                                  cfg["kmeans"]["assign_batch"], args.device, log=log)
            err += e
            bits += b
            del fisher
            torch.cuda.empty_cache()
        assert len(err) == len(names)
        n_layers = model.config.num_hidden_layers
        table = {"Ks": Ks, "names": names, "err": err, "bits": bits, "choices": {}}
        for run in alloc_runs:
            B = al.budget_bits(bits, Ks, run["budget"])
            # alloc_Ks: restrict this run's choices to a subset of Ks (e.g. kernel-friendly sizes);
            # the budget is still the global spec, so the leftover bits are spent within the subset
            allowed = set(run.get("alloc_Ks", Ks))
            run_err = [[e if Ks[i] in allowed else float("inf") for i, e in enumerate(row)] for row in err]
            ch = (al.lagrangian(run_err, bits, B) if run["alloc"] == "fisher"
                  else al.heuristic(names, bits, B, n_layers) if run["alloc"] == "heuristic"
                  else None)
            if ch is None:
                raise ValueError(f"unknown alloc {run['alloc']!r}")
            run["_K_by_name"] = {n: Ks[k] for n, k in zip(names, ch)}
            table["choices"][run["name"]] = run["_K_by_name"]
            hist = {str(K): ch.count(i) for i, K in enumerate(Ks)}
            log(f"  {run['name']}: {run['alloc']} allocation at {run['budget']}: K histogram {hist}")
        (out_dir / "allocation.json").write_text(json.dumps(table))

    rows: dict[str, dict] = {}
    for run in cfg["runs"]:
        name = run["name"]
        if name in finished:
            log(f"skip {name} (already in results.csv)")
            rows[name] = json.loads((out_dir / "runs" / f"{name}.json").read_text())
            continue
        log(f"=== run {name}: { {k: v for k, v in run.items() if not k.startswith('_')} }")
        seed_everything(seed)
        try:
            stats = apply_run(model, originals, run, cfg["kmeans"], seed, log=log, importance=importance,
                              hessians=hessians)
        except Exception as e:  # a broken external reference must not cost the whole run
            if run["method"] not in ("gguf", "vptq"):
                raise
            import traceback
            log(f"  REFERENCE FAILED {name}: {e!r}; skipping")
            log(traceback.format_exc())
            restore_originals(model, originals)
            continue
        fits = stats.pop("_fits", None)
        log(f"  compressed in {stats['compress_seconds']:.1f}s; "
            f"{stats['bits_per_weight_rounded']:.3f} bpw; scoring wikitext")

        def evaluate(row_name: str, ft: bool, extra: dict) -> dict:
            metrics = scorer.score(model)
            log(f"  [{row_name}] mean KL {metrics['mean_kl']:.5f}  p99 {metrics['p99_kl']:.4f}  "
                f"top1 {metrics['top1_agree']:.2f}%  ppl {metrics['perplexity']:.3f}")
            if ecfg.get("deploy"):
                dm = scorer.score_deploy(model, ecfg["deploy"])
                metrics.update(dm)
                names = [v["name"] for v in ecfg["deploy"]]
                if run["method"] == "gguf" and ecfg.get("deploy_gguf_own"):
                    # the GGUF file's own head and embedding (what llama.cpp would run)
                    from shapequant import gguf_ref
                    he = gguf_ref.GGUFSource(gguf_ref.resolve(run)).head_and_embed(*model.lm_head.weight.shape)
                    dm.update(scorer.score_deploy(model, [{"name": "own", "head_w": he["head"], "embed_w": he["embed"]}]))
                    dm.update({k: he[k] for k in ("head_type", "embed_type", "head_bits", "embed_bits")})
                    metrics.update(dm)
                    names.append("own")
                log(f"  [{row_name}] deployed: " + "  ".join(f"{n} KL {dm[f'deploy_{n}_kl']:.5f}" for n in names))
            if problems is not None and run.get("gsm8k", True):
                g = run_gsm8k(model, tok, problems, gcfg, out_dir / "gsm8k" / f"{row_name}.jsonl")
                log(f"  [{row_name}] GSM8K {g['gsm8k_acc']:.1f}% ({g['gsm8k_seconds']:.0f}s)")
                metrics.update(g)
            # standard benchmarks (lm-eval): rows of runs with lm_eval: true; for fine-tuned
            # stacks only the fine-tuned row
            if lcfg.get("tasks") and run.get("lm_eval") and (ft or not run.get("finetune")):
                from shapequant.lmeval import run_lm_eval
                log(f"  [{row_name}] lm-eval: {[t['task'] for t in lcfg['tasks']]}")
                metrics.update(run_lm_eval(model, tok, lcfg, out_dir / "lm_eval" / f"{row_name}.json", log=log))
            row = {
                "stage": run.get("stage", cfg.get("stage")), "run_name": row_name, "method": run["method"],
                "model": cfg["model"], "d": run.get("d"), "K": run.get("K"),
                "shuffle": run.get("shuffle", False), "masks": run.get("M", 1) > 1 if run["method"] in ("masked", "stack") else run.get("masks", False),
                "M": run.get("M"), "G": run.get("G"), "alloc": run.get("alloc"),
                "rotate": run.get("rotate"), "ldlq": bool(run.get("ldlq")),
                "finetune": ft,
                "weighting": ("act_row" if run["method"] in ("masked", "stack") else run.get("weighting", "none"))
                if run["method"] in ("codebook", "masked", "stack") else None,
                "start": run.get("start", "q4") if run["method"] == "codebook" else "fp16",
                "source": ((run.get("path") or f"{run['repo']}/{run['file']}") if run["method"] == "gguf"
                           else run["repo"] if run["method"] == "vptq" else None),
                "seed": seed, **stats, **extra, **metrics,
            }
            rows[row_name] = row
            (out_dir / "runs" / f"{row_name}.json").write_text(json.dumps(row, indent=2, default=str))
            append_row(csv_path, row)
            make_plots(csv_path, out_dir)
            return row

        evaluate(name, False, {})
        if run.get("export") and not run.get("finetune"):
            export_packed(fits, out_dir / "packed" / f"{name}.safetensors", cfg, name, log, model)
        if run.get("finetune"):
            from transformers import AutoModelForCausalLM as _AM
            from shapequant.finetune import finetune
            log(f"  fine-tuning codebooks + row scales: {run['finetune']}")
            # teacher: the original model sharing the student's memory (finetune.SharedTeacher),
            # or teacher: separate for a second loaded copy (the old way; same result)
            teacher = (_AM.from_pretrained(cfg["model"], dtype=dtype).to(args.device).eval()
                       if run["finetune"].get("teacher") == "separate" else None)
            d_by_name = {n: run["d"] for n in fits}
            torch.cuda.reset_peak_memory_stats()
            ft_stats = finetune(model, teacher, fits, d_by_name, seqs, run["finetune"], log=log, originals=originals)
            ft_stats["ft_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
            log(f"  fine-tune peak GPU memory {ft_stats['ft_peak_gb']:.1f} GB")
            if run.get("export"):
                export_packed(fits, out_dir / "packed" / f"{name}_ft.safetensors", cfg, f"{name}_ft", log, model)
            del teacher, fits
            torch.cuda.empty_cache()
            log(f"  fine-tune KL {ft_stats['ft_kl_first']:.4f} -> {ft_stats['ft_kl_last']:.4f}")
            evaluate(f"{name}_ft", True, ft_stats)

    if cfg.get("stage0_checks"):
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
