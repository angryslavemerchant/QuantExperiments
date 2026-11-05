"""Apply a compression method to every linear layer inside the transformer blocks,
write decoded bf16 weights back into the model, and account memory by formula."""

from __future__ import annotations

import math
import time

import torch

from . import codebook as cb
from . import gguf_ref, vptq_ref
from . import masks as mk
from . import memory
from .rtn import rtn_quantize, ternary_quantize

BLOCK_LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def compressible_linears(model) -> list[tuple[str, torch.nn.Linear]]:
    """Attention Q/K/V/O and MLP gate/up/down in every block. Embedding, lm_head,
    norms and biases are left alone."""
    found = []
    for name, mod in model.model.layers.named_modules():
        if isinstance(mod, torch.nn.Linear) and name.split(".")[-1] in BLOCK_LINEARS:
            found.append((f"model.layers.{name}", mod))
    expected = len(BLOCK_LINEARS) * len(model.model.layers)
    if len(found) != expected:
        raise RuntimeError(f"expected {expected} block linears, found {len(found)}")
    return found


def snapshot_originals(model) -> dict[str, torch.Tensor]:
    return {n: m.weight.detach().to("cpu", copy=True) for n, m in compressible_linears(model)}


def restore_originals(model, originals: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for n, m in compressible_linears(model):
            m.weight.copy_(originals[n].to(m.weight.device))


def unique_param_count(model) -> int:
    seen, total = set(), 0
    for p in model.parameters():  # parameters() already de-duplicates tied weights
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    return total


WEIGHTINGS = {"none": (False, False), "act": (True, False), "row": (False, True), "act_row": (True, True)}


def random_orthogonal(n: int, seed: int, device) -> torch.Tensor:
    """Seeded Haar-random orthogonal [n, n] (QR of a Gaussian, sign-corrected). Stands in
    for a randomized Hadamard transform in quality screens (same incoherence effect)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn(n, n, generator=g, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r))[None, :]
    return q.to(torch.float32)


def compress_matrix(w: torch.Tensor, run: dict, kcfg: dict, seed: int, name: str, source=None,
                    importance: dict | None = None, hessians: dict | None = None):
    """Returns (decoded float32 matrix, rounded bits, ideal bits, extra stats)."""
    R, C = w.shape
    method = run["method"]
    if method in ("gguf", "vptq"):
        out, bits, qtype = source.matrix(name, (R, C))
        out = out.to(w.device)
        rel = float(((out - w) ** 2).sum() / (w ** 2).sum())
        limit = gguf_ref.MAX_REL_MSE if method == "gguf" else vptq_ref.MAX_REL_MSE
        if rel > limit:
            raise RuntimeError(f"{name}: {method} rel MSE {rel:.3f} > {limit}; tensor mapping is wrong")
        return out, bits, bits, {"ggml_type": qtype}
    if method == "rtn":
        out = rtn_quantize(w, run["bits"], run.get("group", 32))
        rb, ib = memory.rtn_bits(R, C, run["bits"], run.get("group", 32))
        return out, rb, ib, {}
    if method == "ternary":
        out = ternary_quantize(w, run.get("group", 32))
        rb, ib = memory.ternary_bits(R, C, run.get("group", 32))
        return out, rb, ib, {}
    if method == "masked":
        # Stage 3/4 recipe: importance-weighted (s_r^2 h_c) codebook with M
        # seeded random column orders per matrix; fitted from the 16-bit weights.
        if importance is None or name not in importance:
            raise RuntimeError(f"{name}: masked method needs calibration statistics")
        M = run.get("M", 16)
        fit = mk.compress_masked(
            w, importance[name].to(w.device, torch.float32), run["d"], run["K"],
            M=M, init="identity+random" if M > 1 else "identity",
            rounds=run.get("rounds", 3), select_frac=run.get("select_frac", 0.25),
            iters=kcfg["iters"], max_fit_chunks=kcfg["max_fit_chunks"],
            seed=cb.layer_seed(seed, name), batch=kcfg["assign_batch"],
            keep_tags=bool(run.get("finetune")),
        )
        bits = float(sum(fit.bits.values()))
        out = fit.decoded
        fit.decoded = None
        usage = fit.stats["mask_usage"]
        return out, bits, bits, {"fit": fit, "mask_nonidentity": 1.0 - usage[0] / max(1, sum(usage))}
    if method == "stack":
        # Stage 4 full stack: importance-weighted codebook fitted from the 16-bit
        # weights, with an optional per-matrix K (allocation, run["_K_by_name"]),
        # G codebooks per matrix (mixture) and M seeded column orders (masks),
        # rows choosing on all their chunks unless select_frac < 1.
        if importance is None or name not in importance:
            raise RuntimeError(f"{name}: stack method needs calibration statistics")
        K = run.get("_K_by_name", {}).get(name, run.get("K"))
        if isinstance(K, (list, tuple)):
            # additive codebook: codeword = sum of one entry per book (kernel-
            # friendly tiny books); masks as seeded orders, rows choose on all chunks
            if run.get("G", 1) != 1:
                raise ValueError("additive books do not combine with the mixture (G must be 1)")
            Kt = tuple(int(k) for k in K)
            M = run.get("M", 1)
            fit = mk.compress_masked(
                w, importance[name].to(w.device, torch.float32), run["d"], Kt, M=M,
                init="identity+random" if M > 1 else "identity", rounds=run.get("rounds", 2),
                select_frac=run.get("select_frac", 1.0), iters=kcfg["iters"],
                refit_iters=kcfg.get("refit_iters", 10), max_fit_chunks=kcfg["max_fit_chunks"],
                seed=cb.layer_seed(seed, name), batch=kcfg["assign_batch"],
                keep_tags=bool(run.get("finetune") or run.get("export")),
            )
            fit.stats["pack"] = {"d": run["d"], "K": Kt, "G": 1, "M": M, "seed": cb.layer_seed(seed, name)}
            bits = float(sum(fit.bits.values()))
            out = fit.decoded
            fit.decoded = None
            usage = fit.stats["mask_usage"]
            return out, bits, bits, {"fit": fit, "mask_nonidentity": 1.0 - usage[0] / max(1, sum(usage)),
                                     "log2K": sum(math.log2(k) for k in Kt), "numel": R * C}
        # QTIP/QuIP#-style options: rotate ("in": W V^T, "both": U W V^T with seeded random
        # orthogonal U, V; incoherence processing) and/or LDLQ error feedback (full H).
        rot, use_ldlq = run.get("rotate"), bool(run.get("ldlq"))
        h = importance[name].to(w.device, torch.float32)
        H = Ur = Vr = None
        w_fit = w
        if rot or use_ldlq:
            if hessians is None or name not in hessians:
                raise RuntimeError(f"{name}: rotate/ldlq need full input Hessians")
            H = hessians[name].to(w.device, torch.float32)
        if rot:
            if rot not in ("in", "both"):
                raise ValueError(f"rotate must be 'in' or 'both', got {rot!r}")
            lseed = cb.layer_seed(seed, name)
            Vr = random_orthogonal(C, lseed + 1, w.device)
            w_fit = w @ Vr.T
            if rot == "both":
                Ur = random_orthogonal(R, lseed + 2, w.device)
                w_fit = Ur @ w_fit
            H = Vr @ H @ Vr.T
            H = 0.5 * (H + H.T)
            h = torch.diag(H).clamp_min(0).contiguous()
        if run.get("export") and rot:
            raise ValueError("packed export does not carry rotations")
        fit = mk.compress_mixture(
            w_fit, h, run["d"], K,
            G=run.get("G", 1), M=run.get("M", 1), rounds=run.get("rounds", 2),
            select_frac=run.get("select_frac", 1.0), joint=run.get("joint"),
            iters=kcfg["iters"], refit_iters=kcfg.get("refit_iters", 10),
            max_fit_chunks=kcfg["max_fit_chunks"], seed=cb.layer_seed(seed, name),
            batch=kcfg["assign_batch"], keep_tags=bool(run.get("finetune") or run.get("export")),
            H=H if use_ldlq else None, ldlq_damp=run.get("ldlq_damp", 0.01),
        )
        fit.stats["pack"] = {"d": run["d"], "K": K, "G": run.get("G", 1), "M": run.get("M", 1),
                             "seed": cb.layer_seed(seed, name)}
        bits = float(sum(fit.bits.values()))
        out = fit.decoded
        fit.decoded = None
        if rot:
            out = out @ Vr
            if Ur is not None:
                out = Ur.T @ out
            fit.stats["rot_in"], fit.stats["rot_out"] = Vr, Ur
        usage = fit.stats["mask_usage"]
        return out, bits, bits, {"fit": fit, "mask_nonidentity": 1.0 - usage[0] / max(1, sum(usage)),
                                 "log2K": math.log2(K), "numel": R * C}
    if method == "codebook":
        src = w if run.get("start", "q4") == "fp16" else rtn_quantize(w, 4, 32)
        use_act, use_row = WEIGHTINGS[run.get("weighting", "none")]
        if use_act and (importance is None or name not in importance):
            raise RuntimeError(f"{name}: weighting needs calibration statistics")
        cm, stats = cb.compress(
            src, run["d"], run["K"],
            iters=kcfg["iters"], max_fit_chunks=kcfg["max_fit_chunks"],
            seed=cb.layer_seed(seed, name), batch=kcfg["assign_batch"],
            col_weight=importance[name] if use_act else None, row_weighted=use_row,
        )
        out = cb.decode(cm)
        rb, ib = memory.codebook_bits(R, C, run["d"], run["K"])
        return out, rb, ib, stats
    raise ValueError(f"unknown method {method!r}")


def apply_run(model, originals: dict[str, torch.Tensor], run: dict, kcfg: dict, seed: int,
              log=print, importance: dict | None = None, hessians: dict | None = None) -> dict:
    """Restore originals, then compress every block linear per `run`. Returns stats."""
    restore_originals(model, originals)
    linears = compressible_linears(model)
    n_compressed = sum(m.weight.numel() for _, m in linears)
    n_total = unique_param_count(model)
    original_bits = 16 * n_total
    other_bits = 16 * (n_total - n_compressed)

    if run["method"] == "original":
        return {
            "bits_per_weight_rounded": 16.0, "bits_per_weight_ideal": 16.0,
            "total_mb": original_bits / 8e6, "pct_of_original": 100.0,
            "recon_mse": 0.0, "recon_rel_mse": 0.0, "codebook_usage": None,
            "compress_seconds": 0.0,
        }

    t0 = time.time()
    rbits = ibits = 0.0
    se = sq = 0.0
    hse = hsq = 0.0
    usage = []
    qtypes: dict[str, int] = {}
    fits: dict = {}
    nonid = []
    log2k = []
    source = None
    if run["method"] == "gguf":
        source = gguf_ref.GGUFSource(gguf_ref.resolve(run))
    elif run["method"] == "vptq":
        source = vptq_ref.VPTQSource(run["repo"], run.get("revision"))
    with torch.no_grad():
        for i, (name, mod) in enumerate(linears):
            w = originals[name].to(mod.weight.device, torch.float32)
            out, rb, ib, stats = compress_matrix(w, run, kcfg, seed, name, source, importance, hessians)
            if hessians is not None and name in hessians:   # layer-output error under the full H
                Hd = hessians[name].to(w.device, torch.float64)
                e = (out - w).to(torch.float64)
                w64 = w.to(torch.float64)
                hse += float(((e @ Hd) * e).sum())
                hsq += float(((w64 @ Hd) * w64).sum())
                del Hd
            if "fit" in stats:
                f = stats.pop("fit")
                if run.get("finetune") or run.get("export"):
                    fits[name] = f
                nonid.append(stats.pop("mask_nonidentity"))
            if "log2K" in stats:
                log2k.append((stats.pop("log2K"), stats.pop("numel")))
            if "ggml_type" in stats:
                qtypes[stats["ggml_type"]] = qtypes.get(stats["ggml_type"], 0) + w.numel()
            rbits += rb
            ibits += ib
            se += float(((out - w) ** 2).sum(dtype=torch.float64))
            sq += float((w ** 2).sum(dtype=torch.float64))
            if "codebook_usage" in stats:
                usage.append(stats["codebook_usage"])
            mod.weight.copy_(out.to(mod.weight.dtype))
            if run["method"] in ("codebook", "gguf", "vptq", "masked", "stack") and (i + 1) % 14 == 0:
                log(f"  compressed {i + 1}/{len(linears)} matrices ({time.time() - t0:.0f}s)")
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    return {
        "bits_per_weight_rounded": rbits / n_compressed,
        "bits_per_weight_ideal": ibits / n_compressed,
        "total_mb": (rbits + other_bits) / 8e6,
        "pct_of_original": 100.0 * (rbits + other_bits) / original_bits,
        "recon_mse": se / n_compressed,
        "recon_rel_mse": se / sq,
        **({"out_rel_err_fullH": hse / hsq} if hsq > 0 else {}),
        "codebook_usage": (sum(usage) / len(usage)) if usage else None,
        "compress_seconds": time.time() - t0,
        **({"ggml_type_fractions": {k: v / n_compressed for k, v in sorted(qtypes.items())}} if qtypes else {}),
        **({"mask_nonidentity_frac": sum(nonid) / len(nonid)} if nonid else {}),
        **({"mean_log2K": sum(a * b for a, b in log2k) / sum(b for _, b in log2k)} if log2k else {}),
        **({"_fits": fits} if fits else {}),
    }
