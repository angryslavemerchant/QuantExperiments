"""Apply a compression method to every linear layer inside the transformer blocks,
write decoded bf16 weights back into the model, and account memory by formula."""

from __future__ import annotations

import time

import torch

from . import codebook as cb
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


def compress_matrix(w: torch.Tensor, run: dict, kcfg: dict, seed: int, name: str):
    """Returns (decoded float32 matrix, rounded bits, ideal bits, extra stats)."""
    R, C = w.shape
    method = run["method"]
    if method == "rtn":
        out = rtn_quantize(w, run["bits"], run.get("group", 32))
        rb, ib = memory.rtn_bits(R, C, run["bits"], run.get("group", 32))
        return out, rb, ib, {}
    if method == "ternary":
        out = ternary_quantize(w, run.get("group", 32))
        rb, ib = memory.ternary_bits(R, C, run.get("group", 32))
        return out, rb, ib, {}
    if method == "codebook":
        src = w if run.get("start", "q4") == "fp16" else rtn_quantize(w, 4, 32)
        cm, stats = cb.compress(
            src, run["d"], run["K"],
            iters=kcfg["iters"], max_fit_chunks=kcfg["max_fit_chunks"],
            seed=cb.layer_seed(seed, name), batch=kcfg["assign_batch"],
        )
        out = cb.decode(cm)
        rb, ib = memory.codebook_bits(R, C, run["d"], run["K"])
        return out, rb, ib, stats
    raise ValueError(f"unknown method {method!r}")


def apply_run(model, originals: dict[str, torch.Tensor], run: dict, kcfg: dict, seed: int,
              log=print) -> dict:
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
    usage = []
    with torch.no_grad():
        for i, (name, mod) in enumerate(linears):
            w = originals[name].to(mod.weight.device, torch.float32)
            out, rb, ib, stats = compress_matrix(w, run, kcfg, seed, name)
            rbits += rb
            ibits += ib
            se += float(((out - w) ** 2).sum(dtype=torch.float64))
            sq += float((w ** 2).sum(dtype=torch.float64))
            if "codebook_usage" in stats:
                usage.append(stats["codebook_usage"])
            mod.weight.copy_(out.to(mod.weight.dtype))
            if run["method"] == "codebook" and (i + 1) % 14 == 0:
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
        "codebook_usage": (sum(usage) / len(usage)) if usage else None,
        "compress_seconds": time.time() - t0,
    }
