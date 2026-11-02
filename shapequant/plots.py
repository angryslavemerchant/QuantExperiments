"""The two deliverable plots: mean KL vs bits/weight (log KL) and GSM8K vs bits/weight."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MARKERS = {"rtn": ("s", "#4c72b0"), "ternary": ("v", "#c44e52"), "codebook": ("o", "#55a868")}


def _f(v: str):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def make_plots(csv_path: Path, out_dir: Path) -> None:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for metric, fname, ylabel, logy in [
        ("mean_kl", "kl_vs_bits.png", "mean KL(original || compressed)", True),
        ("gsm8k_acc", "gsm8k_vs_bits.png", "GSM8K accuracy (%)", False),
    ]:
        pts = [r for r in rows if _f(r.get(metric)) is not None]
        if not pts:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        seen = set()
        for r in pts:
            y = _f(r[metric])
            if r["method"] == "original":
                if not logy:  # KL of the original is 0; nothing to draw on a log axis
                    ax.axhline(y, color="gray", ls="--", lw=1, label="original bf16")
                continue
            if logy and y <= 0:
                continue
            marker, color = MARKERS.get(r["method"], ("x", "black"))
            label = r["method"] if r["method"] not in seen else None
            seen.add(r["method"])
            x = _f(r["bits_per_weight_rounded"])
            ax.scatter(x, y, marker=marker, color=color, s=50, label=label, zorder=3)
            ax.annotate(r["run_name"], (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("effective bits per weight (compressed layers, rounded)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=130)
        plt.close(fig)
