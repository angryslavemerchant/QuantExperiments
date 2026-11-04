"""Load a packed quant into its base model and score it (KL vs the 16-bit model,
same wikitext windows as an experiment config).

Run: python scripts/eval_packed.py --config CFG --packed FILE [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shapequant import packed  # noqa: E402
from shapequant.evaluate import Scorer, load_wikitext_windows  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--packed", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = yaml.safe_load(open(a.config))
    pf = packed.PackedFile(a.packed)
    if pf.summary["base_model"] != cfg["model"]:
        raise SystemExit(f"packed base {pf.summary['base_model']} != config model {cfg['model']}")
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=getattr(torch, cfg["reference_dtype"]))
    model = model.to(a.device).eval()
    windows = load_wikitext_windows(tok, cfg["eval"]["wikitext"])
    scorer = Scorer(model, windows, cfg["eval"]["logit_chunk"])
    summary = packed.apply_to_model(model, a.packed)
    metrics = scorer.score(model)
    print(json.dumps({"summary": summary, "metrics": metrics}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
