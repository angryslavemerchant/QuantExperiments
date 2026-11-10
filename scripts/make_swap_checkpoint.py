"""Layer-swap checkpoint: the exported shapequant checkpoint plus the original bf16 weights of the packed
linears of chosen decoder layers, so those layers run unquantized (serve/score with
SHAPEQUANT_BF16_LAYERS set to the same layers; plugin._bf16_layer).

  python scripts/make_swap_checkpoint.py --base ~/models/q27-sq-vl --layers 3,17 --out ~/models/swap_3_17
  SHAPEQUANT_BF16_LAYERS=3,17 MODEL_DIR=~/models/swap_3_17 ... deploy/serve.sh

Everything of the base directory is symlinked; bf16 weights come from the HF cache of Qwen/Qwen3.8-27B
(only the shards holding the chosen layers are read).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


def parse_layers(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        if ":" in part:
            a, b = part.split(":")
            out += list(range(int(a), int(b)))
        elif part.strip():
            out.append(int(part))
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="Qwen/Qwen3.8-27B")
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file
    base, out = Path(a.base).expanduser(), Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for f in base.iterdir():
        dst = out / f.name
        if not dst.exists():
            os.symlink(f.resolve(), dst)
    layers = set(parse_layers(a.layers))
    snap = Path(snapshot_download(a.repo, allow_patterns=["model.safetensors.index.json"]))
    wmap = json.load(open(snap / "model.safetensors.index.json"))["weight_map"]
    packed = {"in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj"}
    want = {}
    for name, shard in wmap.items():
        m = re.match(r"model\.language_model\.layers\.(\d+)\.(\w+)\.(\w+)\.weight$", name)
        if m and int(m.group(1)) in layers and m.group(3) in packed:
            want.setdefault(shard, []).append(name)
    shards = Path(snapshot_download(a.repo, allow_patterns=sorted(want)))
    tensors = {}
    for shard, names in want.items():
        with safe_open(shards / shard, "pt") as g:
            for n in names:
                tensors[n] = g.get_tensor(n).contiguous()
    save_file(tensors, str(out / "bf16_swap.safetensors"))
    gb = sum(t.numel() * t.element_size() for t in tensors.values()) / 2**30
    print(f"{out}: {len(tensors)} bf16 tensors for layers {sorted(layers)} ({gb:.2f} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
