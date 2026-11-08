"""Sweep the standalone runtime's decode launch configs on the current GPU (batch-1 tg, CUDA graph).

engine_from_packed launches each run-kernel matrix with (threads, blocks_per_sm) = `small` for
R <= 4096 and `large` above (threads 0 = one block per SM). The defaults were picked on the PRO
6000 / 5090; this times every (small, large) pair in the grid on this device and writes the
ranking to --out (best first).

  python scripts/tune_decode.py --packed PATH --out artifacts/x/tune_decode.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SMALL = [(0, 1), (256, 2), (512, 1), (256, 4)]
LARGE = [(512, 1), (256, 2), (512, 2), (1024, 1), (256, 4)]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tg", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=1024)
    a = ap.parse_args()
    import torch
    from shapequant import qwen35_runtime as q35
    res = {"gpu": torch.cuda.get_device_name(0), "sms": torch.cuda.get_device_properties(0).multi_processor_count,
           "runs": []}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def one(small, large):
        eng, _ = q35.engine_from_packed(a.packed, max_len=a.max_len, small=small, large=large)
        eng.capture()
        s = eng.time_tokens(a.tg, 3)
        del eng
        gc.collect()
        torch.cuda.empty_cache()
        return 1.0 / s

    # large matrices dominate decode time: sweep `large` at the default small, then `small` at the best large
    for large in LARGE:
        try:
            r = {"small": SMALL[0], "large": large, "tok_s": one(SMALL[0], large)}
        except Exception as e:  # noqa: BLE001 - a config may not launch on this device
            r = {"small": SMALL[0], "large": large, "error": repr(e)[:300]}
        res["runs"].append(r)
        log(r)
        out.write_text(json.dumps(res, indent=1))
    ok = [r for r in res["runs"] if "tok_s" in r]
    if not ok:
        return 1
    best_large = tuple(max(ok, key=lambda r: r["tok_s"])["large"])
    for small in SMALL[1:]:
        try:
            r = {"small": small, "large": best_large, "tok_s": one(small, best_large)}
        except Exception as e:  # noqa: BLE001
            r = {"small": small, "large": best_large, "error": repr(e)[:300]}
        res["runs"].append(r)
        log(r)
        out.write_text(json.dumps(res, indent=1))
    res["runs"].sort(key=lambda r: -r.get("tok_s", 0))
    res["best"] = res["runs"][0]
    out.write_text(json.dumps(res, indent=1))
    log(f"best: {res['best']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
