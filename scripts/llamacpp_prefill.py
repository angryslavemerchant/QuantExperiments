"""Prefill (prompt processing) and decode-at-depth speed of a llama.cpp llama-server, from the server's
own /completion timings: exact-length token-id prompts (no prompt cache), n_predict 1 for prefill,
then n_predict 128 on the same prompt for decode speed with that much context.

  python scripts/llamacpp_prefill.py --endpoint http://127.0.0.1:8100 --lengths 8192,32768,65536 --out x.json

Filler text: this repo's Markdown + Python sources, tokenized by the server (/tokenize) and repeated
as needed; prompt-processing speed does not depend on what the text says.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


def post(url: str, body: dict, timeout: float = 3600) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--lengths", default="8192,32768,65536")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--decode", type=int, default=128)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    text = "\n\n".join(p.read_text(encoding="utf-8", errors="ignore")
                       for pat in ("*.md", "shapequant/**/*.py", "scripts/*.py") for p in sorted(root.glob(pat)))
    toks = post(f"{a.endpoint}/tokenize", {"content": text})["tokens"]
    res = {"endpoint": a.endpoint, "runs": []}
    for L in map(int, a.lengths.split(",")):
        ids = (toks * (L // len(toks) + 1))[:L]
        for rep in range(a.repeats):
            t0 = time.time()
            r = post(f"{a.endpoint}/completion", {"prompt": ids, "n_predict": 1, "cache_prompt": False, "temperature": 0})
            tm = r["timings"]
            run = {"tokens": L, "rep": rep, "wall_s": time.time() - t0, "prompt_n": tm["prompt_n"],
                   "prompt_tps": tm["prompt_per_second"]}
            if rep == a.repeats - 1 and a.decode:
                r = post(f"{a.endpoint}/completion", {"prompt": ids, "n_predict": a.decode, "cache_prompt": False,
                                                       "temperature": 0, "ignore_eos": True})
                run["decode_tps"] = r["timings"]["predicted_per_second"]
            res["runs"].append(run)
            print(f"{L:6d} tokens rep {rep}: prefill {run['prompt_tps']:7.0f} tok/s ({run['wall_s']:.1f}s wall)"
                  + (f", decode at this depth {run['decode_tps']:.1f} tok/s" if "decode_tps" in run else ""), flush=True)
            Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
