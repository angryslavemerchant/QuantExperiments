"""Decode speed with long context already in the prompt, through an OpenAI-compatible server (vLLM /
deploy/serve.sh): streamed /v1/completions with an exact-length token-id prompt (server /tokenize)
followed by "Summary of the above:", 256 generated tokens (ignore_eos, temperature 0).
Prints time to first token (the prefill) and decode tok/s after it.

  python scripts/decode_depth.py --endpoint http://127.0.0.1:8000 --api-key "$SQ_API_KEY" \
      --depths 0,8192,32768,65536 --out x.json

Filler: this repo's Markdown + Python sources, repeated; the same text for every depth.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


def post(url, body, key, stream=False, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    r = urllib.request.urlopen(req, timeout=timeout)
    return r if stream else json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--model")
    ap.add_argument("--depths", default="0,8192,32768,65536")
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--name", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base = a.endpoint.rstrip("/")
    model = a.model or json.load(urllib.request.urlopen(urllib.request.Request(
        f"{base}/v1/models", headers={"Authorization": f"Bearer {a.api_key}"})))["data"][0]["id"]
    root = Path(__file__).resolve().parents[1]
    text = "\n\n".join(p.read_text(encoding="utf-8", errors="ignore")
                       for pat in ("*.md", "shapequant/**/*.py", "scripts/*.py") for p in sorted(root.glob(pat)))
    toks = post(f"{base}/tokenize", {"model": model, "prompt": text}, a.api_key)["tokens"]
    tail = post(f"{base}/tokenize", {"model": model, "prompt": "\n\nSummary of the above, in plain prose:\n",
                                     "add_special_tokens": False}, a.api_key)["tokens"]
    res = {"name": a.name, "model": model, "runs": []}
    for D in map(int, a.depths.split(",")):
        ids = (toks * (D // max(1, len(toks)) + 1))[:D] + tail
        for rep in range(a.repeats):
            body = {"model": model, "prompt": ids, "max_tokens": a.gen, "temperature": 0, "ignore_eos": True,
                    "stream": True, "stream_options": {"include_usage": True}}
            t0 = time.time()
            first = None
            n = 0
            with post(f"{base}/v1/completions", body, a.api_key, stream=True) as resp:
                for line in resp:
                    line = line.decode().strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    d = json.loads(line[5:])
                    if d.get("usage"):
                        n = d["usage"]["completion_tokens"]
                    if d.get("choices") and d["choices"][0].get("text") and first is None:
                        first = time.time()
            t1 = time.time()
            ttft = (first or t1) - t0
            dec = (n - 1) / (t1 - first) if first and n > 1 else 0.0
            run = {"depth": D, "rep": rep, "ttft_s": round(ttft, 2), "decode_tps": round(dec, 1), "tokens": n}
            res["runs"].append(run)
            print(f"[{a.name}] depth {D:6d} rep {rep}: TTFT {ttft:6.1f}s, decode {dec:6.1f} tok/s ({n} tokens)", flush=True)
            Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
