"""Download-speed probe run on the rented box before any setup.

Fetches a byte range of a large public file (by default a Qwen2.5-7B shard
from the Hugging Face Hub, the run's biggest download) and fails fast when
the host's real throughput is far below what the job needs. Exit 0 = fast
enough, 3 = too slow, 4 = probe could not connect at all.

Standard library only: it runs before requirements-remote.txt is installed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

DEFAULT_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/main/"
    "model-00001-of-00004.safetensors"
)


def probe(url: str, max_bytes: int, max_seconds: float, chunk: int = 1 << 20) -> dict:
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{max_bytes - 1}"})
    token = os.environ.get("HF_TOKEN", "")
    if token and urllib.parse.urlsplit(url).hostname == "huggingface.co":
        # Unredirected: sent to huggingface.co only, never to the CDN it redirects to.
        req.add_unredirected_header("Authorization", f"Bearer {token}")
    t0 = time.monotonic()
    got = 0
    with urllib.request.urlopen(req, timeout=min(30.0, max_seconds)) as resp:
        t_first = time.monotonic()
        while got < max_bytes and time.monotonic() - t0 < max_seconds:
            block = resp.read(min(chunk, max_bytes - got))
            if not block:
                break
            got += len(block)
    elapsed = max(time.monotonic() - t_first, 1e-6)
    return {"bytes": got, "seconds": round(elapsed, 2),
            "connect_seconds": round(t_first - t0, 2),
            "mb_per_s": round(got / elapsed / 1e6, 2)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--max-mb", type=float, default=256)
    ap.add_argument("--max-seconds", type=float, default=30)
    ap.add_argument("--min-mb-per-s", type=float, default=25)
    args = ap.parse_args(argv)
    try:
        result = probe(args.url, int(args.max_mb * 1e6), args.max_seconds)
    except Exception as exc:  # noqa: BLE001 - any failure means "host unusable"
        print(json.dumps({"event": "net_probe", "ok": False, "error": repr(exc)[:300]}))
        return 4
    ok = result["mb_per_s"] >= args.min_mb_per_s
    print(json.dumps({"event": "net_probe", "ok": ok, "min_mb_per_s": args.min_mb_per_s, **result}))
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
