"""One-command check that a running deploy/serve.sh server works end to end (stdlib only, so it runs
from any machine with Python 3, e.g. the home PC through a tunnel):

  python scripts/smoke_server.py --endpoint http://127.0.0.1:8000 --api-key "$SQ_API_KEY"

Checks, each printed PASS / FAIL / SKIP with timings:
  models     GET /v1/models lists the served model (and a request without the key is refused)
  chat       a short chat completion (thinking off) answers 7 x 8
  stream     the same as a streamed response (server-sent events): first-token time, tokens/s
  thinking   a reasoning request returns reasoning and an answer (Qwen3 reasoning parser)
  image      a generated PNG (red circle, number) is described, if the server takes images
Exit code 0 when nothing FAILed.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
import urllib.error
import urllib.request
import zlib

RESULTS: list[tuple[str, str, str]] = []


def req(url, body=None, key=None, stream=False, timeout=300):
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    r = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None, headers=h)
    resp = urllib.request.urlopen(r, timeout=timeout)
    return resp if stream else json.load(resp)


def record(name, ok, detail):
    status = "PASS" if ok is True else ("SKIP" if ok is None else "FAIL")
    RESULTS.append((name, status, detail))
    print(f"[{status}] {name:8s} {detail}", flush=True)


def png_red_circle() -> bytes:
    """160x120 RGB PNG: white, a red disc on the left, a blue square on the right (no PIL needed)."""
    w, h = 160, 120
    rows = []
    for y in range(h):
        row = bytearray([0])
        for x in range(w):
            if (x - 45) ** 2 + (y - 60) ** 2 < 30 ** 2:
                row += bytes((220, 30, 30))
            elif 95 <= x <= 145 and 35 <= y <= 85:
                row += bytes((30, 60, 220))
            else:
                row += bytes((255, 255, 255))
        rows.append(bytes(row))

    def chunk(t, d):
        return len(d).to_bytes(4, "big") + t + d + zlib.crc32(t + d).to_bytes(4, "big")
    ihdr = w.to_bytes(4, "big") + h.to_bytes(4, "big") + bytes((8, 2, 0, 0, 0))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b"")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000")
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--model", help="served model name (default: the first one listed)")
    a = ap.parse_args()
    base = a.endpoint.rstrip("/")

    try:
        models = req(f"{base}/v1/models", key=a.api_key)
        ids = [m["id"] for m in models.get("data", [])]
        model = a.model or ids[0]
        try:
            req(f"{base}/v1/models")
            noauth = "WARNING: answered without the API key"
            ok = False
        except urllib.error.HTTPError as e:
            noauth, ok = f"no key -> HTTP {e.code}", e.code == 401
        record("models", ok and model in ids, f"{ids}; {noauth}")
    except Exception as e:  # noqa: BLE001
        record("models", False, f"{type(e).__name__}: {e}")
        return 1

    msgs = [{"role": "user", "content": "What is 7 times 8? Reply with only the number."}]
    off = {"chat_template_kwargs": {"enable_thinking": False}}
    try:
        t0 = time.time()
        r = req(f"{base}/v1/chat/completions", {"model": model, "messages": msgs, "max_tokens": 16, "temperature": 0, **off},
                key=a.api_key)
        text = r["choices"][0]["message"]["content"] or ""
        record("chat", "56" in text, f"{text.strip()!r} in {time.time() - t0:.1f}s")
    except Exception as e:  # noqa: BLE001
        record("chat", False, f"{type(e).__name__}: {e}")

    try:
        t0, first, n, text = time.time(), None, 0, ""
        body = {"model": model, "temperature": 0, "max_tokens": 200, "stream": True, **off,
                "messages": [{"role": "user", "content": "Count from 1 to 40, separated by spaces."}]}
        with req(f"{base}/v1/chat/completions", body, key=a.api_key, stream=True) as resp:
            for line in resp:
                line = line.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                d = json.loads(line[5:])
                delta = (d.get("choices") or [{}])[0].get("delta", {}).get("content")
                if delta:
                    first = first or time.time()
                    n += 1
                    text += delta
        dt = time.time() - (first or t0)
        record("stream", n > 5 and "40" in text,
               f"{n} chunks, first token {((first or t0) - t0):.2f}s, {n / dt if dt > 0 else 0:.0f} chunks/s")
    except Exception as e:  # noqa: BLE001
        record("stream", False, f"{type(e).__name__}: {e}")

    try:
        t0 = time.time()
        r = req(f"{base}/v1/chat/completions", {"model": model, "max_tokens": 1024, "temperature": 0.6,
                "messages": [{"role": "user", "content": "Is 221 prime? Answer yes or no, then one line why."}]},
                key=a.api_key)
        m = r["choices"][0]["message"]
        reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
        ans = (m.get("content") or "").strip()
        record("thinking", bool(reasoning) and "no" in ans.lower(),
               f"{len(reasoning)} reasoning chars, answer {ans[:60]!r} in {time.time() - t0:.1f}s")
    except Exception as e:  # noqa: BLE001
        record("thinking", False, f"{type(e).__name__}: {e}")

    try:
        url = "data:image/png;base64," + base64.b64encode(png_red_circle()).decode()
        t0 = time.time()
        r = req(f"{base}/v1/chat/completions", {"model": model, "max_tokens": 60, "temperature": 0, **off,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": url}},
                    {"type": "text", "text": "What shapes and colors are in this image? One sentence."}]}]},
                key=a.api_key)
        text = (r["choices"][0]["message"]["content"] or "").lower()
        record("image", "red" in text and "blue" in text, f"{text.strip()[:90]!r} in {time.time() - t0:.1f}s")
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="ignore")[:160]
        if e.code == 400 and ("image" in body.lower() or "multimodal" in body.lower() or "limit" in body.lower()):
            record("image", None, "server takes no images (text-only checkpoint)")
        else:
            record("image", False, f"HTTP {e.code}: {body}")
    except Exception as e:  # noqa: BLE001
        record("image", False, f"{type(e).__name__}: {e}")

    fails = [n for n, s, _ in RESULTS if s == "FAIL"]
    print(f"\n{'ALL OK' if not fails else 'FAILED: ' + ', '.join(fails)} ({len(RESULTS)} checks)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
