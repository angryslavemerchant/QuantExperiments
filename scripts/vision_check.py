"""Image understanding through an OpenAI-compatible endpoint: a synthetic image with known content
(a number, shapes, colors; scored by keyword) and a standard COCO photo (two cats on a couch with
remotes; scored by keyword), thinking off, greedy.

  python scripts/vision_check.py --endpoint http://127.0.0.1:8000 --model ours --out x.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
import urllib.request
from pathlib import Path

COCO = "http://images.cocodataset.org/val2017/000000039769.jpg"


def synthetic_png() -> bytes:
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGB", (640, 480), "white")
    d = ImageDraw.Draw(im)
    d.ellipse((40, 60, 240, 260), fill=(220, 30, 30))            # red circle
    d.rectangle((400, 60, 600, 260), fill=(30, 60, 220))         # blue square
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 120)
    except OSError:
        font = ImageFont.load_default(size=120) if hasattr(ImageFont, "load_default") else None
    d.text((250, 300), "73", fill=(0, 0, 0), font=font)
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def ask(endpoint: str, model: str, img: bytes, mime: str, question: str, api_key: str | None) -> tuple[str, float]:
    url = f"data:{mime};base64," + base64.b64encode(img).decode()
    body = {"model": model, "temperature": 0, "max_tokens": 300, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                      {"type": "text", "text": question}]}]}
    hdr = {"Content-Type": "application/json"}
    if api_key:
        hdr["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(endpoint.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(), headers=hdr)
    t = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return r["choices"][0]["message"]["content"] or "", time.time() - t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", default="default")
    ap.add_argument("--api-key")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cases = [("synthetic", synthetic_png(), "image/png",
              "What number is written in this image, and what shapes and colors do you see? Answer in one or two sentences.",
              ["73", "red", "blue", "circle", "square"])]
    try:
        cases.append(("coco_cats", urllib.request.urlopen(COCO, timeout=60).read(), "image/jpeg",
                      "Describe this image in two sentences: what animals are there, how many, and what objects are near them?",
                      ["cat", "two", "remote"]))
    except Exception as e:  # noqa: BLE001
        print("coco download failed:", e)
    rec = []
    for name, img, mime, q, keys in cases:
        try:
            ans, s = ask(a.endpoint, a.model, img, mime, q, a.api_key)
            hit = [k for k in keys if k in ans.lower()]
            rec.append({"case": name, "answer": ans, "keywords": keys, "hit": hit, "seconds": round(s, 1)})
            print(f"[{name}] {len(hit)}/{len(keys)} keywords in {s:.1f}s: {ans[:300]!r}", flush=True)
        except Exception as e:  # noqa: BLE001
            rec.append({"case": name, "error": repr(e)[:300]})
            print(f"[{name}] error {e!r}", flush=True)
    Path(a.out).write_text(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
