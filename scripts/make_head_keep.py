"""Token ids the lm_head keeps when pruning CJK output (SHAPEQUANT_HEAD_KEEP=<this .npy>).

A token is dropped when its text contains a CJK ideograph, kana, Hangul, Bopomofo, CJK
punctuation/symbols or a fullwidth form. Kept: everything else (ASCII, Latin, Greek, Cyrillic,
math, emoji, ...), every added/special token, and byte-level pieces that are not valid UTF-8 on
their own (they spell many scripts, emoji included). Token ids are unchanged: the head only loses
rows, and the plugin writes -inf into the dropped ids' logits.

  python scripts/make_head_keep.py ~/models/q27-sq-vl [--out head_keep_nocjk.npy]
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

CJK = re.compile(
    "[ᄀ-ᇿ⺀-⿟　-〿぀-ヿ㄀-ㄯ㄰-㆏㆐-ㇿ"
    "㈀-㏿㐀-䶿一-鿿ꥠ-꥿가-힯ힰ-퟿豈-﫿"
    "︰-﹏＀-￯\U00020000-\U0003ffff]")


def byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2's bytes_to_unicode (byte-level BPE token strings -> bytes)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--out", default="head_keep_nocjk.npy")
    ap.add_argument("--vocab", type=int, help="lm_head rows (default: config vocab_size)")
    a = ap.parse_args()
    d = Path(a.model_dir).expanduser()
    tok = json.loads((d / "tokenizer.json").read_text(encoding="utf-8"))
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    V = a.vocab or cfg.get("text_config", cfg)["vocab_size"]
    dec = byte_decoder()
    vocab = tok["model"]["vocab"]
    added = {t["id"] for t in tok.get("added_tokens", [])}
    drop, partial = set(), 0
    for s, i in vocab.items():
        if i in added:
            continue
        b = bytes(dec[c] for c in s)
        try:
            text = b.decode("utf-8")
        except UnicodeDecodeError:
            text = b.decode("utf-8", errors="ignore")       # complete chars inside a partial piece
            partial += 1
        if CJK.search(text):
            drop.add(i)
    keep = np.array([i for i in range(V) if i not in drop], dtype=np.int64)
    out = d / a.out
    np.save(out, keep)
    print(f"vocab rows {V}, tokenizer entries {len(vocab)} + {len(added)} added; partial-UTF-8 pieces {partial}")
    print(f"dropped {len(drop)} CJK tokens; keeping {len(keep)} rows -> {out}")
    ex = sorted(drop)[:: max(1, len(drop) // 12)][:12]
    inv = {i: s for s, i in vocab.items()}
    print("dropped e.g.:", [bytes(dec[c] for c in inv[i]).decode("utf-8", "replace") for i in ex])


if __name__ == "__main__":
    main()
