"""Packed on-disk format for the stack quant (codebook + mixture + masks).

One safetensors file holds every compressed block linear; everything else
(embeddings, lm_head, norms, biases) comes from the base checkpoint named in
the metadata. Per matrix `name` (an [R, C] weight, d-dim chunks, G codebooks of
K entries, M column orders):

  {name}.tags       int32 [R, W]   bit-packed local codebook indices, log2(K) bits
                                   each, nch = C/d per row, little-endian bit
                                   stream within each row, rows padded to whole
                                   32-bit words (W = ceil(nch * log2K / 32))
  {name}.table      fp16  [G*K, d] the G codebooks, concatenated
  {name}.row_scale  fp16  [R]      per-row scale (fine-tuned gain included)
  {name}.row_book   uint8 [R]      codebook of each row (only if G > 1)
  {name}.row_mask   uint8 [R]      column order of each row (only if M > 1)

Column orders are not stored: order 0 is the identity and orders 1..M-1 are
regenerated from the matrix seed (masks.seeded_perms); a CRC of the orders is
kept to catch a generator change. Row r decodes as

  u_r[j*d:(j+1)*d] = table[row_book[r]*K + tag[r, j]] * row_scale[r]
  w[r, perm[row_mask[r]][c]] = u_r[c]

i.e. chunk j of row r holds columns perm[j*d], ..., perm[j*d+d-1] in that
row's column order. Rows are stored in their original order; a runtime may
regroup them (by mask, by codebook) at load time.

Storage bits are the real tensor bytes plus 32 bits per matrix for the seed.
"""

from __future__ import annotations

import json
import math
import zlib
from pathlib import Path

import torch

from . import masks as mk

FORMAT = "shapequant-stack-v1"


_ROWS = 64  # rows per step, bounds the bit-expansion memory


def pack_bits(vals: torch.Tensor, bits: int) -> torch.Tensor:
    """[R, n] ints in [0, 2^bits) -> [R, W] int32 (little-endian bit stream per row)."""
    R, n = vals.shape
    W = math.ceil(n * bits / 32)
    if int(vals.min()) < 0 or int(vals.max()) >= (1 << bits):
        raise ValueError(f"values out of range for {bits} bits")
    v = vals.to(torch.int64).cpu()
    out = torch.empty(R, W, dtype=torch.int32)
    bitpos = torch.arange(bits, dtype=torch.int64)
    wordpos = torch.arange(32, dtype=torch.int64)
    for a in range(0, R, _ROWS):
        blk = v[a:a + _ROWS]
        stream = ((blk[..., None] >> bitpos) & 1).reshape(blk.shape[0], n * bits)
        stream = torch.nn.functional.pad(stream, (0, W * 32 - n * bits))
        words = (stream.view(-1, W, 32) << wordpos).sum(-1)
        out[a:a + _ROWS] = torch.where(words >= (1 << 31), words - (1 << 32), words).to(torch.int32)
    return out


def unpack_bits(words: torch.Tensor, bits: int, n: int) -> torch.Tensor:
    """Inverse of pack_bits: [R, W] int32 -> [R, n] int64 (on words' device)."""
    R, W = words.shape
    out = torch.empty(R, n, dtype=torch.int64, device=words.device)
    wordpos = torch.arange(32, device=words.device)
    bitpos = torch.arange(bits, device=words.device)
    for a in range(0, R, _ROWS):
        w = words[a:a + _ROWS].to(torch.int64) & 0xFFFFFFFF
        stream = ((w[..., None] >> wordpos) & 1).reshape(w.shape[0], W * 32)[:, : n * bits]
        out[a:a + _ROWS] = (stream.view(-1, n, bits) << bitpos).sum(-1)
    return out


def perm_crc(perms: torch.Tensor) -> int:
    return zlib.crc32(perms.to(torch.int64).cpu().numpy().tobytes())


def pack_matrix(fit, *, d: int, K: int, G: int, M: int, seed: int) -> tuple[dict, dict]:
    """Tensors + metadata for one compressed matrix. `fit` is a masks.Fit from
    compress_mixture(..., keep_tags=True) (fine-tuned books/scales written back
    into fit.stats['books'] / fit.row_scale, see finetune.write_back)."""
    if fit.tags is None or len(fit.tags) != 1:
        raise ValueError("pack_matrix needs single-stage global tags (keep_tags=True)")
    if K & (K - 1):
        raise ValueError(f"K={K} is not a power of two")
    R = fit.row_scale.shape[0]
    C = fit.perms.shape[1]
    if C % d:
        raise ValueError(f"C={C} not divisible by d={d}")
    nch = C // d
    table = fit.stats["books"][0]
    if table.shape != (G * K, d):
        raise ValueError(f"table shape {tuple(table.shape)} != ({G * K}, {d})")
    gt = fit.tags[0].to(torch.int64).view(R, nch).cpu()
    row_book = gt.div(K, rounding_mode="floor")
    if not torch.all(row_book == row_book[:, :1]):
        raise ValueError("a row's tags span several codebooks")
    row_book = row_book[:, 0]
    local = gt - row_book[:, None] * K
    b = int(math.log2(K))
    t = {"tags": pack_bits(local, b), "table": table.to(torch.float16).cpu().contiguous(),
         "row_scale": fit.row_scale.to(torch.float16).cpu().contiguous()}
    if G > 1:
        t["row_book"] = row_book.to(torch.uint8)
    if M > 1:
        t["row_mask"] = fit.row_mask.cpu().to(torch.uint8)
    perms = mk.seeded_perms(C, M, torch.Generator(device="cpu").manual_seed(seed))
    if not torch.equal(perms, fit.perms.cpu()):
        raise ValueError("fit's column orders do not regenerate from its seed")
    meta = {"R": R, "C": C, "d": d, "K": K, "G": G, "M": M, "seed": seed, "perm_crc": perm_crc(perms),
            "bits": sum(x.numel() * x.element_size() * 8 for x in t.values()) + 32}
    return t, meta


def save(path, packed: dict[str, tuple[dict, dict]], *, base_model: str, extra: dict | None = None) -> dict:
    """Write {name: (tensors, meta)} to one safetensors file; returns the summary."""
    from safetensors.torch import save_file
    tensors, metas = {}, {}
    for name, (t, m) in packed.items():
        for k, v in t.items():
            tensors[f"{name}.{k}"] = v
        metas[name] = m
    total_bits = sum(m["bits"] for m in metas.values())
    numel = sum(m["R"] * m["C"] for m in metas.values())
    summary = {"format": FORMAT, "base_model": base_model, "matrices": len(metas),
               "bits": total_bits, "numel": numel, "bpw": total_bits / numel, **(extra or {})}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata={"summary": json.dumps(summary), "matrices": json.dumps(metas)})
    return summary


class PackedFile:
    """Read access to a packed file: metadata, raw tensors, reference decode."""

    def __init__(self, path):
        from safetensors import safe_open
        self.f = safe_open(str(path), framework="pt", device="cpu")
        md = self.f.metadata()
        self.summary = json.loads(md["summary"])
        if self.summary.get("format") != FORMAT:
            raise ValueError(f"{path}: format {self.summary.get('format')!r} != {FORMAT}")
        self.meta = json.loads(md["matrices"])

    def tensors(self, name: str) -> dict:
        m = self.meta[name]
        keys = ["tags", "table", "row_scale"] + (["row_book"] if m["G"] > 1 else []) \
            + (["row_mask"] if m["M"] > 1 else [])
        return {k: self.f.get_tensor(f"{name}.{k}") for k in keys}

    def perms(self, name: str) -> torch.Tensor:
        m = self.meta[name]
        p = mk.seeded_perms(m["C"], m["M"], torch.Generator(device="cpu").manual_seed(m["seed"]))
        if perm_crc(p) != m["perm_crc"]:
            raise RuntimeError(f"{name}: regenerated column orders do not match the file (torch RNG changed?)")
        return p

    def local_tags(self, name: str, device="cpu") -> torch.Tensor:
        m = self.meta[name]
        return unpack_bits(self.f.get_tensor(f"{name}.tags").to(device), int(math.log2(m["K"])), m["C"] // m["d"])

    def decode(self, name: str, device="cpu") -> torch.Tensor:
        """Reference dense float32 [R, C] weight (original row and column order)."""
        m = self.meta[name]
        t = {k: v.to(device) for k, v in self.tensors(name).items()}
        return decode_tensors(self.local_tags(name, device), t["table"], t["row_scale"],
                              t.get("row_book"), t.get("row_mask"), self.perms(name).to(device), m["K"])


def decode_tensors(local_tags, table, row_scale, row_book, row_mask, perms, K: int) -> torch.Tensor:
    """Reference decode from unpacked parts: local_tags [R, nch] int64, table
    [G*K, d] fp16, row_scale [R], row_book/row_mask [R] or None, perms [M, C]."""
    R, nch = local_tags.shape
    d = table.shape[1]
    C = nch * d
    tags = local_tags if row_book is None else local_tags + row_book.to(torch.int64)[:, None] * K
    u = table.float()[tags].reshape(R, C) * row_scale.float()[:, None]
    idx = perms[row_mask.to(torch.int64)] if row_mask is not None else perms[:1].expand(R, C)
    w = torch.empty_like(u)
    w.scatter_(1, idx, u)
    return w


def apply_to_model(model, path, device=None) -> dict:
    """Write the reference-decoded weights of every packed matrix into `model`."""
    from .pipeline import compressible_linears
    pf = PackedFile(path)
    mods = dict(compressible_linears(model))
    missing = set(mods) - set(pf.meta)
    if missing:
        raise ValueError(f"packed file lacks {len(missing)} block linears, e.g. {sorted(missing)[:3]}")
    with torch.no_grad():
        for name, m in mods.items():
            w = pf.decode(name, device=device or m.weight.device)
            if w.shape != m.weight.shape:
                raise ValueError(f"{name}: packed {tuple(w.shape)} != model {tuple(m.weight.shape)}")
            m.weight.copy_(w.to(m.weight.dtype))
    return pf.summary
