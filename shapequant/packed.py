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

Optional aux section (metadata key "aux"; files without it still load, with the rest taken from
the base checkpoint): everything else the runtime needs, so the file stands alone (tokenizer aside).

  aux.embed.q        uint8 [V, C/2]   4-bit embedding, block 32 symmetric: n = clamp(round(w/s),
                                      -8, 7) + 8, low nibble = even column
  aux.embed.scale    fp16  [V, C/32]  s = fp16(amax / 7) (an all-zero block stores s = 1)
  aux.lm_head.q      int8  [V, C]     8-bit head (Q8_0-like): n = clamp(round(w/s), -127, 127)
  aux.lm_head.scale  fp16  [V, C/32]  s = fp16(amax / 127)
  aux.dense.<name>   the remaining parameters (norms, biases, ...) under their HF names, as stored

metadata "aux" = {"embed": "q4_b32", "lm_head": "q8_b32", "dense": [names], "config": {HF config}}.
Dequantized values equal evaluate.block_quant(w, 4 | 8) exactly (the formats scored by eval.deploy).
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


def q4_embed_pack(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """w [V, C] -> (q uint8 [V, C/2], scale fp16 [V, C/32]); see the module doc."""
    V, C = w.shape
    q = torch.empty(V, C // 2, dtype=torch.uint8, device=w.device)
    sc = torch.empty(V, C // 32, dtype=torch.float16, device=w.device)
    for s in range(0, V, 8192):
        b = w[s:s + 8192].float().view(-1, C // 32, 32)
        k = (b.abs().amax(-1).clamp_min(1e-8) / 7.0).half().float()
        k = torch.where(k == 0, torch.ones_like(k), k)
        n = ((b / k[..., None]).round().clamp(-8, 7) + 8).to(torch.uint8).view(-1, C // 2, 2)
        q[s:s + 8192] = n[..., 0] | (n[..., 1] << 4)
        sc[s:s + 8192] = k.half()
    return q, sc


def q4_embed_rows(q: torch.Tensor, sc: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """fp32 [len(ids), C] dequantized embedding rows."""
    b = q[ids]
    n = torch.stack([b & 15, b >> 4], -1).view(b.shape[0], -1).float() - 8
    return (n.view(b.shape[0], -1, 32) * sc[ids].float()[..., None]).view(b.shape[0], -1)


def q8_rows(q: torch.Tensor, sc: torch.Tensor, s: int = 0, e: int | None = None) -> torch.Tensor:
    """fp32 rows s:e of an 8-bit block-32 matrix."""
    b = q[s:e]
    return (b.float().view(b.shape[0], -1, 32) * sc[s:e].float()[..., None]).view(b.shape[0], -1)


def aux_from_model(model, block_names) -> tuple[dict, dict]:
    """The aux section for `model` (block_names: the packed block linears, left out of dense)."""
    from .kernels.proto_formats import q8_pack
    body = model.model
    E = body.embed_tokens.weight.detach()
    H = model.lm_head.weight.detach()
    eq, es = q4_embed_pack(E)
    hq, hs, _ = q8_pack(H.float().cpu())
    skip = {f"{n}.weight" for n in block_names} | {"lm_head.weight", "model.embed_tokens.weight"}
    dense = {n: p.detach().cpu().contiguous() for n, p in model.named_parameters() if n not in skip}
    t = {"aux.embed.q": eq.cpu(), "aux.embed.scale": es.cpu(), "aux.lm_head.q": hq, "aux.lm_head.scale": hs}
    t.update({f"aux.dense.{n}": v for n, v in dense.items()})
    meta = {"embed": "q4_b32", "lm_head": "q8_b32", "dense": sorted(dense),
            "config": model.config.to_dict(), "vocab": E.shape[0], "hidden": E.shape[1],
            "tied": bool(getattr(model.config, "tie_word_embeddings", False))}
    return t, meta


def aux_bits(meta: dict) -> dict:
    """Storage bits of the aux parts: embed 4.5 bpw, head 8.5 bpw (fp16 scale per 32)."""
    n = meta["vocab"] * meta["hidden"]
    return {"embed": n * 4.5, "lm_head": n * 8.5}


def save(path, packed: dict[str, tuple[dict, dict]], *, base_model: str, extra: dict | None = None,
         aux: tuple[dict, dict] | None = None) -> dict:
    """Write {name: (tensors, meta)} to one safetensors file; returns the summary.
    aux: (tensors, meta) from aux_from_model, to make the file self-contained."""
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
    md = {"summary": None, "matrices": json.dumps(metas)}
    if aux is not None:
        at, am = aux
        tensors.update(at)
        md["aux"] = json.dumps(am, default=str)
        ab = aux_bits(am)
        dense_bits = sum(v.numel() * v.element_size() * 8 for k, v in at.items() if k.startswith("aux.dense."))
        n_all = numel + 2 * am["vocab"] * am["hidden"] + sum(
            v.numel() for k, v in at.items() if k.startswith("aux.dense."))
        summary["bpw_all"] = (total_bits + ab["embed"] + ab["lm_head"] + dense_bits) / n_all
        summary["aux"] = True
    md["summary"] = json.dumps(summary)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=md)
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
        self.aux = json.loads(md["aux"]) if "aux" in md else None

    def aux_tensor(self, name: str) -> torch.Tensor:
        """aux.<name>, e.g. "embed.q" or "dense.model.norm.weight"."""
        return self.f.get_tensor(f"aux.{name}")

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
