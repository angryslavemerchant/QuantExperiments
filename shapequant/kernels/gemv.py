"""Batch-1 matrix-vector product straight from the packed format (Triton).

y[r] = row_scale[r] * sum_j < table[row_book[r]*K + tag[r, j]], x[perm_{row_mask[r]}[j*d : j*d+d]] >

One program handles BR rows and walks the row's chunks BJ at a time:
- tags are read from the packed bit stream in place: chunk j of a row with
  b-bit tags starts at bit j*b, inside word (j*b)>>5, possibly continuing into
  the next word (two 32-bit loads, a funnel shift, a mask);
- the d codebook values are gathered from the (small, cache-resident) table;
- the matching inputs are gathered through that row's column order (perms is
  [M, C] int32, also cache-resident), so no permuted copies of x are built;
- accumulation is fp32, the row scale is applied once at the end.

Weight bytes read per row are the packed tags only; table, perms and x are
re-read from cache. That makes the kernel's DRAM traffic ~ the file size of
the matrix, which is the bandwidth ceiling this is measured against.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _gemv_kernel(words, table, row_scale, book_off, row_mask, perms, x, y,
                 R, C, W, NCH,
                 BITS: tl.constexpr, D: tl.constexpr, BR: tl.constexpr, BJ: tl.constexpr,
                 HAS_MASK: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    rvalid = rows < R
    boff = tl.load(book_off + rows, mask=rvalid, other=0)
    if HAS_MASK:
        pbase = tl.load(row_mask + rows, mask=rvalid, other=0) * C
    dk = tl.arange(0, D)
    acc = tl.zeros([BR], dtype=tl.float32)
    for j0 in range(0, NCH, BJ):
        j = j0 + tl.arange(0, BJ)
        jvalid = j < NCH
        valid = rvalid[:, None] & jvalid[None, :]
        bit = j * BITS
        wi = bit >> 5
        sh = (bit & 31).to(tl.uint32)
        wptr = words + rows[:, None] * W + wi[None, :]
        lo = tl.load(wptr, mask=valid, other=0).to(tl.uint32, bitcast=True)
        hi = tl.load(wptr + 1, mask=valid & (wi[None, :] + 1 < W), other=0).to(tl.uint32, bitcast=True)
        shb = sh[None, :]
        v = (lo >> shb) | tl.where(shb == 0, 0, hi << (32 - shb))
        tag = (v & ((1 << BITS) - 1)).to(tl.int32) + boff[:, None]
        tab = tl.load(table + tag[:, :, None] * D + dk[None, None, :],
                      mask=valid[:, :, None], other=0.0).to(tl.float32)
        col = j[None, :, None] * D + dk[None, None, :]
        if HAS_MASK:
            src = tl.load(perms + pbase[:, None, None] + col, mask=valid[:, :, None], other=0)
        else:
            src = col + rows[:, None, None] * 0
        xv = tl.load(x + src, mask=valid[:, :, None], other=0.0).to(tl.float32)
        acc += tl.sum(tl.sum(tab * xv, axis=2), axis=1)
    s = tl.load(row_scale + rows, mask=rvalid, other=0.0).to(tl.float32)
    tl.store(y + rows, (acc * s).to(y.dtype.element_ty), mask=rvalid)


@triton.jit
def _h(v, k: tl.constexpr):
    """fp16 number k (0..3) of a packed int64 of four fp16s, as fp32."""
    return ((v >> (16 * k)) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _gemv4_kernel(words, table64, row_scale, book_off, row_mask, xp64, y,
                  R, W, NCH,
                  BITS: tl.constexpr, BR: tl.constexpr, BJ: tl.constexpr, HAS_MASK: tl.constexpr):
    """d = 4 version: one 8-byte load per codebook entry (table64 = table viewed
    as int64 [G*K]) and one per input chunk (xp64 = the M permuted copies of x,
    viewed as int64 [M, NCH])."""
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    rvalid = rows < R
    boff = tl.load(book_off + rows, mask=rvalid, other=0)
    if HAS_MASK:
        xbase = tl.load(row_mask + rows, mask=rvalid, other=0) * NCH
    else:
        xbase = rows * 0
    acc = tl.zeros([BR, BJ], dtype=tl.float32)
    for j0 in range(0, NCH, BJ):
        j = j0 + tl.arange(0, BJ)
        valid = rvalid[:, None] & (j < NCH)[None, :]
        bit = j * BITS
        wi = bit >> 5
        sh = (bit & 31).to(tl.uint32)[None, :]
        wptr = words + rows[:, None] * W + wi[None, :]
        lo = tl.load(wptr, mask=valid, other=0).to(tl.uint32, bitcast=True)
        hi = tl.load(wptr + 1, mask=valid & (wi[None, :] + 1 < W), other=0).to(tl.uint32, bitcast=True)
        v = (lo >> sh) | tl.where(sh == 0, 0, hi << (32 - sh))
        tag = (v & ((1 << BITS) - 1)).to(tl.int32) + boff[:, None]
        t = tl.load(table64 + tag, mask=valid, other=0)
        xv = tl.load(xp64 + xbase[:, None] + j[None, :], mask=valid, other=0)
        acc += _h(t, 0) * _h(xv, 0) + _h(t, 1) * _h(xv, 1) + _h(t, 2) * _h(xv, 2) + _h(t, 3) * _h(xv, 3)
    s = tl.load(row_scale + rows, mask=rvalid, other=0.0).to(tl.float32)
    tl.store(y + rows, (tl.sum(acc, axis=1) * s).to(y.dtype.element_ty), mask=rvalid)


class PackedLinear(torch.nn.Module):
    """A block linear run from packed tensors. forward() uses the Triton GEMV
    for a single token; more tokens fall back to a dense decode (for now)."""

    def __init__(self, words, table, row_scale, row_book, row_mask, perms, K: int, d: int,
                 bias=None, BR: int = 16, BJ: int = 64):
        super().__init__()
        R, W = words.shape
        C = perms.shape[1]
        self.R, self.C, self.K, self.d, self.W = R, C, K, d, W
        self.bits = int(math.log2(K))
        self.nch = C // d
        self.BR, self.BJ = BR, BJ
        dev = words.device
        self.register_buffer("words", words.contiguous())
        self.register_buffer("table", table.to(torch.float16).contiguous())
        self.register_buffer("row_scale", row_scale.to(torch.float16).contiguous())
        rb = row_book if row_book is not None else torch.zeros(R, dtype=torch.int64, device=dev)
        self.register_buffer("book_off", (rb.to(torch.int64) * K).to(torch.int32).contiguous())
        self.has_mask = row_mask is not None and perms.shape[0] > 1
        rm = row_mask if self.has_mask else torch.zeros(R, dtype=torch.int64, device=dev)
        self.register_buffer("row_mask", rm.to(torch.int32).contiguous())
        self.register_buffer("perms", perms.to(torch.int32).contiguous())
        self.register_buffer("perms_l", perms.to(torch.int64).contiguous())  # for torch indexing
        self.bias = bias

    @classmethod
    def from_packed(cls, pf, name: str, device="cuda", bias=None, **kw):
        m = pf.meta[name]
        t = pf.tensors(name)
        return cls(t["tags"].to(device), t["table"].to(device), t["row_scale"].to(device),
                   t["row_book"].to(device) if "row_book" in t else None,
                   t["row_mask"].to(device) if "row_mask" in t else None,
                   pf.perms(name).to(device), m["K"], m["d"], bias=bias, **kw)

    def gemv(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """x [C] (fp16/bf16, contiguous) -> y [R]."""
        y = out if out is not None else torch.empty(self.R, dtype=x.dtype, device=x.device)
        grid = (triton.cdiv(self.R, self.BR),)
        if self.d == 4 and x.dtype == torch.float16:
            xp = (x[self.perms_l] if self.has_mask else x[None]).contiguous()   # [M, C]
            _gemv4_kernel[grid](self.words, self.table.view(torch.int64), self.row_scale, self.book_off,
                                self.row_mask, xp.view(torch.int64), y, self.R, self.W, self.nch,
                                BITS=self.bits, BR=self.BR, BJ=self.BJ, HAS_MASK=self.has_mask)
            return y
        _gemv_kernel[grid](self.words, self.table, self.row_scale, self.book_off, self.row_mask,
                           self.perms, x, y, self.R, self.C, self.W, self.nch,
                           BITS=self.bits, D=self.d, BR=self.BR, BJ=self.BJ, HAS_MASK=self.has_mask)
        return y

    def dense(self) -> torch.Tensor:
        from ..packed import decode_tensors, unpack_bits
        tags = unpack_bits(self.words, self.bits, self.nch)
        rb = (self.book_off // self.K).to(torch.int64)
        return decode_tensors(tags, self.table, self.row_scale, rb,
                              self.row_mask if self.has_mask else None, self.perms.to(torch.int64), self.K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        if math.prod(lead) == 1:
            y = self.gemv(x.reshape(-1).contiguous()).reshape(*lead, self.R)
        else:
            y = torch.nn.functional.linear(x, self.dense().to(x.dtype))
        return y if self.bias is None else y + self.bias

    def stored_bytes(self) -> int:
        """Bytes the GEMV must stream from DRAM (packed tags; small side tables
        and the scales are cache-resident after the first touch, counted too)."""
        return sum(t.numel() * t.element_size() for t in (self.words, self.table, self.row_scale))
