"""vLLM-independent half of the vLLM integration: packed codebook matrices as linear layers.

A vLLM linear layer can be several checkpoint matrices stacked along the output dim (qkv_proj =
q|k|v, gate_up_proj = gate|up, Qwen3.5's in_proj_qkvz = in_proj_qkv|in_proj_z). `PackedShards`
holds one run-kernel matrix (proto_runs.RunsLinear) per stacked piece and computes
Y [M, sum R] = X [M, C] . W^T:

  M <= BMAX (decode, small batches): the batched run kernel (kernels.batched_runs.RunsBatch),
      every weight code decoded once per call for all M rows, written straight into its column
      slice of a [M, sum R] fp16 buffer, no dense weight in memory;
  M >  BMAX (prefill, large batches): each piece dequantized on the GPU into a scratch buffer
      shared by every layer (kernels.cuda_dequant), then one fp16 GEMM per piece.

Activations go through the kernels in fp16 (the run kernels' input type) and come back in the
caller's dtype. Everything launches on torch's current stream, so the calls are CUDA-graph
capturable once each kernel variant has been compiled (vLLM's warmup does this).

`linear_op` exposes the forward as the torch custom op `shapequant::linear` (an integer layer
handle in, so torch.compile sees an opaque op with a known output shape).
"""

from __future__ import annotations

import re

import torch

BMAX = 32

# checkpoint suffix of each vLLM (possibly stacked) linear -> the packed pieces, in output order
STACKED = {
    "self_attn.qkv_proj": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    "mlp.gate_up_proj": ("mlp.gate_proj", "mlp.up_proj"),
    "linear_attn.in_proj_qkvz": ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z"),
}
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.(.+)$")


def packed_pieces(prefix: str, names) -> list[str] | None:
    """Packed matrix names (model.layers.N.<kind>) making up the vLLM layer at `prefix`, or None
    when the layer is not (entirely) packed. names: the packed file's matrix names."""
    m = _LAYER.search(prefix)
    if m is None:
        return None
    i, suffix = m.group(1), m.group(2)
    pieces = [f"model.layers.{i}.{s}" for s in STACKED.get(suffix, (suffix,))]
    have = [p in names for p in pieces]
    if all(have):
        return pieces
    if any(have):
        raise ValueError(f"{prefix}: only some of its pieces are packed ({pieces})")
    return None


class _Scratch:
    """One fp16 buffer per device for dequantized weights (grown as needed; layers run one at a
    time on one stream, so a single buffer is shared)."""
    bufs: dict = {}

    @classmethod
    def get(cls, n: int, dev) -> torch.Tensor:
        key = str(dev)
        b = cls.bufs.get(key)
        if b is None or b.numel() < n:
            b = cls.bufs[key] = torch.empty(n, dtype=torch.float16, device=dev)
        return b


class PackedShards:
    def __init__(self, rls: list, *, bmax: int = BMAX, layout: str = "tile", warps: int = 8, bps: int = 2,
                 max_split: int = 8):
        from ..kernels import batched_runs as br
        from ..kernels import cuda_dequant
        self.rls = rls
        C = {rl.nch * 4 for rl in rls}
        if len(C) != 1:
            raise ValueError(f"stacked pieces disagree on the input size: {sorted(C)}")
        self.C = C.pop()
        for rl in rls:
            if rl.has_mask:
                raise ValueError("column orders (M > 1) are not supported")
            if not cuda_dequant.supported(rl):
                raise ValueError("matrix not supported by the GPU dequant kernel")
        self.Rs = [rl.R for rl in rls]
        self.R = sum(self.Rs)
        self.offs = [sum(self.Rs[:i]) for i in range(len(rls))]
        self.bmax = bmax
        self.launchers = [br.RunsBatch(rl, bmax=bmax, warps=warps, bps=bps, max_split=max_split, layout=layout)
                          for rl in rls]
        # the batched kernel writes Y [B, R_piece] contiguous: one buffer per piece, then a copy
        # into the stacked output (skipped for a single piece)
        self.device = rls[0].words.device

    @classmethod
    def from_packed(cls, pf, names: list[str], dev="cuda", **kw) -> "PackedShards":
        from ..runtime import runs_linear
        return cls([runs_linear(pf, n, dev=dev).rl for n in names], **kw)

    def nbytes(self) -> int:
        n = 0
        for rl in self.rls:
            n += rl.words.numel() * 4 + rl.table.numel() * 2 + rl.row_scale.numel() * 4
            tw = getattr(rl, "_tile_words", None)
            if tw is not None:
                n += tw[0].numel() * 4
        return n

    def dense(self) -> torch.Tensor:
        """fp16 [R, C] (testing)."""
        from ..kernels import cuda_dequant
        return torch.cat([cuda_dequant.dequant(rl).clone() for rl in self.rls])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [..., C] -> [..., R] in x's dtype."""
        shp = x.shape
        x2 = x.reshape(-1, self.C)
        M = x2.shape[0]
        x16 = x2.to(torch.float16).contiguous()
        if M == 0:
            y = x16.new_empty(0, self.R)
        elif M <= self.bmax:
            y = self._small(x16, M)
        else:
            y = self._large(x16)
        return y.to(x.dtype).view(*shp[:-1], self.R)

    def _small(self, x16: torch.Tensor, M: int) -> torch.Tensor:
        if len(self.launchers) == 1:
            y = torch.empty(M, self.R, dtype=torch.float16, device=x16.device)
            self.launchers[0].run(x16, y, M)
            return y
        parts = []
        for lb, R in zip(self.launchers, self.Rs):
            yp = torch.empty(M, R, dtype=torch.float16, device=x16.device)
            lb.run(x16, yp, M)
            parts.append(yp)
        return torch.cat(parts, 1)

    def _large(self, x16: torch.Tensor) -> torch.Tensor:
        from ..kernels import cuda_dequant
        scratch = _Scratch.get(max(self.Rs) * self.C, x16.device)
        if len(self.rls) == 1:
            return x16 @ cuda_dequant.dequant(self.rls[0], scratch).view(self.R, self.C).t()
        y = torch.empty(x16.shape[0], self.R, dtype=torch.float16, device=x16.device)
        for rl, off, R in zip(self.rls, self.offs, self.Rs):
            w = cuda_dequant.dequant(rl, scratch).view(R, self.C)
            y[:, off:off + R] = x16 @ w.t()       # the GEMM finishes before the scratch is reused (one stream)
        return y


# ------------------------------------------------------------------ torch custom op

_REGISTRY: list[PackedShards] = []


def register(ps: PackedShards) -> int:
    _REGISTRY.append(ps)
    return len(_REGISTRY) - 1


@torch.library.custom_op("shapequant::linear", mutates_args=())
def linear_op(x: torch.Tensor, handle: int) -> torch.Tensor:
    return _REGISTRY[handle].forward(x)


@linear_op.register_fake
def _(x: torch.Tensor, handle: int) -> torch.Tensor:
    return x.new_empty(*x.shape[:-1], _REGISTRY[handle].R)
