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

import contextlib
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


@contextlib.contextmanager
def neutral_defaults():
    """torch's stock defaults (CPU device, float32) for building the kernels' tensors: vLLM
    constructs models under a CUDA default device and a bf16 default dtype, while the packed-file
    loaders create their CPU / fp32 temporaries without naming a device or dtype."""
    dt = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("cpu"):
            yield
    finally:
        torch.set_default_dtype(dt)


_TUNED: dict = {}


def _time(fn, reps: int, n: int = 10) -> float:
    """GPU ms per call: CUDA-graph replay of n calls (as vLLM's captured decode), median of reps."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()                      # compiles kernels, grows shared buffers before capture
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            fn()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) / n)
    del g
    return sorted(ts)[len(ts) // 2]


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
    def __init__(self, rls: list, *, bmax: int = BMAX, layout: str = "tile", warps: int = 8, bps: int = 2, xs: bool = False,
                 max_split: int = 8, large: str = "dequant", chunk_bytes: int = 24 << 20, dequant_from: int = 384):
        # large-M path: "dequant" (whole matrix to scratch + GEMM), "chunked" (L2-sized row chunks,
        # chunk_bytes of fp16 each), "fused" (kernels.runs_gemm: decode inside the GEMM tiles)
        assert large in ("dequant", "chunked", "fused"), large
        self.large, self.chunk_bytes = large, chunk_bytes
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
        self.cross = bmax        # M <= cross: batched kernel, else the large-M path (tune() moves it)
        self.dequant_from = dequant_from   # large == "fused": M >= this uses dequant + GEMM instead
        self.launchers = [br.RunsBatch(rl, bmax=bmax, warps=warps, bps=bps, max_split=max_split, layout=layout, xs=xs)
                          for rl in rls]
        # the batched kernel writes Y [B, R_piece] contiguous: one buffer per piece, then a copy
        # into the stacked output (skipped for a single piece)
        self.device = rls[0].words.device

    def key(self) -> tuple:
        return (tuple(self.Rs), self.C, tuple(rl.bits for rl in self.rls), self.large, self.bmax)

    @torch.no_grad()
    def tune(self, Ms=(8, 16, 24, 32, 48, 64, 96, 128), reps: int = 5) -> int:
        """Measure the batched kernel against the large-M path at each M <= bmax and set cross =
        the largest M before the large path first wins (the per-shape crossover differs: e.g.
        down_proj's batched kernel wins to 64 rows, qkv's fused GEMM from 48). Cached per shape."""
        k = self.key()
        if k in _TUNED:
            self.cross = _TUNED[k]
            return self.cross
        cross = 0
        for M in Ms:
            if M > self.bmax:
                break
            x = torch.randn(M, self.C, device=self.device, dtype=torch.float16)
            t_small = _time(lambda: self._small(x, M), reps)
            t_large = _time(lambda: self._large(x), reps)
            if t_large < t_small:
                break
            cross = M
        self.cross = _TUNED[k] = cross
        return cross

    @classmethod
    def from_packed(cls, pf, names: list[str], dev="cuda", **kw) -> "PackedShards":
        from ..runtime import runs_linear
        with neutral_defaults():
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
        elif M <= self.cross:
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
        if self.large == "chunked":
            return self._large_chunked(x16)
        if self.large == "fused" and x16.shape[0] < self.dequant_from:
            from ..kernels import runs_gemm
            return runs_gemm.gemm(x16, self.rls)
        # long prefill chunks: dequant + cuBLAS (the decode cost is amortized over many rows; the
        # fused GEMM's mma loop is ~2x slower than cuBLAS there: PRO 6000 s35c, M >= 512)
        from ..kernels import cuda_dequant
        scratch = _Scratch.get(max(self.Rs) * self.C, x16.device)
        if len(self.rls) == 1:
            return x16 @ cuda_dequant.dequant(self.rls[0], scratch).view(self.R, self.C).t()
        y = torch.empty(x16.shape[0], self.R, dtype=torch.float16, device=x16.device)
        for rl, off, R in zip(self.rls, self.offs, self.Rs):
            w = cuda_dequant.dequant(rl, scratch).view(R, self.C)
            y[:, off:off + R] = x16 @ w.t()       # the GEMM finishes before the scratch is reused (one stream)
        return y

    def _large_chunked(self, x16: torch.Tensor) -> torch.Tensor:
        """Row chunks of ~chunk_bytes fp16 decoded and multiplied in turn: a chunk small enough to
        stay in L2 is read back by the GEMM from cache, so DRAM traffic is ~the packed codes plus
        activations. Computes Y^T (row chunks of Y^T are contiguous) and transposes at the end."""
        from ..kernels import cuda_dequant
        M = x16.shape[0]
        rc = max(64, (self.chunk_bytes // (2 * self.C)) // 64 * 64)
        scratch = _Scratch.get(min(rc, max(self.Rs)) * self.C, x16.device)
        yt = torch.empty(self.R, M, dtype=torch.float16, device=x16.device)
        xt = x16.t()
        for rl, off, R in zip(self.rls, self.offs, self.Rs):
            for r0 in range(0, R, rc):
                r1 = min(R, r0 + rc)
                w = cuda_dequant.dequant(rl, scratch, rows=(r0, r1))
                torch.mm(w, xt, out=yt[off + r0:off + r1])
        return yt.t().contiguous()


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
