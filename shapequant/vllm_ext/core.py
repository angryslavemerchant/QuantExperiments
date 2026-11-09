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
import os
import re

import torch

BMAX = 32

# checkpoint suffix of each vLLM (possibly stacked) linear -> the packed pieces, in output order
STACKED = {
    "self_attn.qkv_proj": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    "mlp.gate_up_proj": ("mlp.gate_proj", "mlp.up_proj"),
    "linear_attn.in_proj_qkvz": ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z"),
}
# Diagnostic only: simulate an INT8 prefill GEMM on the large-M path by rounding to int8 and back
# (symmetric, one scale per row: per token for activations, per output channel for weights).
#   SHAPEQUANT_FAKEQ      = "a8" | "w8" | "w8a8"; unset = off
#   SHAPEQUANT_FAKEQ_SKIP = comma list of name substrings left in fp16 (e.g. "down_proj")
#   SHAPEQUANT_FAKEQ_HAD  = block size b (power of 2): rotate X and W by a block-diagonal Hadamard
#                           along C before rounding (X W^T is unchanged by the rotation)
_FAKEQ = os.environ.get("SHAPEQUANT_FAKEQ", "").lower()
_FAKEQ_SKIP = [t for t in os.environ.get("SHAPEQUANT_FAKEQ_SKIP", "").split(",") if t]
_FAKEQ_HAD = int(os.environ.get("SHAPEQUANT_FAKEQ_HAD", "0") or 0)
_HAD: dict = {}
# compact mode, bf16 activations: prefill GEMMs run in bf16 (SHAPEQUANT_BF16_PREFILL=0: the older
# fp16 path with casts around it)
_BF16_PREFILL = os.environ.get("SHAPEQUANT_BF16_PREFILL", "1") != "0"
# INT8 prefill (compact mode, bf16 X, M >= SHAPEQUANT_INT8_MIN_M): W8A8 GEMM with a block-128
# Hadamard on both X and W along C (X W^T = (X H)(W H)^T), dynamic per-token int8 X, per-output-row
# int8 W made from the dequantized tile each call; vLLM's cutlass_scaled_mm applies both scales and
# writes bf16. 27B on wikitext: KL vs the bf16 path 0.0018 (vs 0.12 for 2.5 bpw vs bf16); without
# the rotation 0.027 (scripts/kl_fakeq.py). Needs vLLM's _C ops (only imported on first use).
_INT8_PREFILL = os.environ.get("SHAPEQUANT_INT8_PREFILL", "0") != "0"
_INT8_MIN_M = int(os.environ.get("SHAPEQUANT_INT8_MIN_M", "1024"))
_INT8_ROWS = 2048           # weight rows rotated per step (bf16 temp of _INT8_ROWS x C)
_INT8_FUSED = os.environ.get("SHAPEQUANT_INT8_FUSED", "1") != "0"   # kernels/had_quant.py


class _Int8:
    """Lazily bound vLLM ops + the bf16 H_128 per device."""
    ops = None
    h: dict = {}

    @classmethod
    def get(cls, dev):
        if cls.ops is None:
            import vllm._custom_ops  # noqa: F401  (registers torch.ops._C)
            cls.ops = torch.ops._C
        k = str(dev)
        if k not in cls.h:
            h = torch.ones(1, 1)
            while h.shape[0] < 128:
                h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
            cls.h[k] = (h / 128 ** 0.5).to(dev, torch.bfloat16)
        return cls.ops, cls.h[k]


def _int8_ok(ps) -> bool:
    return ps.C % 128 == 0 and all(R % 16 == 0 for R in ps.Rs)


def _fq_rows(t: torch.Tensor) -> torch.Tensor:
    s = t.abs().amax(dim=1, keepdim=True).float().clamp_min(1e-12) / 127
    return (t.float() / s).round_().clamp_(-127, 127).mul_(s).to(t.dtype)


def _hadamard(t: torch.Tensor) -> torch.Tensor:
    """t [N, C] @ blockdiag(H_b) along C (fp32 math), C % b == 0 or b halves until it divides."""
    b, C = _FAKEQ_HAD, t.shape[1]
    while C % b:
        b //= 2
    key = (b, t.device)
    if key not in _HAD:
        h = torch.ones(1, 1)
        while h.shape[0] < b:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        _HAD[key] = (h / b ** 0.5).to(t.device)
    return (t.float().view(t.shape[0], C // b, b) @ _HAD[key]).view(t.shape[0], C).to(t.dtype)


def _fq_on(name: str) -> bool:
    return bool(_FAKEQ) and not any(k in name for k in _FAKEQ_SKIP)


def _fqx(x: torch.Tensor, name: str) -> torch.Tensor:
    if not _fq_on(name):
        return x
    if _FAKEQ_HAD:
        x = _hadamard(x)
    return _fq_rows(x) if "a8" in _FAKEQ else x


def _fqw(w: torch.Tensor, name: str = "") -> torch.Tensor:
    if not _fq_on(name):
        return w
    if _FAKEQ_HAD:
        w = _hadamard(w)
    return _fq_rows(w) if "w8" in _FAKEQ else w


_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.(.+)$")


def packed_pieces(prefix: str, names) -> list[str] | None:
    """Packed matrix names (model.layers.N.<kind>) making up the vLLM layer at `prefix`, or None
    when the layer is not (entirely) packed. names: the packed file's matrix names."""
    if prefix.startswith("mtp.") or ".mtp." in prefix:
        return None          # the MTP draft's own layers (bf16), not the main model's layers.N
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
_LAUNCH: dict = {}          # (runs_key, bucket, layout) -> (xs, warps, bps, max_split)
BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)
LAUNCH_GRID = [(w, b, m) for w in (4, 8) for b in (1, 2, 4) for m in (1, 2, 4, 8)]


_TIME_STREAM: dict = {}     # device -> the one side stream every _time warmup runs on


def _time(fn, reps: int, n: int = 10) -> float:
    """GPU ms per call: CUDA-graph replay of n calls (as vLLM's captured decode), median of reps.
    The warmup runs on one reused side stream: torch caches a cuBLAS workspace per stream for the
    process lifetime, and a fresh stream per call left 31 x 8 MiB (0.25 GiB) behind on the 27B."""
    dev = torch.cuda.current_device()
    if dev not in _TIME_STREAM:
        _TIME_STREAM[dev] = torch.cuda.Stream()
    s = _TIME_STREAM[dev]
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
                 max_split: int = 8, large: str = "dequant", chunk_bytes: int = 24 << 20, dequant_from: int = 384,
                 compact: bool = False):
        # compact: keep only the TILE-layout words (layout="tile"): the lane-major words are freed
        # (~half the weight memory: 27B 19.8 -> ~11 GB), so the fused GEMM / lane dequant are not
        # available: M <= cross runs the batched kernel, larger M dequantizes from the tile words
        # (batched_runs.dequant_tile) into the shared scratch + one GEMM per piece.
        if compact and layout != "tile":
            raise ValueError("compact needs layout='tile'")
        self.compact = compact
        self.sq_name = ""        # set by the vLLM plugin (layer prefix); used by SHAPEQUANT_FAKEQ_SKIP
        self.int8_ok = False     # set below once Rs / C are known
        # large-M path: "dequant" (whole matrix to scratch + GEMM), "chunked" (L2-sized row chunks,
        # chunk_bytes of fp16 each), "fused" (kernels.runs_gemm: decode inside the GEMM tiles)
        assert large in ("dequant", "chunked", "fused"), large
        self.large, self.chunk_bytes = large, chunk_bytes
        self.layout = layout
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
        self.int8_ok = _int8_ok(self)
        self.bmax = bmax
        self.cross = bmax        # M <= cross: batched kernel, else the large-M path (tune() moves it)
        self.dequant_from = dequant_from   # large == "fused": M >= this uses dequant + GEMM instead
        self.launchers = [br.RunsBatch(rl, bmax=bmax, warps=warps, bps=bps, max_split=max_split, layout=layout, xs=xs)
                          for rl in rls]
        if compact:
            for rl in rls:
                rl.words = rl.words.new_empty(0)       # every compact path reads the tile copy
        # the batched kernel writes Y [B, R_piece] contiguous: one buffer per piece, then a copy
        # into the stacked output (skipped for a single piece)
        self.device = rls[0].words.device
        self.buckets, self.blaunch = None, None          # tune_launch(): per batch-size bucket launchers

    @torch.no_grad()
    def tune_launch(self, reps: int = 3) -> list:
        """Per piece and batch-size bucket (1, 2, 4, ... up to bmax), the fastest batched-kernel launch
        (X in shared memory or not x warps x blocks/SM x split-K), timed on bf16 X / Y as vLLM calls
        it, cached per matrix shape (s40 3090: 15-30% over the default config at B 1-8)."""
        from ..kernels import batched_runs as br
        bl = [b for b in BUCKETS if b < self.bmax] + [self.bmax]
        blaunch = []
        for rl, base in zip(self.rls, self.launchers):
            per = []
            for b in bl:
                k = (br.runs_key(rl), b, self.layout)
                cfg = _LAUNCH.get(k)
                if cfg is None:
                    x = torch.randn(b, self.C, device=self.device, dtype=torch.bfloat16)
                    y = torch.empty(b, rl.R, device=self.device, dtype=torch.bfloat16)
                    best, seen = None, set()
                    for xs in (False, True):
                        if not xs and b > br.BMAX:
                            continue
                        for w, bps, m in LAUNCH_GRID:
                            lb = br.RunsBatch(rl, bmax=b, warps=w, bps=bps, max_split=m, layout=self.layout, xs=xs)
                            eff = (xs, w, lb.grid, lb.S)
                            if eff in seen:
                                continue
                            seen.add(eff)
                            t = _time(lambda: lb.run(x, y, b), reps)
                            if best is None or t < best[0]:
                                best = (t, (xs, w, bps, m))
                    cfg = _LAUNCH[k] = best[1]
                xs, w, bps, m = cfg
                lb = br.RunsBatch(rl, bmax=b, warps=w, bps=bps, max_split=m, layout=self.layout, xs=xs)
                if self.layout == "tile" and not torch.equal(lb.tiles, base.tiles):
                    raise RuntimeError("tile order differs between launch configs (shared tile words)")
                per.append(lb)
            blaunch.append(per)
        self.buckets, self.blaunch = bl, blaunch
        return [_LAUNCH[(br.runs_key(rl), b, self.layout)] for rl in self.rls for b in bl]

    def _launchers_for(self, M: int) -> list:
        if self.blaunch is None:
            return self.launchers
        import bisect
        i = bisect.bisect_left(self.buckets, M)
        return [per[i] for per in self.blaunch]

    def key(self) -> tuple:
        return (tuple(self.Rs), self.C, tuple(rl.bits for rl in self.rls), self.large, self.bmax, self.compact)

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
            x = torch.randn(M, self.C, device=self.device, dtype=torch.bfloat16)      # vLLM's activations
            t_small = _time(lambda: self._small(x, M), reps)
            t_large = _time(lambda: self._large(x.to(torch.float16)), reps)
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
        if self.compact:
            from ..kernels import batched_runs as br
            return torch.cat([br.dequant_tile(lb).clone() for lb in self.launchers])
        return torch.cat([cuda_dequant.dequant(rl).clone() for rl in self.rls])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [..., C] -> [..., R] in x's dtype."""
        shp = x.shape
        x2 = x.reshape(-1, self.C)
        M = x2.shape[0]
        if M == 0:
            y = x2.new_empty(0, self.R)
        elif M <= self.cross:
            # the batched kernel reads bf16 / fp16 X and writes Y in the same dtype, every stacked
            # piece straight into its column slice: no cast or concat kernels around it
            xin = x2 if x2.dtype in (torch.float16, torch.bfloat16) else x2.to(torch.float16)
            y = self._small(xin.contiguous(), M)
        elif (self.compact and _INT8_PREFILL and x2.dtype == torch.bfloat16 and M >= _INT8_MIN_M
              and not _FAKEQ and self.int8_ok):
            y = self._large_int8(x2.contiguous())
        elif self.compact and _BF16_PREFILL and x2.dtype == torch.bfloat16:
            # bf16 end to end: weights dequantized straight to bf16, GEMM in bf16 (fp32 accumulate),
            # no cast kernels on X or Y
            y = self._large(x2.contiguous())
        else:
            y = self._large(x2.to(torch.float16).contiguous())
        return y.to(x.dtype).view(*shp[:-1], self.R)

    def _small(self, x: torch.Tensor, M: int) -> torch.Tensor:
        y = torch.empty(M, self.R, dtype=x.dtype, device=x.device)
        lbs = self._launchers_for(M)
        if len(lbs) == 1:
            lbs[0].run(x, y, M)
            return y
        for lb, off, R in zip(lbs, self.offs, self.Rs):
            lb.run(x, y[:, off:off + R], M)
        return y

    def _large(self, x16: torch.Tensor) -> torch.Tensor:
        if _FAKEQ:
            x16 = _fqx(x16, self.sq_name)
        if self.compact:
            return self._large_compact(x16)
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
            return x16 @ _fqw(cuda_dequant.dequant(self.rls[0], scratch).view(self.R, self.C), self.sq_name).t()
        y = torch.empty(x16.shape[0], self.R, dtype=torch.float16, device=x16.device)
        for rl, off, R in zip(self.rls, self.offs, self.Rs):
            w = _fqw(cuda_dequant.dequant(rl, scratch).view(R, self.C), self.sq_name)
            y[:, off:off + R] = x16 @ w.t()       # the GEMM finishes before the scratch is reused (one stream)
        return y

    def _large_compact(self, x: torch.Tensor) -> torch.Tensor:
        """x fp16 or bf16; the weights are dequantized to x's dtype, Y comes out in x's dtype."""
        from ..kernels import batched_runs as br
        scratch = _Scratch.get(max(self.Rs) * self.C, x.device).view(x.dtype)
        if len(self.rls) == 1:
            return x @ _fqw(br.dequant_tile(self.launchers[0], scratch), self.sq_name).t()
        y = torch.empty(x.shape[0], self.R, dtype=x.dtype, device=x.device)
        for lb, off, R in zip(self.launchers, self.offs, self.Rs):
            w = _fqw(br.dequant_tile(lb, scratch), self.sq_name)
            # the column slice has row stride self.R and unit column stride, which cuBLAS takes as
            # its output directly (ldc = self.R): no temporary + copy kernel per piece
            torch.mm(x, w.t(), out=y[:, off:off + R])
        return y

    def _large_int8(self, x: torch.Tensor) -> torch.Tensor:
        """x bf16 [M, C] -> bf16 [M, R] by the W8A8 path (see _INT8_PREFILL)."""
        from ..kernels import batched_runs as br
        ops, H = _Int8.get(x.device)
        M, C = x.shape
        if _INT8_FUSED:
            # rotation + per-row int8 in one Triton kernel (kernels/had_quant.py): no bf16 temporaries
            from ..kernels.had_quant import had_quant
            xq, sx = had_quant(x)
            scratch = _Scratch.get(max(self.Rs) * self.C, x.device).view(torch.bfloat16)
            wq = torch.empty(self.R, C, dtype=torch.int8, device=x.device)
            sw = torch.empty(self.R, 1, dtype=torch.float32, device=x.device)
            # the weights never change, so neither do their rotated rows' scales: computed on the
            # first call (two passes), then reused (one pass)
            known = getattr(self, "_int8_sw", None)
            for lb, off, R in zip(self.launchers, self.offs, self.Rs):
                if known is not None:
                    sw[off:off + R] = known[off:off + R]
                had_quant(br.dequant_tile(lb, scratch), wq[off:off + R], sw[off:off + R], given=known is not None)
            if known is None:
                self._int8_sw = sw.clone()
            y = torch.empty(M, self.R, dtype=torch.bfloat16, device=x.device)
            ops.cutlass_scaled_mm(y, xq, wq.t(), sx, sw.view(1, self.R), None)
            return y
        # X: rotate, then dynamic per-token int8
        xr = torch.mm(x.view(-1, 128), H).view(M, C)
        xq = torch.empty(M, C, dtype=torch.int8, device=x.device)
        sx = torch.empty(M, 1, dtype=torch.float32, device=x.device)
        ops.dynamic_scaled_int8_quant(xq, xr, sx, None)
        del xr
        # W: every stacked piece dequantized (bf16 scratch), rotated in row chunks, int8 per row,
        # into one [R, C] int8 matrix, so one GEMM covers all pieces
        scratch = _Scratch.get(max(self.Rs) * self.C, x.device).view(torch.bfloat16)
        wq = torch.empty(self.R, C, dtype=torch.int8, device=x.device)
        sw = torch.empty(self.R, 1, dtype=torch.float32, device=x.device)
        rows = min(_INT8_ROWS, max(self.Rs))
        tmp = torch.empty(rows * C, dtype=torch.bfloat16, device=x.device)
        for lb, off, R in zip(self.launchers, self.offs, self.Rs):
            w = br.dequant_tile(lb, scratch)
            for r0 in range(0, R, rows):
                r1 = min(R, r0 + rows)
                wr = tmp[:(r1 - r0) * C].view(-1, 128)
                torch.mm(w[r0:r1].reshape(-1, 128), H, out=wr)
                ops.dynamic_scaled_int8_quant(wq[off + r0:off + r1], wr.view(r1 - r0, C),
                                              sw[off + r0:off + r1], None)
        y = torch.empty(M, self.R, dtype=torch.bfloat16, device=x.device)
        ops.cutlass_scaled_mm(y, xq, wq.t(), sx, sw.view(1, self.R), None)
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


class Q8Head:
    """The packed file's 8-bit block-32 lm_head (aux.lm_head: q int8 [V, C], scale fp16 [V, C/32])
    as Y [M, R] = X [M, C] . W^T, R >= V (vLLM pads the vocab; padded logits are 0).

    C % 64 == 0 (27B: 248k x 5120): the weights are kept only in kernels.batched_runs' tile-major
    layout (q8_tile_layout; same bytes), so the decode GEMV reads at ~95% of DRAM bandwidth
    (3090: 1.53 ms per token vs 3.69 ms on the natural layout). Otherwise the natural layout.
      M <= small_max: the batched q8 GEMV (1.35 GB read for 27B vs 2.5 GB bf16);
      M >  small_max: row chunks dequantized to fp16 into the shared scratch, then a GEMM each.

    Same forward / R interface as PackedShards, so it runs through the shapequant::linear op."""

    def __init__(self, q: torch.Tensor, sc: torch.Tensor, R: int | None = None, small_max: int = 32,
                 chunk_rows: int = 8192, tiled: bool | None = None):
        self.V, self.C = q.shape
        if self.C % 32 or sc.shape != (self.V, self.C // 32):
            raise ValueError(f"q8 head: q {tuple(q.shape)}, scale {tuple(sc.shape)}")
        self.R = self.V if R is None else R
        if self.R < self.V:
            raise ValueError(f"q8 head has {self.V} rows, layer wants {self.R}")
        self.small_max = small_max
        self.tiled = self.C % 64 == 0 if tiled is None else tiled
        if self.tiled:
            from ..kernels import batched_runs as br
            self.qt, self.st = br.q8_tile_layout(q, sc)       # the caller's q / sc can be freed
            self.q = self.sc = None
            self.chunk_rows = max(16, chunk_rows // 16 * 16)
        else:
            self.q, self.sc = q, sc
            self.chunk_rows = chunk_rows

    def nbytes(self) -> int:
        if self.tiled:
            return self.qt.numel() + self.st.numel() * 2
        return self.q.numel() + self.sc.numel() * 2

    def dense(self) -> torch.Tensor:
        """fp16 [V, C] (testing)."""
        return torch.cat([self._rows(s, min(self.V, s + self.chunk_rows)).clone()
                          for s in range(0, self.V, self.chunk_rows)])

    def _parts(self, s: int, e: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.tiled:
            return self.q[s:e], self.sc[s:e]
        from ..kernels import batched_runs as br
        assert s % 16 == 0
        q, sc = br.q8_tile_rows(self.qt, self.st, s // 16, (e + 15) // 16)
        return q[:e - s], sc[:e - s]

    def _rows(self, s: int, e: int, out: torch.Tensor | None = None) -> torch.Tensor:
        n = e - s
        q, sc = self._parts(s, e)
        w = q.reshape(n, self.C // 32, 32).to(torch.float16)
        o = w if out is None else out[:n * self.C].view(n, self.C // 32, 32)
        torch.mul(w, sc[:, :, None], out=o)
        return o.view(n, self.C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        x16 = x.reshape(-1, self.C).to(torch.float16).contiguous()
        M = x16.shape[0]
        y = torch.empty(M, self.R, dtype=torch.float16, device=x16.device)
        if self.R > self.V:
            y[:, self.V:] = 0
        if 0 < M <= self.small_max:
            from ..kernels import batched_runs as br
            yv = y if self.R == self.V else torch.empty(M, self.V, dtype=torch.float16, device=x16.device)
            if self.tiled:
                br.gemv_q8_tile(self.qt, self.st, x16, yv, self.V, self.C, M)
            else:
                br.gemv_q8_batched(self.q, self.sc, x16, yv, self.V, self.C, M)
            if yv is not y:
                y[:, :self.V] = yv
        elif M > 0:
            scratch = _Scratch.get(self.chunk_rows * self.C, x16.device)
            for s in range(0, self.V, self.chunk_rows):
                e = min(self.V, s + self.chunk_rows)
                y[:, s:e] = x16 @ self._rows(s, e, scratch).t()
        return y.to(x.dtype).view(*shp[:-1], self.R)


def q4_embed(q: torch.Tensor, sc: torch.Tensor, ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Rows `ids` (any shape) of the packed file's 4-bit block-32 embedding (aux.embed), [*ids, C]
    in dtype. Plain torch ops (traceable by torch.compile, CUDA-graph capturable)."""
    flat = ids.reshape(-1)
    b = q[flat]
    n = torch.stack([b & 15, b >> 4], -1).view(flat.shape[0], -1, 32).to(torch.float32) - 8
    rows = (n * sc[flat].to(torch.float32)[..., None]).view(flat.shape[0], -1)
    return rows.to(dtype).view(*ids.shape, -1)


class Q4Embedding:
    """The 4-bit embedding behind the shapequant::embed op (an integer handle, like the linears:
    vLLM's model build / load / torch.compile never see its tensors; s38: as registered module
    buffers it came back all zeros inside vLLM)."""

    def __init__(self, q: torch.Tensor, sc: torch.Tensor, dtype: torch.dtype):
        self.q, self.sc, self.dtype = q, sc, dtype
        self.C = q.shape[1] * 2

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return q4_embed(self.q, self.sc, ids, self.dtype)


# ------------------------------------------------------------------ torch custom op

_REGISTRY: list = []        # PackedShards | Q8Head


def register(ps) -> int:
    _REGISTRY.append(ps)
    return len(_REGISTRY) - 1


_POOL_ALIGN = 64            # int32 elements (256 bytes) between pooled matrices


def pool_tile_words() -> int:
    """Move the tile words of every registered matrix not pooled yet into one exact-size buffer per
    device, repoint its launchers, and free the per-matrix copies. Each matrix's tile words were a
    separate ~10-60 MB allocation, which the caching allocator rounds up inside 2 MB segments: on the
    27B, 800 blocks held 8.20 GiB for 7.38 GiB of data. Returns the bytes now pooled (0 if nothing
    new). The kernels read `RunsBatch.wt` at every call, so nothing else needs to change; call it
    before any CUDA graph that uses these matrices is captured."""
    rls, seen = [], set()
    for ps in _REGISTRY:
        for rl in getattr(ps, "rls", ()):
            tw = getattr(rl, "_tile_words", None)
            if tw is None or id(rl) in seen or getattr(rl, "_tw_pooled", False):
                continue
            seen.add(id(rl))
            rls.append(rl)
    if not rls:
        return 0
    pooled = 0
    for dev in {rl._tile_words[0].device for rl in rls}:
        lst = [rl for rl in rls if rl._tile_words[0].device == dev]
        sizes = [-(-rl._tile_words[0].numel() // _POOL_ALIGN) * _POOL_ALIGN for rl in lst]
        arena = torch.empty(sum(sizes), dtype=torch.int32, device=dev)
        remap, off = {}, 0
        for rl, n in zip(lst, sizes):
            wt, toff = rl._tile_words
            assert wt.dtype == torch.int32 and wt.is_contiguous()
            new = arena[off:off + wt.numel()].view(wt.shape)
            new.copy_(wt)
            remap[id(wt)] = (wt, new)
            rl._tile_words, rl._tw_pooled = (new, toff), True
            off += n
        for ps in _REGISTRY:
            lbs = list(getattr(ps, "launchers", None) or ())
            lbs += [lb for per in (getattr(ps, "blaunch", None) or ()) for lb in per]
            for lb in lbs:
                w = getattr(lb, "wt", None)
                if w is not None and id(w) in remap:
                    lb.wt = remap[id(w)][1]
        pooled += arena.numel() * 4
        del remap
    return pooled


_DEBUG = {"on": __import__("os").environ.get("SHAPEQUANT_DEBUG") == "1", "n": 0}


def _debug(kind: str, handle: int, x: torch.Tensor, y: torch.Tensor) -> None:
    """SHAPEQUANT_DEBUG=1 (eager runs): report non-finite outputs / inputs over the fp16 range."""
    if not _DEBUG["on"] or torch.cuda.is_current_stream_capturing() or _DEBUG["n"] > 60:
        return
    xa = x.float().abs().max().item() if x.is_floating_point() and x.numel() else 0.0
    yf = bool(torch.isfinite(y).all().item()) if y.numel() else True
    seen = _DEBUG.setdefault("seen", set())
    if handle not in seen and x.shape[0] > 1:          # one line per layer on the first real batch
        seen.add(handle)
        print(f"[shapequant trace] {handle} M {x.shape[0]} x absmax {xa:.4g} rms {x.float().pow(2).mean().sqrt().item():.4g}"
              f" -> y rms {y.float().pow(2).mean().sqrt().item():.4g} finite {yf}", flush=True)
    if not yf or xa > 6e4 or not (xa == xa):
        _DEBUG["n"] += 1
        print(f"[shapequant debug] {kind} handle {handle} {type(_REGISTRY[handle]).__name__} x {tuple(x.shape)} "
              f"{x.dtype} absmax {xa:.4g} y finite {yf} y absmax {y.float().abs().max().item():.4g}", flush=True)


@torch.library.custom_op("shapequant::linear", mutates_args=())
def linear_op(x: torch.Tensor, handle: int) -> torch.Tensor:
    y = _REGISTRY[handle].forward(x)
    _debug("linear", handle, x, y)
    return y


@linear_op.register_fake
def _(x: torch.Tensor, handle: int) -> torch.Tensor:
    return x.new_empty(*x.shape[:-1], _REGISTRY[handle].R)


@torch.library.custom_op("shapequant::embed", mutates_args=())
def embed_op(ids: torch.Tensor, handle: int) -> torch.Tensor:
    y = _REGISTRY[handle].forward(ids)
    if _DEBUG["on"] and not torch.cuda.is_current_stream_capturing() and _DEBUG["n"] < 3 and y.numel():
        _DEBUG["n"] += 1
        print(f"[shapequant debug] embed ids {tuple(ids.shape)} first {ids.flatten()[:4].tolist()} "
              f"y rms {y.float().pow(2).mean().sqrt().item():.4g}", flush=True)
    return y


@embed_op.register_fake
def _(ids: torch.Tensor, handle: int) -> torch.Tensor:
    e = _REGISTRY[handle]
    return torch.empty(*ids.shape, e.C, dtype=e.dtype, device=ids.device)
