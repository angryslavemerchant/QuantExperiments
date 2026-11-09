"""Fused block-128 Hadamard rotation + dynamic per-row int8 quantization (Triton).

For the INT8 prefill path (vllm_ext/core.py, _large_int8): Y = (X H)(W H)^T with H = blockdiag(H_128)
along C, X and W rounded to int8 with one scale per row (per token / per output channel). This
replaces torch.mm(t.view(-1, 128), H) into a bf16 temporary followed by vLLM's
dynamic_scaled_int8_quant: one kernel, reading t twice (absmax pass + quantize pass, the second
mostly from L2) and writing only int8 + the scales, no [rows, C] bf16 temporary. The rotation is
done in fp32 (tl.dot, fp32 accumulate) rather than rounded to bf16 first.

  q, s = had_quant(t)        # t bf16/fp16 [N, C], C % 128 == 0 -> int8 [N, C], fp32 [N, 1]
  had_quant(t, q_out, s_out) # into given buffers (rows of a larger int8 matrix are fine)
  had_quant(t, q_out, s_known, given=True)   # fixed per-row scales (weights): single pass
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_H: dict = {}


def hadamard128(device, dtype=torch.bfloat16) -> torch.Tensor:
    k = (str(device), dtype)
    if k not in _H:
        h = torch.ones(1, 1)
        while h.shape[0] < 128:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        _H[k] = (h / 128 ** 0.5).to(device, dtype).contiguous()
    return _H[k]


@triton.jit
def _had_q8_kernel(X, Q, S, H, stride_x, stride_q, NBLK: tl.constexpr, BB: tl.constexpr,
                   GIVEN: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    rb = tl.arange(0, BB)[:, None]
    cc = tl.arange(0, 128)[None, :]
    h = tl.load(H + tl.arange(0, 128)[:, None] * 128 + cc)
    xrow = X + row * stride_x
    if GIVEN:                       # scale known (fixed weights): one pass
        s = tl.load(S + row)
    else:
        amax = tl.zeros((), dtype=tl.float32)
        for b0 in range(0, NBLK, BB):
            m = (b0 + rb) < NBLK
            x = tl.load(xrow + (b0 + rb) * 128 + cc, mask=m, other=0.0)
            y = tl.dot(x, h)
            amax = tl.maximum(amax, tl.max(tl.abs(y)))
        s = tl.maximum(amax, 1e-10) / 127.0
    inv = 1.0 / s
    qrow = Q + row * stride_q
    for b0 in range(0, NBLK, BB):
        m = (b0 + rb) < NBLK
        x = tl.load(xrow + (b0 + rb) * 128 + cc, mask=m, other=0.0)
        y = tl.dot(x, h) * inv
        q = tl.extra.cuda.libdevice.rint(y)
        q = tl.minimum(tl.maximum(q, -127.0), 127.0)
        tl.store(qrow + (b0 + rb) * 128 + cc, q.to(tl.int8), mask=m)
    if not GIVEN:
        tl.store(S + row, s)


def had_quant(t: torch.Tensor, q: torch.Tensor | None = None, s: torch.Tensor | None = None,
              given: bool = False):
    """t [N, C] (bf16/fp16, unit column stride), C % 128 == 0 -> (int8 [N, C], fp32 [N, 1]).
    given: s already holds each row's scale (e.g. computed once for fixed weights): one pass."""
    N, C = t.shape
    assert C % 128 == 0 and t.stride(1) == 1
    if q is None:
        q = torch.empty(N, C, dtype=torch.int8, device=t.device)
    if s is None:
        s = torch.empty(N, 1, dtype=torch.float32, device=t.device)
    assert q.shape == (N, C) and q.stride(1) == 1 and s.numel() == N and s.is_contiguous()
    if N:
        _had_q8_kernel[(N,)](t, q, s, hadamard128(t.device, t.dtype), t.stride(0), q.stride(0),
                             NBLK=C // 128, BB=16, GIVEN=given, num_warps=4)
    return q, s
