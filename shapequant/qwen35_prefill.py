"""Fast batched prefill for the Qwen3.5 engine (qwen35_runtime.prefill(impl="fast"), the default).

Same math as the reference prefill in qwen35_runtime (fp32 residual and norms, fp16 GEMM inputs
and outputs, fp32 DeltaNet), restructured for fewer and larger launches:

  - one GEMM per group of matrices sharing an input: in_proj_qkv | in_proj_z, q | k | v, gate | up
    (prefill.Group; the a / b projections as one fp32 GEMM);
  - fused kernels (kernels.prefill_kernels) for residual add + RMSNorm, the causal conv, the
    DeltaNet input prep, gated RMSNorm, q/k norm + rope + cache write, the attention output gate
    and SiLU * up;
  - the chunked gated delta rule as batched intra-chunk math plus one sequential scan kernel
    instead of HF's per-row / per-chunk Python loops (torch_chunk_gated_delta_rule);
  - attention in fp16 SDPA (q rounded to fp16; the reference runs SDPA in fp32).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .kernels import decode_e2e as de
from .prefill import Group, group_scratch, head_fast, needs_scratch


def groups(eng):
    """Per layer {name: Group | tensor} from eng.prefill_lins (cached on the engine)."""
    gs = getattr(eng, "_prefill_groups", None)
    if gs is not None and gs[0] is eng.prefill_lins:
        return gs[1]
    out = []
    for L, P in zip(eng.layers, eng.prefill_lins):
        G = {"gu": Group([P["gate"], P["up"]]), "down": Group([P["down"]])}
        if "qkv" in P:
            G.update(qkvz=Group([P["qkv"], P["z"]]), out=Group([P["out"]]),
                     wab=torch.cat([L["wa"], L["wb"]]).contiguous())
        else:
            G.update(qkv=Group([P["q"], P["k"], P["v"]]), o=Group([P["o"]]))
        out.append(G)
    eng._prefill_groups = (eng.prefill_lins, out)
    return out


@torch.no_grad()
def prefill(eng, tokens, logits: str = "last", gemm: str = "dequant") -> torch.Tensor:
    from .kernels import prefill_kernels as pk
    c = eng.cfg
    dev = eng.h.device
    ids = torch.as_tensor(tokens, dtype=torch.int64, device=dev).view(-1)
    T = ids.numel()
    p0 = int(eng.pos)
    if p0 + T > c.max_len:
        raise ValueError(f"context {p0 + T} > max_len {c.max_len}")
    G_all = groups(eng)
    scratch = group_scratch(eng, G_all) if needs_scratch(G_all, gemm) else None
    rid = ids % eng.embed.shape[0]
    h = (eng.embed.rows(rid) if isinstance(eng.embed, de.Q4Embed) else eng.embed[rid]).float().contiguous()
    pos = torch.arange(p0, p0 + T, device=dev)
    fr = pos.float()[:, None] * eng.inv_freq[None]
    cs, sn = fr.cos().contiguous(), fr.sin().contiguous()                     # [T, rot/2] fp32
    mask = None if p0 == 0 else (torch.arange(p0 + T, device=dev)[None, :] <= pos[:, None])
    NH, NKV, HD = c.n_heads, c.n_kv, c.head_dim
    jl = ja = 0
    add = None
    for i, (L, G) in enumerate(zip(eng.layers, G_all)):
        x = pk.addnorm(h, add, L["ln1"], c.eps)
        if c.layer_types[i] == "linear_attention":
            y = G["qkvz"].matmul(x, scratch, gemm)                    # [T, conv_dim + hv*dv] fp16
            ab = x.float() @ G["wab"].T                               # [T, 2hv] fp32
            u = pk.conv(y, eng.conv[jl], L["convw"], eng.conv_dim)    # [T, conv_dim] fp32, state updated
            q, k, v, g, beta = pk.gdn_prep(u, ab, L["A"], L["dt_bias"], c.hk, c.hv, c.dk, c.dv)
            o = pk.chunk_gated_delta_rule(q, k, v, g, beta, eng.S[jl])
            o16 = pk.gated_norm(o, y[:, eng.conv_dim:], L["gnorm"], c.eps, T)
            add = G["out"].matmul(o16, scratch, gemm)
            jl += 1
        else:
            y = G["qkv"].matmul(x, scratch, gemm)                     # [T, NH*2HD + 2*NKV*HD]
            q = pk.attn_prep(y, L["qn"], L["kn"], cs, sn, NH, NKV, HD, c.rot, c.eps, p0, eng.kc[ja], eng.vc[ja])
            rep = NH // NKV
            kk = eng.kc[ja, :, :p0 + T].repeat_interleave(rep, 0)
            vv = eng.vc[ja, :, :p0 + T].repeat_interleave(rep, 0)
            o = F.scaled_dot_product_attention(q[None], kk[None], vv[None],
                                               attn_mask=None if mask is None else mask[None, None],
                                               is_causal=mask is None, scale=HD ** -0.5)[0]   # [NH, T, HD]
            add = G["o"].matmul(pk.gate_out(o.contiguous(), y, NH, HD), scratch, gemm)
            ja += 1
        x = pk.addnorm(h, add, L["ln2"], c.eps)
        act = pk.silu_mul(G["gu"].matmul(x, scratch, gemm), 1)
        add = G["down"].matmul(act, scratch, gemm)
    if logits == "all":
        x = pk.addnorm(h, add, eng.final_norm, c.eps)
    else:
        x = pk.addnorm(h[-1:], add[-1:], eng.final_norm, c.eps)
    lg = head_fast(eng, x)
    eng.logits.copy_(lg[-1].half())
    eng.tok.copy_(lg[-1].half().float().argmax().view(1))
    eng.pos.fill_(p0 + T)
    return lg if logits == "all" else lg[-1]
