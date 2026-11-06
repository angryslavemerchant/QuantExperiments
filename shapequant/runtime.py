"""Real-weight runtime: the batch-1 decode engine (kernels/decode_e2e.py) built from a
packed file (shapequant.packed) plus the base checkpoint's other tensors.

Every block linear runs on the lane-interleaved run kernel from the file's codes,
books and scales; embeddings, norms and q/k/v biases come from the base checkpoint
named in the file. The lm_head is not part of the packed format (the pipeline only
compresses block linears): `head` picks fp16 (cuBLAS) or our Q4_0-like kernel
(4.5 bpw, the usual GGUF practice of keeping the output matrix at higher precision).

    eng, info = engine_from_packed("model.safetensors", head="q4", max_len=2048)
    out = generate(eng, prompt_ids, 64)
"""

from __future__ import annotations

import torch

from .kernels import decode_e2e as de
from .kernels import proto_runs
from .kernels.gemv import PackedLinear
from .packed import PackedFile

NAMES = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj",
         "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}


def cfg_from_hf(c, max_len: int) -> de.Cfg:
    theta = getattr(c, "rope_theta", None)
    if theta is None:
        rp = getattr(c, "rope_parameters", None) or {}
        theta = rp.get("rope_theta", 1e6)
        if rp.get("rope_type", "default") != "default":
            raise ValueError(f"rope_type {rp.get('rope_type')!r} not supported by the engine")
    hd = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    return de.Cfg(hidden=c.hidden_size, n_heads=c.num_attention_heads, n_kv=c.num_key_value_heads, head_dim=hd,
                  inter=c.intermediate_size, vocab=c.vocab_size, layers=c.num_hidden_layers,
                  rope_theta=float(theta), eps=float(c.rms_norm_eps), max_len=max_len)


def runs_linear(pf: PackedFile, name: str, dev="cuda", ru: int = 1, small=(0, 1), large=(512, 1)) -> de.RunsLin:
    """One packed matrix as an engine linear (launch config by size, as in random_engine)."""
    m = pf.meta[name]
    if m["d"] != 4 or (m["K"].bit_length() - 1) % 2:
        raise ValueError(f"{name}: run kernel needs d=4 and even tag bits (d={m['d']}, K={m['K']})")
    lin = PackedLinear.from_packed(pf, name, device=dev)
    rl = proto_runs.RunsLinear(lin, pf.local_tags(name, dev), ru)
    th, bps = small if m["R"] <= 4096 else large
    return de.RunsLin(rl, threads=th, blocks_per_sm=bps)


def engine_from_packed(packed_path, *, model=None, head: str = "q4", max_len: int = 2048, dev="cuda",
                       ru: int = 1, small=(0, 1), large=(512, 1), **engine_kw):
    """Build a DecodeEngine from a packed file. model: an already-loaded HF causal LM to take the
    non-block tensors from (default: the file's base model, loaded on the CPU in bf16).
    Returns (engine, info) where info has the file summary and the engine's weight bytes."""
    pf = PackedFile(packed_path)
    if model is None:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(pf.summary["base_model"], dtype=torch.bfloat16)
    cfg = cfg_from_hf(model.config, max_len)
    body = model.model
    half = lambda t: None if t is None else t.detach().to(dev, torch.float16).contiguous()  # noqa: E731
    layers = []
    for i, blk in enumerate(body.layers):
        L = {k: runs_linear(pf, f"model.layers.{i}.{n}", dev, ru, small, large) for k, n in NAMES.items()}
        a = blk.self_attn
        L.update(ln1=half(blk.input_layernorm.weight), ln2=half(blk.post_attention_layernorm.weight),
                 bq=half(a.q_proj.bias), bk=half(a.k_proj.bias), bv=half(a.v_proj.bias))
        layers.append(L)
    hw = model.lm_head.weight.detach()
    if head == "fp16":
        lm_head = de.DenseLin(hw.to(dev))
    elif head == "q4":
        lm_head = de.Q4Lin.from_weight(hw, dev=dev)
    else:
        raise ValueError(f"head {head!r}: fp16 | q4")
    eng = de.DecodeEngine(cfg, half(body.embed_tokens.weight), layers, half(body.norm.weight), lm_head, dev,
                          small=small, large=large, **engine_kw)
    info = {"summary": pf.summary, "head": head, "weight_bytes": eng.weight_bytes(),
            "embed_bytes": eng.embed.numel() * 2}
    return eng, info


def head_reference_weight(model, head: str) -> torch.Tensor:
    """The lm_head weight the engine actually multiplies by (for reference comparisons)."""
    w = model.lm_head.weight.detach().float().cpu()
    if head == "q4":
        from .kernels import proto_formats as pf
        return pf.q4_pack(w)[2]
    return w.half().float()


@torch.no_grad()
def feed(eng: de.DecodeEngine, tokens: list[int], keep_logits: bool = False) -> list:
    """Run a prompt through the engine one token at a time from position 0 (eager steps).
    Afterwards eng.tok holds the greedy next token. keep_logits: the logits after each step
    (the prediction for the following position)."""
    if len(tokens) > eng.cfg.max_len:
        raise ValueError(f"prompt of {len(tokens)} tokens > max_len {eng.cfg.max_len}")
    eng.reset(tokens[0])
    out = []
    for i in range(len(tokens)):
        eng.step()
        if keep_logits:
            out.append(eng.logits.float().clone())
        if i + 1 < len(tokens):
            eng.tok.fill_(tokens[i + 1])
    return out


@torch.no_grad()
def generate(eng: de.DecodeEngine, prompt: list[int], n: int, graph: bool = True) -> list[int]:
    """Greedy continuation of `prompt` (n new tokens). graph: replay the captured step graph for
    the generated tokens (the prompt goes through eager steps)."""
    if len(prompt) + n > eng.cfg.max_len:
        raise ValueError("prompt + n exceeds max_len")
    if graph and eng.graph is None:
        eng.capture()             # capture advances the state: feed afterwards
    feed(eng, prompt)
    out = [int(eng.tok)]
    for _ in range(n - 1):
        if graph:
            eng.graph.replay()
        else:
            eng.step()
        out.append(int(eng.tok))
    return out
