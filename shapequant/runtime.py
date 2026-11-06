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


def runs_linear(pf: PackedFile, name: str, dev="cuda", ru: int = 1, small=(0, 1), large=(512, 1),
                tags: torch.Tensor | None = None) -> de.RunsLin:
    """One packed matrix as an engine linear (launch config by size, as in random_engine).
    tags: its unpacked local tags, if already at hand."""
    m = pf.meta[name]
    if m["d"] != 4 or (m["K"].bit_length() - 1) % 2:
        raise ValueError(f"{name}: run kernel needs d=4 and even tag bits (d={m['d']}, K={m['K']})")
    lin = PackedLinear.from_packed(pf, name, device=dev)
    rl = proto_runs.RunsLinear(lin, pf.local_tags(name, dev) if tags is None else tags, ru)
    th, bps = small if m["R"] <= 4096 else large
    return de.RunsLin(rl, threads=th, blocks_per_sm=bps)


def engine_from_packed(packed_path, *, model=None, head: str = "q4", max_len: int = 2048, dev="cuda",
                       ru: int = 1, small=(0, 1), large=(512, 1), prefill: bool = False, **engine_kw):
    """Build a DecodeEngine from a packed file. model: an already-loaded HF causal LM to take the
    non-block tensors from (default: the file's base model, loaded on the CPU in bf16).
    prefill: also keep what batched prefill needs (shapequant.prefill: int16 tags, +4 bits per
    weight, and a dense fp16 lm_head copy).
    Returns (engine, info) where info has the file summary and the engine's weight bytes."""
    pf = PackedFile(packed_path)
    if model is None:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(pf.summary["base_model"], dtype=torch.bfloat16)
    cfg = cfg_from_hf(model.config, max_len)
    body = model.model
    half = lambda t: None if t is None else t.detach().to(dev, torch.float16).contiguous()  # noqa: E731
    layers, names, tag_cache = [], [], {}
    for i, blk in enumerate(body.layers):
        L = {}
        names.append({k: f"model.layers.{i}.{n}" for k, n in NAMES.items()})
        for k, full in names[-1].items():
            t = pf.local_tags(full, dev)
            if prefill:
                tag_cache[full] = t.to(torch.int16)
            L[k] = runs_linear(pf, full, dev, ru, small, large, tags=t)
            del t
        a = blk.self_attn
        L.update(ln1=half(blk.input_layernorm.weight), ln2=half(blk.post_attention_layernorm.weight),
                 bq=half(a.q_proj.bias), bk=half(a.k_proj.bias), bv=half(a.v_proj.bias))
        layers.append(L)
    hw = model.lm_head.weight.detach()
    if head == "fp16":
        lm_head = de.DenseLin(hw.to(dev))
        dense_head = lm_head.w if prefill else None
    elif head == "q4":
        from .kernels import proto_formats
        q, sc, deq = proto_formats.q4_pack(hw.float().cpu())
        lm_head = de.Q4Lin(q.to(dev), sc.to(dev), *hw.shape)
        dense_head = deq.half() if prefill else None
        del deq
    else:
        raise ValueError(f"head {head!r}: fp16 | q4")
    eng = de.DecodeEngine(cfg, half(body.embed_tokens.weight), layers, half(body.norm.weight), lm_head, dev,
                          small=small, large=large, **engine_kw)
    eng.prefill_lins = None
    if prefill:
        from . import prefill as pre
        pre.attach(eng, pf, names, dense_head, tag_cache, dev)
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


def has_prefill(eng) -> bool:
    return getattr(eng, "prefill_lins", None) is not None


@torch.no_grad()
def start(eng: de.DecodeEngine, prompt: list[int], all_logits: bool = False, use_prefill: bool | None = None):
    """Fresh context: run `prompt` from position 0 (batched prefill when the engine has it, else
    eager steps). Returns fp32 logits: [T, vocab] if all_logits (row i predicts token i+1), else
    the last row [vocab]. Afterwards eng.tok = greedy next token, eng.pos = len(prompt)."""
    if use_prefill is None:
        use_prefill = has_prefill(eng)
    if use_prefill:
        from .prefill import prefill
        eng.reset(0)
        return prefill(eng, prompt, logits="all" if all_logits else "last")
    lg = feed(eng, prompt, keep_logits=all_logits)
    return torch.stack(lg) if all_logits else eng.logits.float().clone()


def _sample(logits: torch.Tensor, temperature: float, top_p: float, top_k: int, gen: torch.Generator | None) -> int:
    z = logits.float() / max(temperature, 1e-6)
    if top_k and top_k < z.numel():
        kth = torch.topk(z, top_k).values[-1]
        z = torch.where(z < kth, torch.full_like(z, float("-inf")), z)
    if top_p < 1.0:
        sz, si = torch.sort(z, descending=True)
        cp = torch.softmax(sz, -1).cumsum(-1)
        drop = cp - torch.softmax(sz, -1) > top_p          # keep the smallest prefix reaching top_p
        sz[drop] = float("-inf")
        z = torch.full_like(z, float("-inf")).scatter(0, si, sz)
    return int(torch.multinomial(torch.softmax(z, -1), 1, generator=gen))


@torch.no_grad()
def generate(eng: de.DecodeEngine, prompt: list[int], n: int, graph: bool = True, *, temperature: float = 0.0,
             top_p: float = 1.0, top_k: int = 0, seed: int | None = None, stop_ids=(), stop_fn=None,
             return_logprobs: bool = False, use_prefill: bool | None = None):
    """Continue `prompt` by up to n tokens. temperature 0: greedy (the graph's argmax); else
    sampled with top-k / top-p (seeded by `seed`). Stops after a token in stop_ids (included in
    the output) or when stop_fn(tokens so far) is true. graph: replay the captured step graph.
    Returns the tokens, or (tokens, logprobs) with each chosen token's log-probability under the
    unscaled model distribution if return_logprobs."""
    if len(prompt) + n > eng.cfg.max_len:
        raise ValueError("prompt + n exceeds max_len")
    if graph and eng.graph is None:
        eng.capture()             # capture advances the state: start afterwards
    gen = None
    if temperature > 0:
        gen = torch.Generator(device=eng.logits.device)
        gen.manual_seed(seed if seed is not None else torch.seed() % (2 ** 63))
    stop_ids = set(stop_ids or ())
    logits = start(eng, prompt, use_prefill=use_prefill)
    out, lps = [], []
    while True:
        if temperature > 0:
            t = _sample(logits, temperature, top_p, top_k, gen)
            eng.tok.fill_(t)
        else:
            t = int(eng.tok)
        out.append(t)
        if return_logprobs:
            lps.append(float(torch.log_softmax(logits.float(), -1)[t]))
        if len(out) >= n or t in stop_ids or (stop_fn is not None and stop_fn(out)):
            break
        if graph:
            eng.graph.replay()
        else:
            eng.step()
        logits = eng.logits.float()
    return (out, lps) if return_logprobs else out


@torch.no_grad()
def bench(eng: de.DecodeEngine, pp: int = 512, tg: int = 128, reps: int = 3) -> dict:
    """llama-bench-style speeds: pp = prompt tokens/s of one batched prefill of `pp` tokens
    (needs prefill), tg = generated tokens/s over `tg` graph-replayed steps from an empty
    context. Median of reps."""
    res = {}
    if pp and has_prefill(eng):
        from .prefill import prefill
        ids = torch.randint(0, eng.cfg.vocab, (pp,), device=eng.kc.device)
        ts = []
        for _ in range(reps + 1):           # first rep warms up (allocator, cuBLAS heuristics)
            eng.reset(0)
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            prefill(eng, ids, logits="last")
            b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b) / 1e3)
        ts = sorted(ts[1:])
        res["pp"] = pp
        res["pp_tok_s"] = pp / ts[len(ts) // 2]
    if tg:
        res["tg"] = tg
        res["tg_tok_s"] = 1.0 / eng.time_tokens(tg, reps)
    return res
