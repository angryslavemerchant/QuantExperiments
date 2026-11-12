"""Calibration statistics for importance-weighted fitting (Stage 2c).

For every block linear, h_c = mean over calibration tokens of x_c^2, the
diagonal of the input second-moment matrix (what llama.cpp's imatrix and the
diagonal of GPTQ's Hessian estimate). Collected once on the ORIGINAL model.

Calibration text is never the evaluation text: the default is FineWeb-Edu
(web text), not WikiText-2, so the weighting gets no in-distribution edge.
"""

from __future__ import annotations

import torch

from .pipeline import compressible_linears


def doc_texts(src: dict, tok):
    """Texts of one source, in order. src: {dataset, subset, split, field (default "text")} for a
    Hugging Face dataset (streamed), with field "messages" rendered by the chat template; or
    {jsonl: path} with a "text" (or src["field"]) per line. skip_docs drops the first n documents."""
    import json
    from itertools import islice

    field = src.get("field", "text")
    if src.get("jsonl"):
        def it():
            with open(src["jsonl"], encoding="utf-8") as f:
                for line in f:
                    yield json.loads(line)
        rows = it()
    else:
        from datasets import load_dataset
        rows = load_dataset(src["dataset"], src.get("subset"), split=src["split"], streaming=True)
    for row in islice(rows, src.get("skip_docs", 0), None):
        if field == "messages":
            yield tok.apply_chat_template(row["messages"], tokenize=False)
        elif src.get("template"):
            yield src["template"].format(**row)
        else:
            yield row[field]


def _source_sequences(src: dict, tok, context: int):
    """Endless-until-exhausted generator of [context] token lists from one source (docs joined with EOS)."""
    eos = tok.eos_token_id
    buf: list[int] = []
    for text in doc_texts(src, tok):
        buf.extend(tok(text, add_special_tokens=False)["input_ids"])
        buf.append(eos)
        while len(buf) >= context:
            yield buf[:context]
            buf = buf[context:]


def mixture_schedule(weights: list[float], n: int) -> list[int]:
    """Source index of each of n sequences: deterministic weighted round robin (largest deficit first),
    so any prefix of the schedule is the schedule of that shorter length (fine-tune data taken after the
    calibration sequences never overlaps them)."""
    tot = sum(weights)
    got = [0] * len(weights)
    out = []
    for i in range(n):
        j = max(range(len(weights)), key=lambda k: (weights[k] / tot) * (i + 1) - got[k])
        got[j] += 1
        out.append(j)
    return out


def load_calibration_sequences(tok, ccfg: dict, seed: int = 0) -> torch.Tensor:
    """[num_sequences, context] token ids, from streamed documents joined with EOS. With
    ccfg["sources"] = [{..., weight}], a deterministic mixture: each sequence comes whole from one
    source (see mixture_schedule and doc_texts)."""
    if ccfg.get("sources"):
        srcs = ccfg["sources"]
        sched = mixture_schedule([float(s.get("weight", 1.0)) for s in srcs], ccfg["num_sequences"])
        gens = [_source_sequences(s, tok, ccfg["context"]) for s in srcs]
        rows = []
        for j in sched:
            try:
                rows.append(next(gens[j]))
            except StopIteration:
                raise RuntimeError(f"calibration source {srcs[j]} too small") from None
        return torch.tensor(rows)
    from datasets import load_dataset

    ds = load_dataset(ccfg["dataset"], ccfg.get("subset"), split=ccfg["split"], streaming=True)
    if ccfg.get("shuffle_buffer"):
        ds = ds.shuffle(seed=seed, buffer_size=ccfg["shuffle_buffer"])
    need = ccfg["num_sequences"] * ccfg["context"]
    eos = tok.eos_token_id
    ids: list[int] = []
    for doc in ds:
        ids.extend(tok(doc[ccfg.get("text_field", "text")], add_special_tokens=False)["input_ids"])
        ids.append(eos)
        if len(ids) >= need:
            break
    if len(ids) < need:
        raise RuntimeError(f"calibration source too small: {len(ids)} < {need} tokens")
    return torch.tensor(ids[:need]).reshape(ccfg["num_sequences"], ccfg["context"])


@torch.no_grad()
def column_importance(model, sequences: torch.Tensor, batch_size: int = 4) -> dict[str, torch.Tensor]:
    """Mean squared input activation per input column, for every block linear (CPU fp32)."""
    sums: dict[str, torch.Tensor] = {}
    count = 0
    hooks = []

    def make_hook(name):
        def hook(_mod, args):
            x = args[0].detach().reshape(-1, args[0].shape[-1]).to(torch.float32)
            s = (x * x).sum(0)
            sums[name] = s if name not in sums else sums[name] + s
        return hook

    for name, mod in compressible_linears(model):
        hooks.append(mod.register_forward_pre_hook(make_hook(name)))
    try:
        dev = next(model.parameters()).device
        for s in range(0, sequences.shape[0], batch_size):
            b = sequences[s : s + batch_size].to(dev)
            model.model(input_ids=b, use_cache=False)  # skip lm_head: not needed
            count += b.numel()
    finally:
        for h in hooks:
            h.remove()
    return {n: (v / count).cpu() for n, v in sums.items()}


def _layer_index(name: str) -> int:
    return int(name.split(".")[2])


def layer_groups(sizes: list[tuple[int, int]], budget: int) -> list[list[int]]:
    """Contiguous groups of layer indices whose summed sizes stay within budget (bytes).
    sizes: (layer index, bytes) in layer order; a single layer over budget gets its own group."""
    groups: list[list[int]] = []
    cur, tot = [], 0
    for li, b in sizes:
        if cur and tot + b > budget:
            groups.append(cur)
            cur, tot = [], 0
        cur.append(li)
        tot += b
    if cur:
        groups.append(cur)
    return groups


class _StopForward(Exception):
    pass


@torch.no_grad()
def input_hessians(model, sequences: torch.Tensor, batch_size: int = 4,
                   max_gpu_bytes: int | None = None, log=None) -> dict[str, torch.Tensor]:
    """Full input second-moment matrix H = E[x x^T] per block linear (CPU fp32), for LDLQ
    and rotations. Linears reading the same input (q/k/v, gate/up, in_proj_qkv/z) share one tensor.
    diag(H) equals `column_importance`.

    The fp32 accumulators of all layers at once (~100 GB at 27B) need not fit on the GPU: layers
    are collected in groups of at most max_gpu_bytes (default: a third of the free GPU memory),
    one calibration pass per group, each pass stopping after the group's last layer. Every
    accumulator sums the same batches in the same order either way, so the result is identical."""
    rep: dict[str, str] = {}          # linear name -> name whose input it shares
    lins = compressible_linears(model)
    mods = dict(lins)
    for name, _ in lins:
        kind = name.rsplit(".", 1)[1]
        first = {"q_proj": "q_proj", "k_proj": "q_proj", "v_proj": "q_proj", "gate_proj": "gate_proj",
                 "up_proj": "gate_proj", "in_proj_z": "in_proj_qkv"}.get(kind, kind)
        prefix = name[: -len(kind)]
        rep[name] = prefix + first
    owners = [n for n, _ in lins if rep[n] == n]
    dev = next(model.parameters()).device
    per_layer: dict[int, int] = {}
    for n in owners:
        C = mods[n].weight.shape[1]
        per_layer[_layer_index(n)] = per_layer.get(_layer_index(n), 0) + C * C * 4
    if max_gpu_bytes is None:
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            max_gpu_bytes = torch.cuda.mem_get_info(dev)[0] // 3
        else:
            max_gpu_bytes = 1 << 62
    groups = layer_groups(sorted(per_layer.items()), max_gpu_bytes)
    if log and len(groups) > 1:
        log(f"  input Hessians: {sum(per_layer.values()) / 1e9:.1f} GB of accumulators in {len(groups)} passes "
            f"(<= {max_gpu_bytes / 1e9:.1f} GB each)")
    layers = model.model.layers
    out: dict[str, torch.Tensor] = {}
    for g in groups:
        gset = set(g)
        sums: dict[str, torch.Tensor] = {}
        count = 0

        def make_hook(name):
            def hook(_mod, args):
                x = args[0].detach().reshape(-1, args[0].shape[-1]).to(torch.float32)
                s = x.T @ x
                if name in sums:
                    sums[name] += s
                else:
                    sums[name] = s
            return hook

        def stop(_mod, _args, _out):
            raise _StopForward

        hooks = [mods[n].register_forward_pre_hook(make_hook(n)) for n in owners if _layer_index(n) in gset]
        if g[-1] < len(layers) - 1:
            hooks.append(layers[g[-1]].register_forward_hook(stop))
        try:
            for s in range(0, sequences.shape[0], batch_size):
                b = sequences[s : s + batch_size].to(dev)
                try:
                    model.model(input_ids=b, use_cache=False)
                except _StopForward:
                    pass
                count += b.numel()
        finally:
            for h in hooks:
                h.remove()
        for n, v in sums.items():
            out[n] = (v / count).cpu()
        del sums
    return {n: out[rep[n]] for n, _ in lins}
