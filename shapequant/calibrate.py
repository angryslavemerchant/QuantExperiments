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


def load_calibration_sequences(tok, ccfg: dict, seed: int = 0) -> torch.Tensor:
    """[num_sequences, context] token ids, from streamed documents joined with EOS."""
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


@torch.no_grad()
def input_hessians(model, sequences: torch.Tensor, batch_size: int = 4) -> dict[str, torch.Tensor]:
    """Full input second-moment matrix H = E[x x^T] per block linear (CPU fp32), for LDLQ
    and rotations. Linears reading the same input (q/k/v, gate/up) share one tensor.
    diag(H) equals `column_importance`."""
    sums: dict[str, torch.Tensor] = {}
    rep: dict[str, str] = {}          # linear name -> name whose input it shares
    count = 0
    hooks = []
    lins = compressible_linears(model)
    for name, _ in lins:
        kind = name.rsplit(".", 1)[1]
        first = {"q_proj": "q_proj", "k_proj": "q_proj", "v_proj": "q_proj", "gate_proj": "gate_proj",
                 "up_proj": "gate_proj"}.get(kind, kind)
        prefix = name[: -len(kind)]
        rep[name] = prefix + first

    def make_hook(name):
        def hook(_mod, args):
            x = args[0].detach().reshape(-1, args[0].shape[-1]).to(torch.float32)
            s = x.T @ x
            if name in sums:
                sums[name] += s
            else:
                sums[name] = s
        return hook

    for name, mod in lins:
        if rep[name] == name:
            hooks.append(mod.register_forward_pre_hook(make_hook(name)))
    try:
        dev = next(model.parameters()).device
        for s in range(0, sequences.shape[0], batch_size):
            b = sequences[s : s + batch_size].to(dev)
            model.model(input_ids=b, use_cache=False)
            count += b.numel()
    finally:
        for h in hooks:
            h.remove()
    out = {n: (v / count).cpu() for n, v in sums.items()}
    return {n: out[rep[n]] for n, _ in lins}
