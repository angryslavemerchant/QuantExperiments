"""Reference points from published VPTQ quantizations (VPTQ-community on the Hub).

Same protocol as the GGUF references: only the block linears are taken from
the checkpoint (decoded to dense weights); embedding, lm_head, norms and
biases stay at the original 16-bit values. Bits per weight come from the
checkpoint's real storage for each linear: packed indices, main and residual
codebooks, outlier parts, permutation and the per-column scale/bias.

The decode follows VPTQ's own pure-PyTorch path (`vptq/ops/quant_gemm.py:
dequant` and `vptq/utils/pack.py:unpack_index_tensor`, Copyright (c)
Microsoft Corporation, MIT License), restricted to what the published
Qwen2.5 checkpoints use, with every unsupported feature rejected loudly.
"""

from __future__ import annotations

import json
import math

import torch

# A correctly decoded matrix is far below this relative error; a wrong layout
# lands near 1-2.
MAX_REL_MSE = 0.5


def unpack_indices(packed: torch.Tensor, index_bits: int, num_elements: int,
                   res_bits: int = 0) -> tuple[torch.Tensor, torch.Tensor | None]:
    """[G, N, P] int32 bit-packed -> ([G, N, num_elements] main, residual or None) int64.
    Bits are little-endian within each int32; each element holds
    index_bits (low) + res_bits (high)."""
    total = index_bits + res_bits
    wf = torch.arange(32, device=packed.device).view(1, 1, 1, -1)
    bits = torch.bitwise_and(torch.bitwise_right_shift(packed.unsqueeze(-1), wf), 1)
    bits = bits.reshape(*packed.shape[:-1], -1)[..., : total * num_elements]
    bits = bits.reshape(*packed.shape[:-1], num_elements, total).to(torch.int64)
    vals = (bits << torch.arange(total, device=packed.device)).sum(-1)
    main = vals & ((1 << index_bits) - 1)
    res = (vals >> index_bits) & ((1 << res_bits) - 1) if res_bits else None
    return main, res


def decode_linear(t: dict, cfg: dict) -> torch.Tensor:
    """Dense float32 [out, in] weight from one VQuantLinear's tensors."""
    if cfg.get("vector_lens", [-1])[0] > 1 and cfg["num_centroids"][0] > 0:
        raise NotImplementedError("VPTQ outlier component")
    if cfg.get("enable_perm"):
        raise NotImplementedError("VPTQ permutation")
    if not cfg.get("is_indice_packed", False):
        raise NotImplementedError("unpacked VPTQ indices")
    v = cfg["vector_lens"][1]
    K = cfg["num_centroids"][1]
    Kr = cfg["num_res_centroids"][1]
    G = cfg["group_num"]
    gs = cfg["group_size"]
    out_f, in_f = cfg["out_features"], cfg["in_features"]
    if G * gs != in_f:
        raise ValueError(f"group_num*group_size {G}*{gs} != in_features {in_f}")
    padding = (-out_f) % v
    ib = int(math.log2(K))
    rb = int(math.log2(Kr)) if Kr > 0 else 0
    idx, ridx = unpack_indices(t["indices"], ib, gs, rb)       # [G, N, gs]

    def gather(cent, ind, Kc):
        c = cent.float().view(G, Kc, v)
        sel = torch.gather(c, 1, ind.reshape(G, -1, 1).expand(-1, -1, v))  # [G, N*gs, v]
        sel = sel.view(G, -1, gs, v).permute(0, 1, 3, 2)                  # [G, N, v, gs]
        return sel.reshape(G, -1, gs).permute(1, 0, 2).reshape(-1, G * gs)  # [N*v, in]

    w = gather(t["centroids.weight"], idx, K)
    if Kr > 0:
        w = w + gather(t["res_centroids.weight"], ridx, Kr)
    if padding:
        w = w[:-padding]
    if cfg.get("enable_norm"):
        w = w * t["weight_scale"].float() + t["weight_bias"].float()
    if w.shape != (out_f, in_f):
        raise ValueError(f"decoded {tuple(w.shape)} != ({out_f}, {in_f})")
    return w


_PARTS = ("indices", "centroids.weight", "res_centroids.weight", "weight_scale", "weight_bias",
          "outlier_indices", "outlier_centroids.weight", "perm")


class VPTQSource:
    """Lazily opened VPTQ checkpoint; decodes one named block linear at a time."""

    def __init__(self, repo: str, revision: str | None = None):
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        cfg = json.load(open(hf_hub_download(repo, "config.json", revision=revision)))
        qc = cfg.get("quantization_config", {})
        if qc.get("quant_method") != "vptq":
            raise ValueError(f"{repo} is not a VPTQ checkpoint")
        self.layers = qc["config_for_layers"]
        self.f = safe_open(hf_hub_download(repo, "model.safetensors", revision=revision),
                           framework="pt", device="cpu")
        self.keys = set(self.f.keys())

    def matrix(self, hf_name: str, shape: tuple[int, int], device=None):
        """Returns (decoded float32 [R, C], storage bits, type label)."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg = self.layers[hf_name]
        if (cfg["out_features"], cfg["in_features"]) != tuple(shape):
            raise ValueError(f"{hf_name}: VPTQ shape {cfg['out_features']}x{cfg['in_features']} != {shape}")
        t, bits = {}, 0
        for p in _PARTS:
            k = f"{hf_name}.{p}"
            if k in self.keys:
                x = self.f.get_tensor(k)
                bits += x.numel() * x.element_size() * 8
                t[p] = x.to(device)
        w = decode_linear(t, cfg)
        label = f"vptq_v{cfg['vector_lens'][1]}_k{cfg['num_centroids'][1]}_r{cfg['num_res_centroids'][1]}"
        return w, float(bits), label
