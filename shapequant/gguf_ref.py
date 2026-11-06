"""Reference points from published llama.cpp GGUF quantizations.

Only the block linears are taken from the GGUF file (dequantized with gguf-py);
embedding, lm_head, norms and biases stay at the original 16-bit values, which
is the same protocol as every other method here. Bits per weight come from the
GGUF tensors' real storage size, so mixed-precision files (Q2_K, IQ2_M, ...)
are reported at their true average over the compressed layers.
"""

from __future__ import annotations

import numpy as np
import torch

# HF suffix -> GGUF tensor name suffix (llama.cpp naming, block index i).
_GGUF_NAMES = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
    # Qwen3.5 / 3.8 (qwen35) Gated DeltaNet layers
    "linear_attn.in_proj_qkv": "attn_qkv",
    "linear_attn.in_proj_z": "attn_gate",
    "linear_attn.out_proj": "ssm_out",
}

# A correctly mapped matrix from any real quant is far below this relative error;
# a transposed or permuted mapping lands near 2.0.
MAX_REL_MSE = 0.5


def gguf_tensor_name(hf_name: str) -> str:
    """'model.layers.3.mlp.down_proj' -> 'blk.3.ffn_down.weight'."""
    parts = hf_name.split(".")
    if parts[:2] != ["model", "layers"]:
        raise ValueError(f"not a block linear: {hf_name}")
    suffix = ".".join(parts[3:])
    return f"blk.{int(parts[2])}.{_GGUF_NAMES[suffix]}.weight"


class GGUFSource:
    """Lazily opened GGUF file; dequantizes one named block linear at a time."""

    def __init__(self, path: str):
        from gguf import GGUFReader

        self.path = path
        self.reader = GGUFReader(path)
        self.tensors = {t.name: t for t in self.reader.tensors}
        self.vhead_rows = self._vhead_rows()

    def _field(self, key: str):
        f = self.reader.fields.get(key)
        return None if f is None else int(f.parts[-1][0])

    def _vhead_rows(self) -> torch.Tensor | None:
        """llama.cpp stores Gated DeltaNet value heads reordered when there are more value than
        key heads (qwen35 / qwen3next, num_v_heads > num_k_heads): it tiles the key heads
        (value head j uses key head j % HK) where HF repeats them (head h uses key head h // rep),
        so GGUF value head j is HF head rep * (j % HK) + j // HK. Returns, per GGUF value row, the
        HF row it holds (None when no reorder applies). Verified on Qwen3.8-27B (HK 16, HV 48)."""
        arch = None
        f = self.reader.fields.get("general.architecture")
        if f is not None:
            arch = bytes(f.parts[-1]).decode()
        if arch is None:
            return None
        hk = self._field(f"{arch}.ssm.group_count")
        hv = self._field(f"{arch}.ssm.time_step_rank")
        inner = self._field(f"{arch}.ssm.inner_size")
        if not hk or not hv or not inner or hv == hk:
            return None
        if hv % hk:
            raise ValueError(f"{self.path}: {hv} value heads not a multiple of {hk} key heads")
        dv, rep = inner // hv, hv // hk
        perm = torch.tensor([rep * (j % hk) + j // hk for j in range(hv)])
        return (perm[:, None] * dv + torch.arange(dv)[None]).reshape(-1)

    def matrix(self, hf_name: str, shape: tuple[int, int]) -> tuple[torch.Tensor, float, str]:
        """Returns (dequantized float32 [R, C], storage bits, ggml type name)."""
        from gguf import dequantize

        t = self.tensors[gguf_tensor_name(hf_name)]
        w = np.asarray(dequantize(t.data, t.tensor_type), dtype=np.float32)
        if w.size != shape[0] * shape[1]:
            raise ValueError(f"{t.name}: {w.shape} does not match HF shape {shape}")
        out = torch.from_numpy(w.reshape(shape).copy())
        rows = self.vhead_rows
        if rows is not None and ".linear_attn." in hf_name:
            kind = hf_name.rsplit(".", 1)[1]
            nv = rows.numel()
            fixed = out.clone()
            if kind == "in_proj_z":                       # value_dim rows
                fixed[rows] = out
            elif kind == "in_proj_qkv":                   # [q | k | v] rows: only v is reordered
                off = shape[0] - nv
                fixed[off + rows] = out[off:]
            elif kind == "out_proj":                      # value_dim input columns
                fixed[:, rows] = out
            out = fixed
        return out, float(t.n_bytes) * 8, t.tensor_type.name


    def head_and_embed(self, vocab: int, hidden: int) -> dict:
        """The file's own output / token_embd, dequantized float32 [vocab, hidden] (output falls
        back to token_embd for tied files), with their ggml types and storage bits."""
        from gguf import dequantize

        out = {}
        for key, name in (("embed", "token_embd.weight"), ("head", "output.weight")):
            t = self.tensors.get(name) or self.tensors["token_embd.weight"]
            w = np.asarray(dequantize(t.data, t.tensor_type), dtype=np.float32)
            if w.size != vocab * hidden:
                raise ValueError(f"{t.name}: {w.shape} does not match [{vocab}, {hidden}]")
            out[key] = torch.from_numpy(w.reshape(vocab, hidden).copy())
            out[f"{key}_type"] = t.tensor_type.name
            out[f"{key}_bits"] = float(t.n_bytes) * 8
        return out


def resolve(run: dict) -> str:
    """Local path of the run's GGUF file, downloading from the Hub if needed."""
    if run.get("path"):
        return run["path"]
    from huggingface_hub import hf_hub_download

    return hf_hub_download(run["repo"], run["file"])
