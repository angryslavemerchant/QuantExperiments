"""vLLM quantization method "shapequant": the block linears of a packed codebook file run on the
shapequant kernels (vllm_ext.core); everything else (attention, DeltaNet, norms, embedding, head,
scheduling, KV cache, sampling) is stock vLLM, except that with "embed_head": "packed" the token
embedding and lm_head also come from the packed file (4-bit embedding, 8-bit head: ~3 GB less
than bf16 on 27B).

Checkpoint layout (scripts/export_vllm_checkpoint.py): an HF directory whose safetensors hold only
the non-block tensors, plus the packed file itself under a non-.safetensors name (vLLM loads every
*.safetensors), named in config.json:

  "quantization_config": {"quant_method": "shapequant", "packed_file": "shapequant.packed"}

Block linears get no vLLM parameters at all (nothing to load or check); their kernels are built
from the packed file when vLLM constructs each layer. Loaded in every vLLM process through the
`vllm.general_plugins` entry point (pyproject.toml) -> register().
"""

from __future__ import annotations

import os
from typing import Any

import torch

from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding

from . import core


def _model_path() -> tuple[str, str | None]:
    from vllm.config import get_current_vllm_config
    mc = get_current_vllm_config().model_config
    return mc.model, getattr(mc, "revision", None)


def _resolve(file: str) -> str:
    if os.path.isabs(file) and os.path.exists(file):
        return file
    model, rev = _model_path()
    if os.path.isdir(model):
        return os.path.join(model, file)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(model, file, revision=rev)


@register_quantization_config("shapequant")
class ShapeQuantConfig(QuantizationConfig):
    def __init__(self, packed_file: str, bmax: int = core.BMAX, layout: str = "tile", large: str = "dequant",
                 chunk_mb: int = 24, xs: bool = False, warps: int = 8, tune: bool = False,
                 embed_head: str = "bf16", compact: str | bool = "auto"):
        super().__init__()
        self.packed_file, self.bmax = packed_file, bmax
        # "tile": the batched kernel reads a repacked copy of every matrix's words (faster warp loads,
        # but a second copy of the weights: 27B 19.8 GB at load vs ~11 on a 3090, s38); "lane" reads
        # them in place. SHAPEQUANT_LAYOUT overrides.
        self.layout = os.environ.get("SHAPEQUANT_LAYOUT", layout)
        # SHAPEQUANT_LARGE / SHAPEQUANT_CHUNK_MB / SHAPEQUANT_BMAX / SHAPEQUANT_XS / SHAPEQUANT_WARPS
        # override the checkpoint's choice (benchmarking)
        self.bmax = int(os.environ.get("SHAPEQUANT_BMAX", bmax))
        self.xs = os.environ.get("SHAPEQUANT_XS", "1" if xs else "0") == "1"
        self.warps = int(os.environ.get("SHAPEQUANT_WARPS", warps))
        # per-shape crossover between the batched kernel and the large-M path, measured at load
        self.tune = os.environ.get("SHAPEQUANT_TUNE", "1" if tune else "0") == "1"
        self.large = os.environ.get("SHAPEQUANT_LARGE", large)
        self.chunk_mb = int(os.environ.get("SHAPEQUANT_CHUNK_MB", chunk_mb))
        # "packed": embedding (q4) and lm_head (q8) from the packed file's aux section; the export
        # then leaves them out of model.safetensors. "bf16": stock vLLM layers loading them.
        if embed_head not in ("bf16", "packed"):
            raise ValueError(f"embed_head {embed_head!r}: bf16 | packed")
        self.embed_head = embed_head
        # compact: drop the lane-major words, keep the tile copy only (half the weight memory; no
        # fused GEMM). "auto": on GPUs under 40 GB. SHAPEQUANT_COMPACT=0/1 overrides.
        # decided lazily (in the engine process, at the first layer): the config is also built in
        # vLLM's front-end process, which must not initialize CUDA
        self._compact_req = os.environ.get("SHAPEQUANT_COMPACT", str(compact)).lower()
        self._compact = None
        self._pf = None
        if embed_head == "packed":
            _patch_qwen35_embedding()

    def __setstate__(self, state):
        # vLLM builds the config in its front-end process and pickles it into the engine process:
        # __init__ never runs there, so the embedding patch must also be installed on unpickle
        # (s38 / s40c: without it embed_tokens was a stock, never-loaded layer -> NLL = log(vocab))
        self.__dict__.update(state)
        if self.embed_head == "packed":
            _patch_qwen35_embedding()

    def __repr__(self) -> str:
        return (f"ShapeQuantConfig(packed_file={self.packed_file!r}, bmax={self.bmax}, large={self.large!r}, "
                f"chunk_mb={self.chunk_mb}, xs={self.xs}, warps={self.warps}, layout={self.layout!r}, "
                f"compact={self._compact_req}, embed_head={self.embed_head!r})")

    @classmethod
    def get_name(cls) -> str:
        return "shapequant"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ShapeQuantConfig":
        return cls(packed_file=config["packed_file"], bmax=int(config.get("bmax", core.BMAX)),
                   layout=config.get("layout", "tile"), large=config.get("large", "dequant"),
                   chunk_mb=int(config.get("chunk_mb", 24)), xs=bool(config.get("xs", False)),
                   warps=int(config.get("warps", 8)), tune=bool(config.get("tune", False)),
                   embed_head=config.get("embed_head", "bf16"), compact=config.get("compact", "auto"))

    @property
    def compact(self) -> bool:
        if self._compact is None:
            c = self._compact_req
            self._compact = c in ("1", "true") or (
                c == "auto" and self.layout == "tile"
                and torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory < 40e9)
        return self._compact

    @property
    def pf(self):
        if self._pf is None:
            from ..packed import PackedFile
            self._pf = PackedFile(_resolve(self.packed_file))
        return self._pf

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        if isinstance(layer, ParallelLMHead):
            return ShapeQuantHeadMethod(self) if self.embed_head == "packed" else None
        if isinstance(layer, VocabParallelEmbedding):
            return ShapeQuantEmbedMethod(self) if self.embed_head == "packed" else None
        if not isinstance(layer, LinearBase):
            return None
        pieces = core.packed_pieces(prefix, self.pf.meta)
        if pieces is None:
            return UnquantizedLinearMethod()
        return ShapeQuantLinearMethod(self, prefix, pieces)


class ShapeQuantLinearMethod(LinearMethodBase):
    def __init__(self, cfg: ShapeQuantConfig, prefix: str, pieces: list[str]):
        self.cfg, self.prefix, self.pieces = cfg, prefix, pieces

    def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int, output_size: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        if input_size_per_partition != input_size or sum(output_partition_sizes) != output_size:
            raise NotImplementedError("shapequant: tensor parallelism is not supported (use TP = 1)")
        dev = torch.device("cuda", torch.cuda.current_device())
        ps = core.PackedShards.from_packed(self.cfg.pf, self.pieces, dev=dev, bmax=self.cfg.bmax,
                                           layout=self.cfg.layout, large=self.cfg.large,
                                           chunk_bytes=self.cfg.chunk_mb << 20, xs=self.cfg.xs,
                                           warps=self.cfg.warps, compact=self.cfg.compact)
        if ps.C != input_size or ps.R != output_size:
            raise ValueError(f"{self.prefix}: packed pieces are [{ps.R}, {ps.C}], vLLM layer is "
                             f"[{output_size}, {input_size}] (partitions {output_partition_sizes})")
        if self.cfg.tune:
            with core.neutral_defaults():
                ps.tune_launch()
                ps.tune()
        layer.sq_handle = core.register(ps)
        layer.sq_shards = ps            # keeps the kernels' tensors alive with the module

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        y = torch.ops.shapequant.linear(x, layer.sq_handle)
        return y if bias is None else y + bias


def _aux_check(cfg: ShapeQuantConfig, rows: int, cols: int, what: str) -> None:
    aux = cfg.pf.aux or {}
    want = {"embed": "q4_b32", "lm_head": "q8_b32"}[what]
    if aux.get(what) != want:
        raise ValueError(f"shapequant embed_head=packed: the packed file's {what} is {aux.get(what)!r}, not {want}")
    if aux.get("vocab") is not None and (int(aux["vocab"]) > rows or int(aux["hidden"]) != cols):
        raise ValueError(f"shapequant: packed {what} is [{aux['vocab']}, {aux['hidden']}], layer is [{rows}, {cols}]")


class ShapeQuantHeadMethod(QuantizeMethodBase):
    """lm_head from the packed file's 8-bit head (core.Q8Head) through the shapequant::linear op."""

    def __init__(self, cfg: ShapeQuantConfig):
        self.cfg = cfg

    def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int, output_size: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        if sum(output_partition_sizes) != output_size:
            raise NotImplementedError("shapequant: tensor parallelism is not supported (use TP = 1)")
        _aux_check(self.cfg, output_size, input_size, "lm_head")
        if not _EMBED_BUILT:          # Qwen3.5 builds embed_tokens before lm_head
            raise RuntimeError("shapequant embed_head=packed: the token embedding was not built by the shapequant "
                               "method (the Qwen3.5 embedding patch did not apply); it would run unloaded")
        dev = torch.device("cuda", torch.cuda.current_device())
        with core.neutral_defaults():
            head = core.Q8Head(self.cfg.pf.aux_tensor("lm_head.q").to(dev).contiguous(),
                               self.cfg.pf.aux_tensor("lm_head.scale").to(dev).contiguous(), R=output_size)
        if head.C != input_size:
            raise ValueError(f"lm_head: packed C {head.C}, layer {input_size}")
        layer.sq_handle = core.register(head)
        layer.sq_shards = head

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        y = torch.ops.shapequant.linear(x, layer.sq_handle)
        return y if bias is None else y + bias

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("shapequant: the lm_head is not an embedding (tie_word_embeddings is unsupported)")


class ShapeQuantEmbedMethod(QuantizeMethodBase):
    """Token embedding from the packed file's 4-bit embedding (core.q4_embed: plain torch ops)."""

    def __init__(self, cfg: ShapeQuantConfig):
        self.cfg = cfg

    def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int, output_size: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        if sum(output_partition_sizes) != output_size:
            raise NotImplementedError("shapequant: tensor parallelism is not supported (use TP = 1)")
        _aux_check(self.cfg, output_size, input_size, "embed")
        dev = torch.device("cuda", torch.cuda.current_device())
        with core.neutral_defaults():
            q = self.cfg.pf.aux_tensor("embed.q").to(dev).contiguous()
            sc = self.cfg.pf.aux_tensor("embed.scale").to(dev).contiguous()
        if q.shape[1] * 2 != input_size:
            raise ValueError(f"embedding: packed C {q.shape[1] * 2}, layer {input_size}")
        emb = core.Q4Embedding(q, sc, params_dtype)
        global _EMBED_BUILT
        _EMBED_BUILT = True
        layer.sq_handle = core.register(emb)
        layer.sq_shards = emb

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        raise RuntimeError("shapequant: the 4-bit embedding is not an output head")

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return torch.ops.shapequant.embed(input_, layer.sq_handle)


_EMBED_BUILT = False


def _patch_qwen35_embedding() -> None:
    """vLLM's Qwen3.5 text model builds embed_tokens without the quant config (only lm_head gets
    it), so a quantization method is never asked about the embedding. Wrap the name its module
    uses so that, under a shapequant config with embed_head=packed, the embedding is built with
    our method (passed as VocabParallelEmbedding's preselected quant_method)."""
    try:
        import vllm.model_executor.models.qwen3_5 as q35
    except ImportError:
        return
    real = q35.VocabParallelEmbedding
    if getattr(real, "_sq_wrapped", False):
        return

    def VocabParallelEmbedding_(*args, **kw):
        from vllm.config import get_current_vllm_config
        qc = get_current_vllm_config().quant_config
        if (isinstance(qc, ShapeQuantConfig) and qc.embed_head == "packed"
                and kw.get("quant_method") is None and kw.get("quant_config") is None):
            kw["quant_method"] = ShapeQuantEmbedMethod(qc)
        return real(*args, **kw)

    VocabParallelEmbedding_._sq_wrapped = True
    q35.VocabParallelEmbedding = VocabParallelEmbedding_


def register() -> None:
    """vllm.general_plugins entry point: importing this module registers the config."""
