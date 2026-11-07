"""vLLM quantization method "shapequant": the block linears of a packed codebook file run on the
shapequant kernels (vllm_ext.core); everything else (attention, DeltaNet, norms, embedding, head,
scheduling, KV cache, sampling) is stock vLLM.

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
                 chunk_mb: int = 24, xs: bool = False, warps: int = 8):
        super().__init__()
        self.packed_file, self.bmax, self.layout = packed_file, bmax, layout
        # SHAPEQUANT_LARGE / SHAPEQUANT_CHUNK_MB / SHAPEQUANT_BMAX / SHAPEQUANT_XS / SHAPEQUANT_WARPS
        # override the checkpoint's choice (benchmarking)
        self.bmax = int(os.environ.get("SHAPEQUANT_BMAX", bmax))
        self.xs = os.environ.get("SHAPEQUANT_XS", "1" if xs else "0") == "1"
        self.warps = int(os.environ.get("SHAPEQUANT_WARPS", warps))
        self.large = os.environ.get("SHAPEQUANT_LARGE", large)
        self.chunk_mb = int(os.environ.get("SHAPEQUANT_CHUNK_MB", chunk_mb))
        self._pf = None

    def __repr__(self) -> str:
        return (f"ShapeQuantConfig(packed_file={self.packed_file!r}, bmax={self.bmax}, large={self.large!r}, "
                f"chunk_mb={self.chunk_mb}, xs={self.xs}, warps={self.warps})")

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
                   warps=int(config.get("warps", 8)))

    @property
    def pf(self):
        if self._pf is None:
            from ..packed import PackedFile
            self._pf = PackedFile(_resolve(self.packed_file))
        return self._pf

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
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
                                           warps=self.cfg.warps)
        if ps.C != input_size or ps.R != output_size:
            raise ValueError(f"{self.prefix}: packed pieces are [{ps.R}, {ps.C}], vLLM layer is "
                             f"[{output_size}, {input_size}] (partitions {output_partition_sizes})")
        layer.sq_handle = core.register(ps)
        layer.sq_shards = ps            # keeps the kernels' tensors alive with the module

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        y = torch.ops.shapequant.linear(x, layer.sq_handle)
        return y if bias is None else y + bias


def register() -> None:
    """vllm.general_plugins entry point: importing this module registers the config."""
