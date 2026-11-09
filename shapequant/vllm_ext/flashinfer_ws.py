"""One FlashInfer float workspace per GPU, allocated before vLLM's memory profiling (vLLM 0.33.x).

With fp8 KV on Ampere vLLM runs attention through FlashInfer. Each FlashInferMetadataBuilder
gets a float workspace of max(394 MiB, max_num_batched_tokens * q_heads * head_dim * 16 B):
805 MiB for Qwen3.8-27B at an 8192-token prefill chunk. vLLM's V2 model runner shares one
buffer across the target model's attention groups, but allocates it when the builders are
created, after the KV cache was sized from the memory profile, and a speculative drafter's
builders (DSpark) get a second one. On the work 3090 that was ~1.6-1.8 GB outside the
gpu_memory_utilization budget: VRAM spilled to system RAM under WSL and the server slowed ~5x.

Here the buffer is allocated once per device when the first FlashInferImpl is constructed
(model init, i.e. before profiling, so the budget counts it), sized for the largest layer seen
by then, and every builder (target and drafter) uses it. The float workspace is scratch for a
kernel run; all of vLLM's attention runs on one stream, as the V2 runner's own sharing assumes.

Installed from the plugin's register(); SHAPEQUANT_FI_WORKSPACE=0 leaves vLLM untouched.
"""

from __future__ import annotations

import os

import torch
from vllm.logger import init_logger

logger = init_logger("vllm.shapequant")
_WS: dict = {}          # device str -> uint8 tensor


def _size(tokens: int, heads: int, head_dim: int) -> int:
    from vllm import envs
    return max(envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE, tokens * heads * head_dim * 16)


def _get(dev, nbytes: int) -> torch.Tensor:
    k = str(dev)
    b = _WS.get(k)
    if b is None or b.numel() < nbytes:
        if b is not None:
            logger.warning("shapequant: FlashInfer workspace grown %.0f -> %.0f MiB after start-up "
                           "(not in the memory budget)", b.numel() / 2**20, nbytes / 2**20)
        b = _WS[k] = torch.zeros(nbytes, dtype=torch.uint8, device=dev)
    return b


def install() -> bool:
    if os.environ.get("SHAPEQUANT_FI_WORKSPACE", "1") == "0":
        return False
    import vllm
    if not vllm.__version__.startswith("0.33."):
        return False
    try:
        from vllm.v1.attention.backends import flashinfer as fi
    except Exception as e:  # noqa: BLE001 - FlashInfer not installed / importable
        logger.info("shapequant: FlashInfer workspace patch skipped (%r)", e)
        return False
    if getattr(fi, "_shapequant_ws", False):
        return True
    from vllm.config import get_current_vllm_config
    if envs_batch_invariant():
        return False

    orig_init = fi.FlashInferImpl.__init__

    def impl_init(self, num_heads, head_size, *args, **kwargs):
        orig_init(self, num_heads, head_size, *args, **kwargs)
        try:
            cfg = get_current_vllm_config()
            tokens = cfg.scheduler_config.max_num_batched_tokens
            dev = torch.device("cuda", torch.cuda.current_device())
            n = _size(tokens, num_heads, head_size)
            if str(dev) not in _WS or _WS[str(dev)].numel() < n:
                _get(dev, n)
                logger.info("shapequant: shared FlashInfer workspace %.0f MiB (allocated before profiling)",
                            n / 2**20)
        except Exception as e:  # noqa: BLE001
            logger.warning("shapequant: FlashInfer workspace pre-allocation failed (%r)", e)

    def get_ws(self):
        if self._workspace_buffer is None:
            n = _size(self.max_num_batched_tokens, self.num_qo_heads, self.head_dim)
            self._workspace_buffer = _get(self.device, n)
        return self._workspace_buffer

    fi.FlashInferImpl.__init__ = impl_init
    fi.FlashInferMetadataBuilder._get_workspace_buffer = get_ws
    fi._shapequant_ws = True
    return True


def envs_batch_invariant() -> bool:
    from vllm import envs
    return bool(getattr(envs, "VLLM_BATCH_INVARIANT", False))
