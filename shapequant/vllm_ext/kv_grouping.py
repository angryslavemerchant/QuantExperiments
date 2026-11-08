"""Least-waste group size for vLLM's hybrid KV cache groups (vLLM 0.33.x).

vLLM splits a hybrid model's layers into KV cache groups of one common size, padding the last
group of each layer type (`_get_kv_cache_groups_uniform_page_size`). It takes the smallest layer
count among the types as that size. With a speculative drafter that is the drafter's handful of
layers: Qwen3.8-27B (16 full attention + 48 Gated DeltaNet layers) plus the DSpark drafter (5
sliding-window layers) got group size 5, so the 16 attention layers were padded to 20 and every
context token paid for 4 phantom layers (~25% of the attention KV, measured on the work 3090).

Here every group size from that minimum up to the largest count is scored by the padded pages
it costs one request at max_model_len: each layer type's padding layers times the blocks that
type needs (its spec's max_memory_usage_bytes / page size). The cheapest wins (ties: the larger
size, i.e. fewer groups). For 16 + 48 + 5 that is 8: no padding for the target's layers, 3 padding
layers in the drafter's small sliding-window group. Without a drafter (16 + 48) the choice is 16,
the same as vLLM's.

Installed from the shapequant plugin's register() in every vLLM process; SHAPEQUANT_KV_GROUPING=0
leaves vLLM untouched. Falls back to vLLM's own grouping on any error.
"""

from __future__ import annotations

import os
from collections import defaultdict

from vllm.logger import init_logger

logger = init_logger("vllm.shapequant")
_CFG: dict = {}


def _least_waste_groups(kp, kv_cache_spec: dict, vllm_config) -> list:
    # same bucketing as vLLM: group layers by spec, then merge mergeable specs
    same_type: dict = defaultdict(list)
    for name, spec in kv_cache_spec.items():
        same_type[spec].append(name)
    buckets, spec_buckets = [], []
    for spec, names in same_type.items():
        for b, sb in zip(buckets, spec_buckets):
            try:
                type(sb[0]).merge([*sb, spec])
            except (AssertionError, ValueError):
                continue
            b.extend(names)
            sb.append(spec)
            break
        else:
            buckets.append(list(names))
            spec_buckets.append([spec])
    blocks = [max(-(-s.max_memory_usage_bytes(vllm_config) // s.page_size_bytes) for s in sb)
              for sb in spec_buckets]
    lo, hi = min(len(b) for b in buckets), max(len(b) for b in buckets)
    best = None
    for g in range(lo, hi + 1):
        waste = sum(((-len(b)) % g) * nb for b, nb in zip(buckets, blocks))
        if best is None or waste < best[0] or (waste == best[0] and g > best[1]):
            best = (waste, g)
    waste, g = best
    grouped = []
    for b in buckets:
        n = -(-len(b) // g)
        grouped.extend(b[i::n] for i in range(n))
    stock = lo if hi >= lo * 1.5 else hi
    stock_waste = sum(((-len(b)) % stock) * nb for b, nb in zip(buckets, blocks))
    logger.info("shapequant: KV cache group size %d (vLLM's rule: %d); padded pages per max-length "
                "request %d (vLLM's rule: %d); layers per type %s", g, stock, waste, stock_waste,
                [len(b) for b in buckets])
    return kp.create_kv_cache_group_specs(kv_cache_spec, grouped)


def install() -> bool:
    if os.environ.get("SHAPEQUANT_KV_GROUPING", "1") == "0":
        return False
    import vllm
    if not vllm.__version__.startswith("0.33."):
        logger.warning("shapequant: KV grouping patch written for vLLM 0.33.x, not %s: skipped", vllm.__version__)
        return False
    from vllm.v1.core import kv_cache_planning as kp
    if getattr(kp, "_shapequant_grouping", False):
        return True
    orig_uniform = kp._get_kv_cache_groups_uniform_page_size
    orig_groups = kp.get_kv_cache_groups

    def get_kv_cache_groups(vllm_config, kv_cache_spec):
        _CFG["cfg"] = vllm_config
        try:
            return orig_groups(vllm_config, kv_cache_spec)
        finally:
            _CFG.pop("cfg", None)

    def uniform_page_size(kv_cache_spec):
        cfg = _CFG.get("cfg")
        if cfg is None:
            return orig_uniform(kv_cache_spec)
        try:
            return _least_waste_groups(kp, kv_cache_spec, cfg)
        except Exception as e:  # noqa: BLE001 - never break vLLM over an optimization
            logger.warning("shapequant: KV grouping patch failed (%r), using vLLM's", e)
            return orig_uniform(kv_cache_spec)

    kp.get_kv_cache_groups = get_kv_cache_groups
    kp._get_kv_cache_groups_uniform_page_size = uniform_page_size
    kp._shapequant_grouping = True
    return True
