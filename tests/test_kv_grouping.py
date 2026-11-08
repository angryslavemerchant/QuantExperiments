"""shapequant.vllm_ext.kv_grouping: least-waste group size, with stub KV specs (no GPU)."""

from dataclasses import dataclass
import types

import pytest

pytest.importorskip("vllm")
from shapequant.vllm_ext import kv_grouping  # noqa: E402


@dataclass(frozen=True)
class _Spec:
    kind: str
    blocks: int                 # blocks one max-length request needs
    page_size_bytes: int = 1000

    def max_memory_usage_bytes(self, cfg):
        return self.blocks * self.page_size_bytes

    @classmethod
    def merge(cls, specs):
        if len({s.kind for s in specs}) > 1:
            raise ValueError("different kinds")
        return specs[0]


def _kp():
    return types.SimpleNamespace(create_kv_cache_group_specs=lambda spec, grouped: grouped)


def _specs(n_attn=16, n_gdn=48, n_draft=5, attn_blocks=20, gdn_blocks=9, draft_blocks=14):
    s = {f"attn.{i}": _Spec("full", attn_blocks) for i in range(n_attn)}
    s |= {f"gdn.{i}": _Spec("mamba", gdn_blocks) for i in range(n_gdn)}
    s |= {f"draft.{i}": _Spec("swa", draft_blocks) for i in range(n_draft)}
    return s


def test_qwen38_dspark_picks_8_without_target_padding():
    groups = kv_grouping._least_waste_groups(_kp(), _specs(), cfg := object())
    sizes = sorted({len(g) for g in groups})
    attn = [g for g in groups if g[0].startswith("attn")]
    gdn = [g for g in groups if g[0].startswith("gdn")]
    draft = [g for g in groups if g[0].startswith("draft")]
    assert len(attn) == 2 and all(len(g) == 8 for g in attn)      # 16 = 2 x 8, no padding
    assert len(gdn) == 6 and all(len(g) == 8 for g in gdn)        # 48 = 6 x 8
    assert len(draft) == 1 and len(draft[0]) == 5                 # padded to 8 by vLLM
    assert sizes == [5, 8]
    assert sorted(n for g in groups for n in g) == sorted(_specs())      # every layer exactly once


def test_no_drafter_matches_vllm_rule():
    s = _specs(n_draft=0)
    groups = kv_grouping._least_waste_groups(_kp(), s, object())
    assert all(len(g) == 16 for g in groups) and len(groups) == 4       # vLLM's own choice: 16


def test_ties_prefer_fewer_groups():
    s = {f"a.{i}": _Spec("a", 10) for i in range(8)} | {f"b.{i}": _Spec("b", 10) for i in range(8)}
    groups = kv_grouping._least_waste_groups(_kp(), s, object())
    assert len(groups) == 2
