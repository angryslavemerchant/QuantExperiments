"""shapequant.runtime_batch: B sequences decoded together == each alone through the batch-1
Qwen2 DecodeEngine (tiny random Qwen2 model, as in test_runtime)."""

import pytest
import torch

cp = pytest.importorskip("cupy")
transformers = pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from shapequant import qwen35_batch as qb, runtime  # noqa: E402
from shapequant.runtime_batch import Qwen2BatchEngine  # noqa: E402
from test_runtime import _hf, _write_packed  # noqa: E402


def _prompts(B, T, seed, ragged=False):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 1024, (T - (3 * b if ragged else 0),), generator=g).tolist() for b in range(B)]


def _single(eng, prompt, n):
    lg = runtime.feed(eng, prompt, keep_logits=True)
    toks = [int(eng.tok)]
    for _ in range(n - 1):
        eng.step()
        lg.append(eng.logits.float().clone())
        toks.append(int(eng.tok))
    return torch.stack(lg), toks


@pytest.mark.parametrize("head,B,layout", [("q8", 3, "tile"), ("fp16", 8, "tile"), ("q4", 2, "lane"), ("q8", 1, "tile")])
def test_batch_matches_single(tmp_path, head, B, layout):
    path = tmp_path / "m.safetensors"
    _write_packed(path, 1, 4)
    eng, _ = runtime.engine_from_packed(path, model=_hf(False), head=head, max_len=64)
    T, n = 9, 8
    prompts = _prompts(B, T, seed=B)
    ref = [_single(eng, p, n) for p in prompts]
    beng = Qwen2BatchEngine(eng, B, layout=layout, nsplit=2 if B > 2 else None)
    lg = qb.feed_batch(beng, prompts, keep_logits=True)
    toks = [[int(t)] for t in beng.tok.tolist()]
    for _ in range(n - 1):
        beng.step()
        lg.append(beng.logits.float().clone())
        for b, t in enumerate(beng.tok.tolist()):
            toks[b].append(int(t))
    got = torch.stack(lg, 1)
    for b in range(B):
        r = ref[b][0]
        rel = float((got[b] - r).norm() / r.norm())
        assert rel < 2e-3, (b, rel)
        assert toks[b] == ref[b][1], b


def test_ragged_slots_and_graph(tmp_path):
    path = tmp_path / "m.safetensors"
    _write_packed(path, 1, 4)
    eng, _ = runtime.engine_from_packed(path, model=_hf(True), head="q8", max_len=64)
    prompts = _prompts(4, 12, seed=5, ragged=True)
    n = 8
    ref = [_single(eng, p, n)[1] for p in prompts]
    beng = Qwen2BatchEngine(eng, 4)
    assert qb.generate_batch(beng, prompts, n, graph=False, feed="slots") == ref
    assert qb.generate_batch(beng, prompts, n, graph=True, feed="slots") == ref


def test_rejects_column_orders(tmp_path):
    path = tmp_path / "m.safetensors"
    _write_packed(path, 4, 1)
    eng, _ = runtime.engine_from_packed(path, model=_hf(False), head="q8", max_len=64)
    with pytest.raises(ValueError):
        Qwen2BatchEngine(eng, 2)
