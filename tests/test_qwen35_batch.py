"""shapequant.qwen35_batch: B sequences decoded together == each sequence alone through the
batch-1 Qwen35Engine (tiny random Qwen3.5 model, as in test_qwen35_runtime)."""

import pytest
import torch

cp = pytest.importorskip("cupy")
transformers = pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from shapequant import qwen35_batch as qb, qwen35_runtime as q35  # noqa: E402
from test_qwen35_runtime import _hf, _write_packed  # noqa: E402


def _prompts(B, T, seed, ragged=False):
    g = torch.Generator().manual_seed(seed)
    lens = [T - (3 * b if ragged else 0) for b in range(B)]
    return [torch.randint(0, 1024, (n,), generator=g).tolist() for n in lens]


@pytest.fixture(scope="module")
def engines(tmp_path_factory):
    out = {}
    for hv in (4, 2):
        path = tmp_path_factory.mktemp(f"q35b{hv}") / "m.safetensors"
        _write_packed(path, _hf(hv=hv), aux=True)
        out[hv] = q35.engine_from_packed(path, max_len=64)[0]
    return out


def _single(eng, prompt, n):
    """Per-step logits while feeding `prompt`, then n greedy tokens with their logits (eager)."""
    lg = q35.feed(eng, prompt, keep_logits=True)
    toks = [int(eng.tok)]
    for _ in range(n - 1):
        eng.step()
        lg.append(eng.logits.float().clone())
        toks.append(int(eng.tok))
    return torch.stack(lg), toks


@pytest.mark.parametrize("hv", [4, 2])
@pytest.mark.parametrize("B", [1, 3, 8])
def test_batch_matches_single(engines, hv, B):
    eng = engines[hv]
    T, n = 9, 8
    prompts = _prompts(B, T, seed=B + hv)
    ref = [_single(eng, p, n) for p in prompts]
    beng = qb.Qwen35BatchEngine(eng, B)
    lg = qb.feed_batch(beng, prompts, keep_logits=True)
    toks = [[int(t)] for t in beng.tok.tolist()]
    for _ in range(n - 1):
        beng.step()
        lg.append(beng.logits.float().clone())
        for b, t in enumerate(beng.tok.tolist()):
            toks[b].append(int(t))
    got = torch.stack(lg, 1)                                     # [B, T + n - 1, vocab]
    for b in range(B):
        r = ref[b][0]
        rel = float((got[b] - r).norm() / r.norm())
        if B == 1:
            assert torch.equal(got[b], r)                        # batch-1 kernels throughout
        assert rel < 2e-3, (b, rel)
        assert toks[b] == ref[b][1], b


@pytest.mark.parametrize("B", [4])
def test_ragged_slots_and_graph(engines, B):
    """Prompts of different lengths loaded per slot (different positions per sequence); graph
    replay == eager steps == each sequence alone."""
    eng = engines[4]
    prompts = _prompts(B, 12, seed=7, ragged=True)
    n = 8
    ref = [_single(eng, p, n)[1] for p in prompts]
    beng = qb.Qwen35BatchEngine(eng, B)
    eager = qb.generate_batch(beng, prompts, n, graph=False, feed="slots")
    graph = qb.generate_batch(beng, prompts, n, graph=True, feed="slots")
    assert eager == graph
    assert eager == ref
    # equal-length prompts through the graph-replayed batched step
    eq = _prompts(B, 10, seed=8)
    assert qb.generate_batch(beng, eq, n, graph=True, feed="batch") == [_single(eng, p, n)[1] for p in eq]


def test_prefill_slots(engines):
    """Slots loaded with the batch-1 engine's batched prefill."""
    eng = engines[2]
    q35.attach_prefill(eng)
    try:
        prompts = _prompts(3, 20, seed=11, ragged=True)
        n = 6
        ref = [_single(eng, p, n)[1] for p in prompts]
        beng = qb.Qwen35BatchEngine(eng, 3)
        assert qb.generate_batch(beng, prompts, n, graph=True) == ref
    finally:
        eng.prefill_lins = None
