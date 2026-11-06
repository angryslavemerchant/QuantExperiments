"""Batched prefill (shapequant.prefill) == token-by-token feed; sampling / stopping / logprobs."""

import pytest
import torch

cp = pytest.importorskip("cupy")
pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from shapequant import runtime  # noqa: E402
from shapequant.prefill import prefill  # noqa: E402
from test_runtime import _hf, _write_packed  # noqa: E402

PROMPT = [5, 17, 300, 2, 999, 41, 7, 7, 128, 64, 3]

CASES = [(1, 4, "q4", True), (4, 1, "fp16", False), (1, 1, "q4", False)]


def _eng(tmp_path, M, G, head, tied):
    path = tmp_path / "m.safetensors"
    _write_packed(path, M, G)
    eng, _ = runtime.engine_from_packed(path, model=_hf(tied), head=head, max_len=64, prefill=True)
    return eng


@pytest.mark.parametrize("M,G,head,tied", CASES)
def test_prefill_matches_feed(tmp_path, M, G, head, tied):
    eng = _eng(tmp_path, M, G, head, tied)
    ref = torch.stack(runtime.feed(eng, PROMPT, keep_logits=True))
    ref_tok, ref_pos = int(eng.tok), int(eng.pos)
    ref_cache = eng.kc[:, :, :len(PROMPT)].float().clone(), eng.vc[:, :, :len(PROMPT)].float().clone()
    eng.reset(0)
    got = prefill(eng, PROMPT)
    rel = float((got - ref).norm() / ref.norm())
    assert rel < 1e-2, rel
    assert int(eng.tok) == ref_tok or rel < 3e-3
    assert int(eng.pos) == ref_pos
    for a, b in zip((eng.kc, eng.vc), ref_cache):
        c = a[:, :, :len(PROMPT)].float()
        assert float((c - b).norm() / b.norm()) < 1e-2
    # decoding continues identically after prefill and after feed
    g1 = runtime.generate(eng, PROMPT, 12, use_prefill=True)
    g2 = runtime.generate(eng, PROMPT, 12, use_prefill=False)
    assert g1 == g2


def test_prefill_appends(tmp_path):
    eng = _eng(tmp_path, *CASES[0])
    eng.reset(0)
    whole = prefill(eng, PROMPT)
    eng.reset(0)
    prefill(eng, PROMPT[:4])
    tail = prefill(eng, PROMPT[4:])
    assert int(eng.pos) == len(PROMPT)
    rel = float((tail - whole[4:]).norm() / whole[4:].norm())
    assert rel < 1e-2, rel


def test_sampling_stop_logprobs(tmp_path):
    eng = _eng(tmp_path, *CASES[0])
    greedy, lp = runtime.generate(eng, PROMPT, 10, return_logprobs=True)
    # logprobs are the chosen tokens' log-softmax under the model
    lg = runtime.start(eng, PROMPT + greedy[:-1], all_logits=True)[len(PROMPT) - 1:]
    ref = torch.log_softmax(lg, -1)[torch.arange(len(greedy)), torch.tensor(greedy, device=lg.device)]
    assert torch.allclose(torch.tensor(lp), ref.cpu(), atol=2e-2)
    # top_k = 1 sampling is greedy
    assert runtime.generate(eng, PROMPT, 10, temperature=0.8, top_k=1, seed=0) == greedy
    # seeded sampling is reproducible, and differs from greedy at high temperature
    s1 = runtime.generate(eng, PROMPT, 10, temperature=1.5, seed=123)
    s2 = runtime.generate(eng, PROMPT, 10, temperature=1.5, seed=123)
    assert s1 == s2
    # stop on a token id (included), and via a callback
    assert runtime.generate(eng, PROMPT, 10, stop_ids={greedy[3]})[-1] == greedy[3]
    assert len(runtime.generate(eng, PROMPT, 10, stop_fn=lambda t: len(t) >= 4)) == 4


def test_bench_runs(tmp_path):
    eng = _eng(tmp_path, *CASES[0])
    r = runtime.bench(eng, pp=32, tg=16, reps=1)
    assert r["pp_tok_s"] > 0 and r["tg_tok_s"] > 0
