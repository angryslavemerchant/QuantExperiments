"""scripts/serve_runtime.py: OpenAI-style endpoints on the tiny packed model, parsed by lm-eval."""

import sys
from pathlib import Path

import pytest
import torch

cp = pytest.importorskip("cupy")
pytest.importorskip("transformers")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fastapi.testclient import TestClient  # noqa: E402
import serve_runtime  # noqa: E402
from shapequant import runtime  # noqa: E402
from test_runtime import _hf, _write_packed  # noqa: E402


class CharTok:
    """Minimal tokenizer over the tiny vocab: one token per character (ord % 1000), 1000 = eos."""
    eos_token_id = 1000

    def encode(self, s, add_special_tokens=False):
        return [ord(c) % 1000 for c in s]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(32 + (t % 90)) for t in ids if not (skip_special_tokens and t == 1000))

    def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=False):
        return "".join(f"<{m['role']}>{m['content']}" for m in msgs) + ("<assistant>" if add_generation_prompt else "")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    path = tmp_path_factory.mktemp("srv") / "m.safetensors"
    _write_packed(path, 1, 4)
    eng, _ = runtime.engine_from_packed(path, model=_hf(True), head="q4", max_len=64, prefill=True)
    eng.capture()
    app = serve_runtime.create_app(eng, CharTok(), "tiny")
    return TestClient(app), eng


def test_models(client):
    c, _ = client
    assert c.get("/v1/models").json()["data"][0]["id"] == "tiny"


def test_echo_logprobs_match_model_and_parse_in_lm_eval(client):
    c, eng = client
    ids = [[5, 17, 300, 2, 999, 41], [8, 9, 10]]
    r = c.post("/v1/completions", json={"model": "tiny", "prompt": ids, "temperature": 0, "max_tokens": 1,
                                        "logprobs": 1, "seed": 1234, "echo": True}).json()
    assert len(r["choices"]) == 2
    for ch, p in zip(r["choices"], ids):
        lp = ch["logprobs"]["token_logprobs"]
        assert lp[0] is None and len(lp) == len(p) + 1
        ref = torch.log_softmax(runtime.start(eng, p, all_logits=True), -1)
        want = ref[torch.arange(len(p) - 1), torch.tensor(p[1:], device=ref.device)].tolist()
        assert all(abs(a - b) < 1e-4 for a, b in zip(lp[1:len(p)], want))
    from lm_eval.models.openai_completions import LocalCompletionsAPI
    res = LocalCompletionsAPI.parse_logprobs(r, tokens=ids, ctxlens=[3, 1])
    assert len(res) == 2
    lp0 = r["choices"][0]["logprobs"]["token_logprobs"]
    assert abs(res[0][0] - sum(lp0[3:-1])) < 1e-6
    assert isinstance(res[0][1], bool)


def test_greedy_flag_true_on_greedy_continuation(client):
    c, eng = client
    p = [5, 17, 300]
    g = runtime.generate(eng, p, 4)
    r = c.post("/v1/completions", json={"prompt": [p + g], "max_tokens": 1, "logprobs": 1, "echo": True}).json()
    from lm_eval.models.openai_completions import LocalCompletionsAPI
    (_, greedy), = LocalCompletionsAPI.parse_logprobs(r, tokens=[p + g], ctxlens=[len(p)])
    assert greedy


def test_generate_text_stop_and_chat(client):
    c, eng = client
    p = [5, 17, 300, 2]
    r = c.post("/v1/completions", json={"prompt": p, "max_tokens": 8, "temperature": 0, "logprobs": 1}).json()
    ch = r["choices"][0]
    toks = runtime.generate(eng, p, 8, stop_ids={1000})
    assert ch["text"] == CharTok().decode(toks, skip_special_tokens=True)
    assert len(ch["logprobs"]["token_logprobs"]) == len(toks)
    stop = ch["text"][2:4]
    r2 = c.post("/v1/completions", json={"prompt": p, "max_tokens": 8, "stop": [stop]}).json()["choices"][0]
    assert stop not in r2["text"] and r2["finish_reason"] == "stop"
    r3 = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}],
                                              "max_tokens": 5}).json()
    assert r3["choices"][0]["message"]["role"] == "assistant"
    s = c.post("/v1/completions", json={"prompt": "hello", "max_tokens": 6, "temperature": 1.2, "seed": 7}).json()
    s2 = c.post("/v1/completions", json={"prompt": "hello", "max_tokens": 6, "temperature": 1.2, "seed": 7}).json()
    assert s["choices"][0]["text"] == s2["choices"][0]["text"]


def test_too_long_is_400(client):
    c, _ = client
    r = c.post("/v1/completions", json={"prompt": list(range(1, 80)), "max_tokens": 1, "logprobs": 1, "echo": True})
    assert r.status_code == 400
