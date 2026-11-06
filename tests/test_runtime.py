"""shapequant.runtime: engine built from a packed file == HF model with the decoded weights."""

import math

import pytest
import torch

cp = pytest.importorskip("cupy")
transformers = pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from shapequant import masks as mk  # noqa: E402
from shapequant import packed, runtime  # noqa: E402

H, I, NH, NKV, V, NL = 512, 1024, 4, 2, 1024, 2


def _hf(tied):
    c = transformers.Qwen2Config(hidden_size=H, intermediate_size=I, num_hidden_layers=NL, num_attention_heads=NH,
                                 num_key_value_heads=NKV, vocab_size=V, max_position_embeddings=64,
                                 rope_parameters={"rope_type": "default", "rope_theta": 1e6},
                                 tie_word_embeddings=tied, initializer_range=0.05)
    c._attn_implementation = "eager"
    torch.manual_seed(0)
    m = transformers.Qwen2ForCausalLM(c).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            elif "bias" in n:
                p.copy_(0.1 * torch.randn_like(p))
            p.copy_(p.half().float())
    return m


def _write_packed(path, M, G, aux_model=None):
    """Random codes for every block linear: K cycles 256/1024/4096, G books, M column orders."""
    items = {}
    g = torch.Generator().manual_seed(1)
    Ks = [256, 1024, 4096]
    for i in range(NL):
        for j, (n, (R, C)) in enumerate({"self_attn.q_proj": (H, H), "self_attn.k_proj": (NKV * 128, H),
                                         "self_attn.v_proj": (NKV * 128, H), "self_attn.o_proj": (H, H),
                                         "mlp.gate_proj": (I, H), "mlp.up_proj": (I, H),
                                         "mlp.down_proj": (H, I)}.items()):
            K = Ks[(i * 7 + j) % 3]
            # q/k/v and gate/up share column orders (the engine requires it): same seed per group
            seed = 1000 * i + {"q": 1, "k": 1, "v": 1, "o": 2, "g": 3, "u": 3, "d": 4}[n.split(".")[1][0]]
            nch = C // 4
            local = torch.randint(0, K, (R, nch), generator=g)
            t = {"tags": packed.pack_bits(local, int(math.log2(K))),
                 "table": (torch.randn(G * K, 4, generator=g) * 0.5).half(),
                 "row_scale": (torch.rand(R, generator=g) * 0.02 + 0.01).half()}
            if G > 1:
                t["row_book"] = torch.randint(0, G, (R,), generator=g).to(torch.uint8)
            if M > 1:
                t["row_mask"] = torch.randint(0, M, (R,), generator=g).to(torch.uint8)
            perms = mk.seeded_perms(C, M, torch.Generator().manual_seed(seed))
            meta = {"R": R, "C": C, "d": 4, "K": K, "G": G, "M": M, "seed": seed, "perm_crc": packed.perm_crc(perms),
                    "bits": 0}
            items[f"model.layers.{i}.{n}"] = (t, meta)
    aux = packed.aux_from_model(aux_model, list(items)) if aux_model is not None else None
    return packed.save(path, items, base_model="tiny", aux=aux)


@pytest.mark.parametrize("M,G,head,tied", [(1, 4, "q4", True), (4, 1, "fp16", False), (1, 1, "q4", False),
                                                  (1, 4, "q8", False)])
def test_engine_from_packed_matches_hf(tmp_path, M, G, head, tied):
    path = tmp_path / "m.safetensors"
    _write_packed(path, M, G)
    hf = _hf(tied)
    eng, info = runtime.engine_from_packed(path, model=hf, head=head, max_len=64)
    assert info["weight_bytes"] > 0
    ref = _hf(tied)
    packed.apply_to_model(ref, path)
    with torch.no_grad():
        ref.lm_head.weight = torch.nn.Parameter(runtime.head_reference_weight(ref, head).half().float())
    ref = ref.cuda()
    prompt = [5, 17, 300, 2]
    gen = runtime.generate(eng, prompt, 10, graph=True)
    lg = runtime.feed(eng, prompt + gen[:-1], keep_logits=True)[len(prompt) - 1:]
    with torch.no_grad():
        r = ref(torch.tensor([prompt + gen[:-1]], device="cuda")).logits[0, len(prompt) - 1:].float()
    got = torch.stack(lg)
    rel = float((got - r).norm() / r.norm())
    assert rel < 2e-2, rel
    assert gen == r.argmax(-1).tolist() or rel < 5e-3
    # eager and graph decoding agree
    assert runtime.generate(eng, prompt, 10, graph=False) == gen


@pytest.mark.parametrize("R", [1000, 999, 17, 1])
def test_q8_gemv_matches_dense(R):
    from shapequant.kernels import proto_formats as pf
    g = torch.Generator().manual_seed(0)
    w = torch.randn(R, 512, generator=g) * 0.05
    q, sc, deq = pf.q8_pack(w)
    x = (torch.randn(512, generator=g)).half()
    y = torch.full((R + 1,), 7.0, dtype=torch.float16, device="cuda")
    pf.gemv_q8(q.cuda(), sc.cuda(), x.cuda(), y, R, 512)
    assert float(y[R]) == 7.0                    # nothing written past R
    y = y[:R]
    ref = deq @ x.float()
    assert float((y.float().cpu() - ref).norm() / ref.norm()) < 2e-3


@pytest.mark.parametrize("tied", [False, True])
def test_self_contained_file(tmp_path, tied):
    """A file with the aux section runs without the HF model: 4-bit embedding, 8-bit head, the
    norms/biases/config from the file. Matches HF with exactly those weights, decode and prefill."""
    from shapequant.evaluate import block_quant
    path = tmp_path / "m.safetensors"
    summary = _write_packed(path, 1, 4, aux_model=_hf(tied))
    assert summary["aux"] and summary["bpw_all"] > 0
    eng, info = runtime.engine_from_packed(path, max_len=64, prefill=True)
    assert info["self_contained"] and info["embed_bytes"] == V * H // 2 + V * H // 32 * 2
    ref = _hf(tied)
    packed.apply_to_model(ref, path)
    with torch.no_grad():
        e0 = ref.model.embed_tokens.weight.detach().clone()
        ref.lm_head.weight = torch.nn.Parameter(block_quant(ref.lm_head.weight.detach(), 8).half().float())
        ref.model.embed_tokens.weight.data = block_quant(e0, 4).half().float()
    ref = ref.cuda()
    prompt = [5, 17, 300, 2, 999, 64]
    gen = runtime.generate(eng, prompt, 8, graph=True, use_prefill=False)
    seq = prompt + gen[:-1]
    lg = torch.stack(runtime.feed(eng, seq, keep_logits=True))
    with torch.no_grad():
        r = ref(torch.tensor([seq], device="cuda")).logits[0].float()
    rel = float((lg - r).norm() / r.norm())
    assert rel < 2e-2, rel
    assert gen == r[len(prompt) - 1:].argmax(-1).tolist() or rel < 5e-3
    pre = runtime.start(eng, seq, all_logits=True, use_prefill=True)
    rel_p = float((pre - r).norm() / r.norm())
    assert rel_p < 2e-2, rel_p
