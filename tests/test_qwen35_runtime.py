"""shapequant.qwen35_runtime: the Qwen3.5 decode engine == HF Qwen3_5ForCausalLM with the same
(decoded) weights, on a tiny random model with Gated DeltaNet and gated full-attention layers."""

import math

import pytest
import torch

cp = pytest.importorskip("cupy")
transformers = pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from shapequant import masks as mk  # noqa: E402
from shapequant import packed, qwen35_runtime as q35  # noqa: E402
from shapequant.pipeline import compressible_linears  # noqa: E402


def _hf(tied=True, hv=4):
    from transformers.models.qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    c = Qwen3_5TextConfig(hidden_size=512, intermediate_size=1024, num_hidden_layers=4, num_attention_heads=4,
                          num_key_value_heads=2, head_dim=128, vocab_size=1024, max_position_embeddings=128,
                          layer_types=["linear_attention"] * 3 + ["full_attention"],
                          linear_num_key_heads=2, linear_num_value_heads=hv, linear_key_head_dim=128,
                          linear_value_head_dim=128, linear_conv_kernel_dim=4, tie_word_embeddings=tied,
                          rope_parameters={"rope_type": "default", "rope_theta": 1e6, "partial_rotary_factor": 0.25,
                                           "mrope_section": [11, 11, 10]},
                          initializer_range=0.05, rms_norm_eps=1e-6)
    c._attn_implementation = "eager"
    torch.manual_seed(0)
    m = Qwen3_5ForCausalLM(c).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.copy_(0.1 * torch.randn_like(p) + (1.0 if "linear_attn.norm" in n else 0.0))
            elif "A_log" in n:
                p.copy_(torch.log(torch.rand_like(p) * 4 + 0.5))
            elif "dt_bias" in n:
                p.copy_(torch.randn_like(p) * 0.5)
            elif "conv1d" in n:
                p.copy_(torch.randn_like(p) * 0.3)
            p.copy_(p.half().float())
    return m


def _write_packed(path, model, aux=False):
    items = {}
    g = torch.Generator().manual_seed(1)
    Ks = [256, 1024, 4096]
    for j, (name, mod) in enumerate(compressible_linears(model)):
        R, C = mod.weight.shape
        K, G = Ks[j % 3], 1 + (j % 2) * 3
        local = torch.randint(0, K, (R, C // 4), generator=g)
        t = {"tags": packed.pack_bits(local, int(math.log2(K))),
             "table": (torch.randn(G * K, 4, generator=g) * 0.5).half(),
             "row_scale": (torch.rand(R, generator=g) * 0.02 + 0.01).half()}
        if G > 1:
            t["row_book"] = torch.randint(0, G, (R,), generator=g).to(torch.uint8)
        perms = mk.seeded_perms(C, 1, torch.Generator().manual_seed(j))
        items[name] = (t, {"R": R, "C": C, "d": 4, "K": K, "G": G, "M": 1, "seed": j,
                           "perm_crc": packed.perm_crc(perms), "bits": 0})
    a = packed.aux_from_model(model, list(items)) if aux else None
    packed.save(path, items, base_model="tiny35", aux=a)


def _reference(model, path, aux):
    from shapequant.evaluate import block_quant
    packed.apply_to_model(model, path)
    with torch.no_grad():
        e0 = model.model.embed_tokens.weight.detach().clone()
        model.lm_head.weight = torch.nn.Parameter(block_quant(model.lm_head.weight.detach(), 8).half().float())
        model.model.embed_tokens.weight.data = (block_quant(e0, 4) if aux else e0).half().float()
    return model.cuda()


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("aux,hv", [(False, 4), (True, 4), (False, 2)])
def test_matches_hf(tmp_path, aux, hv, fused):
    path = tmp_path / "m.safetensors"
    _write_packed(path, _hf(hv=hv), aux=aux)
    eng, info = q35.engine_from_packed(path, model=None if aux else _hf(hv=hv), max_len=64, fused=fused)
    assert eng.fused == fused
    assert info["self_contained"] == aux and info["weight_bytes"] > 0
    ref = _reference(_hf(hv=hv), path, aux)
    seq = [5, 17, 300, 2, 999, 64, 7, 123, 512, 33, 1, 800]
    got = torch.stack(q35.feed(eng, seq, keep_logits=True))
    with torch.no_grad():
        r = ref(torch.tensor([seq], device="cuda")).logits[0].float()
    rel = float((got - r).norm() / r.norm())
    assert rel < 5e-3, rel
    assert bool((got.argmax(-1) == r.argmax(-1)).all())
    # graph replay == eager steps
    assert q35.generate(eng, seq[:4], 8, graph=True) == q35.generate(eng, seq[:4], 8, graph=False)


@pytest.mark.parametrize("hv", [4, 2])
def test_prefill_matches_steps(tmp_path, hv):
    """Batched prefill == step-by-step feeding (logits, then continued decode from its states),
    including a second prefill continuing from the first."""
    path = tmp_path / "m.safetensors"
    _write_packed(path, _hf(hv=hv), aux=True)
    eng, _ = q35.engine_from_packed(path, max_len=256)
    q35.attach_prefill(eng)
    seq = [int(t) for t in torch.randint(0, 1024, (150,), generator=torch.Generator().manual_seed(3))]
    ref = torch.stack(q35.feed(eng, seq, keep_logits=True))
    ref_next = [int(eng.tok)] + [None] * 5
    for i in range(1, 6):
        eng.step()
        ref_next[i] = int(eng.tok)
    eng.reset()
    lg = q35.prefill(eng, seq[:100], logits="all")
    lg2 = q35.prefill(eng, seq[100:], logits="all")
    got = torch.cat([lg, lg2])
    rel = float((got - ref).norm() / ref.norm())
    assert rel < 5e-3, rel
    nxt = [int(eng.tok)]
    for _ in range(5):
        eng.step()
        nxt.append(int(eng.tok))
    assert nxt == ref_next
