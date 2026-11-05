"""End-to-end decode engine (shapequant/kernels/decode_e2e.py) vs HF Qwen2 on a tiny
config: same weights (fp16-rounded, or the dequantized packed / Q4 weights), the
engine's greedy tokens teacher-forced through HF, logits compared per step."""

import pytest
import torch

cp = pytest.importorskip("cupy")
transformers = pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from shapequant import masks as mk  # noqa: E402
from shapequant.kernels import decode_e2e as de  # noqa: E402
from shapequant.kernels import proto_runs  # noqa: E402
from shapequant.kernels.gemv import PackedLinear  # noqa: E402
from shapequant.packed import pack_bits  # noqa: E402

CFG = de.Cfg(hidden=512, n_heads=4, n_kv=2, head_dim=128, inter=1024, vocab=1024, layers=2, rope_theta=1e6,
             max_len=64)
NAMES = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj",
         "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}


def _hf():
    c = transformers.Qwen2Config(hidden_size=CFG.hidden, intermediate_size=CFG.inter, num_hidden_layers=CFG.layers,
                                 num_attention_heads=CFG.n_heads, num_key_value_heads=CFG.n_kv, vocab_size=CFG.vocab,
                                 max_position_embeddings=CFG.max_len, rms_norm_eps=CFG.eps,
                                 rope_parameters={"rope_type": "default", "rope_theta": CFG.rope_theta},
                                 tie_word_embeddings=False, initializer_range=0.05)
    c._attn_implementation = "eager"
    torch.manual_seed(0)
    m = transformers.Qwen2ForCausalLM(c).float().cuda().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            elif "bias" in n:
                p.copy_(0.1 * torch.randn_like(p))
            p.copy_(p.half().float())                   # engine stores fp16
    return m


def _packed(R, C, perms, seed, bits=10):
    g = torch.Generator().manual_seed(seed)
    K = 1 << bits
    tags = torch.randint(0, K, (R, C // 4), generator=g)
    table = (torch.randn(K, 4, generator=g) * 0.5).half()
    rs = (torch.rand(R, generator=g) * 0.02 + 0.01).half()
    rm = torch.randint(0, perms.shape[0], (R,), generator=g)
    lin = PackedLinear(pack_bits(tags, bits).cuda(), table.cuda(), rs.cuda(), None, rm.cuda(), perms.cuda(), K, 4)
    return lin, tags


def _engine(hf, kind, fuse=True):
    """Engine with `kind` linears; HF weights are replaced by what the engine actually computes with."""
    sd = hf.model
    gc = torch.Generator().manual_seed(3)
    seed = [100]

    def make(mod, perms):
        w = mod.weight.data
        if kind == "dense":
            return de.DenseLin(w)
        if kind == "q4":
            lin = de.Q4Lin.from_weight(w)
            from shapequant.kernels import proto_formats as pf
            mod.weight.data.copy_(pf.q4_pack(w.float().cpu())[2].cuda())
            return lin
        seed[0] += 1
        lin, tags = _packed(*w.shape, perms, seed[0])
        mod.weight.data.copy_(lin.dense().float())
        return de.RunsLin(proto_runs.RunsLinear(lin, tags, ru=2), threads=0 if w.shape[0] <= 512 else 128,
                          blocks_per_sm=1)

    def perms(C):
        return mk.seeded_perms(C, 16, gc) if kind == "runs" else None

    layers = []
    with torch.no_grad():
        for blk in sd.layers:
            pa, po, pg, pd = perms(CFG.hidden), perms(CFG.hidden), perms(CFG.hidden), perms(CFG.inter)
            a, mlp = blk.self_attn, blk.mlp
            layers.append(dict(ln1=blk.input_layernorm.weight.half(), ln2=blk.post_attention_layernorm.weight.half(),
                               q=make(a.q_proj, pa), k=make(a.k_proj, pa), v=make(a.v_proj, pa),
                               o=make(a.o_proj, po), gate=make(mlp.gate_proj, pg), up=make(mlp.up_proj, pg),
                               down=make(mlp.down_proj, pd), bq=a.q_proj.bias.half(), bk=a.k_proj.bias.half(),
                               bv=a.v_proj.bias.half()))
        head = make(hf.lm_head, perms(CFG.hidden))
    return de.DecodeEngine(CFG, sd.embed_tokens.weight.half().contiguous(), layers, sd.norm.weight.half(), head,
                           fuse=fuse)


@pytest.mark.parametrize("kind", ["dense", "q4", "runs"])
@pytest.mark.parametrize("fuse", [True, False])
def test_engine_matches_hf(kind, fuse):
    hf = _hf()
    eng = _engine(hf, kind, fuse)
    if fuse and kind == "runs":    # q/k/v and gate/up share bits and orders here: one launch each
        assert all(len(G["q"]) == 1 and len(G["gate"]) == 1 for G in eng.groups)
    eng.reset(7)
    toks, logits = [7], []
    for _ in range(12):
        eng.step()
        logits.append(eng.logits.float().clone())
        toks.append(int(eng.tok))
    with torch.no_grad():
        ref = hf(torch.tensor([toks[:-1]], device="cuda")).logits[0].float()
    got = torch.stack(logits)
    rel = float((got - ref).norm() / ref.norm())
    assert rel < 2e-2, rel
    assert toks[1:] == ref.argmax(-1).tolist() or rel < 5e-3     # near-ties may flip


def test_graph_replay_matches_eager():
    hf = _hf()
    eng = _engine(hf, "runs")
    eager = eng.generate(10, tok=3)
    eng.capture()
    eng.reset(3)
    replay = []
    for _ in range(10):
        eng.graph.replay()
        replay.append(int(eng.tok))
    assert replay == eager


def test_fused_matches_unfused_exactly():
    """Fusion changes launches, not arithmetic: same logits bit for bit over 20 steps (tokens
    from the unfused engine fed to both)."""
    hf = _hf()
    a, b = _engine(hf, "runs", fuse=False), _engine(hf, "runs", fuse=True)
    a.reset(5)
    b.reset(5)
    for _ in range(20):
        a.step()
        b.step()
        assert torch.equal(a.logits, b.logits)
        b.tok.copy_(a.tok)


@pytest.mark.parametrize("graph", [False, True])
def test_pdl_matches_plain_launches(graph):
    """Programmatic dependent launch changes scheduling only: bit-identical logits and tokens
    (on sm_90+ this exercises the real overlap; below it, the cuLaunchKernelEx path)."""
    hf = _hf()
    a, b = _engine(hf, "runs"), _engine(hf, "runs")
    a.pdl = False
    b.pdl = True
    for eng in (a, b):
        for lin in [eng.lm_head] + [l for G in eng.groups for st in ("q", "gate") for l, _ in G[st]] + \
                [L[n] for L in eng.layers for n in ("o", "down")]:
            lin.pdl = eng.pdl
    if graph:
        a.capture()
        b.capture()
    a.reset(9)
    b.reset(9)
    for _ in range(24):
        for eng in (a, b):
            eng.graph.replay() if graph else eng.step()
        assert torch.equal(a.logits, b.logits)
        assert int(a.tok) == int(b.tok)
