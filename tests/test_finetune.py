import torch
from torch import nn

from shapequant.finetune import CodebookLinear, kl_loss
from shapequant.masks import compress_masked


def _fit(M):
    g = torch.Generator().manual_seed(0)
    w = torch.randn(24, 40, generator=g) * 0.05
    h = torch.rand(40, generator=g) + 0.5
    fit = compress_masked(w, h, 4, 8, M=M, init="identity+random" if M > 1 else "identity",
                          rounds=2, iters=5, seed=1, keep_tags=True)
    return w, fit


def test_wrapper_matches_decoded_weight_and_bakes_exactly():
    for M in (1, 4):
        w, fit = _fit(M)
        lin = nn.Linear(40, 24)
        decoded = fit.decoded.clone()
        wrap = CodebookLinear(lin, fit, 4)
        x = torch.randn(3, 5, 40)
        y = wrap(x)
        ref = torch.nn.functional.linear(x, decoded, lin.bias)
        assert torch.allclose(y, ref, atol=1e-5), M
        baked = wrap.bake()  # fp16 rounding of books / scales is already in `decoded`
        assert torch.allclose(baked, decoded, atol=1e-6), M


def test_gradients_reach_codebooks_and_gains():
    w, fit = _fit(4)
    wrap = CodebookLinear(nn.Linear(40, 24), fit, 4)
    wrap(torch.randn(2, 40)).pow(2).sum().backward()
    assert wrap.books[0].grad is not None and wrap.books[0].grad.abs().sum() > 0
    assert wrap.log_gain.grad.abs().sum() > 0


def test_kl_loss_zero_for_identical_logits():
    a = torch.randn(4, 7, 11)
    assert float(kl_loss(a, a)) < 1e-6
    assert float(kl_loss(a, torch.zeros_like(a))) > 0


def test_shared_teacher_matches_separate_teacher():
    """finetune(teacher=None, originals=...) (SharedTeacher: one model copy) == a separately
    loaded original teacher: same losses, same baked weights."""
    import copy
    import transformers
    from shapequant.finetune import finetune
    from shapequant.pipeline import compressible_linears
    c = transformers.Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                                 num_key_value_heads=2, vocab_size=128, max_position_embeddings=64)
    c._attn_implementation = "eager"
    torch.manual_seed(0)
    orig = transformers.Qwen2ForCausalLM(c).float().eval()
    originals = {n: m.weight.detach().clone() for n, m in compressible_linears(orig)}
    fits = {}
    for n, m in compressible_linears(orig):
        w = m.weight.detach()
        fits[n] = compress_masked(w, torch.ones(w.shape[1]), 4, 8, M=1, init="identity", rounds=1, iters=3,
                                  seed=1, keep_tags=True)
    seqs = torch.randint(0, 128, (4, 16), generator=torch.Generator().manual_seed(5))
    cfg = {"steps": 4, "seq_len": 16, "log_every": 100}
    res = []
    for shared in (False, True):
        student = copy.deepcopy(orig)
        with torch.no_grad():
            for n, m in compressible_linears(student):
                m.weight.copy_(fits[n].decoded)
        f = copy.deepcopy(fits)
        teacher = None if shared else copy.deepcopy(orig)
        torch.manual_seed(0)
        st = finetune(student, teacher, f, {n: 4 for n in f}, seqs, cfg, log=lambda *_: None, originals=originals)
        res.append((st, {n: m.weight.detach().clone() for n, m in compressible_linears(student)}))
    (a, wa), (b, wb) = res
    assert a["ft_kl_first"] == b["ft_kl_first"] and a["ft_kl_last"] == b["ft_kl_last"]
    for n in wa:
        assert torch.equal(wa[n], wb[n]), n


def test_keep_best_heldout_restores_best_step():
    """With keep_best, a fine-tune that only gets worse on the held-out set bakes its step-0
    books + gains: the same weights as a zero-step fine-tune; the held-out log covers every check."""
    import copy
    import transformers
    from shapequant.finetune import finetune
    from shapequant.pipeline import compressible_linears
    c = transformers.Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                                 num_key_value_heads=2, vocab_size=128, max_position_embeddings=64)
    c._attn_implementation = "eager"
    torch.manual_seed(0)
    orig = transformers.Qwen2ForCausalLM(c).float().eval()
    originals = {n: m.weight.detach().clone() for n, m in compressible_linears(orig)}
    fits = {}
    for n, m in compressible_linears(orig):
        w = m.weight.detach()
        fits[n] = compress_masked(w, torch.ones(w.shape[1]), 4, 8, M=1, init="identity", rounds=1, iters=3,
                                  seed=1, keep_tags=True)
    g = torch.Generator().manual_seed(5)
    seqs = torch.randint(0, 128, (4, 16), generator=g)
    held = {"a": torch.randint(0, 128, (2, 16), generator=g)}

    def run(steps, cfg_extra):
        student = copy.deepcopy(orig)
        with torch.no_grad():
            for n, m in compressible_linears(student):
                m.weight.copy_(fits[n].decoded)
        cfg = {"steps": steps, "seq_len": 16, "log_every": 100, **cfg_extra}
        st = finetune(student, None, copy.deepcopy(fits), {n: 4 for n in fits}, seqs, cfg, log=lambda *_: None,
                      originals=originals, heldout=held)
        return st, {n: m.weight.detach().clone() for n, m in compressible_linears(student)}

    # absurd learning rates: every step makes the held-out KL worse
    st, w = run(4, {"lr_books": 10.0, "lr_gain": 10.0, "heldout_every": 2, "keep_best": "a"})
    assert [r["step"] for r in st["ft_heldout"]] == [0, 2, 4]
    assert st["ft_best_step"] == 0
    assert st["ft_heldout"][-1]["a"] > st["ft_heldout"][0]["a"]
    _, w0 = run(0, {})
    for n in w:
        assert torch.equal(w[n], w0[n]), n
