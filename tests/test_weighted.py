import torch

from shapequant import codebook as cb
from shapequant.calibrate import column_importance
from shapequant.pipeline import compressible_linears


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def test_unit_weights_match_unweighted():
    x = torch.randn(4000, 4, generator=_gen(1))
    a = cb.kmeans(x, 16, 10, _gen(2))
    b = cb.kmeans(x, 16, 10, _gen(2), wt=torch.ones_like(x))
    assert torch.allclose(a, b, atol=1e-4)
    ia, ea = cb.assign(x, a)
    ib, eb = cb.assign(x, a, wt=torch.ones_like(x))
    assert torch.equal(ia, ib) and torch.allclose(ea, eb, atol=1e-4)


def test_weighted_fit_lowers_weighted_error():
    x = torch.randn(6000, 4, generator=_gen(3))
    wt = torch.ones_like(x)
    wt[:, 0] = 50.0  # first coordinate matters far more
    plain = cb.kmeans(x, 16, 15, _gen(4))
    weighted = cb.kmeans(x, 16, 15, _gen(4), wt=wt)
    e_plain = cb.assign(x, plain, wt=wt)[1].mean()
    e_weighted = cb.assign(x, weighted, wt=wt)[1].mean()
    assert e_weighted < 0.8 * e_plain


def test_weighted_centroid_is_weighted_mean():
    x = torch.tensor([[0.0], [1.0]])
    wt = torch.tensor([[3.0], [1.0]])
    c = cb.kmeans(x, 1, 1, _gen(0), wt=wt)
    assert torch.allclose(c, torch.tensor([[0.25]]))


def test_compress_weighted_decodes_and_respects_weights():
    g = _gen(5)
    w = torch.randn(64, 30, generator=g) * 0.02  # 30 % 4 != 0 -> padding
    h = torch.ones(30)
    h[:5] = 100.0
    cm, _ = cb.compress(w, 4, 16, iters=10, max_fit_chunks=10**6, seed=0, col_weight=h, row_weighted=True)
    out = cb.decode(cm)
    assert out.shape == w.shape
    cm0, _ = cb.compress(w, 4, 16, iters=10, max_fit_chunks=10**6, seed=0)
    err_w = ((cb.decode(cm) - w)[:, :5] ** 2).mean()
    err_0 = ((cb.decode(cm0) - w)[:, :5] ** 2).mean()
    assert err_w < err_0  # heavily weighted columns are reconstructed better


def test_column_importance_matches_manual():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1)
    if getattr(cfg, "layer_types", None):
        cfg.layer_types = cfg.layer_types[:1]
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(cfg).eval()
    seqs = torch.randint(0, 64, (3, 10))
    imp = column_importance(m, seqs, batch_size=2)
    assert set(imp) == {n for n, _ in compressible_linears(m)}
    captured = []
    h = m.model.layers[0].mlp.down_proj.register_forward_pre_hook(lambda mod, a: captured.append(a[0]))
    with torch.no_grad():
        m.model(input_ids=seqs)
    h.remove()
    x = captured[0].reshape(-1, 64)
    assert torch.allclose(imp["model.layers.0.mlp.down_proj"], (x * x).mean(0), rtol=1e-4, atol=1e-7)
