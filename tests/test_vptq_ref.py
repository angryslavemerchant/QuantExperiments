import math

import torch

from shapequant.vptq_ref import decode_linear, unpack_indices


def _pack(vals: torch.Tensor, bits_per: int) -> torch.Tensor:
    """Inverse of unpack_indices: [G, N, n] ints -> [G, N, P] int32, little-endian bits."""
    G, N, n = vals.shape
    b = ((vals.unsqueeze(-1) >> torch.arange(bits_per)) & 1).reshape(G, N, -1)
    P = math.ceil(n * bits_per / 32)
    b = torch.nn.functional.pad(b, (0, P * 32 - b.shape[-1])).reshape(G, N, P, 32)
    packed = (b.to(torch.int64) << torch.arange(32)).sum(-1)
    return ((packed + 2**31) % 2**32 - 2**31).to(torch.int32)  # wrap to signed int32


def test_unpack_roundtrip_with_residual():
    g = torch.Generator().manual_seed(0)
    main = torch.randint(0, 2**16, (1, 3, 10), generator=g)
    res = torch.randint(0, 2**8, (1, 3, 10), generator=g)
    m, r = unpack_indices(_pack(main | (res << 16), 24), 16, 10, 8)
    assert torch.equal(m, main) and torch.equal(r, res)


def test_decode_layout_matches_reference_construction():
    # out=6 rows (vector_len 4 -> 2 index rows, padding 2), in=5 columns, K=16, residual K=4
    g = torch.Generator().manual_seed(1)
    v, K, Kr, out_f, in_f = 4, 16, 4, 6, 5
    cent = torch.randn(K, v, generator=g)
    rcent = torch.randn(Kr, v, generator=g) * 0.1
    N = 2
    idx = torch.randint(0, K, (1, N, in_f), generator=g)
    ridx = torch.randint(0, Kr, (1, N, in_f), generator=g)
    scale = torch.rand(in_f, generator=g) + 0.5
    bias = torch.randn(in_f, generator=g) * 0.01
    # reference: column c, index row n holds output rows n*v .. n*v+v-1
    W = torch.zeros(N * v, in_f)
    for n in range(N):
        for c in range(in_f):
            W[n * v:(n + 1) * v, c] = cent[idx[0, n, c]] + rcent[ridx[0, n, c]]
    W = W[:out_f] * scale + bias
    t = {"indices": _pack(idx | (ridx << 4), 6),
         "centroids.weight": cent.reshape(1, -1).to(torch.bfloat16).float(),
         "res_centroids.weight": rcent.reshape(1, -1).to(torch.bfloat16).float(),
         "weight_scale": scale, "weight_bias": bias}
    cfg = {"vector_lens": [-1, v], "num_centroids": [-1, K], "num_res_centroids": [-1, Kr],
           "group_num": 1, "group_size": in_f, "in_features": in_f, "out_features": out_f,
           "enable_norm": True, "enable_perm": False, "is_indice_packed": True}
    t["centroids.weight"] = cent.reshape(1, -1)
    t["res_centroids.weight"] = rcent.reshape(1, -1)
    assert torch.allclose(decode_linear(t, cfg), W, atol=1e-6)


def test_source_reads_sharded_checkpoints(tmp_path, monkeypatch):
    import json
    import huggingface_hub
    from safetensors.torch import save_file
    from shapequant.vptq_ref import VPTQSource
    a, b = torch.arange(6.0).view(2, 3), torch.ones(4)
    save_file({"x.a": a}, str(tmp_path / "s1.safetensors"))
    save_file({"y.b": b}, str(tmp_path / "s2.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(
        {"quantization_config": {"quant_method": "vptq", "config_for_layers": {}}}))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"x.a": "s1.safetensors", "y.b": "s2.safetensors"}}))

    def fake_download(repo, filename, revision=None):
        p = tmp_path / filename
        if not p.exists():
            raise FileNotFoundError(filename)
        return str(p)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    src = VPTQSource("fake/repo")
    assert src.keys == {"x.a", "y.b"}
    assert torch.equal(src.f.get_tensor("x.a"), a) and torch.equal(src.f.get_tensor("y.b"), b)
