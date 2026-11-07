"""kernels.prefill_kernels: fused prefill kernels == the torch reference ops they replace."""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cupy")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from shapequant.kernels import prefill_kernels as pk  # noqa: E402


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


@pytest.mark.parametrize("T,H,bv", [(1, 2, 32), (64, 2, 16), (150, 4, 32), (300, 3, 16)])
def test_chunk_gated_delta_rule_matches_hf(T, H, bv):
    tr = pytest.importorskip("transformers")  # noqa: F841
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
    g_ = torch.Generator(device="cuda").manual_seed(T + H)
    DK = DV = 128
    q = torch.randn(1, T, H, DK, device="cuda", generator=g_)
    k = torch.randn(1, T, H, DK, device="cuda", generator=g_)
    v = torch.randn(1, T, H, DV, device="cuda", generator=g_)
    g = -torch.rand(1, T, H, device="cuda", generator=g_) * 2
    beta = torch.rand(1, T, H, device="cuda", generator=g_)
    S0 = torch.randn(1, H, DK, DV, device="cuda", generator=g_) * 0.1
    ref, Sref = torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=S0.clone(),
                                             output_final_state=True, use_qk_l2norm_in_kernel=True)
    Tp = (T + 63) // 64 * 64

    def hm(x):            # [1, T, H, d] -> [H, Tp, d] zero padded
        x = x[0].transpose(0, 1)
        return F.pad(x, (0, 0, 0, Tp - T)).contiguous() if x.dim() == 3 else F.pad(x, (0, Tp - T)).contiguous()

    qn = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * DK ** -0.5
    kn = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    S = S0[0].clone()
    o = pk.chunk_gated_delta_rule(hm(qn), hm(kn), hm(v), hm(g), hm(beta), S, bv=bv)
    got = o[:, :T].transpose(0, 1)
    assert _rel(got, ref[0]) < 1e-4
    assert _rel(S, Sref[0]) < 1e-4


def test_gdn_prep_conv():
    """conv + gdn_prep == the torch ops of the reference prefill."""
    T, HK, HV, DK, DV, K = 70, 2, 4, 128, 128, 4
    D = 2 * HK * DK + HV * DV
    g_ = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(T, D + 64, device="cuda", generator=g_).half()          # extra columns: row stride
    state = torch.randn(D, K, device="cuda", generator=g_)
    w = torch.randn(D, K, device="cuda", generator=g_) * 0.3
    seq = torch.cat([state, x[:, :D].float().T], 1)
    ref_u = F.silu(F.conv1d(seq[None], w[:, None], groups=D)[0, :, 1:]).T
    st = state.clone()
    u = pk.conv(x, st, w, D)
    assert _rel(u, ref_u) < 1e-6
    assert torch.equal(st, seq[:, -K:])
    # short prompt (T < K - 1): the window reaches into the old state
    st2 = state.clone()
    u2 = pk.conv(x[:2], st2, w, D)
    assert _rel(u2, ref_u[:2]) < 1e-6
    assert torch.equal(st2, torch.cat([state, x[:2, :D].float().T], 1)[:, -K:])
    ab = torch.randn(T, 2 * HV, device="cuda", generator=g_)
    A = torch.rand(HV, device="cuda", generator=g_) + 0.5
    dtb = torch.randn(HV, device="cuda", generator=g_)
    q, k, v, g, beta = pk.gdn_prep(u, ab, A, dtb, HK, HV, DK, DV)
    assert q.shape == (HV, 128, DK)
    kd = HK * DK
    rq = ref_u[:, :kd].view(T, HK, DK).repeat_interleave(HV // HK, 1)
    rk = ref_u[:, kd:2 * kd].view(T, HK, DK).repeat_interleave(HV // HK, 1)
    rq = rq * torch.rsqrt((rq * rq).sum(-1, keepdim=True) + 1e-6) * DK ** -0.5
    rk = rk * torch.rsqrt((rk * rk).sum(-1, keepdim=True) + 1e-6)
    assert _rel(q[:, :T].transpose(0, 1), rq) < 1e-5 and _rel(k[:, :T].transpose(0, 1), rk) < 1e-5
    assert torch.equal(v[:, :T].transpose(0, 1), ref_u[:, 2 * kd:].view(T, HV, DV))
    assert _rel(g[:, :T].T, -A * F.softplus(ab[:, :HV] + dtb)) < 1e-5
    assert _rel(beta[:, :T].T, torch.sigmoid(ab[:, HV:])) < 1e-5
    assert float(q[:, T:].abs().sum() + g[:, T:].abs().sum() + beta[:, T:].abs().sum()) == 0.0


@pytest.mark.parametrize("wdt", [torch.float16, torch.float32])
def test_addnorm_silu(wdt):
    T, C = 33, 1000
    g_ = torch.Generator(device="cuda").manual_seed(1)
    h = torch.randn(T, C, device="cuda", generator=g_)
    add = torch.randn(T, C + 24, device="cuda", generator=g_).half()[:, :C]
    w = (torch.randn(C, device="cuda", generator=g_) * 0.2 + 1).to(wdt)
    href = h + add.float()
    ref = (href * torch.rsqrt(href.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).half()
    got = pk.addnorm(h, add, w, 1e-6)
    assert torch.allclose(h, href)
    assert (got.float() - ref.float()).abs().max() <= 2e-3 * ref.float().abs().max()
    gu = torch.randn(T, 2 * C, device="cuda", generator=g_).half()
    gg, uu = gu[:, :C], gu[:, C:]
    assert torch.equal(pk.silu_mul(gu, 0), F.silu(gg.float()).half() * uu)
    r1 = (F.silu(gg.float()) * uu.float()).half()
    assert (pk.silu_mul(gu, 1).float() - r1.float()).abs().max() <= 1e-3 * r1.float().abs().max()
