"""Stage 5: codebook fine-tune.

Tags and masks are frozen; only the codebook entries and per-row scales train,
to minimize KL(original || compressed) on calibration text (never evaluation
text). Each block linear is temporarily replaced by a `CodebookLinear` that
re-decodes its weight from the trainable codebook(s) on every forward, so the
gradient reaches the codewords. Afterwards everything is rounded back to fp16
(the stored format), decoded once more, and written into the ordinary
nn.Linear weights, so evaluation is unchanged.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch import nn

from .pipeline import compressible_linears


class _GatherRows(torch.autograd.Function):
    """books[idx] with an atomic index_add_ backward. PyTorch's default index
    backward sorts all indices every step (~11x slower here); tags are fixed, so
    plain atomics are fine. Float summation order is not deterministic."""

    @staticmethod
    def forward(ctx, books, idx):
        ctx.save_for_backward(idx)
        ctx.K = books.shape[0]
        return books[idx]

    @staticmethod
    def backward(ctx, grad):
        (idx,) = ctx.saved_tensors
        g = torch.zeros(ctx.K, grad.shape[1], device=grad.device, dtype=grad.dtype)
        g.index_add_(0, idx, grad)
        return g, None


class CodebookLinear(nn.Module):
    """Block linear whose weight is decoded from trainable codebooks.

    Masks are applied on the INPUT side, as inference would: rows are grouped by
    their column order m; group m computes x[..., perm_m] @ W_m^T, where W_m holds
    that group's rows in permuted column order, exactly as the tags store them.
    So the weight decodes with one gather (no column un-shuffling, whose backward
    is a slow accumulate-scatter), and only the small outputs are reordered."""

    def __init__(self, lin: nn.Linear, fit, d: int):
        super().__init__()
        R, C = lin.weight.shape
        self.R, self.C, self.d = R, C, d
        self.lin = [lin]  # keep a reference without registering it as a submodule
        self.books = nn.ParameterList([nn.Parameter(b.detach().float().clone())
                                       for b in fit.stats["books"]])
        row_mask = fit.row_mask
        order = torch.argsort(row_mask, stable=True)
        self.register_buffer("order", order)
        self.register_buffer("unsort", torch.argsort(order))
        self.register_buffer("rs", fit.row_scale.detach().float()[order].clone())
        self.log_gain = nn.Parameter(torch.zeros(R, device=lin.weight.device))  # sorted-row order
        for i, t in enumerate(fit.tags):
            self.register_buffer(f"tags{i}", t.view(R, -1)[order].reshape(-1).contiguous())
        # rotated fits (pipeline `rotate`): W = U^T W'' V, applied as x -> x V^T and y'' -> y'' U
        rin, rout = fit.stats.get("rot_in"), fit.stats.get("rot_out")
        self.register_buffer("rot_in", None if rin is None else rin.detach().float().clone())
        self.register_buffer("rot_out", None if rout is None else rout.detach().float().clone())
        counts = torch.bincount(row_mask, minlength=fit.perms.shape[0]).tolist()
        self.groups = []
        start = 0
        for m, n in enumerate(counts):
            if n:
                perm = fit.perms[m]
                identity = bool(torch.equal(perm, torch.arange(C, device=perm.device)))
                if not identity:
                    self.register_buffer(f"perm{m}", perm.clone())
                self.groups.append((start, start + n, None if identity else f"perm{m}"))
                start += n

    def sorted_weight(self) -> torch.Tensor:
        """[R, C] decoded weight, rows in group order, columns in each group's order."""
        flat = None
        for i, b in enumerate(self.books):
            part = _GatherRows.apply(b, getattr(self, f"tags{i}").long())
            flat = part if flat is None else flat + part
        u = flat.view(self.R, -1)[:, : self.C]
        return u * (self.rs * self.log_gain.exp())[:, None]

    def forward(self, x):
        w = self.sorted_weight().to(x.dtype)
        if self.rot_in is not None:
            x = F.linear(x, self.rot_in.to(x.dtype))
        outs = []
        for a, b, perm in self.groups:
            xin = x if perm is None else x.index_select(-1, getattr(self, perm))
            outs.append(F.linear(xin, w[a:b]))
        y = torch.cat(outs, dim=-1) if len(outs) > 1 else outs[0]
        if len(self.groups) > 1:
            y = y.index_select(-1, self.unsort)
        if self.rot_out is not None:
            y = y @ self.rot_out.to(y.dtype)
        bias = self.lin[0].bias
        return y if bias is None else y + bias

    @torch.no_grad()
    def bake(self) -> torch.Tensor:
        """Round to the stored fp16 format; return the decoded weight in the
        original row and column order."""
        for b in self.books:
            b.copy_(b.half().float())
        self.rs.copy_((self.rs * self.log_gain.exp()).half().float())
        self.log_gain.zero_()
        ws = self.sorted_weight()
        out = torch.empty_like(ws)
        for a, b, perm in self.groups:
            if perm is None:
                out[a:b] = ws[a:b]
            else:
                out[a:b][:, getattr(self, perm)] = ws[a:b]  # column j holds original column perm[j]
        out = out[self.unsort]
        if self.rot_in is not None:
            out = out @ self.rot_in
        if self.rot_out is not None:
            out = self.rot_out.T @ out
        return out


def write_back(fit, wrap: "CodebookLinear") -> None:
    """After bake(): store the trained (fp16-rounded) books and row scales in the
    fit, in original row order, so the packed export carries them."""
    fit.stats["books"] = [b.detach().half() for b in wrap.books]
    fit.row_scale = wrap.rs[wrap.unsort].detach().clone()


def _swap(model, fits: dict, d_by_name: dict):
    wrappers = {}
    for name, lin in compressible_linears(model):
        parent_name, child = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        wrap = CodebookLinear(lin, fits[name], d_by_name[name])
        setattr(parent, child, wrap)
        wrappers[name] = (parent, child, lin, wrap)
    return wrappers


def kl_loss(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    tlp = F.log_softmax(teacher_logits.float(), dim=-1)
    slp = F.log_softmax(student_logits.float(), dim=-1)
    return (tlp.exp() * (tlp - slp)).sum(-1).mean()


def finetune(model, teacher, fits: dict, d_by_name: dict, sequences: torch.Tensor,
             fcfg: dict, log=print) -> dict:
    """Train codebooks + row gains in place; write baked weights into `model`."""
    steps = fcfg.get("steps", 256)
    seq_len = fcfg.get("seq_len", 1024)
    seqs = sequences.reshape(-1, seq_len)
    dev = next(model.parameters()).device
    for p in model.parameters():
        p.requires_grad_(False)
    wrappers = _swap(model, fits, d_by_name)
    books = [p for _, _, _, w in wrappers.values() for p in w.books]
    gains = [w.log_gain for _, _, _, w in wrappers.values()]
    opt = torch.optim.Adam([
        {"params": books, "lr": fcfg.get("lr_books", 1e-4)},
        {"params": gains, "lr": fcfg.get("lr_gain", 1e-3)},
    ])
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    teacher.eval()
    losses = []
    t0 = time.time()
    try:
        for step in range(steps):
            ids = seqs[step % seqs.shape[0]][None].to(dev)
            with torch.no_grad():
                t_logits = teacher(input_ids=ids, use_cache=False).logits
            s_logits = model(input_ids=ids, use_cache=False).logits
            loss = kl_loss(t_logits, s_logits)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            if step % fcfg.get("log_every", 16) == 0 or step == steps - 1:
                recent = sum(losses[-16:]) / len(losses[-16:])
                log(f"  finetune step {step + 1}/{steps}: KL {recent:.4f} ({time.time() - t0:.0f}s)")
    finally:
        model.gradient_checkpointing_disable()
        model.eval()
        with torch.no_grad():
            for name, (parent, child, lin, wrap) in wrappers.items():
                w = wrap.bake()
                lin.weight.copy_(w.to(lin.weight.dtype))
                write_back(fits[name], wrap)
                setattr(parent, child, lin)
        for p in model.parameters():
            p.requires_grad_(False)
    k = min(16, len(losses))
    return {"ft_kl_first": sum(losses[:k]) / max(1, k), "ft_kl_last": sum(losses[-k:]) / max(1, k),
            "ft_steps": len(losses), "ft_seconds": time.time() - t0}
