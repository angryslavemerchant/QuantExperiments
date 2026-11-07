"""Stage 5: codebook fine-tune.

Masks are frozen; the codebook entries and per-row scales train by Adam. Tags are
frozen too unless `repick` is set: then every `every` steps the last `accum` steps'
weight gradients drive a discrete code re-pick (PV-tuning style; see CodebookLinear.repick),
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
        self.teacher = False  # forward with lin's own weight (the original, see SharedTeacher)
        self.books = nn.ParameterList([nn.Parameter(b.detach().float().clone())
                                       for b in fit.stats["books"]])
        row_mask = fit.row_mask
        order = torch.argsort(row_mask, stable=True)
        self.register_buffer("order", order)
        self.register_buffer("unsort", torch.argsort(order))
        self.register_buffer("rs", fit.row_scale.detach().float()[order].clone())
        rb = fit.stats.get("row_book")
        self.register_buffer("row_book", None if rb is None else rb.to(order.device)[order].clone())
        self._cap = None   # gradient / curvature statistics while capturing for a re-pick
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

    # --- discrete code re-pick (PV-tuning-style V step) -------------------------------
    # Two backward passes per captured step: "fisher" (log-likelihood of tokens sampled from
    # the student: per-token output gradients whose squares estimate the curvature) and
    # "grad" (the real loss: the gradient). Hooks file each into its own statistic.
    def _accum_w(self, g):
        if self._cap["mode"] == "grad":
            self._cap["g"].add_(g.to(self._cap["g"].dtype))
            self._cap["n"] += 1

    def _accum_y(self, gy):
        if self._cap["mode"] == "fisher":
            self._cap["A"].add_(gy.float().pow(2).reshape(-1, gy.shape[-1]).mean(0))
            self._cap["nA"] += 1

    def start_capture(self, dtype=torch.bfloat16):
        if (len(self.groups) != 1 or self.groups[0][2] is not None or self.rot_in is not None
                or self.rot_out is not None):
            raise ValueError("repick supports M=1 without rotations")
        dev = self.rs.device
        self._cap = {"g": torch.zeros(self.R, self.C, device=dev, dtype=dtype), "n": 0,
                     "A": torch.zeros(self.R, device=dev), "nA": 0, "B": torch.zeros(self.C, device=dev), "nf": 0,
                     "mode": "grad"}

    @torch.no_grad()
    def propose(self, frac: float, damp: float = 0.1, etas=(1.0,), batch: int = 16384) -> list:
        """Candidate code changes from the captured statistics (capture stops); nothing applied.

        Loss model around the current weights: gradient g (mean over the captured steps)
        and diagonal K-FAC curvature F_ij = A_i B_j with A_i = mean_t dy_ti^2 from the
        sampled-token ("true Fisher") backward and B_j = mean_t x_tj^2 (the Gauss-Newton
        curvature of the token-mean KL; unlike the empirical Fisher of the KL gradient it
        does not scale with the current error). In the unscaled codeword space u (w = u * s
        per row) each chunk gets a target t = u - eta * g/F and score
        |c' - t|_F^2 - |c - t|_F^2; F is damped by damp * mean. Each chunk's best codeword in
        its row's book is found exhaustively; the top `frac` by decrease are proposed.

        The diagonal model ignores input correlations, so eta = 1 (the diagonal Newton step)
        overshoots badly when many correlated weights each try to cancel the same gradient;
        the caller picks eta by the real loss (see _repick_round). Returns, per eta,
        (chunk indices, new tags, predicted decrease), or None if nothing was captured."""
        cap, self._cap = self._cap, None
        if len(self.books) != 1 or cap["n"] == 0 or cap["nf"] == 0 or cap["nA"] == 0:
            return [None for _ in etas]
        d, R, C = self.d, self.R, self.C
        book = self.books[0].detach()
        tags = self.tags0
        nch = tags.numel() // R
        Cp = nch * d
        dev = book.device
        s = self.rs * self.log_gain.exp()
        A = cap["A"] / cap["nA"] * s ** 2                    # per row, into u space
        Bc = cap["B"] / cap["nf"]
        Fr = A[:, None] * Bc[None, :]
        Fr = Fr + damp * Fr.mean()
        step = torch.zeros(R, Cp, device=dev)
        F = torch.full((R, Cp), float(Fr.mean()), device=dev)   # padded columns: no gradient
        step[:, :C] = cap["g"].float() / cap["n"] * s[:, None] / Fr
        F[:, :C] = Fr
        del cap, Fr
        cur = tags.long().view(-1)
        u = book[cur]
        step = step.view(-1, d)
        F = F.view(-1, d)
        G = 1 if self.row_book is None else int(self.row_book.max()) + 1
        K = book.shape[0] // G
        chunk_book = (torch.zeros(R, dtype=torch.long, device=dev) if self.row_book is None
                      else self.row_book.long())[:, None].expand(R, nch).reshape(-1)
        groups = [(gi, (chunk_book == gi).nonzero().squeeze(1)) for gi in range(G)]
        out = []
        for eta in etas:
            t = u - eta * step
            cost_cur = (F * (u - t) ** 2).sum(1)
            best = cur.clone()
            cost_best = cost_cur.clone()
            for gi, idx in groups:
                if idx.numel() == 0:
                    continue
                B = book[gi * K:(gi + 1) * K]
                B2T = (B * B).T.contiguous()
                BT = B.T.contiguous()
                for b0 in range(0, idx.numel(), batch):
                    ii = idx[b0:b0 + batch]
                    f, tt = F[ii], t[ii]
                    sc = torch.addmm(f @ B2T, f * tt, BT, alpha=-2.0)   # sum F c^2 - 2 sum F t c
                    m, a = sc.min(1)
                    cb_ = m + (f * tt * tt).sum(1)
                    better = cb_ < cost_best[ii]
                    best[ii[better]] = a[better] + gi * K
                    cost_best[ii[better]] = cb_[better]
            # the expanded search form cancels badly near zero: re-score winners directly
            cost_best = (F * (book[best] - t) ** 2).sum(1)
            gain = torch.where(best != cur, cost_cur - cost_best, torch.zeros_like(cost_cur))
            gain = torch.where(gain > 1e-6 * cost_cur.abs().mean().clamp_min(1e-30), gain,
                               torch.zeros_like(gain))
            k = int(frac * gain.numel())
            if k == 0:
                out.append((cur[:0], cur[:0].to(tags.dtype), 0.0))
                continue
            top = torch.topk(gain, k).indices
            top = top[gain[top] > 0]
            out.append((top, best[top].to(tags.dtype), float(gain[top].sum() / 2)))
            del t, best, cost_best, cost_cur, gain
        return out

    @torch.no_grad()
    def swap_tags(self, idx: torch.Tensor, vals: torch.Tensor) -> torch.Tensor:
        """Write tags at idx; return the previous values (to revert)."""
        flat = self.tags0.view(-1)
        old = flat[idx].clone()
        flat[idx] = vals
        return old

    @torch.no_grad()
    def repick(self, frac: float, damp: float = 0.1, eta: float = 1.0) -> dict:
        """propose() with one step size, applied unconditionally."""
        p = self.propose(frac, damp, (eta,))[0]
        if p is None:
            return {"changed": 0, "chunks": 0, "pred_drop": 0.0}
        self.swap_tags(p[0], p[1])
        return {"changed": int(p[0].numel()), "chunks": self.tags0.numel(), "pred_drop": p[2]}

    def forward(self, x):
        if self.teacher:
            return F.linear(x, self.lin[0].weight, self.lin[0].bias)
        w = self.sorted_weight()
        cap = self._cap is not None and torch.is_grad_enabled()
        if cap:
            if w.requires_grad:
                w.register_hook(self._accum_w)
            self._cap["B"].add_(x.detach().float().pow(2).reshape(-1, x.shape[-1]).mean(0))
            self._cap["nf"] += 1
        w = w.to(x.dtype)
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
        y = y if bias is None else y + bias
        if cap and y.requires_grad:
            y.register_hook(self._accum_y)
        return y

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


@torch.no_grad()
def _repick_round(model, teacher, wrappers, seqs, step, dev, rp, log) -> dict:
    """One V step: a proposal per step size eta, each scored by the real KL on
    `eval_seqs` held calibration sequences (half the epoch away from the ones just
    trained on); the best is kept, or none if no proposal beats the current codes. A
    trust region on the actual loss, since the diagonal model's predictions are not."""
    etas = tuple(rp.get("etas", (1.0, 0.1, 0.01, 0.001)))
    props = {n: w.propose(rp.get("frac", 0.02), rp.get("damp", 0.1), etas)
             for n, (_, _, _, w) in wrappers.items()}
    ne = rp.get("eval_seqs", 2)
    pick = [(step + 1 + seqs.shape[0] // 2 + i) % seqs.shape[0] for i in range(ne)]
    batches = []
    for i in pick:
        ids = seqs[i][None].to(dev)
        batches.append((ids, teacher(input_ids=ids, use_cache=False).logits))

    def kl_now():
        return sum(float(kl_loss(t, model(input_ids=ids, use_cache=False).logits))
                   for ids, t in batches) / len(batches)

    base = kl_now()
    kls = []
    for e in range(len(etas)):
        olds = {n: wrappers[n][3].swap_tags(p[e][0], p[e][1])
                for n, p in props.items() if p[e] is not None}
        kls.append(kl_now())
        for n, old in olds.items():
            wrappers[n][3].swap_tags(props[n][e][0], old)
    e_best = min(range(len(etas)), key=lambda e: kls[e])
    ch = 0
    if kls[e_best] < base:
        for n, p in props.items():
            if p[e_best] is not None:
                wrappers[n][3].swap_tags(p[e_best][0], p[e_best][1])
                ch += int(p[e_best][0].numel())
    tot = sum(w.tags0.numel() for _, _, _, w in wrappers.values())
    pred = [sum(p[e][2] for p in props.values() if p[e] is not None) for e in range(len(etas))]
    kept = etas[e_best] if ch else None
    trials = ", ".join(f"eta {e:g}: {k:.4f} (pred -{pr:.3g})" for e, k, pr in zip(etas, kls, pred))
    kept_s = "none" if kept is None else f"eta {kept:g}"
    log(f"  repick at step {step + 1}: held-out KL {base:.4f} -> {trials}; kept {kept_s}, {ch}/{tot} chunks")
    return {"step": step + 1, "base_kl": base, "etas": list(etas), "kls": kls, "pred_drop": pred,
            "kept_eta": kept, "changed": ch, "chunks": tot}


def set_capture_mode(wrappers: dict, mode: str) -> None:
    for _, _, _, w in wrappers.values():
        if w._cap is not None:
            w._cap["mode"] = mode


def write_back(fit, wrap: "CodebookLinear") -> None:
    """After bake(): store the trained (fp16-rounded) books and row scales in the
    fit, in original row order, so the packed export carries them."""
    fit.stats["books"] = [b.detach().half() for b in wrap.books]
    fit.row_scale = wrap.rs[wrap.unsort].detach().clone()


class SharedTeacher:
    """The original model as the fine-tune teacher without a second copy of it: while the
    wrappers train, the replaced nn.Linear weights are unused (the wrappers decode their own),
    so they hold the ORIGINAL weights and a teacher forward runs the same model with every
    wrapper passing through to them. Same logits as a separately loaded original model (same
    weights and ops; only block linears differ between student and teacher), half the memory."""

    def __init__(self, model, wrappers: dict, originals: dict):
        self.model, self.wrappers = model, wrappers
        with torch.no_grad():
            for name, (_, _, lin, _) in wrappers.items():
                lin.weight.copy_(originals[name].to(lin.weight.device, lin.weight.dtype))

    def eval(self):
        return self

    @torch.no_grad()
    def __call__(self, **kw):
        for *_, w in self.wrappers.values():
            w.teacher = True
        try:
            return self.model(**kw)
        finally:
            for *_, w in self.wrappers.values():
                w.teacher = False


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


@torch.no_grad()
def heldout_kl(model, teacher, seqs: torch.Tensor, dev) -> float:
    """Mean token KL(teacher || student) over held-out sequences (one at a time)."""
    tot = 0.0
    for i in range(seqs.shape[0]):
        ids = seqs[i][None].to(dev)
        t = teacher(input_ids=ids, use_cache=False).logits
        tot += float(kl_loss(t, model(input_ids=ids, use_cache=False).logits))
        del t
    return tot / seqs.shape[0]


def _snapshot(wrappers: dict) -> dict:
    """The trainable state (books + row gains) on the CPU, to restore the best one."""
    return {n: ([b.detach().cpu().clone() for b in w.books], w.log_gain.detach().cpu().clone())
            for n, (*_, w) in wrappers.items()}


@torch.no_grad()
def _restore(wrappers: dict, snap: dict) -> None:
    for n, (books, gain) in snap.items():
        w = wrappers[n][3]
        for b, v in zip(w.books, books):
            b.copy_(v.to(b.device))
        w.log_gain.copy_(gain.to(w.log_gain.device))


def finetune(model, teacher, fits: dict, d_by_name: dict, sequences: torch.Tensor,
             fcfg: dict, log=print, originals: dict | None = None, heldout: dict | None = None) -> dict:
    """Train codebooks + row gains in place; write baked weights into `model`.
    teacher: the original model, or None with `originals` (the block linears' original
    weights) for a SharedTeacher (no second model copy).
    heldout: {name: [n, seq_len] token ids} scored (KL vs the teacher) before training and
    every fcfg["heldout_every"] steps; with fcfg["keep_best"] (a heldout name) the books and
    gains with the lowest KL on that set are the ones baked, so training cannot end worse than
    it started on it."""
    steps = fcfg.get("steps", 256)
    seq_len = fcfg.get("seq_len", 1024)
    seqs = sequences.reshape(-1, seq_len)
    dev = next(model.parameters()).device
    for p in model.parameters():
        p.requires_grad_(False)
    wrappers = _swap(model, fits, d_by_name)
    if teacher is None:
        if originals is None:
            raise ValueError("finetune: teacher=None needs originals")
        teacher = SharedTeacher(model, wrappers, originals)
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
    rp = fcfg.get("repick")
    rp_log = []
    heldout = heldout or {}
    h_every = fcfg.get("heldout_every", 64)
    keep = fcfg.get("keep_best")
    if keep is not None and keep not in heldout:
        raise ValueError(f"finetune: keep_best {keep!r} is not a held-out set ({sorted(heldout)})")
    h_log: list[dict] = []
    best = {"kl": float("inf"), "step": None, "snap": None}

    def score_heldout(step):
        if not heldout:
            return
        rec = {"step": step, **{k: heldout_kl(model, teacher, v, dev) for k, v in heldout.items()}}
        h_log.append(rec)
        msg = "  ".join(f"{k} {rec[k]:.4f}" for k in heldout)
        if keep is not None and rec[keep] < best["kl"]:
            best.update(kl=rec[keep], step=step, snap=_snapshot(wrappers))
            msg += "  (best)"
        log(f"  held-out KL at step {step}: {msg}")

    t0 = time.time()
    try:
        score_heldout(0)
        for step in range(steps):
            # capture over steps [k*every - accum, k*every); re-pick after step k*every, leaving
            # at least `every` steps after the last re-pick for the books to settle
            if rp and ((step + rp.get("accum", 8)) % rp["every"] == 0
                       and step + rp.get("accum", 8) + rp["every"] <= steps):
                for _, _, _, w in wrappers.values():
                    w.start_capture()
            ids = seqs[step % seqs.shape[0]][None].to(dev)
            with torch.no_grad():
                t_logits = teacher(input_ids=ids, use_cache=False).logits
            s_logits = model(input_ids=ids, use_cache=False).logits
            loss = kl_loss(t_logits, s_logits)
            opt.zero_grad(set_to_none=True)
            if rp and next(iter(wrappers.values()))[3]._cap is not None:
                set_capture_mode(wrappers, "fisher")
                with torch.no_grad():
                    lp = F.log_softmax(s_logits.detach().float(), dim=-1)
                    yhat = torch.distributions.Categorical(logits=lp).sample()
                    del lp
                fl = F.log_softmax(s_logits.float(), dim=-1).gather(-1, yhat[..., None]).sum()
                fl.backward(retain_graph=True)
                del fl
                opt.zero_grad(set_to_none=True)
                set_capture_mode(wrappers, "grad")
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            if rp and (step + 1) % rp["every"] == 0 and next(iter(wrappers.values()))[3]._cap is not None:
                rp_log.append(_repick_round(model, teacher, wrappers, seqs, step, dev, rp, log))
            if step % fcfg.get("log_every", 16) == 0 or step == steps - 1:
                recent = sum(losses[-16:]) / len(losses[-16:])
                log(f"  finetune step {step + 1}/{steps}: KL {recent:.4f} ({time.time() - t0:.0f}s)")
            if heldout and ((step + 1) % h_every == 0 or step == steps - 1):
                score_heldout(step + 1)
        if keep is not None and best["snap"] is not None and best["step"] != steps:
            log(f"  keeping the books + gains of step {best['step']} (best {keep} KL {best['kl']:.4f})")
            _restore(wrappers, best["snap"])
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
            "ft_steps": len(losses), "ft_seconds": time.time() - t0,
            **({"ft_repick": rp_log} if rp else {}),
            **({"ft_heldout": h_log} if h_log else {}),
            **({"ft_best_step": best["step"]} if keep is not None else {})}
