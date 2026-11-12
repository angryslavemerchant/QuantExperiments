"""Soft thinking budget against runaway thinking loops (vLLM model-runner-V2 logits processor).

Why: the model's p(</think>) is ~1e-7 at every line break and snaps to ~1 at one line when it is done; runaway
traces never reach that line (scripts/loopmech_*.py, artifacts/loopmech/). A small ramping bonus on </think> after
long thinking ends the loops (0/30 runaways vs 16/30) with no measurable quality loss (HumanEval+, MMLU-Pro).

Rule, per request with thinking on: once the thinking has run for `gate` tokens, add
    min(cap, slope * (n - gate) / 1000)
to the </think> logit, only right after a line break, never inside a ``` code block and never right after a
markdown table row (lines starting with "|"), so it does not cut a table or code being drafted. Nothing changes
before the gate, after </think>, or for requests with thinking off.

Serve:   vllm serve ... --logits-processors shapequant.vllm_ext.think_stop:ThinkStop
Env defaults: THINK_STOP_GATE (8000; 0 disables), THINK_STOP_SLOPE (6), THINK_STOP_CAP (30).
Per request (OpenAI API body): "vllm_xargs": {"think_stop_gate": 12000, "think_stop_slope": 6}; gate 0 = off.

Cost: host-side check per step; the processor reads token history only for requests past gate - 256.
Under speculative decoding the line/code-block state is tracked on accepted tokens only (lags the drafts by at
most k+1 tokens).
"""

from __future__ import annotations

import os

from vllm.v1.worker.gpu.sample.logits_processor import LogitsProcessor

_TABLES: dict = {}


def _token_tables(path: str):
    """Per-token text, plus vocab masks: ends with a newline / contains a code fence. Cached per tokenizer."""
    if path not in _TABLES:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(path)
        texts = [tok.decode([i]) for i in range(len(tok))]
        _TABLES[path] = (texts, tok.convert_tokens_to_ids("</think>"))
    return _TABLES[path]


class _Req:
    __slots__ = ("gate", "slope", "done", "scanned", "fence", "table", "prev_table", "line_start", "last_ch",
                 "prev_colon")

    def __init__(self, gate: int, slope: float, scanned: int):
        self.gate, self.slope = gate, slope
        self.done = gate <= 0          # off, thinking closed, or thinking disabled
        self.scanned = scanned         # absolute position up to which tokens were processed
        self.fence = False             # inside a ``` block
        self.table = False             # current line is a table row
        self.prev_table = False        # the previous line was a table row
        self.line_start = True
        self.last_ch = ""              # last non-space character of the current line
        self.prev_colon = False        # the last finished non-empty line ended with ":" (announces what follows)


class ThinkStop(LogitsProcessor):
    def __init__(self, vllm_config, req_states):
        import torch
        self.rs = req_states
        self.dev = req_states.device
        mc = vllm_config.model_config
        self.texts, self.end_id = _token_tables(mc.tokenizer or mc.model)
        V = req_states.vocab_size
        self.nl = torch.zeros(V, dtype=torch.bool)
        self.fence_tok = torch.zeros(V, dtype=torch.bool)
        for i, t in enumerate(self.texts[:V]):
            if t.endswith("\n"):
                self.nl[i] = True
            if "```" in t:
                self.fence_tok[i] = True
        self.nl, self.fence_tok = self.nl.to(self.dev), self.fence_tok.to(self.dev)
        self.def_gate = int(os.environ.get("THINK_STOP_GATE", 8000))
        self.def_slope = float(os.environ.get("THINK_STOP_SLOPE", 6))
        self.cap = float(os.environ.get("THINK_STOP_CAP", 30))
        sc = vllm_config.speculative_config
        self.lag = (sc.num_speculative_tokens if sc is not None else 0) + 1
        n = req_states.max_num_reqs
        self.req: dict[int, _Req] = {}
        self.gate_t = torch.zeros(n, dtype=torch.float32, device=self.dev)
        self.slope_t = torch.zeros(n, dtype=torch.float32, device=self.dev)
        self.elig_host = torch.zeros(n, dtype=torch.bool)
        self.elig = torch.zeros(n, dtype=torch.bool, device=self.dev)
        # device-side "</think> seen" per slot: set from every row's input token (no host sync, no reliance on
        # the host view of the token history being current). A rejected draft </think> can set it too, which
        # only switches the bonus off early (fail-safe).
        self.closed = torch.zeros(n, dtype=torch.int32, device=self.dev)
        self.uva = getattr(req_states.all_token_ids, "_uva_buf", None)

    @classmethod
    def validate_params(cls, sampling_params) -> None:
        x = sampling_params.extra_args or {}
        for k in ("think_stop_gate", "think_stop_slope"):
            if k in x and not isinstance(x[k], (int, float)):
                raise ValueError(f"{k} must be a number")

    def add_request(self, req_idx: int, sampling_params) -> bool:
        x = sampling_params.extra_args or {}
        gate = int(x.get("think_stop_gate", self.def_gate))
        slope = float(x.get("think_stop_slope", self.def_slope))
        plen = int(self.rs.prompt_len.np[req_idx])
        r = _Req(gate, slope, plen)
        self.req[req_idx] = r
        self.gate_t[req_idx] = float(max(gate, 0))
        self.slope_t[req_idx] = slope / 1000.0
        self.elig_host[req_idx] = False
        self.closed[req_idx] = 0
        return not r.done

    def _tokens(self, slot: int, a: int, b: int) -> list[int]:
        if self.uva is not None:
            return self.uva.np[slot, a:b].tolist()
        return self.rs.all_token_ids.gpu[slot, a:b].tolist()   # dense layout: device read (rare: past gate only)

    def _advance(self, slot: int, r: _Req, upto: int) -> None:
        if r.scanned == int(self.rs.prompt_len.np[slot]) and r.scanned > 0:
            # first look: thinking disabled if the prompt already closes it (template's empty think block)
            plen = r.scanned
            if self.end_id in self._tokens(slot, max(0, plen - 6), plen):
                r.done = True
                return
        if upto <= r.scanned:
            return
        texts = self.texts
        for t_id in self._tokens(slot, r.scanned, upto):
            if t_id == self.end_id:
                r.done = True
                break
            t = texts[t_id] if 0 <= t_id < len(texts) else ""
            if "```" in t:
                r.fence ^= bool(t.count("```") % 2)
            if "\n" in t:
                head, tail = t.split("\n", 1)[0], t.rsplit("\n", 1)[1]
                if head.strip():
                    r.last_ch = head.rstrip()[-1]
                if r.last_ch:                      # a non-empty line ended (blank lines keep the previous state)
                    r.prev_colon = r.last_ch == ":"
                    r.prev_table = r.table
                r.last_ch = tail.strip()[-1:] if tail.strip() else ""
                r.line_start = tail.strip() == ""
                r.table = tail.lstrip().startswith("|")
            else:
                if r.line_start and t.strip():
                    r.table = t.lstrip().startswith("|")
                    r.line_start = False
                if t.strip():
                    r.last_ch = t.rstrip()[-1]
        r.scanned = upto

    def apply(self, logits, ctx):
        if not self.req:
            return logits
        plen = self.rs.prompt_len.np
        any_on = False
        touched = False
        for b, slot in enumerate(ctx.idx_mapping_np.tolist()):
            r = self.req.get(slot)
            if r is None or r.done:
                continue
            upper = int(ctx.seq_lens_upper_bound_np[b])
            if upper - int(plen[slot]) < r.gate - 256:
                if self.elig_host[slot]:
                    self.elig_host[slot] = False
                    touched = True
                continue
            # accepted tokens only, with a margin for the asynchronously written history
            self._advance(slot, r, max(r.scanned, upper - self.lag - 8))
            on = not (r.done or r.fence or r.table or r.prev_table or r.prev_colon or r.last_ch == ":")
            if bool(self.elig_host[slot]) != on:
                self.elig_host[slot] = on
                touched = True
            any_on |= on
        if touched:
            self.elig = self.elig_host.to(self.dev)   # fresh tensor: no race with kernels still reading the old one
        import torch
        slots = ctx.expanded_idx_mapping.long()
        tok = ctx.input_ids.long()
        self.closed.scatter_reduce_(0, slots, (tok == self.end_id).to(torch.int32), reduce="amax")
        if not any_on:
            return logits
        n_out = ctx.pos.long() + 1 - self.rs.prompt_len.gpu[slots].long()
        g = self.gate_t[slots]
        on = (self.elig[slots] & (self.closed[slots] == 0) & self.nl[tok] & ~self.fence_tok[tok]
              & (n_out.float() >= g))
        bonus = ((n_out.float() - g) * self.slope_t[slots]).clamp(0, self.cap)
        logits[:, self.end_id] += torch.where(on, bonus, torch.zeros_like(bonus))
        return logits
