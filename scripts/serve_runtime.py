"""OpenAI-compatible HTTP server for the packed-model runtime (shapequant.runtime).

    python scripts/serve_runtime.py --packed model.safetensors [--head q4] [--max-len 4096] [--port 8000]

Endpoints (one request at a time; batched prompts run one after another):
  POST /v1/completions       prompt: string | token ids | list of either; max_tokens, temperature,
                             top_p, top_k, seed, stop (string or list), logprobs (int), echo.
                             echo + logprobs returns the prompt tokens' logprobs, as lm-eval-harness's
                             `local-completions` backend uses for loglikelihood tasks:
                               lm_eval --model local-completions --model_args \\
                                 base_url=http://localhost:8000/v1/completions,model=<name>,\\
                                 tokenizer=<base HF repo>,tokenized_requests=True,max_length=<max-len>
  POST /v1/chat/completions  messages (chat template applied), same sampling fields.
  GET  /v1/models
No /tokenizer_info endpoint: lm-eval then tokenizes locally with the HF tokenizer it is given.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import uuid
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shapequant import packed, runtime  # noqa: E402


def _as_prompts(p):
    """prompt field -> list of prompts, each a str or a list of token ids."""
    if isinstance(p, str):
        return [p]
    if isinstance(p, list) and p and all(isinstance(t, int) for t in p):
        return [p]
    if isinstance(p, list):
        return list(p)
    raise ValueError("prompt must be a string, a token list, or a list of those")


class Server:
    def __init__(self, eng, tok, model_name: str):
        self.eng, self.tok, self.model_name = eng, tok, model_name
        self.lock = threading.Lock()
        eos = getattr(tok, "eos_token_id", None)
        self.eos_ids = set(eos if isinstance(eos, (list, tuple)) else ([] if eos is None else [eos]))

    def _ids(self, p) -> list[int]:
        return list(p) if isinstance(p, list) else list(self.tok.encode(p, add_special_tokens=False))

    def _piece(self, t: int) -> str:
        return self.tok.decode([t])

    def _generate(self, ids, body, n):
        stops = body.get("stop") or []
        stops = [stops] if isinstance(stops, str) else list(stops)
        stop_fn = None
        if stops:
            def stop_fn(out):
                text = self.tok.decode(out)
                return any(s in text for s in stops)
        toks, lps = runtime.generate(self.eng, ids, n, temperature=float(body.get("temperature") or 0.0),
                                     top_p=float(body.get("top_p", 1.0) or 1.0), top_k=int(body.get("top_k", 0) or 0),
                                     seed=body.get("seed"), stop_ids=self.eos_ids, stop_fn=stop_fn,
                                     return_logprobs=True)
        text = self.tok.decode(toks, skip_special_tokens=True)
        finish = "length"
        for s in stops:
            if s in text:
                text, finish = text[:text.index(s)], "stop"
        if finish == "length" and toks and toks[-1] in self.eos_ids:
            finish = "stop"
        return toks, lps, text, finish

    def completion(self, body: dict) -> dict:
        prompts = _as_prompts(body.get("prompt", ""))
        n = int(body.get("max_tokens", 16) if body.get("max_tokens") is not None else 16)
        echo = bool(body.get("echo", False))
        want_lp = body.get("logprobs")
        choices, usage_p, usage_c = [], 0, 0
        with self.lock:
            for i, p in enumerate(prompts):
                ids = self._ids(p)
                if not ids:
                    raise ValueError("empty prompt")
                usage_p += len(ids)
                if echo:
                    choices.append(self._echo_choice(i, ids, n, want_lp, body))
                    usage_c += min(n, 1) if n else 0
                    continue
                toks, lps, text, finish = self._generate(ids, body, max(n, 1))
                usage_c += len(toks)
                ch = {"index": i, "text": text, "finish_reason": finish, "logprobs": None}
                if want_lp is not None:
                    pieces = [self._piece(t) for t in toks]
                    ch["logprobs"] = {"tokens": pieces, "token_logprobs": lps,
                                      "top_logprobs": [{pc: lp} for pc, lp in zip(pieces, lps)],
                                      "text_offset": _offsets(pieces, 0)}
                choices.append(ch)
        return {"id": f"cmpl-{uuid.uuid4().hex[:12]}", "object": "text_completion", "created": int(time.time()),
                "model": self.model_name, "choices": choices,
                "usage": {"prompt_tokens": usage_p, "completion_tokens": usage_c,
                          "total_tokens": usage_p + usage_c}}

    def _echo_choice(self, i, ids, n, want_lp, body) -> dict:
        """Prompt logprobs (first token: None), plus up to one greedy continuation token."""
        if len(ids) + max(n, 1) > self.eng.cfg.max_len:
            raise ValueError(f"prompt of {len(ids)} tokens exceeds max_len {self.eng.cfg.max_len}")
        lg = runtime.start(self.eng, ids, all_logits=True)              # [T, V], row j predicts j+1
        lsm = torch.log_softmax(lg, -1)
        tgt = torch.tensor(ids[1:], device=lsm.device)
        tok_lp = [None] + lsm[:-1].gather(1, tgt[:, None]).squeeze(1).tolist()
        top_v, top_i = lsm.max(-1)
        top = [None] + [{self._piece(int(a)): float(b)} for a, b in zip(top_i[:-1].tolist(), top_v[:-1].tolist())]
        toks = list(ids)
        if n >= 1:
            g = int(top_i[-1])
            toks.append(g)
            tok_lp.append(float(top_v[-1]))
            top.append({self._piece(g): float(top_v[-1])})
        # actual tokens' own logprob must be comparable to the top value: put it in the dict too
        for j in range(1, len(ids)):
            top[j].setdefault(self._piece(ids[j]), tok_lp[j])
        pieces = [self._piece(t) for t in toks]
        text = self.tok.decode(toks)
        ch = {"index": i, "text": text, "finish_reason": "length", "logprobs": None}
        if want_lp is not None:
            ch["logprobs"] = {"tokens": pieces, "token_logprobs": tok_lp, "top_logprobs": top,
                              "text_offset": _offsets(pieces, 0)}
        return ch

    def chat(self, body: dict) -> dict:
        msgs = body.get("messages") or []
        text = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ids = self._ids(text)
        n = int(body.get("max_tokens") or body.get("max_completion_tokens") or 256)
        with self.lock:
            toks, lps, out, finish = self._generate(ids, body, n)
        return {"id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion", "created": int(time.time()),
                "model": self.model_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": out}, "finish_reason": finish}],
                "usage": {"prompt_tokens": len(ids), "completion_tokens": len(toks),
                          "total_tokens": len(ids) + len(toks)}}


def _offsets(pieces, start):
    out, o = [], start
    for p in pieces:
        out.append(o)
        o += len(p)
    return out


def create_app(eng, tok, model_name: str):
    from fastapi import FastAPI, HTTPException
    srv = Server(eng, tok, model_name)
    app = FastAPI()

    def guard(fn, body):
        try:
            return fn(body)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "shapequant"}]}

    @app.post("/v1/completions")
    def completions(body: dict):
        return guard(srv.completion, body)

    @app.post("/v1/chat/completions")
    def chat(body: dict):
        return guard(srv.chat, body)

    app.state.server = srv
    return app


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", required=True)
    ap.add_argument("--head", default="q8", choices=("q4", "q8", "fp16"))
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--name", default=None, help="model id reported by /v1/models (default: file stem)")
    args = ap.parse_args()
    import uvicorn
    from transformers import AutoTokenizer
    from shapequant.packed import PackedFile
    base = PackedFile(args.packed).summary["base_model"]
    tok = AutoTokenizer.from_pretrained(base)
    if packed.PackedFile(args.packed).aux is not None:     # self-contained file: its own q8 head
        args.head = "q8"
    eng, info = runtime.engine_from_packed(args.packed, head=args.head, max_len=args.max_len, prefill=True)
    eng.capture()
    name = args.name or Path(args.packed).stem
    print(f"serving {name} ({base}, {info['summary'].get('bpw', 0):.3f} bpw, head {args.head}) on "
          f"http://{args.host}:{args.port}", flush=True)
    uvicorn.run(create_app(eng, tok, name), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
