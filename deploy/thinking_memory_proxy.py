"""OpenAI-API pass-through that gives chat clients "preserved thinking": it remembers the thinking of every
reply it streams out and puts it back into later requests, so earlier turns reach the model exactly as it
generated them.

Why: Qwen3.8's chat template keeps past thinking by default (preserve_thinking), but chat clients such as
Cherry Studio send old assistant turns back without it (vLLM only reads a `reasoning` field on assistant
messages). The template then renders an empty <think></think> for every past turn:
  - the model never sees its earlier reasoning, and
  - the prompt differs from what was generated, so vLLM's prefix cache misses from the last reply onward.
With this proxy the history matches the generated tokens. Cost: thinking stays in the context, so long
chats fill the 98k window faster.

  python deploy/thinking_memory_proxy.py --upstream http://127.0.0.1:8000 --port 8002
Point the client at http://127.0.0.1:8002/v1 (same API key). Stdlib only; remembered thinking is kept in
memory (and in --store if given), keyed by the reply text.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _key(content) -> str | None:
    t = _text(content).strip()
    return hashlib.sha256(t.encode()).hexdigest() if t else None


class Memory:
    def __init__(self, store: str | None, cap: int = 5000):
        self.d: collections.OrderedDict[str, str] = collections.OrderedDict()
        self.cap, self.store, self.lock = cap, store, threading.Lock()
        if store and os.path.exists(store):
            for line in open(store, encoding="utf-8"):
                try:
                    k, v = json.loads(line)
                    self.d[k] = v
                except ValueError:
                    pass
            while len(self.d) > cap:
                self.d.popitem(last=False)

    def put(self, content: str, reasoning: str):
        k = _key(content)
        if not k or not reasoning.strip():
            return
        with self.lock:
            self.d[k] = reasoning
            self.d.move_to_end(k)
            while len(self.d) > self.cap:
                self.d.popitem(last=False)
            if self.store:
                with open(self.store, "a", encoding="utf-8") as f:
                    f.write(json.dumps([k, reasoning]) + "\n")

    def get(self, content) -> str | None:
        k = _key(content)
        with self.lock:
            return self.d.get(k) if k else None


def make_handler(upstream: str, mem: Memory, stats: dict):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _restore(self, d: dict):
            hit = miss = 0
            for m in d.get("messages") or []:
                if m.get("role") != "assistant" or m.get("reasoning"):
                    continue
                r = m.pop("reasoning_content", None) or mem.get(m.get("content"))
                if r:
                    m["reasoning"] = r
                    hit += 1
                elif _text(m.get("content")).strip():
                    miss += 1
            stats["restored"] += hit
            stats["missing"] += miss

        def _fwd(self, body: bytes | None):
            chat = body is not None and self.path.rstrip("/").endswith("/chat/completions")
            if chat:
                try:
                    d = json.loads(body)
                    self._restore(d)
                    body = json.dumps(d).encode()
                except ValueError:
                    chat = False
            hdr = {"Content-Type": "application/json"}
            if self.headers.get("Authorization"):
                hdr["Authorization"] = self.headers["Authorization"]
            req = urllib.request.Request(upstream + self.path, data=body, headers=hdr,
                                         method="POST" if body is not None else "GET")
            try:
                r = urllib.request.urlopen(req, timeout=7200)
                code = r.status
            except urllib.error.HTTPError as e:
                r, code = e, e.code
            ctype = r.headers.get("Content-Type", "application/json")
            sse = ctype.startswith("text/event-stream")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            reasoning, content, whole = [], [], []
            for chunk in iter((r.readline if sse else (lambda: r.read(65536))), b""):
                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
                if not (chat and code == 200):
                    continue
                if not sse:
                    whole.append(chunk)
                elif chunk.startswith(b"data: ") and not chunk.startswith(b"data: [DONE]"):
                    try:
                        for ch in json.loads(chunk[6:]).get("choices") or []:
                            if ch.get("index", 0) != 0:
                                continue
                            dl = ch.get("delta") or {}
                            reasoning.append(dl.get("reasoning") or dl.get("reasoning_content") or "")
                            content.append(dl.get("content") or "")
                    except ValueError:
                        pass
            self.wfile.write(b"0\r\n\r\n")
            if chat and code == 200:
                if whole:
                    try:
                        msg = json.loads(b"".join(whole))["choices"][0]["message"]
                        reasoning = [msg.get("reasoning") or msg.get("reasoning_content") or ""]
                        content = [msg.get("content") or ""]
                    except (ValueError, KeyError, IndexError):
                        pass
                mem.put("".join(content), "".join(reasoning))

        def do_GET(self):
            if self.path == "/proxy-stats":
                b = json.dumps({**stats, "remembered": len(mem.d)}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
                return
            self._fwd(None)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            self._fwd(self.rfile.read(n))

    return H


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default="http://127.0.0.1:8000")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--store", default=os.path.expanduser("~/.cache/thinking_memory.jsonl"),
                    help="file to keep remembered thinking across restarts ('' = memory only)")
    a = ap.parse_args()
    if a.store:
        os.makedirs(os.path.dirname(a.store), exist_ok=True)
    stats = {"restored": 0, "missing": 0}
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(a.upstream.rstrip("/"), Memory(a.store or None), stats))
    print(f"thinking-memory proxy on http://{a.host}:{a.port} -> {a.upstream}", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
