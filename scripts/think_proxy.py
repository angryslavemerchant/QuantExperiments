"""Tiny OpenAI-API pass-through that merges fixed fields into every /v1/chat/completions body, e.g. thinking off
(chat_template_kwargs.enable_thinking=false) or sampling settings, for eval tools that cannot send them
(lm-eval). Works in front of vLLM and llama-server. Stdlib only, streams responses through unchanged.

  python scripts/think_proxy.py --upstream http://127.0.0.1:8000 --port 8001 \
      --merge '{"chat_template_kwargs": {"enable_thinking": false}}'
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def make_handler(upstream: str, merge: dict, key: str | None):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # quiet
            pass

        def _fwd(self, body: bytes | None):
            if body is not None and self.path.rstrip("/").endswith("/chat/completions"):
                try:
                    d = json.loads(body)
                    for k, v in merge.items():
                        if isinstance(v, dict) and isinstance(d.get(k), dict):
                            d[k] = {**d[k], **v}
                        else:
                            d[k] = v
                    body = json.dumps(d).encode()
                except ValueError:
                    pass
            hdr = {"Content-Type": "application/json"}
            auth = self.headers.get("Authorization")
            if key:
                hdr["Authorization"] = f"Bearer {key}"
            elif auth:
                hdr["Authorization"] = auth
            req = urllib.request.Request(upstream + self.path, data=body, headers=hdr,
                                         method="POST" if body is not None else "GET")
            try:
                r = urllib.request.urlopen(req, timeout=7200)
                code = r.status
            except urllib.error.HTTPError as e:
                r, code = e, e.code
            data_iter = iter(lambda: r.read(65536) if r.headers.get("Content-Type", "").startswith(
                "text/event-stream") is False else r.readline(), b"")
            self.send_response(code)
            self.send_header("Content-Type", r.headers.get("Content-Type", "application/json"))
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for chunk in data_iter:
                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")

        def do_GET(self):
            self._fwd(None)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            self._fwd(self.rfile.read(n))

    return H


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--merge", default='{"chat_template_kwargs": {"enable_thinking": false}}')
    ap.add_argument("--api-key", help="key to send upstream (default: pass the client's through)")
    a = ap.parse_args()
    ThreadingHTTPServer(("127.0.0.1", a.port),
                        make_handler(a.upstream.rstrip("/"), json.loads(a.merge), a.api_key)).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
