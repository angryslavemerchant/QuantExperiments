"""scripts/serve_eval.py against a fake OpenAI-compatible server (CPU only)."""

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("serve_eval", ROOT / "scripts" / "serve_eval.py")
se = importlib.util.module_from_spec(spec)
spec.loader.exec_module(se)


def _fake_items(task, style="lighteval"):
    return [{"id": i, "prompt": f"question {i}", "doc": {"gold": "ABCD"[i % 4]}} for i in range(6)]


class _H(BaseHTTPRequestHandler):
    seen = []
    split_reasoning = False

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _H.seen.append(body)
        qid = int(body["messages"][0]["content"].split()[-1])
        letter = "ABCD"[qid % 4] if qid % 3 else "A"          # q0, q3 answered "A" (q3 wrong: gold D)
        if _H.split_reasoning:
            msg = {"content": f"\n\nAnswer: {letter}", "reasoning_content": "thinking..."}
        else:
            msg = {"content": f"thinking...</think>\n\nAnswer: {letter}"}
        out = {"choices": [{"message": msg, "finish_reason": "stop"}], "usage": {"completion_tokens": 7}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _serve():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def test_round_major_scoring_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(se.te, "load_task", _fake_items)
    srv, url = _serve()
    try:
        _H.seen.clear()
        out = tmp_path / "o"
        assert se.main(["--endpoints", f"{url},{url}", "--conc", "2", "--name", "t", "--out", str(out),
                        "--n", "2", "--max-tokens", "64"]) == 0
        rows = [json.loads(x) for x in (out / "samples.jsonl").read_text().splitlines()]
        assert len(rows) == 12 and {(r["id"], r["k"]) for r in rows} == {(i, k) for i in range(6) for k in range(2)}
        b = _H.seen[0]
        assert b["chat_template_kwargs"] == {"enable_thinking": True} and b["top_k"] == 20 and b["max_tokens"] == 64
        assert len({x["seed"] for x in _H.seen}) == 12
        s = json.loads((out / "summary.json").read_text())["tasks"]["gpqa"]
        assert abs(s["exact_match"] - 5 / 6) < 1e-9 and s["coverage"] == {"2": 6} and s["rounds"]["0"]["n"] == 6
        _H.seen.clear()                                      # rerun: nothing left to do
        assert se.main(["--endpoints", url, "--name", "t", "--out", str(out), "--n", "2"]) == 0
        assert _H.seen == []
        _H.split_reasoning = True                            # server-side reasoning split is undone
        out2 = tmp_path / "o2"
        se.main(["--endpoints", url, "--name", "t", "--out", str(out2), "--n", "1"])
        r = json.loads((out2 / "samples.jsonl").read_text().splitlines()[0])
        assert "</think>" in r["text"] and r["scores"]["think_closed"] == 1
    finally:
        _H.split_reasoning = False
        srv.shutdown()


def test_unreachable_endpoint_counts_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(se.te, "load_task", _fake_items)
    rc = se.main(["--endpoints", "http://127.0.0.1:9", "--name", "t", "--out", str(tmp_path), "--retries", "0",
                  "--limit", "2"])
    assert rc == 1
    assert json.loads((tmp_path / "summary.json").read_text())["errors"] == 2


def test_response_text_variants():
    assert se.response_text({"message": {"content": "a</think>b"}}) == "a</think>b"
    assert se.response_text({"message": {"content": "b", "reasoning_content": "a"}}) == "a</think>b"
    assert se.response_text({"message": {"content": "", "reasoning_content": "a"}, "finish_reason": "length"}) == "a"
