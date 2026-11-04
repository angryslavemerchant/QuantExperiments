"""HF_TOKEN forwarding to rented boxes (user-approved 2026-11-04; HF_TOKEN only).

The token travels on SSH stdin into the bootstrap's environment, never on a
command line; the download probe sends it to huggingface.co only (not to the
CDN hosts it redirects to)."""

import shutil
import subprocess
import urllib.request

import pytest

from vast import net_probe, ops


def test_stdin_line_carries_hf_token_only(monkeypatch):
    monkeypatch.setattr(ops, "load_secrets", lambda: {"VAST_API_KEY": "vast-secret", "HF_TOKEN": "hf_abc"})
    line = ops._hf_token_stdin()
    assert line == "hf_abc\n"
    assert "vast-secret" not in line


def test_missing_token_sends_empty_line(monkeypatch):
    monkeypatch.setattr(ops, "load_secrets", lambda: {"VAST_API_KEY": "x"})
    assert ops._hf_token_stdin() == "\n"


def test_multiline_token_rejected(monkeypatch):
    monkeypatch.setattr(ops, "load_secrets", lambda: {"HF_TOKEN": "a\nb"})
    with pytest.raises(ops.OpsError):
        ops._hf_token_stdin()


def test_token_value_never_in_command_text(monkeypatch):
    monkeypatch.setattr(ops, "load_secrets", lambda: {"HF_TOKEN": "hf_abc"})
    ops._hf_token_stdin()
    assert isinstance(ops._HF_TOKEN_FROM_STDIN, str) and "hf_abc" not in ops._HF_TOKEN_FROM_STDIN


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize("stdin,expected", [("hf_abc\n", "set:hf_abc"), ("\n", "unset"), ("", "unset")])
def test_remote_prefix_exports_only_nonempty(stdin, expected):
    script = ops._HF_TOKEN_FROM_STDIN + ' bash -c \'if [ -n "${HF_TOKEN+x}" ]; then echo "set:$HF_TOKEN"; else echo unset; fi\''
    out = subprocess.run([shutil.which("bash"), "-c", script], input=stdin, capture_output=True, text=True)
    assert out.stdout.strip() == expected


class _Resp:
    def __init__(self):
        self._sent = False

    def read(self, n):
        if self._sent:
            return b""
        self._sent = True
        return b"x" * 10

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.mark.parametrize("url,authed", [
    ("https://huggingface.co/Qwen/x/resolve/main/f", True),
    ("https://example.com/f", False),
    ("https://huggingface.co.evil.example/f", False),
])
def test_probe_sends_token_unredirected_to_hf_only(monkeypatch, url, authed):
    seen = {}

    def fake_urlopen(req, timeout):
        seen["req"] = req
        return _Resp()

    monkeypatch.setenv("HF_TOKEN", "hf_abc")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    net_probe.probe(url, 10, 5)
    req = seen["req"]
    assert "Authorization" not in req.headers                 # never forwarded on redirects
    assert (req.unredirected_hdrs.get("Authorization") == "Bearer hf_abc") == authed


def test_diagnostic_tail_makes_progress_output_readable():
    raw = "Cloning into 'repo'...\rReceiving objects:  50%\rReceiving objects: 100%\r\n\x1b[31mERROR: boom\x1b[0m\n"
    out = ops._diagnostic_tail(raw)
    assert "\r" not in out and "\x1b" not in out
    assert out.endswith("ERROR: boom")
    assert ops._redacted_diagnostic_text(out) == out          # no longer withheld wholesale


def test_diagnostic_tail_keeps_only_the_end():
    out = ops._diagnostic_tail("x" * 10_000 + "\nlast line", limit=100)
    assert len(out) == 100 and out.endswith("last line")


def test_secrets_still_redacted_after_normalizing(monkeypatch):
    monkeypatch.setattr(ops, "load_secrets", lambda: {"HF_TOKEN": "hf_supersecretvalue123", "VAST_API_KEY": "vk_abcdef0123456789"})
    raw = "progress\rauth hf_supersecretvalue123 failed\r\nkey vk_abcdef0123456789\n"
    out = ops._redacted_diagnostic_text(ops._diagnostic_tail(raw))
    assert "hf_supersecretvalue123" not in out and "vk_abcdef0123456789" not in out
