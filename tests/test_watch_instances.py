"""Tests for vast/watch_instances.py — subprocess is always stubbed."""
import json
import subprocess
from types import SimpleNamespace

import pytest

import vast.watch_instances as wi

FAKE_KEY = "test-key-not-real-123"


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    monkeypatch.setattr(wi, "cli", lambda: ["vastai"])
    clock = SimpleNamespace(t=5_000.0)
    monkeypatch.setattr(wi.time, "time", lambda: clock.t)
    monkeypatch.setattr(wi.time, "sleep", lambda s: setattr(clock, "t", clock.t + s))
    secrets = tmp_path / "secrets.env"   # absent unless a test writes it
    return SimpleNamespace(clock=clock, secrets=secrets, monkeypatch=monkeypatch)


def listing(**instances):
    return json.dumps([{"id": int(k[1:]), "actual_status": v} for k, v in instances.items()])


def stub(env, replies):
    calls = []

    def run(argv, **kw):
        calls.append((argv, kw))
        r = replies[min(len(calls) - 1, len(replies) - 1)]
        if isinstance(r, Exception):
            raise r
        if r is None:
            return subprocess.CompletedProcess(argv, 1, "", "HTTP 500 boom")
        return subprocess.CompletedProcess(argv, 0, r, "")

    env.monkeypatch.setattr(wi.subprocess, "run", run)
    return calls


def test_no_key_anywhere_exits_2_without_assert(env, capsys):
    stub(env, [listing()])
    assert wi.main(["--once"], secrets_file=env.secrets) == 2
    assert "NO_API_KEY" in capsys.readouterr().err


def test_key_loaded_from_secrets_file_bom_quotes(env, capsys):
    env.secrets.write_bytes(
        b"\xef\xbb\xbf# comment\nOTHER=1\nVAST_API_KEY=\"" + FAKE_KEY.encode() + b"\"\n")
    calls = stub(env, [listing(i1="running")])
    assert wi.main(["--once"], secrets_file=env.secrets) == 0
    argv, kw = calls[0]
    assert kw["env"]["VAST_API_KEY"] == FAKE_KEY
    assert FAKE_KEY not in " ".join(argv)
    out = capsys.readouterr()
    assert FAKE_KEY not in out.out + out.err


def test_read_secret_file_variants(tmp_path):
    p = tmp_path / "s.env"
    p.write_text("export VAST_API_KEY = 'abc'\n", encoding="utf-8")
    assert wi.read_secret_file(p) == "abc"
    p.write_text("VAST_API_KEY=\n", encoding="utf-8")
    assert wi.read_secret_file(p) is None
    assert wi.read_secret_file(tmp_path / "missing.env") is None


def test_env_key_takes_precedence(env):
    env.monkeypatch.setenv("VAST_API_KEY", "from-env")
    env.secrets.write_text("VAST_API_KEY=from-file\n", encoding="utf-8")
    calls = stub(env, [listing()])
    assert wi.main(["--once"], secrets_file=env.secrets) == 0
    assert calls[0][1]["env"]["VAST_API_KEY"] == "from-env"


def test_consecutive_api_failures_exit_5(env, capsys):
    env.monkeypatch.setenv("VAST_API_KEY", FAKE_KEY)
    calls = stub(env, [None])
    assert wi.main(["7"], secrets_file=env.secrets) == 5
    assert len(calls) == 5
    assert "VAST_API_TRIPWIRE" in capsys.readouterr().out


def test_failures_reset_on_success(env):
    env.monkeypatch.setenv("VAST_API_KEY", FAKE_KEY)
    replies = [None] * 4 + [listing(i7="running")] + [None] * 4 + [listing()]
    calls = stub(env, replies)
    assert wi.main(["7"], secrets_file=env.secrets) == 0   # gone -> done
    assert len(calls) == 10


def test_timeout_counts_as_failure(env):
    env.monkeypatch.setenv("VAST_API_KEY", FAKE_KEY)
    stub(env, [subprocess.TimeoutExpired("vastai", 120)])
    assert wi.main(["7", "--max-fails", "2"], secrets_file=env.secrets) == 5


def test_deadline_exit_6(env, capsys):
    env.monkeypatch.setenv("VAST_API_KEY", FAKE_KEY)
    calls = stub(env, [listing(i7="running")])
    assert wi.main(["7", "--interval", "60", "--deadline-min", "10"],
                   secrets_file=env.secrets) == 6
    assert 10 <= len(calls) <= 12
    assert "WATCH_DEADLINE" in capsys.readouterr().out


def test_deadline_must_be_positive(env):
    env.monkeypatch.setenv("VAST_API_KEY", FAKE_KEY)
    stub(env, [listing()])
    assert wi.main(["--deadline-min", "0"], secrets_file=env.secrets) == 2


def test_not_running_reported_once_per_status(env, capsys):
    env.monkeypatch.setenv("VAST_API_KEY", FAKE_KEY)
    stub(env, [listing(i7="running"), listing(i7="exited"), listing(i7="exited"),
               listing(i7="stopped"), listing()])
    assert wi.main(["7"], secrets_file=env.secrets) == 0
    out = capsys.readouterr().out
    assert out.count("INSTANCE_NOT_RUNNING 7 exited (still billing storage)") == 1
    assert "INSTANCE_NOT_RUNNING 7 stopped (still billing storage)" in out
    assert "INSTANCE_STATUS 7 running -> exited" in out
    assert "INSTANCE_DOWN 7 (was: stopped)" in out
    assert "ALL_WATCHED_INSTANCES_GONE" in out


def test_not_running_reported_in_all_mode_once(env, capsys):
    env.monkeypatch.setenv("VAST_API_KEY", FAKE_KEY)
    stub(env, [listing(i1="running", i2="exited")])
    assert wi.main(["--once"], secrets_file=env.secrets) == 0
    out = capsys.readouterr().out
    assert "INSTANCE_NOT_RUNNING 2 exited (still billing storage)" in out
    assert "INSTANCE_NOT_RUNNING 1" not in out


def test_api_error_redacts_key(env, capsys):
    env.monkeypatch.setenv("VAST_API_KEY", FAKE_KEY)

    def run(argv, **kw):
        return subprocess.CompletedProcess(argv, 1, "", f"bad key {FAKE_KEY}")

    env.monkeypatch.setattr(wi.subprocess, "run", run)
    assert wi.main(["7", "--max-fails", "1"], secrets_file=env.secrets) == 5
    assert FAKE_KEY not in capsys.readouterr().out
