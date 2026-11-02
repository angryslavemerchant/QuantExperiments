"""Tests for vast/mayfly/mayfly.py — subprocess is always stubbed."""
import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

MAYFLY_PATH = Path(__file__).resolve().parent.parent / "vast" / "mayfly" / "mayfly.py"


@pytest.fixture
def mayfly(monkeypatch):
    spec = importlib.util.spec_from_file_location("mayfly_under_test", MAYFLY_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    clock = SimpleNamespace(t=1_000_000.0)
    monkeypatch.setattr(mod.time, "time", lambda: clock.t)

    def fake_sleep(sec):
        clock.t += sec

    monkeypatch.setattr(mod.time, "sleep", fake_sleep)
    mod.clock = clock
    return mod


class FakeRemote:
    """Emulates the remote shell one-liner against an in-memory file."""

    def __init__(self, content=b"", script=None):
        self.content = content
        self.script = script or []   # per-poll callables mutating self
        self.calls = []
        self.polls = 0

    def __call__(self, argv, **kw):
        self.calls.append((argv, kw))
        if self.polls < len(self.script) and self.script[self.polls]:
            self.script[self.polls](self)
        self.polls += 1
        remote = argv[-1]
        if self.content is None:
            return subprocess.CompletedProcess(argv, 44, b"", b"mayfly: cannot read remote log\n")
        size = len(self.content)
        m = re.search(r"; o=(\d+);", remote)
        if m:
            o = int(m.group(1))
        else:  # baseline: last CTX bytes
            o = max(0, size - 4096)
        if size < o:
            o = 0
        out = f"@@MAYFLY_SIZE {size}\n@@MAYFLY_BASE {o}\n@@MAYFLY_DATA\n".encode()
        return subprocess.CompletedProcess(argv, 0, out + self.content[o:], b"")


def run_main(mayfly, monkeypatch, argv, fake):
    monkeypatch.setattr(mayfly.subprocess, "run", fake)
    with pytest.raises(SystemExit) as exc:
        mayfly.main(argv)
    return exc.value.code


BASE = ["--ssh", "ssh -p 2222 root@1.2.3.4", "--log", "/workspace/quant_experiments_ops/train.log",
        "--interval", "60", "--stall-min", "5", "--deadline-min", "60"]


def append(data):
    def f(remote):
        remote.content += data
    return f


# 1. re-arm self-trigger -------------------------------------------------------

def test_old_traceback_does_not_refire(mayfly, monkeypatch, capsys):
    old = b"step 1\nTraceback (most recent call last)\nRuntimeError: boom\n"
    fake = FakeRemote(old, script=[None, append(b"step 2\n"), append(b"all DONE\n")])
    code = run_main(mayfly, monkeypatch, BASE, fake)
    assert code == 0
    assert "EVENT:DONE" in capsys.readouterr().out


def test_new_error_fires(mayfly, monkeypatch, capsys):
    fake = FakeRemote(b"old ok\n", script=[None, append(b"CUDA out of memory\n")])
    assert run_main(mayfly, monkeypatch, BASE, fake) == 3
    assert "CUDA out of memory" in capsys.readouterr().out


def test_from_start_scans_existing(mayfly, monkeypatch):
    fake = FakeRemote(b"Traceback (most recent call last)\n")
    assert run_main(mayfly, monkeypatch, BASE + ["--from-start"], fake) == 3
    assert "; o=0;" in fake.calls[0][0][-1]


def test_offsets_advance_and_only_new_bytes_requested(mayfly, monkeypatch):
    fake = FakeRemote(b"a" * 10 + b"\n", script=[None, append(b"xyz\n"), append(b"DONE\n")])
    assert run_main(mayfly, monkeypatch, BASE, fake) == 0
    offs = [re.search(r"; o=([^;]+);", c[0][-1]).group(1) for c in fake.calls]
    assert offs[0].startswith("$((")      # baseline
    assert offs[1:] == ["11", "15"]


def test_truncation_resets_and_rescans(mayfly, monkeypatch, capsys):
    def truncate(remote):
        remote.content = b"DONE\n"   # shorter than old offset
    fake = FakeRemote(b"x" * 200 + b"\n", script=[None, truncate])
    assert run_main(mayfly, monkeypatch, BASE, fake) == 0
    assert "truncated/rotated" in capsys.readouterr().out


def test_match_split_across_polls(mayfly, monkeypatch):
    fake = FakeRemote(b"", script=[None, append(b"Traceback (most rec"),
                                   append(b"ent call last)\n")])
    assert run_main(mayfly, monkeypatch, BASE, fake) == 3


def test_context_tail_bounded(mayfly, monkeypatch, capsys):
    fake = FakeRemote(b"", script=[None, append(b"z" * 50_000 + b"\nDONE\n")])
    assert run_main(mayfly, monkeypatch, BASE, fake) == 0
    out = capsys.readouterr().out
    assert out.count("z") <= 4096


# 2. shlex ssh / quoted log ----------------------------------------------------

def test_ssh_shlex_and_known_hosts(mayfly, monkeypatch):
    kh = "C:/Users/me/documents/Python Projects/Q/.vast/known_hosts"
    ssh = ("ssh -i C:/Users/me/.ssh/id_ed25519 -p 2222 "
           f"-o 'UserKnownHostsFile=\"{kh}\"' root@1.2.3.4")
    fake = FakeRemote(b"", script=[None, append(b"DONE\n")])
    argv = ["--ssh", ssh, "--log", "/workspace/my dir/train.log", "--deadline-min", "60"]
    assert run_main(mayfly, monkeypatch, argv, fake) == 0
    sent = fake.calls[0][0]
    assert sent[:-1] == ["ssh", "-i", "C:/Users/me/.ssh/id_ed25519", "-p", "2222",
                         "-o", f'UserKnownHostsFile="{kh}"', "root@1.2.3.4"]
    assert "f='/workspace/my dir/train.log'" in sent[-1]


def test_known_hosts_flag_inserts_quoted_option(mayfly, monkeypatch):
    fake = FakeRemote(b"", script=[None, append(b"DONE\n")])
    argv = BASE + ["--known-hosts", r"C:\Users\me\Python Projects\.vast\known_hosts"]
    assert run_main(mayfly, monkeypatch, argv, fake) == 0
    sent = fake.calls[0][0]
    assert sent[:5] == ["ssh", "-o", "StrictHostKeyChecking=accept-new", "-o",
                        'UserKnownHostsFile="C:/Users/me/Python Projects/.vast/known_hosts"']
    assert sent[-2] == "root@1.2.3.4"


def test_bad_ssh_quoting_is_badargs(mayfly, monkeypatch):
    argv = ["--ssh", "ssh 'unterminated", "--log", "/x.log", "--deadline-min", "5"]
    assert run_main(mayfly, monkeypatch, argv, FakeRemote()) == 2


def test_msys_mangled_log_is_badargs(mayfly, monkeypatch, capsys):
    argv = ["--ssh", "ssh h", "--log", "C:/Program Files/Git/workspace/x.log"]
    assert run_main(mayfly, monkeypatch, argv, FakeRemote()) == 2
    assert "MSYS" in capsys.readouterr().out


# 3. progress = byte growth by default ----------------------------------------

def test_stall_when_log_stops_growing_despite_dates(mayfly, monkeypatch, capsys):
    # an old-style regex would see "2026/11" and "it/s" forever; size mode
    # must stall because no bytes are added.
    fake = FakeRemote(b"2026/11/01 12.3 it/s\n")
    assert run_main(mayfly, monkeypatch, BASE, fake) == 4
    assert "log did not grow" in capsys.readouterr().out


def test_growing_log_never_stalls(mayfly, monkeypatch):
    script = [None] + [append(b"tick\n")] * 20 + [append(b"DONE\n")]
    fake = FakeRemote(b"", script=script)
    assert run_main(mayfly, monkeypatch, BASE, fake) == 0   # 21 min > stall 5


def test_progress_re_mode_kept(mayfly, monkeypatch, capsys):
    # log grows but the step token does not change -> STALL in regex mode
    script = [None, append(b"step 5\n")] + [append(b"noise\n")] * 10
    fake = FakeRemote(b"", script=script)
    code = run_main(mayfly, monkeypatch, BASE + ["--progress-re", r"step \d+"], fake)
    assert code == 4
    assert "progress-re match unchanged" in capsys.readouterr().out


# 4. deadline must be positive ---------------------------------------------------

def test_deadline_zero_rejected(mayfly, monkeypatch):
    argv = ["--ssh", "ssh h", "--log", "/x.log", "--deadline-min", "0"]
    assert run_main(mayfly, monkeypatch, argv, FakeRemote()) == 2


def test_deadline_zero_with_no_deadline_allowed(mayfly, monkeypatch):
    fake = FakeRemote(b"", script=[None, append(b"DONE\n")])
    argv = ["--ssh", "ssh h", "--log", "/x.log", "--deadline-min", "0", "--no-deadline"]
    assert run_main(mayfly, monkeypatch, argv, fake) == 0


def test_default_deadline_is_bounded(mayfly, monkeypatch):
    script = [None] + [append(b"t\n")] * 2000
    fake = FakeRemote(b"", script=script)
    argv = ["--ssh", "ssh h", "--log", "/x.log", "--interval", "60"]
    assert run_main(mayfly, monkeypatch, argv, fake) == 6
    assert 720 <= fake.polls <= 725


def test_wait_up_refuses_unbounded(mayfly, monkeypatch):
    argv = ["--ssh", "ssh h", "--wait-up", "--deadline-min", "0", "--no-deadline"]
    assert run_main(mayfly, monkeypatch, argv, FakeRemote()) == 2


def test_wait_up_default_15(mayfly, monkeypatch):
    def fail(argv, **kw):
        return subprocess.CompletedProcess(argv, 255, "", "refused")
    argv = ["--ssh", "ssh h", "--wait-up", "--interval", "60"]
    assert run_main(mayfly, monkeypatch, argv, fail) == 6


# 5. deadman ignores stale heartbeat --------------------------------------------

def test_deadman_ignores_old_terminal_line(mayfly, monkeypatch, tmp_path):
    hb = tmp_path / "mayfly.hb"
    old = int(mayfly.clock.t) - 3600
    hb.write_text(f"{old} TERMINAL DONE -\n", encoding="utf-8")
    os.utime(hb, (old, old))
    code = run_main(mayfly, monkeypatch,
                    ["--deadman", str(hb), "--interval", "60", "--stale-min", "3",
                     "--deadline-min", "60"], FakeRemote())
    assert code == 7      # stale after 3 min from OUR start, not 0, not instant


def test_deadman_old_file_not_instantly_stale(mayfly, monkeypatch, tmp_path):
    hb = tmp_path / "mayfly.hb"
    old = int(mayfly.clock.t) - 3600
    hb.write_text(f"{old} OK -\n", encoding="utf-8")
    os.utime(hb, (old, old))
    sleeps = []
    monkeypatch.setattr(mayfly.time, "sleep",
                        lambda s: (sleeps.append(s), setattr(mayfly.clock, "t", mayfly.clock.t + s)))
    code = run_main(mayfly, monkeypatch,
                    ["--deadman", str(hb), "--interval", "60", "--stale-min", "3",
                     "--deadline-min", "60"], FakeRemote())
    assert code == 7 and len(sleeps) >= 3


def test_deadman_fresh_terminal_exits_0(mayfly, monkeypatch, tmp_path):
    hb = tmp_path / "mayfly.hb"

    def sleep(s):
        mayfly.clock.t += s
        hb.write_text(f"{int(mayfly.clock.t)} TERMINAL DONE -\n", encoding="utf-8")
        os.utime(hb, (mayfly.clock.t, mayfly.clock.t))

    monkeypatch.setattr(mayfly.time, "sleep", sleep)
    code = run_main(mayfly, monkeypatch,
                    ["--deadman", str(hb), "--interval", "60", "--stale-min", "3",
                     "--deadline-min", "60"], FakeRemote())
    assert code == 0


def test_watch_rotates_old_heartbeat(mayfly, monkeypatch, tmp_path):
    hb = tmp_path / "mayfly.hb"
    hb.write_text("1 TERMINAL DONE -\n", encoding="utf-8")
    fake = FakeRemote(b"", script=[None, append(b"DONE\n")])
    assert run_main(mayfly, monkeypatch, BASE + ["--heartbeat", str(hb)], fake) == 0
    assert (tmp_path / "mayfly.hb.prev").read_text(encoding="utf-8") == "1 TERMINAL DONE -\n"
    assert "1 TERMINAL DONE" not in hb.read_text(encoding="utf-8")


def test_deadman_deadline_default_bounded(mayfly, monkeypatch, tmp_path):
    hb = tmp_path / "hb"

    def sleep(s):
        mayfly.clock.t += s
        hb.write_text(f"{int(mayfly.clock.t)} OK -\n", encoding="utf-8")
        os.utime(hb, (mayfly.clock.t, mayfly.clock.t))

    monkeypatch.setattr(mayfly.time, "sleep", sleep)
    assert run_main(mayfly, monkeypatch, ["--deadman", str(hb), "--interval", "600"],
                    FakeRemote()) == 6


# 6. events mode -----------------------------------------------------------------

def ev(name, **kw):
    return (json.dumps({"event": name, "run_id": "r", **kw}) + "\n").encode()


EV = ["--ssh", "ssh h", "--events", "--deadline-min", "60"]


def test_events_mode_default_path_and_done(mayfly, monkeypatch, capsys):
    old = ev("run_failed", exit_code=1)   # previous run: must be ignored
    fake = FakeRemote(old, script=[None, append(ev("run_started") + ev("heartbeat")),
                                   append(ev("command_finished", exit_code=0)
                                          + ev("artifacts_ready", exit_code=0))])
    assert run_main(mayfly, monkeypatch, EV, fake) == 0
    assert "f=/workspace/quant_experiments_ops/events.jsonl" in fake.calls[0][0][-1]
    assert "artifacts_ready" in capsys.readouterr().out


def test_events_mode_failure(mayfly, monkeypatch):
    fake = FakeRemote(b"", script=[None, append(ev("artifacts_ready", exit_code=2))])
    assert run_main(mayfly, monkeypatch, EV, fake) == 3
    fake = FakeRemote(b"", script=[None, append(ev("runner_exception", error="x"))])
    assert run_main(mayfly, monkeypatch, EV, fake) == 3


def test_events_mode_ignores_regex_words_and_partial_lines(mayfly, monkeypatch):
    line = ev("command_started", argv=["echo", "DONE", "RuntimeError"])
    done = ev("run_succeeded")
    fake = FakeRemote(b"", script=[None, append(line), append(done[:10]),
                                   append(done[10:])])
    assert run_main(mayfly, monkeypatch, EV, fake) == 0
    assert fake.polls == 4


# unreachable ----------------------------------------------------------------------

def test_missing_remote_log_counts_as_failure(mayfly, monkeypatch):
    fake = FakeRemote(None)
    assert run_main(mayfly, monkeypatch, BASE + ["--max-ssh-fails", "3"], fake) == 5
    assert fake.polls == 3


def test_ssh_timeout_counts_as_failure(mayfly, monkeypatch):
    def boom(argv, **kw):
        raise subprocess.TimeoutExpired(argv, 30)
    assert run_main(mayfly, monkeypatch, BASE + ["--max-ssh-fails", "2"], boom) == 5
