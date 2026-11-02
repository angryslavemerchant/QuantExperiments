#!/usr/bin/env python3
"""mayfly - a dumb, loud, disposable remote-log watcher.

The WATCHTOWER mechanism layer (see ../WATCHTOWER.md) as one portable
file with zero dependencies. It polls a log on a remote box over ssh at
a fixed interval, applies regex rules, and — this is the whole design —
**exits the moment anything decision-worthy happens**. It never judges,
never fixes, never retries beyond its patience. In an agent harness,
run it as a *background task from the main session*: the process
exiting is the one wake-up signal a harness never loses. The agent that
wakes then does the judging (or forwards to a Sonnet judge agent).

Exit codes ARE the protocol:
    0  DONE        done-re matched / terminal success event (the run finished)
    3  ERROR       error-re matched / failure event (context printed)
    4  STALL       no progress for --stall-min minutes
    5  UNREACHABLE ssh (or remote log read) failed --max-ssh-fails times
    6  DEADLINE    --deadline-min elapsed without DONE (absolute cap)
    7  DEADMAN     (deadman mode) watched heartbeat file went stale
    2  BADARGS     misconfiguration (fail fast, fail loud)

Only log content that is NEW since mayfly started is scanned (so a
re-armed mayfly does not re-fire on an old Traceback); pass
--from-start to scan the existing content too. The remote file size is
tracked as a byte offset; a shrinking file (truncate/rotate) resets the
offset to 0 and rescans. By default "progress" means "the log grew";
--progress-re switches to "the last regex match in the tail changed".

The last ~4 KB of log context and the event line are printed to stdout
on exit, so the woken agent has context without another round-trip.

Typical launch (agent, background task; reuse the ssh options ops.py
uses: key, BatchMode, accept-new + the repo known-hosts file):
    python vast/mayfly/mayfly.py \
        --ssh "ssh -p 54019 -o BatchMode=yes root@1.2.3.4" \
        --known-hosts "C:/path/with spaces/.vast/known_hosts" \
        --events \
        --interval 60 --stall-min 20 --deadline-min 480 \
        --heartbeat mayfly.hb

Shadow deadman (second background task; watches the FILE, not the agent):
    python vast/mayfly/mayfly.py --deadman mayfly.hb \
        --interval 90 --stale-min 6 --deadline-min 500

Every poll appends one line to --heartbeat (if given): epoch, status,
last progress token. The deadman only ever reads that file — no ssh, no
process inspection, no agent-watching-agent.
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time

DEFAULT_ERROR_RE = (r"Traceback \(most recent call last\)"
                    r"|CUDA out of memory|CUDA error"
                    r"|loss(?:=|: ?)nan|\bNaN\b detected"
                    r"|torch\.OutOfMemoryError|RuntimeError")
DEFAULT_OPS_DIR = "/workspace/quant_experiments_ops"
DEFAULT_EVENTS_LOG = DEFAULT_OPS_DIR + "/events.jsonl"
# ops.py / remote_runner.py events.jsonl vocabulary
EVENTS_DONE = {"run_succeeded"}
EVENTS_FAIL = {"run_failed", "runner_exception", "bootstrap_failed",
               "artifact_packaging_failed"}
DEFAULT_DEADLINE = {"watch": 720.0, "deadman": 750.0, "wait-up": 15.0}
CTX_BYTES = 4096            # bounded context kept for the exit printout
MAX_FETCH = 2_000_000       # per-poll cap; the rest is fetched next poll
MARK = "@@MAYFLY_"


def say(msg):
    print(f"[mayfly {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bail(code, event, tail=""):
    if tail:
        print("---- log tail ----", flush=True)
        print(tail[-CTX_BYTES:], flush=True)
        print("------------------", flush=True)
    say(f"EVENT:{event}")
    sys.exit(code)


def ssh_cmd(a, remote):
    return list(a.ssh_argv) + [remote]


def run_wait_up(a):
    """WAIT-UP MODE: the inverse watch — ssh failure is EXPECTED, first
    success is the event. For box provisioning, reboots, key
    propagation. Exits 0 (DONE) when --ready-cmd first succeeds over
    ssh, 6 (DEADLINE) at the bound. A bare `until ssh; do sleep; done`
    loop is this mode done wrong: no deadline, no heartbeat, no loud
    exit — an agent behind one is indistinguishable from a dead one
    (observed live 2026-08-14; a human had to poke)."""
    t0 = time.time()
    dl = a.deadline_min
    tries = 0
    rotate_heartbeat(a)
    while True:
        tries += 1
        try:
            r = subprocess.run(ssh_cmd(a, a.ready_cmd),
                               capture_output=True, text=True,
                               timeout=a.ssh_timeout,
                               encoding="utf-8", errors="replace")
            if r.returncode == 0:
                heartbeat(a, "TERMINAL DONE", f"up after {tries} tries")
                bail(0, f"DONE box is up ({tries} tries, "
                        f"{(time.time() - t0) / 60:.1f} min)")
        except (subprocess.TimeoutExpired, OSError):
            pass
        heartbeat(a, "WAITING", f"try {tries}")
        if (time.time() - t0) / 60 > dl:
            bail(6, f"DEADLINE box not up after {dl} min "
                    f"({tries} tries) - check provider status / "
                    "destroy and re-rent")
        time.sleep(a.interval)


def read_last_heartbeat(path):
    """Return (epoch or None, last non-empty line) of a heartbeat file."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        return None, "(heartbeat file empty)"
    last = lines[-1]
    try:
        return int(last.split()[0]), last
    except (ValueError, IndexError):
        return None, last


def run_deadman(a):
    """Watch a heartbeat FILE. Exits 7 when it goes stale, 0 if the
    watched mayfly wrote a terminal status, 6 at deadline.

    Heartbeat lines older than the deadman's own start are ignored: a
    leftover file from a previous run (possibly ending in TERMINAL)
    must neither end the deadman instantly nor trip it instantly."""
    t0 = time.time()
    t0_epoch = int(t0)
    while True:
        if a.deadline_min and (time.time() - t0) / 60 > a.deadline_min:
            bail(6, "DEADLINE deadman outlived its watch window")
        try:
            mtime = os.path.getmtime(a.deadman)
            epoch, last = read_last_heartbeat(a.deadman)
        except OSError:
            # file not written yet: age from OUR start, not infinity —
            # the watched mayfly only writes after its first poll
            # (instant false DEADMAN observed live 2026-08-14)
            mtime, epoch, last = t0, None, "(heartbeat file not created yet)"
        fresh = epoch is not None and epoch >= t0_epoch
        if fresh and "TERMINAL" in last:
            bail(0, f"DONE watched mayfly ended cleanly: {last}")
        if epoch is not None and not fresh:
            last += " (predates deadman start; ignored)"
        # a leftover file's old mtime must not trip us instantly either
        age_min = (time.time() - max(mtime, t0)) / 60
        if age_min > a.stale_min:
            bail(7, f"DEADMAN heartbeat stale {age_min:.1f} min "
                    f"(> {a.stale_min}); last: {last}")
        time.sleep(a.interval)


def remote_script(a, offset):
    """Shell one-liner: print size/base headers, then bytes from base.

    offset None = baseline: fetch only the last CTX_BYTES as context.
    If the file shrank below offset (truncate/rotate), base resets to 0.
    """
    f = shlex.quote(a.log)
    if offset is None:
        o = f"$(( s > {CTX_BYTES} ? s - {CTX_BYTES} : 0 ))"
    else:
        o = str(int(offset))
    parts = [
        f"f={f}",
        f'[ -r "$f" ] || {{ echo "mayfly: cannot read remote log $f" >&2;'
        f" exit 44; }}",
        "s=$(wc -c < \"$f\" | tr -d ' ')",
        f"o={o}",
        'if [ "$s" -lt "$o" ]; then o=0; fi',
        f'echo "{MARK}SIZE $s"',
        f'echo "{MARK}BASE $o"',
    ]
    if a.gpu:
        parts.append(f'echo "{MARK}GPU $(nvidia-smi --query-gpu='
                     'utilization.gpu,memory.used --format=csv,noheader '
                     '2>/dev/null | head -n 1)"')
    parts.append(f'echo "{MARK}DATA"')
    parts.append(f'tail -c +$((o + 1)) "$f" | head -c {MAX_FETCH}')
    return "; ".join(parts)


def parse_poll(raw):
    """Parse remote_script stdout bytes -> dict or None."""
    sep = (MARK + "DATA\n").encode()
    head, found, data = raw.partition(sep)
    if not found:
        return None
    info = {"data": data, "gpu": None}
    for line in head.decode("utf-8", "replace").splitlines():
        if line.startswith(MARK + "SIZE "):
            info["size"] = int(line.split()[1])
        elif line.startswith(MARK + "BASE "):
            info["base"] = int(line.split()[1])
        elif line.startswith(MARK + "GPU"):
            info["gpu"] = line[len(MARK) + 3:].strip() or None
    if "size" not in info or "base" not in info:
        return None
    return info


def poll_once(a, offset):
    """One ssh round-trip. Returns (ok, info_dict | error_text)."""
    try:
        r = subprocess.run(ssh_cmd(a, remote_script(a, offset)),
                           capture_output=True, timeout=a.ssh_timeout)
    except (subprocess.TimeoutExpired, OSError) as e:
        return False, str(e)
    out = r.stdout or b""
    err = (r.stderr or b"").decode("utf-8", "replace")
    if r.returncode != 0:
        return False, (err or out.decode("utf-8", "replace")
                       or f"exit {r.returncode}")
    try:
        info = parse_poll(out)
    except (ValueError, IndexError):
        info = None
    if info is None:
        return False, "unparseable remote reply: " + (
            out.decode("utf-8", "replace") + err)[-300:]
    return True, info


def rotate_heartbeat(a):
    """Move a leftover heartbeat file aside so a deadman never reads a
    previous run's TERMINAL line as ours."""
    if a.heartbeat and os.path.exists(a.heartbeat):
        try:
            os.replace(a.heartbeat, a.heartbeat + ".prev")
        except OSError as e:
            say(f"could not rotate old heartbeat file: {e}")


def heartbeat(a, status, progress):
    if not a.heartbeat:
        return
    with open(a.heartbeat, "a", encoding="utf-8") as f:
        f.write(f"{int(time.time())} {status} {progress}\n")


def scan_events(lines):
    """events.jsonl lines -> (code, event) for the first terminal event."""
    for line in lines:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if not isinstance(rec, dict):
            continue
        ev = rec.get("event")
        if ev in EVENTS_DONE:
            return 0, f"DONE event {ev}"
        if ev == "artifacts_ready":
            if rec.get("exit_code", 0) == 0:
                return 0, "DONE event artifacts_ready (exit_code 0)"
            return 3, (f"ERROR event artifacts_ready exit_code="
                       f"{rec.get('exit_code')}")
        if ev in EVENTS_FAIL:
            return 3, f"ERROR event {ev}: {line.strip()[:300]}"
    return None


def run_watch(a):
    done_re = re.compile(a.done_re)
    err_re = re.compile(a.error_re)
    prog_re = re.compile(a.progress_re) if a.progress_re else None
    t0 = time.time()
    last_prog, last_prog_t = None, time.time()
    ssh_fails = 0
    offset = 0 if a.from_start else None
    carry, ctx = "", ""
    rotate_heartbeat(a)
    say(f"watching {a.log} every {a.interval}s "
        f"({'events' if a.events else 'regex'} mode, "
        f"{'from start' if a.from_start else 'new content only'}, "
        f"stall {a.stall_min}m, deadline "
        f"{a.deadline_min or 'NONE'}m)")
    while True:
        ok, res = poll_once(a, offset)
        if not ok:
            ssh_fails += 1
            heartbeat(a, f"SSH_FAIL_{ssh_fails}", last_prog or "-")
            say(f"ssh fail {ssh_fails}/{a.max_ssh_fails}: "
                f"{res.strip()[:200]}")
            if ssh_fails >= a.max_ssh_fails:
                heartbeat(a, "TERMINAL UNREACHABLE", last_prog or "-")
                bail(5, f"UNREACHABLE {ssh_fails} consecutive ssh "
                        f"failures (last: {res.strip()[:200]})", ctx)
            time.sleep(a.interval)
            continue
        ssh_fails = 0
        baseline = offset is None
        data = res["data"]
        text = data.decode("utf-8", "replace")
        if not baseline and res["base"] < offset:
            say(f"log shrank ({offset} -> {res['size']} bytes): "
                "truncated/rotated, rescanning from start")
            carry, ctx = "", ""
        offset = res["base"] + len(data)
        grew = bool(data) and not baseline
        ctx = (ctx + text)[-CTX_BYTES:]
        if not baseline:
            scan = carry + text
            nl = scan.rfind("\n")
            complete, carry = scan[:nl + 1], scan[nl + 1:][-CTX_BYTES:]
            if a.events:
                hit = scan_events(complete.splitlines())
                if hit:
                    code, event = hit
                    status = "DONE" if code == 0 else "ERROR"
                    heartbeat(a, f"TERMINAL {status}", last_prog or "-")
                    bail(code, event, ctx)
            else:
                m = err_re.search(scan)
                if m:
                    heartbeat(a, "TERMINAL ERROR", last_prog or "-")
                    excerpt = scan[max(0, m.start() - 1500):
                                   m.start() + 2500]
                    bail(3, f"ERROR pattern {err_re.pattern!r} matched",
                         excerpt)
                if done_re.search(scan):
                    heartbeat(a, "TERMINAL DONE", last_prog or "-")
                    bail(0, "DONE", ctx)
        if prog_re is not None:
            ms = list(prog_re.finditer(ctx))
            prog = ms[-1].group(0) if ms else None
            if prog != last_prog:
                last_prog, last_prog_t = prog, time.time()
        else:
            if grew:
                last_prog_t = time.time()
            last_prog = f"bytes={offset}"
        stalled_min = (time.time() - last_prog_t) / 60
        if stalled_min > a.stall_min:
            heartbeat(a, "TERMINAL STALL", last_prog or "-")
            gpu = f"; gpu: {res['gpu']}" if res.get("gpu") else ""
            what = ("progress-re match unchanged" if prog_re is not None
                    else "log did not grow")
            bail(4, f"STALL {what} for {stalled_min:.1f} min "
                    f"(last: {last_prog!r}){gpu}", ctx)
        if a.deadline_min and (time.time() - t0) / 60 > a.deadline_min:
            heartbeat(a, "TERMINAL DEADLINE", last_prog or "-")
            bail(6, f"DEADLINE {a.deadline_min} min elapsed", ctx)
        heartbeat(a, "OK", last_prog or "-")
        time.sleep(a.interval)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ssh", help='full ssh prefix, shell-quoted (parsed '
                   'with shlex, POSIX rules; write Windows paths with '
                   'forward slashes), e.g. '
                   '"ssh -p 54019 -o BatchMode=yes root@1.2.3.4"')
    p.add_argument("--known-hosts", metavar="PATH",
                   help="adds -o StrictHostKeyChecking=accept-new and "
                        "-o UserKnownHostsFile=\"PATH\" (quoted for "
                        "OpenSSH, so paths with spaces work) - use the "
                        "repo's .vast/known_hosts like ops.py does")
    p.add_argument("--log", help="remote log path to watch (default with "
                   f"--events: {DEFAULT_EVENTS_LOG})")
    p.add_argument("--events", action="store_true",
                   help="watch ops.py's events.jsonl: DONE on "
                        "run_succeeded / artifacts_ready(exit_code 0); "
                        "ERROR on run_failed / runner_exception / "
                        "bootstrap_failed / artifact_packaging_failed / "
                        "artifacts_ready(exit_code!=0). Regexes unused.")
    p.add_argument("--from-start", action="store_true",
                   help="also scan content that existed before mayfly "
                        "started (default: only new bytes)")
    p.add_argument("--done-re", default=r"\bDONE\b")
    p.add_argument("--error-re", default=DEFAULT_ERROR_RE)
    p.add_argument("--progress-re", default=None,
                   help="stall detection by regex: the last match in the "
                        "~4 KB tail must CHANGE (default: log byte size "
                        "must grow)")
    p.add_argument("--interval", type=float, default=60)
    p.add_argument("--stall-min", type=float, default=20)
    p.add_argument("--deadline-min", type=float, default=None,
                   help="absolute cap in minutes; must be > 0 (default: "
                        "720 watch, 750 deadman, 15 wait-up)")
    p.add_argument("--no-deadline", action="store_true",
                   help="explicitly allow an unbounded watch/deadman "
                        "(discouraged; not allowed with --wait-up)")
    p.add_argument("--max-ssh-fails", type=int, default=5)
    p.add_argument("--ssh-timeout", type=float, default=30)
    p.add_argument("--gpu", action="store_true",
                   help="also fetch a one-line nvidia-smi each poll")
    p.add_argument("--heartbeat", help="local file to append liveness to "
                   "(an existing file is moved to <file>.prev at start)")
    p.add_argument("--deadman", metavar="HB_FILE",
                   help="DEADMAN MODE: watch this heartbeat file "
                        "instead of a remote log")
    p.add_argument("--stale-min", type=float, default=6,
                   help="deadman: minutes without heartbeat = dead")
    p.add_argument("--wait-up", action="store_true",
                   help="WAIT-UP MODE: exit 0 when ssh first succeeds "
                        "(box provisioning); deadline-bounded (15 min "
                        "default)")
    p.add_argument("--ready-cmd", default="true",
                   help="wait-up: remote command that must succeed to "
                        "count as up")
    return p


def badargs(msg):
    say(f"BADARGS: {msg}")
    sys.exit(2)


def resolve_deadline(a, mode):
    if a.deadline_min is None:
        if a.no_deadline and mode != "wait-up":
            return 0
        return DEFAULT_DEADLINE[mode]
    if a.deadline_min < 0:
        badargs("--deadline-min must be positive")
    if a.deadline_min == 0:
        if mode == "wait-up":
            badargs("--wait-up must be bounded; --deadline-min 0 refused")
        if not a.no_deadline:
            badargs("--deadline-min 0 means unbounded; there is no "
                    "legitimate unbounded wait. Give a positive cap or "
                    "pass --no-deadline explicitly")
    return a.deadline_min


def parse_ssh(a):
    if not a.ssh:
        return None
    try:
        argv = shlex.split(a.ssh, posix=True)
    except ValueError as e:
        badargs(f"cannot parse --ssh ({e})")
    if not argv:
        badargs("--ssh is empty")
    if a.known_hosts:
        kh = a.known_hosts.replace("\\", "/")
        argv[1:1] = ["-o", "StrictHostKeyChecking=accept-new",
                     "-o", f'UserKnownHostsFile="{kh}"']
    return argv


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.interval <= 0 or a.stall_min <= 0 or a.max_ssh_fails < 1:
        badargs("--interval/--stall-min must be > 0, --max-ssh-fails >= 1")
    if a.deadman:
        a.deadline_min = resolve_deadline(a, "deadman")
        run_deadman(a)
    a.ssh_argv = parse_ssh(a)
    if a.wait_up:
        if not a.ssh_argv:
            badargs("wait-up needs --ssh")
        a.deadline_min = resolve_deadline(a, "wait-up")
        run_wait_up(a)
    if a.events and not a.log:
        a.log = DEFAULT_EVENTS_LOG
    if not a.ssh_argv or not a.log:
        badargs("need --ssh and --log (or --events / --deadman / --wait-up)")
    if re.match(r"^[A-Za-z]:[\\/]", a.log) or "Program Files/Git" in a.log:
        badargs(f"remote log path {a.log!r} looks like a Windows path - "
                "Git-Bash MSYS path mangling? Prefix the launch with "
                "MSYS_NO_PATHCONV=1 or run from PowerShell/cmd")
    a.deadline_min = resolve_deadline(a, "watch")
    run_watch(a)


if __name__ == "__main__":
    main()
