"""Emit an event line whenever a watched vast.ai instance goes down.

    python vast/watch_instances.py [ID ...] [--interval 120] [--once]
                                   [--deadline-min 720]

Polls `vastai show instances --raw` and prints one line per state
transition, so it slots straight into a Monitor/watcher:

    INSTANCE_DOWN 47225408 (was: running)        <- destroyed / reaped
    INSTANCE_STATUS 47205967 running -> exited   <- stopped, crashed host
    INSTANCE_NOT_RUNNING 47205967 exited (still billing storage)
    INSTANCE_UP 47230000 running                 <- new id appeared (no-arg mode)
    VAST_API_UNREACHABLE (xN)                    <- CLI errors

VAST_API_KEY comes from the environment, else from vast/secrets.env
(KEY=VALUE lines); it is handed to the vastai subprocess via its
environment only and never printed.

With explicit IDs, exits 0 once ALL watched ids are gone (natural end for
a single-notification watcher). With no IDs it watches everything on the
account until the deadline. API-unreachable never counts as down — a box
is only DOWN when a successful listing omits it (the 2026-08-09 bench box
self-destroyed unseen; SSH-silence is ambiguous, the account listing is
authoritative). An instance that is listed but not `running` (exited,
stopped, ...) is reported as INSTANCE_NOT_RUNNING: it still bills storage.

Exit codes: 0 done (all watched gone / --once), 2 bad args or no API key,
5 API unreachable for --max-fails consecutive polls, 6 deadline reached.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SECRETS_FILE = Path(__file__).resolve().parent / "secrets.env"
KEY_NAME = "VAST_API_KEY"


def cli():
    exe_dir = os.path.dirname(sys.executable)
    for d in (None, exe_dir, os.path.join(exe_dir, "Scripts"), "/venv/main/bin"):
        for cand in ("vastai", "vastai.exe"):
            p = shutil.which(cand, path=d) if d else shutil.which(cand)
            if p:
                return [p]
    raise RuntimeError("vastai CLI not found near " + sys.executable)


def read_secret_file(path, name=KEY_NAME):
    """Return the value of `name` from a KEY=VALUE file, or None.

    Tolerates a UTF-8 BOM, blank lines, `#` comments, `export ` prefixes
    and surrounding single/double quotes. Never logs the value."""
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return None
    for raw in text.splitlines():
        line = raw.strip().lstrip("﻿")
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        if key != name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1].strip()
        return value or None
    return None


def resolve_api_key(secrets_file=SECRETS_FILE):
    key = (os.environ.get(KEY_NAME) or "").strip()
    if key:
        return key
    return read_secret_file(secrets_file)


def poll(api_key):
    env = dict(os.environ)
    env[KEY_NAME] = api_key
    proc = subprocess.run(cli() + ["show", "instances", "--raw"],
                          capture_output=True, text=True, timeout=120,
                          env=env)
    if proc.returncode != 0:
        msg = (proc.stderr or "").strip()[-200:]
        raise RuntimeError(msg.replace(api_key, "<redacted>") if api_key
                           else msg)
    return {int(d["id"]): d.get("actual_status") or "unknown"
            for d in json.loads(proc.stdout)}


def main(argv=None, secrets_file=SECRETS_FILE):
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", type=int)
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--deadline-min", type=float, default=720,
                    help="bounded watch: exit 6 after this many minutes")
    ap.add_argument("--max-fails", type=int, default=5,
                    help="consecutive API failures before exit 5")
    args = ap.parse_args(argv)
    if args.deadline_min <= 0 or args.interval <= 0 or args.max_fails < 1:
        print("BADARGS: --deadline-min and --interval must be > 0, "
              "--max-fails >= 1", file=sys.stderr, flush=True)
        return 2
    api_key = resolve_api_key(secrets_file)
    if not api_key:
        print(f"NO_API_KEY: {KEY_NAME} not in env and not found in "
              f"{secrets_file}", file=sys.stderr, flush=True)
        return 2

    t0 = time.time()
    prev, fails = None, 0
    reported_not_running = {}
    while True:
        try:
            cur = poll(api_key)
            fails = 0
        except Exception as e:  # noqa: BLE001 - transient API errors
            fails += 1
            print(f"VAST_API_UNREACHABLE (x{fails}): {e}", flush=True)
            if fails >= args.max_fails:
                print(f"VAST_API_TRIPWIRE: {fails} consecutive failures "
                      "- giving up (instance state UNKNOWN, check the "
                      "Vast console)", flush=True)
                return 5
            cur = None
        if cur is not None:
            watched = args.ids or (sorted(prev) if prev is not None else [])
            if prev is not None:
                for i in watched:
                    if i in prev and i not in cur:
                        print(f"INSTANCE_DOWN {i} (was: {prev[i]})",
                              flush=True)
                    elif i in prev and i in cur and prev[i] != cur[i]:
                        print(f"INSTANCE_STATUS {i} {prev[i]} -> {cur[i]}",
                              flush=True)
                if not args.ids:
                    for i in sorted(set(cur) - set(prev)):
                        print(f"INSTANCE_UP {i} {cur[i]}", flush=True)
            else:
                for i in (args.ids or sorted(cur)):
                    print(f"WATCHING {i} {cur.get(i, 'ABSENT')}", flush=True)
                    if args.ids and i not in cur:
                        print(f"INSTANCE_DOWN {i} (was: unseen)", flush=True)
            for i in (args.ids or sorted(cur)):
                status = cur.get(i)
                if status is None or status == "running":
                    reported_not_running.pop(i, None)
                elif reported_not_running.get(i) != status:
                    print(f"INSTANCE_NOT_RUNNING {i} {status} "
                          "(still billing storage)", flush=True)
                    reported_not_running[i] = status
            prev = cur
            if args.ids and all(i not in cur for i in args.ids):
                print("ALL_WATCHED_INSTANCES_GONE", flush=True)
                return 0
            if args.once:
                return 0
        if (time.time() - t0) / 60 >= args.deadline_min:
            print(f"WATCH_DEADLINE {args.deadline_min:g} min elapsed - "
                  "re-arm or escalate", flush=True)
            return 6
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
