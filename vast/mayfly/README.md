# mayfly — drop-in remote-run watcher for agentic projects

One file, stdlib-only, cross-platform (Windows/Linux, any Python 3.8+).
It polls a remote log over ssh at a fixed interval and **exits loudly**
the moment something decision-worthy happens. That's the entire trick:
in every agent harness, a *background process exiting* is the one wake
signal that can't be forgotten, rate-limited, or silently dropped —
unlike "an agent that promised to check back."

Design provenance: `../WATCHTOWER.md` (mechanism/judgment/authority
layers, the anti-stall invariant, the failure-mode table). mayfly is the
mechanism layer; judgment stays with whatever model wakes up.

## Quick start (agent harness)

Launch as a **main-session background task** (not from a subagent —
subagent→background-task wake chains have been observed to silently
drop; main-session task exits always re-invoke).

Reuse the ssh options `ops.py` itself uses (key, `BatchMode=yes`,
`StrictHostKeyChecking=accept-new` and the repo-local known-hosts file
`.vast/known_hosts`) — never disable host-key checking. `--known-hosts`
adds the accept-new + `UserKnownHostsFile="..."` pair for you, quoted
for OpenSSH so a path with a space (`Python Projects`) works:

```
python vast/mayfly/mayfly.py \
  --ssh "ssh -i C:/Users/JmgLi/.ssh/id_ed25519 -p PORT -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 root@IP" \
  --known-hosts "C:/Users/JmgLi/documents/Python Projects/QuantExperiments/.vast/known_hosts" \
  --events \
  --interval 60 --stall-min 20 --deadline-min 480 \
  --heartbeat .vast/monitors/mayfly.hb
```

`--events` watches `/workspace/quant_experiments_ops/events.jsonl`
(written by `vast/remote_runner.py`): DONE (0) on `run_succeeded` or
`artifacts_ready` with `exit_code` 0; ERROR (3) on `run_failed`,
`runner_exception`, `bootstrap_failed`, `artifact_packaging_failed`, or
`artifacts_ready` with a non-zero `exit_code`. The runner writes a
`heartbeat` event every 5 min, so file growth doubles as progress
(keep `--stall-min` above 5). Regexes are not applied in events mode.
To watch the combined workload output with regexes instead:

```
python vast/mayfly/mayfly.py --ssh "..." --known-hosts "..." \
  --log /workspace/quant_experiments_ops/train.log \
  --done-re "RUN_DONE|SPEEDBENCH_DONE" \
  --interval 60 --stall-min 20 --deadline-min 480
```

`--ssh` is split with `shlex` (POSIX rules): write Windows paths with
**forward slashes** (backslashes are escape characters) and quote any
token containing spaces. If you put `UserKnownHostsFile` inside `--ssh`
yourself it needs OpenSSH's inner double quotes —
`-o 'UserKnownHostsFile="C:/.../Python Projects/.../known_hosts"'` —
which `--known-hosts` spares you. The remote log path is shell-quoted
by mayfly.

Only bytes appended **after** mayfly starts are scanned: it records the
remote file size at start and fetches `tail -c +OFFSET` each poll, so a
re-armed mayfly does not re-fire on an old Traceback. A shrinking file
(truncate/rotation) resets the offset and rescans from byte 0.
`--from-start` opts into scanning the existing content too. The exit
printout keeps a bounded ~4 KB context tail.

Optionally arm the **shadow deadman** as a second background task. It
watches the heartbeat *file* (mtime + last line), never the process —
no agent-watching-agent, no ssh of its own. Heartbeat lines older than
the deadman's own start are ignored (a previous run's `TERMINAL` line
cannot end it, an old mtime cannot trip it), and watch/wait-up mode
moves an existing heartbeat file to `<file>.prev` at start:

```
python vast/mayfly/mayfly.py --deadman .vast/monitors/mayfly.hb \
  --interval 90 --stale-min 6 --deadline-min 500
```

## Exit codes are the protocol

| code | event | meaning | typical playbook |
|---|---|---|---|
| 0 | DONE | done-re matched / success event | pull results, tear down box |
| 3 | ERROR | error-re matched / failure event (context printed) | read tail; judge: fix vs abort |
| 4 | STALL | log did not grow (or `--progress-re` match unchanged) for `--stall-min` | check GPU util; hung → restart run |
| 5 | UNREACHABLE | N consecutive ssh / remote-log-read failures | check instance state via provider CLI |
| 6 | DEADLINE | absolute time cap | escalate to human — never rearm blindly |
| 7 | DEADMAN | heartbeat stale | the watcher itself died; relaunch both |
| 2 | BADARGS | misconfigured | fix the invocation |

On every exit the log tail (~4 KB) plus a final `EVENT:<NAME>` line is
on stdout, so the woken agent has context with zero extra round-trips.

## The prime rule: every wait is a watch

If you are about to write ANY wait loop — for a log, a box, an ssh
port, a file, a process — launch a mayfly instead. A bare
`until <check>; do sleep; done` background loop is a mayfly with every
safety feature deleted: no deadline, no heartbeat, no loud exit, no
context on wake. An agent sitting behind one is indistinguishable from
a dead agent, and a human ends up poking it ("is it stuck?") — which is
the exact failure this kit exists to abolish (observed live
2026-08-14, twice: box-provisioning waits done as bare loops while a
mayfly watched only the training log).

Waiting for a box to come up is `--wait-up` (ssh success IS the event,
15-min default deadline). Waiting for a run is the normal watch mode.
Waiting for anything else: normal watch mode pointed at whatever file
exposes the state. Every mode is deadline-bounded by default (watch
720 min, deadman 750, wait-up 15); `--deadline-min 0` is rejected
(BADARGS) unless `--no-deadline` is passed explicitly, and wait-up
refuses to be unbounded. There is no legitimate unbounded silent wait.

## Rules for the agent that launches it

1. **Write the playbook before sleeping.** For each exit code, one
   imperative action (table above is the template). Judgment calls go
   to Sonnet; Opus (or the main session) wakes only for terminal
   events or repeated watcher death.
2. **Never end a turn "waiting"** without a live harness-tracked
   mayfly. An armed intention is not a mechanism (anti-stall invariant).
3. **Rearm after every wake** that isn't terminal: relaunch mayfly (and
   deadman) before doing anything else. A handled event with no new
   watcher is a silent hole. The re-armed mayfly only scans new bytes,
   so the event that woke you will not instantly re-fire.
4. **Progress defaults to "the log grew".** Any new bytes since the
   last poll reset the stall timer. If a hung run still writes chatter
   (keep-alive lines), pass `--progress-re`: then the LAST match in the
   ~4 KB tail must change while the run is healthy. Test it against
   the real log — dates (`2026/11`) and constant text (`it/s`) make
   useless tokens.
5. `--error-re` should over-trigger rather than under-trigger; a false
   ERROR wake costs one cheap judgment, a missed one costs a night.
   (The default includes bare `RuntimeError` on purpose.)
6. ssh must be non-interactive: key-based auth, `BatchMode=yes`. Reuse
   the ops.py ssh options: `StrictHostKeyChecking=accept-new` plus the
   repo `.vast/known_hosts` (via `--known-hosts`). Do not use
   `StrictHostKeyChecking=no` or `UserKnownHostsFile=/dev/null`.
7. **Git-Bash on Windows mangles the remote path** (MSYS rewrites
   `/workspace/x.log` to `C:/Program Files/Git/workspace/x.log` before
   Python sees it — observed live 2026-08-14). Prefix the launch with
   `MSYS_NO_PATHCONV=1` or run from PowerShell/cmd. mayfly refuses a
   `--log` that looks like a Windows path (BADARGS); other symptoms are
   UNREACHABLE with `cannot read remote log ...` in the fail lines.

## Reusing outside this repo

Copy the single file. There is no config file, no state beyond the
heartbeat, no imports beyond the stdlib — the point is that it works
identically from a Windows laptop, a Linux CI runner, or inside any
agent sandbox that has `ssh` on PATH.
