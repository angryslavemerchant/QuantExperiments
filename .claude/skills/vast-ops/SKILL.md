---
name: vast-ops
description: Operate guarded QuantExperiments GPU jobs on Vast.ai — readiness checks, offer search, capped rental, exact-source upload, monitoring, artifact recovery, reconciliation, and termination. Use for any QuantExperiments request involving Vast, rented GPUs, launching a remote experiment, remote logs, GPU cost control, or recovering remote results. The project agent should hand Vast work to a run-scoped operator subagent that follows this skill.
---

# Vast GPU operations

`vast/ops.py` is the only supported lifecycle entry point. `vast/README.md`
has setup and command examples.

## Two roles

**Project agent** (main session): owns the experiment — code, configs,
seeds, data-download logic, scientific gates, interpretation. It does not
rent, monitor, or destroy anything. It may at any time *read* experiment
outputs on a live run:

```powershell
python vast/ops.py status                       # which runs exist
python vast/ops.py peek <run_id>                # ls artifacts/
python vast/ops.py peek <run_id> artifacts/<run_id>/metrics.json
python vast/ops.py peek <run_id> train.log --tail 200
python vast/ops.py ssh-command <run_id>         # exact ssh for manual read-only look
```

`peek` is bounded and restricted to `artifacts/` and `train.log`. Over raw
SSH, look only — never kill processes, edit files, or touch
`/workspace/quant_experiments_ops/`; the operator and local monitor own the
box.

**Vast operator** (run-scoped subagent, this skill): owns everything
infrastructure for one authorized run — manifest pinning, `doctor`, search,
launch, watching, recovery, verification, destruction, final report. Spawn
it as the `vast-operator` agent (`.claude/agents/vast-operator.md`, pinned
to **Opus**). The spawn prompt must include the
user's exact authorizing message verbatim — never a paraphrase — plus the
experiment command, artifact paths, and data-download needs. It terminates
after verified destruction and reconciliation and returns one report.

## Authorization

- Read-only, always allowed: `doctor`, `validate-manifest`, `search`,
  `status`, `logs`, `diagnose`, `peek`, `ssh-command`, `reconcile`
  (without `--restart-monitors`).
- Renting is a purchase. It requires a concrete experiment and the user's
  explicit maximum total-dollar cap. The user never has to invent a run ID;
  the operator generates a unique one and reports it.
- Termination only as predeclared by the manifest, or on explicit request
  with the tracked instance ID.
- `race` and `adopt-existing` are disabled; do not attempt them even if
  asked.
- An unattended retry campaign needs one explicit approval naming the run
  count/stopping rule, pinned commit, host predicate, cumulative cap, and
  which operational failures are retryable. Never auto-retry a scientific
  failure. Note: `max_total_usd` is per manifest — track cumulative spend
  across a campaign yourself.

## Launch gates

Do not launch unless:

1. `python vast/ops.py doctor` reports ready (tools, credentials, Git, SSH);
2. the worktree is clean and manifest `source_commit` equals `HEAD`;
3. the manifest records `seed_policy`, `checkpoint_cadence`, `data_policy`
   (where the job downloads data from), artifact paths, `max_dph`,
   `max_total_usd`, `ttl_hours`, and `stall_minutes`;
4. `validate-manifest` passes; and
5. `max_total_usd` is within the user's approved cap.

## Data

Jobs download their own data at runtime (first step of the command or a
`post_commands` entry). Input-artifact upload is disabled. Only tracked
source from the pinned commit is bundled — never commit datasets, caches,
weights, or secrets. Extra pip dependencies go in `requirements-remote.txt`
at the repo root (installed with deps at bootstrap); the image already has
torch/CUDA. Size `disk_gb` and `min_inet_down_mbps` for the download.

## Lifecycle

1. Project agent finishes code and tests locally and commits.
2. Operator generates a run ID, pins the commit in a new manifest under
   `vast/manifests/` (copy `example.json`), runs `doctor`,
   `validate-manifest`, `search`.
3. Confirm the user's authorization covers the configuration, host
   constraints, TTL, and max spend. Ask only for missing authority.
4. `python vast/ops.py launch <manifest> --detach` — exactly one instance.
5. Watch (see Monitoring). The local monitor enforces TTL, spend, stall,
   stopped-instance, recovery, and destruction on its own.
6. Success: archive is pulled to `artifacts/remote/<run_id>/`, every
   SHA-256 verified, then the instance is destroyed per manifest.
7. Failure: evidence kept for `failure_grace_minutes`, bounded recovery,
   then destroy.
8. After any local interruption (reboot, sleep, closed terminal): run
   `python vast/ops.py reconcile --restart-monitors` immediately.

`keep_alive_after_verify` keeps the box after verified success under the
same cap schedule; end it with `finalize-session --confirm-instance <id>`.
Follow-up commits use `session-stage`; never edit the live checkout.

## Monitoring

The detached `ops.py` monitor is the safety layer — it survives agent loss.
Agent-side watching is only for waking up and reporting, and follows
`vast/WATCHTOWER.md`: scripts poll, models judge.

- Never wait with a bare sleep loop or a promise to check back. Launch a
  harness-tracked background watcher whose *exit* wakes you:
  - box coming up: `python vast/mayfly/mayfly.py --wait-up --ssh "<ssh-command posix>" --deadline-min 20`
  - run progress: `python vast/mayfly/mayfly.py --events --ssh "<ssh-command posix>" --stall-min 30 --deadline-min <ttl minutes> --heartbeat .vast/mayfly_<run_id>.hb`
  - instance disappearance: `python vast/watch_instances.py <instance_id> --deadline-min <ttl minutes>`
  Get the ssh string from `python vast/ops.py ssh-command <run_id>` (`posix`
  field). See `vast/mayfly/README.md` for exit codes.
- On each wake: read the tail mayfly printed, check `status`/`diagnose`,
  decide benign vs real, report only changes, then **re-arm** a watcher
  before ending the turn. Ambiguous = real.
- Report: provisioning/training/recovery/destruction transitions, stalls,
  cap pressure, monitor or recovery failures, and scientifically notable
  milestones (to the project agent, without deciding them).
- Stop watching right after verified destruction.

## Safety

- Never print, log, or transfer values from `vast/secrets.env`.
- The account API key stays local. The box only gets Vast's instance-scoped
  `CONTAINER_API_KEY`, used by the emergency self-destruct watchdog —
  defense in depth, not a guarantee (it has not been verified against a real
  instance yet; see README "Known gaps").
- Transfer an exact Git bundle, never an uncommitted tree.
- SSH uses `.vast/known_hosts` with accept-new; stale keys for recycled
  endpoints are cleared automatically on a fresh instance.
- Never destroy a successful run before artifact verification.
- Hard TTL/spend cleanup attempts recovery, then destroys regardless.
- The deadline schedule is persisted before purchase and never renewed.
- `.vast/instances.json` is a hint; the live Vast API is authoritative.

Final report: run ID, instance ID, source commit, elapsed time, estimated
cost, terminal event, artifact destination, hash verification, destruction
status, unresolved exceptions.
