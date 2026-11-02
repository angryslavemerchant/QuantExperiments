# Vast.ai operations for QuantExperiments

`vast/ops.py` is the supported entry point for rented-GPU experiments. It keeps
the Vast account key local, transfers an exact clean Git commit as a bundle,
monitors machine-readable remote events, verifies recovered artifacts, and enforces
approved time and spending limits.

Read `.claude/skills/vast-ops/SKILL.md` before any Vast operation.

## Two roles

- **Project agent** (main session) designs and interprets experiments. It
  never rents, monitors, or destroys. It can read live outputs at any time:
  `ops.py peek <run_id> [artifacts/... | train.log] [--tail N]` and
  `ops.py ssh-command <run_id>` (exact SSH for a manual read-only look).
- **Vast operator** (the `vast-operator` agent in `.claude/agents/`, Opus)
  owns one authorized run end to end and exits after verified destruction.

## Installed files

- `ops.py`: guarded local lifecycle and deterministic monitor.
- `remote_bootstrap.sh`: installs the exact uploaded checkout.
- `remote_runner.py`: executes only manifest commands, enforces the remote
  workload deadline, emits JSONL events, and packages declared artifacts.
- `manifests/example.json`: non-launchable QuantExperiments manifest template.
- `mayfly/`: harness-tracked watcher that exits loudly on DONE / ERROR /
  STALL / UNREACHABLE / DEADLINE (see `mayfly/README.md`).
- `watch_instances.py`: account-level tripwire for instances disappearing or
  entering a non-running (still storage-billing) state.
- `WATCHTOWER.md`: monitoring design — scripts poll, models judge.
- `secrets.env.example`: credential shape only; contains no key.
- `blacklist.json`: local known-bad physical host IDs; ignored by Git.

## Local setup

QuantExperiments must be a Git repository before launch because source
transfer is pinned to one exact clean commit.

Install the Vast CLI in the local Python environment, then copy
`vast/secrets.env.example` to the ignored `vast/secrets.env` and add the local
account key:

```text
VAST_API_KEY=...
```

Never commit, display, or transfer that file. The remote host never receives the
account key. On ordinary creates, the on-start command also arms a detached,
last-resort watchdog using only Vast's provider-injected per-container
`CONTAINER_API_KEY` and `CONTAINER_ID`; that scoped key is not the account key.

## Safety boundary

- `doctor`, `validate-manifest`, `search`, `status`, `logs`,
  `diagnose`, `peek`, `ssh-command`, and `reconcile` (without
  `--restart-monitors`) are read-only.
- `launch` is the purchase command. A paid campaign requires concrete experiment
  intent/configuration and an explicit maximum total-dollar boundary, but the
  user does not have to invent a run ID or repeat internal lifecycle metadata.
  The operator generates a unique internal run ID before launch and uses it
  consistently throughout the lifecycle.
- `adopt-existing` and `race` are fail-closed in this release pending separate
  setup-supervisor and aggregate-deadline audits. Their helper code and tests
  are preserved for future protocol work, but neither CLI path may control a
  paid campaign.
- The default is one instance.
- Offer selection avoids the cheapest-host lemons: among eligible offers it
  considers those within `OFFER_PRICE_BAND` (20%) of the cheapest price and
  picks the highest `reliability2`; price only breaks ties. `max_dph` and
  `max_total_usd` still bound spend.
- Per-GPU hourly caps apply before the band (`OFFER_MAX_DPH_BY_GPU`):
  RTX 4090 at most $0.50/h, RTX 5090 at most $0.70/h.
- The local monitor reserves time and budget for recovery and termination.
- At a hard TTL or spend boundary, a running job is first asked to package its
  latest atomic checkpoint and declared artifacts before the best-effort pull
  and mandatory destruction.
- Ordinary success termination occurs only after the artifact archive, run ID,
  source commit, declared paths, and every file hash verify locally.
- Ordinary Vast instances use the local supervisor for normal destruction and
  reconciliation. The detached in-instance watchdog is only a last-resort
  fallback at the precomputed cap/TTL boundary; it retries a provider DELETE
  using the scoped container key and never receives `VAST_API_KEY`. After a
  machine or network interruption, run reconciliation immediately.

For delegated operation, the project agent spawns the `vast-operator`
agent (pinned to Opus) with the
user's exact authorizing message included verbatim. It owns the campaign end
to end and terminates after recovery, hash verification, destruction, and
reconciliation.

The lifecycle uses Vast's PyTorch template configuration with the compatible
`vastai/pytorch:cuda-12.8.1-auto` image and direct SSH.

## Manifest

Copy `vast/manifests/example.json`, replace the placeholder commit with the full
40-character `git rev-parse HEAD`, and predeclare:

- the exact Python command and any sequential post-commands;
- artifact paths, all constrained to `artifacts/`;
- compressed and unpacked artifact-byte ceilings sized for the declared outputs;
- an empty `input_artifacts` mapping; this project disables remote inputs;
- GPU, hourly price, total cost, TTL, failure grace, disk, CPU, RAM, and network
  constraints, including the exact `num_gpus` count for multi-GPU contracts;
- optional hard consumer-CPU and exact geolocation constraints for host-pinned
  campaigns;
- `seed_policy`, `checkpoint_cadence`, and `data_policy` (where the job
  downloads its data from); `stall_minutes` (default 45, `0` disables): the
  monitor recovers and destroys a run whose remote events — including the
  5-minute heartbeat — stop changing for that long; and
- `destroy_after_verify` or `keep_alive_after_verify`.

The example intentionally fails validation until its placeholder commit and
scientific settings are replaced.

Only tracked source from the pinned commit is bundled; datasets, caches,
weights, and secrets must never be committed. Jobs download their data at
runtime. Extra pip dependencies go in `requirements-remote.txt` at the repo
root (installed with dependencies during bootstrap; the image already ships
torch/CUDA). A `pyproject.toml`, if present, is installed editable without
dependencies; otherwise the repo root is on `PYTHONPATH`.

## Read-only preparation

```powershell
python vast/ops.py doctor
python vast/ops.py validate-manifest vast/manifests/my_run.json
python vast/ops.py search --gpu RTX_5090 --max-dph 0.50
python vast/ops.py search --gpu RTX_5090 --num-gpus 4 --max-dph 2.00
python vast/ops.py status
python vast/ops.py reconcile
```

`doctor` returning `ready_to_launch: false` for a dirty or non-Git project is a
launch gate; it does not prevent the other read-only inspection steps whose own
requirements are satisfied.

## Approved lifecycle

After the user approves a concrete experiment and maximum total cost, the
operator generates the unique internal run ID, builds the pinned manifest, and
runs the guarded launch. Generated IDs and lifecycle metadata are implementation
details, not values the user must choose or echo.

```powershell
python vast/ops.py launch vast/manifests/generated_run.json
```

`adopt-existing` and `race` intentionally return an operational error before
manifest loading or provider access. Do not enable either for a paid campaign.
The admitted portability path is a single newly created instance through
`launch`.

Use attached monitoring by default or `--detach` for a hidden local supervisor.
Resume and inspect with:

```powershell
python vast/ops.py monitor my_run
python vast/ops.py reconcile --restart-monitors
python vast/ops.py logs my_run
python vast/ops.py diagnose my_run --metrics-tail 10
python vast/ops.py pull my_run
```

Agent-side watching uses harness-tracked watchers whose exit wakes the
agent — never bare sleep loops:

```powershell
python vast/ops.py ssh-command my_run          # copy the "posix" field
python vast/mayfly/mayfly.py --events --ssh "<posix>" --stall-min 30 --deadline-min 300 --heartbeat .vast/mayfly_my_run.hb
python vast/watch_instances.py 12345678 --deadline-min 300
```

These only wake and inform the agent. The detached local monitor enforces
TTL, spend, stall, stopped-instance cleanup, recovery, and destruction, and
survives agent, task, or app loss.

For a verified retained session:

```powershell
python vast/ops.py finalize-session my_run --confirm-instance 12345678
```

For a stalled run whose declared artifacts are ready, use
`recover-destroy`; it snapshots, pulls, verifies, and only then destroys. An
intentional early stop uses `destroy` with the exact tracked instance ID and
does not promise successful recovery.

Follow-up work on a retained session uses `session-stage` with a new committed
source and complete manifest. Each stage has its own checkout, monitor, and
verified destination under
`artifacts/remote/<parent>/stages/<stage>/`.

If a client dies during purchase, `reconcile` can recover lifecycle ownership of
the uniquely labeled `quant-experiments-<run_id>` instance; this is not the
disabled `adopt-existing` command. Cleanup of an otherwise untracked
project-labeled instance requires the explicit manual `destroy-orphan` command
with both exact live identifiers. It rejects any ID or label already referenced
by local ordinary, quarantined, reconciliation, race, or session-stage state;
those targets must use their owning cleanup path.

## Remote events and recovery

The remote runner emits `run_started`, `command_started`, sparse
`heartbeat`, `command_finished`, `artifacts_ready`, and terminal events
under `/workspace/quant_experiments_ops/`. Command output is stored in
`train.log`.

Verified downloads are stored under
`artifacts/remote/<run_id>/`. The local `.vast/` directory contains lifecycle
state, known hosts, monitors, and staging data and is ignored by Git. Live Vast
state remains authoritative.

The monitor also:

- treats an instance Vast reports as `exited`/`stopped`/`offline` (or
  intended `stopped`) for 3 consecutive polls as a hard failure — recover,
  then destroy (stopped instances still bill storage and cannot run their
  own watchdog);
- retries transient local state-lock errors instead of exiting;
- re-anchors its deadlines to the persisted UTC schedule every poll, so
  laptop sleep cannot delay cleanup past the wall-clock boundary.

The remote workload is stopped 5 minutes before the persisted
`cleanup_start_at` (anchored at purchase, not at runner start) so it can
package artifacts before recovery begins.

## Known gaps

- **Emergency watchdog unverified live.** The in-box self-destruct uses
  Vast's `CONTAINER_API_KEY`; whether that key may DELETE its own instance
  has not been confirmed. Before relying on it, run one short, cheap
  instance and confirm the watchdog destroys it at its deadline.
- **Watchdog checked once.** It is verified at SSH readiness only; a
  workload running as root could kill it unnoticed.
- **Per-manifest cap only.** `max_total_usd` (≤ $100) applies per launch;
  there is no account-wide cumulative cap.
- **Bandwidth charges** (`inet_up_cost`/`inet_down_cost`) are neither
  filtered nor budgeted — check them in `search` output for download-heavy
  jobs.
- **Archive size** is checked after download, not before.
- **No auto-restart after reboot.** Run `reconcile --restart-monitors`
  after any reboot, logoff, or long sleep.
- **Maintainability.** `ops.py` is one ~10.6k-line file; the disabled
  race/adopt code (~1.5k lines) is still reachable from `reconcile` for old
  records.

## Host checks at bootstrap

Before installing anything, `remote_bootstrap.sh` runs `vast/net_probe.py`: a
30-second ranged download of a Qwen2.5-7B shard from the Hugging Face Hub.
Below 25 MB/s (`QE_NET_PROBE_MIN_MB_PER_S`) bootstrap fails, and the launcher
records a provisioning failure and destroys the box. Dependency install is
capped at 360 s (`QE_SETUP_TIMEOUT_SECONDS`), and the whole bootstrap at
`BOOTSTRAP_TIMEOUT_SECONDS` = 600 s (was 1,200 s).
