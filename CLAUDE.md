# QuantExperiments

## Remote GPU runs (Vast.ai)

The main session owns experiment code, configs, seeds, data-download logic,
and interpretation, **and operates Vast directly** (user decision, 2026-11-02:
a delegated operator was too slow to react to messages).

- Follow the `vast-ops` skill (`.claude/skills/vast-ops/SKILL.md`) for every
  rental: launch gates, `vast/ops.py` as the only lifecycle entry point,
  harness-tracked watchers, verified recovery, then destroy and reconcile.
  Where the skill says "operator", read "main session".
- Renting needs the user's explicit max-dollar cap; track cumulative spend
  across a campaign yourself (`max_total_usd` is per manifest).
- The `vast-operator` agent (`.claude/agents/vast-operator.md`) remains
  available if the user asks to delegate a run.
- Local Python is the conda env ToastEnv
  (`C:\Users\JmgLi\anaconda3\envs\ToastEnv\python.exe`). Run pytest with
  `--basetemp` pointing into a writable folder if the default temp dir is
  access-denied.
- While a run is live, read experiment outputs directly, read-only:
  - `python vast/ops.py status`
  - `python vast/ops.py peek <run_id> [artifacts/... | train.log] [--tail N]`
  - `python vast/ops.py ssh-command <run_id>` for a manual SSH look —
    look only; never kill, edit, or touch `/workspace/quant_experiments_ops/`.
- Jobs download their own data at runtime. Never commit datasets, weights,
  or anything from `vast/secrets.env`. Extra remote pip deps go in
  `requirements-remote.txt`.
- Never print, log, or transfer values from `vast/secrets.env`, with one
  exception (user decision, 2026-11-04): `HF_TOKEN` (a read-only Hugging Face
  token) may go to rented boxes, only the way `vast/ops.py` sends it — over
  SSH stdin into the bootstrap's environment, never on a command line, in a
  file, a log, or an artifact. `VAST_API_KEY` and every other value stay local.
- Remote job outputs must be written under `artifacts/` to be recovered;
  verified results land in `artifacts/remote/<run_id>/`.
- After a reboot or long sleep with a run live:
  `python vast/ops.py reconcile --restart-monitors`.

Tests: `python -m pytest tests -q`.
