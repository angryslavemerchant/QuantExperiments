# QuantExperiments

## Remote GPU runs (Vast.ai)

The main session is the **project agent**: it owns experiment code, configs,
seeds, data-download logic, and interpretation. It does not operate Vast.

- To run something on a rented GPU, spawn the **`vast-operator`** agent
  (`.claude/agents/vast-operator.md`, runs on Opus), which follows the
  `vast-ops` skill (`.claude/skills/vast-ops/SKILL.md`). Include
  the user's authorizing message verbatim, the exact command, artifact
  paths, and the data source. Renting needs the user's explicit max-dollar cap.
- While a run is live, read experiment outputs directly, read-only:
  - `python vast/ops.py status`
  - `python vast/ops.py peek <run_id> [artifacts/... | train.log] [--tail N]`
  - `python vast/ops.py ssh-command <run_id>` for a manual SSH look —
    look only; never kill, edit, or touch `/workspace/quant_experiments_ops/`.
- Jobs download their own data at runtime. Never commit datasets, weights,
  or anything from `vast/secrets.env`. Extra remote pip deps go in
  `requirements-remote.txt`.
- Remote job outputs must be written under `artifacts/` to be recovered;
  verified results land in `artifacts/remote/<run_id>/`.
- After a reboot or long sleep with a run live:
  `python vast/ops.py reconcile --restart-monitors`.

Tests: `python -m pytest tests -q`.
