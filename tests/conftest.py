"""Seal the test suite off from the real Vast account and the real runtime state.

Without this, tests that only partially patch vast.ops wrote lifecycle events
into the repo's real `.vast/` directory and at least one
(`test_hard_limit_diagnostic_failures_cannot_precede_destroy`) reached the real
`vastai` CLI with the account key from `vast/secrets.env`.

Every test gets a private runtime directory and a secrets path that does not
exist, so any unpatched provider call fails before a subprocess is started.
Tests that patch these names themselves still override this fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_vast_runtime(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    try:
        import vast.ops as ops
    except Exception:  # pragma: no cover - ops not importable in this environment
        yield
        return
    runtime: Path = tmp_path_factory.mktemp("vast_runtime")
    monkeypatch.setattr(ops, "RUNTIME_DIR", runtime)
    monkeypatch.setattr(ops, "STATE_FILE", runtime / "instances.json")
    monkeypatch.setattr(ops, "STATE_LOCK_FILE", runtime / "instances.state.lock")
    monkeypatch.setattr(ops, "STAGE_STATE_FILE", runtime / "stages.json")
    monkeypatch.setattr(ops, "EVENT_FILE", runtime / "events.jsonl")
    monkeypatch.setattr(ops, "KNOWN_HOSTS", runtime / "known_hosts")
    monkeypatch.setattr(ops, "STAGING_DIR", runtime / "staging")
    monkeypatch.setattr(ops, "MONITOR_DIR", runtime / "monitors")
    monkeypatch.setattr(ops, "SECRETS_FILE", runtime / "no-secrets-in-tests.env")
    yield
