"""Regression tests for the remote runner's process and manifest boundaries."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from vast.remote_runner import (
    EventWriter,
    _artifact_packaging_lock,
    package_artifacts,
    read_manifest,
)


def valid_manifest() -> dict[str, object]:
    return {
        "format_version": 1,
        "run_id": "runner_hardening",
        "source_commit": "a" * 40,
        "command": ["python", "-c", "pass"],
        "artifact_paths": ["artifacts/run"],
        "num_gpus": 1,
        "require_cuda": False,
        "min_cpu_cores_effective": 1,
        "min_cpu_ram_gb": 1,
        "min_inet_down_mbps": 1,
        "max_dph": 0.5,
        "max_total_usd": 1.0,
        "ttl_hours": 1.0,
        "failure_grace_minutes": 0,
        "disk_gb": 10,
        "poll_seconds": 15,
        "min_cuda": 12.8,
        "avoid_machine_ids": [],
        "input_artifacts": {},
    }


@pytest.mark.parametrize("field", ["ttl_hours", "max_dph", "max_total_usd"])
@pytest.mark.parametrize("value", [True, False, "1.0", None])
def test_manifest_rejects_malformed_spend_numbers(
    tmp_path: Path, field: str, value: object
) -> None:
    manifest = valid_manifest()
    manifest[field] = value
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        read_manifest(path)


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_manifest_rejects_nonfinite_json_numbers(tmp_path: Path, literal: str) -> None:
    manifest = valid_manifest()
    manifest["max_total_usd"] = 1.0
    text = json.dumps(manifest).replace("1.0", literal, 1)
    path = tmp_path / "manifest.json"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=r"finite|non-finite"):
        read_manifest(path)


@pytest.mark.parametrize(
    "field",
    [
        "format_version",
        "num_gpus",
        "failure_grace_minutes",
        "disk_gb",
        "min_inet_down_mbps",
        "min_cpu_cores_effective",
        "min_cpu_ram_gb",
        "max_artifact_archive_bytes",
        "max_artifact_unpacked_bytes",
        "min_cuda",
        "poll_seconds",
    ],
)
def test_manifest_rejects_boolean_optional_numeric_fields(
    tmp_path: Path, field: str
) -> None:
    manifest = valid_manifest()
    manifest[field] = True
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        read_manifest(path)


def test_manifest_rejects_boolean_machine_identity_numbers(tmp_path: Path) -> None:
    manifest = valid_manifest()
    manifest["avoid_machine_ids"] = [False]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="avoid_machine_ids"):
        read_manifest(path)


def test_packaging_lock_ignores_stale_lock_path(tmp_path: Path) -> None:
    ops_dir = tmp_path / "ops"
    ops_dir.mkdir()
    lock_path = ops_dir / ".artifact-packaging.lock"
    lock_path.write_text("stale pid from an interrupted runner", encoding="ascii")

    with _artifact_packaging_lock(ops_dir):
        assert lock_path.is_file()
    # The inode remains as the stable advisory-lock target; no stale PID is
    # consulted and no cleanup race can replace it underneath another runner.
    assert lock_path.is_file()


def test_packaging_lock_is_released_when_process_exits(tmp_path: Path) -> None:
    ops_dir = tmp_path / "ops"
    project_root = Path(__file__).resolve().parents[1]
    code = (
        "import os; "
        "from pathlib import Path; "
        "from vast.remote_runner import _artifact_packaging_lock; "
        f"ctx = _artifact_packaging_lock(Path({str(ops_dir)!r})); "
        "ctx.__enter__(); os._exit(0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    with _artifact_packaging_lock(ops_dir):
        pass


def test_packaging_rejects_declared_unpacked_bytes_over_limit(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    artifact = repo / "artifacts" / "run" / "receipt.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"x" * 11)
    manifest = valid_manifest()
    manifest["max_artifact_archive_bytes"] = 100
    manifest["max_artifact_unpacked_bytes"] = 10
    with pytest.raises(ValueError, match="unpacked manifest limit"):
        package_artifacts(
            manifest,
            repo=repo,
            ops_dir=tmp_path / "ops",
            exit_code=0,
            events=EventWriter(tmp_path / "events.jsonl", "runner_hardening"),
        )


def test_packaging_rejects_compressed_archive_over_limit(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    artifact = repo / "artifacts" / "run" / "receipt.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(os.urandom(4096))
    manifest = valid_manifest()
    manifest["max_artifact_archive_bytes"] = 100
    manifest["max_artifact_unpacked_bytes"] = 8192
    with pytest.raises(ValueError, match="compressed manifest limit"):
        package_artifacts(
            manifest,
            repo=repo,
            ops_dir=tmp_path / "ops",
            exit_code=0,
            events=EventWriter(tmp_path / "events.jsonl", "runner_hardening"),
        )
