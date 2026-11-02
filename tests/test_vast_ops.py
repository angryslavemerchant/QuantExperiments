import base64
import errno
import json
import os
import re
import subprocess
import sys
import tarfile
import threading
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import vast.ops as vast_ops
from vast.ops import (
    OpsError,
    RunManifest,
    _claim_supervisor_lock,
    _destroy_reconciliation,
    _monitor_run,
    _monitor_write_if_active,
    _purchase_reconciliation,
    _record_provisioning_setup_failure,
    _race_cleanup_losers,
    _race_cleanup_pending,
    _release_supervisor_lock,
    _supervisor_lock_owner_pid,
    _validate_live_contract_constraints,
    attach_local_ssh_key,
    build_parser,
    campaign_deadline_schedule,
    cleanup_reserve_hours,
    command_adopt_existing,
    command_launch,
    command_race,
    command_reconcile,
    create_instance,
    destroy_attempt_budget_seconds,
    destroy_instance,
    destroy_reconciliation_lease_seconds,
    known_hosts_option,
    label_adopted_instance,
    load_state,
    local_ssh_public_key,
    monitor_run,
    pick_offer,
    remote_watchdog_deadline_epoch,
    remote_watchdog_onstart_command,
    require_adoption_approval,
    require_local_ssh_identity,
    retain_failed_debug_session,
    scp_download,
    scp_upload,
    search_offers,
    state_file_lock,
    utc_now,
    validate_adoption_instance,
    validate_offer_for_purchase,
    vast_cli_path,
    verify_adoption_host_idle,
    verify_created_instance,
    wait_for_ssh,
)
from vast.remote_runner import (
    EventWriter,
    package_artifacts,
    run_command,
    runtime_details,
    verify_input_artifacts,
)


@pytest.fixture(autouse=True)
def enable_separately_excluded_legacy_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise internal race/adoption contracts while production gates stay shut."""

    monkeypatch.setattr(vast_ops, "ENABLE_CANDIDATE_RACE", True)
    monkeypatch.setattr(vast_ops, "ENABLE_EXISTING_INSTANCE_ADOPTION", True)


def test_vast_cli_path_finds_windows_environment_scripts_entrypoint(
    tmp_path: Path,
) -> None:
    interpreter = tmp_path / "python.exe"
    entrypoint = tmp_path / "Scripts" / "vastai.exe"
    entrypoint.parent.mkdir()
    entrypoint.write_bytes(b"placeholder")

    with (
        patch.dict(os.environ, {}, clear=True),
        patch("vast.ops.sys.executable", str(interpreter)),
        patch("vast.ops.shutil.which", return_value=None),
    ):
        assert vast_cli_path() == str(entrypoint)


@pytest.mark.parametrize(
    "stdout",
    [
        '{"id":1,"id":2}',
        '[{"id":1,"machine_id":2,"machine_id":3}]',
        '[{"id":1,"label":"first","label":"second"}]',
    ],
)
def test_provider_json_reader_rejects_duplicate_authority_keys(stdout: str) -> None:
    response = SimpleNamespace(returncode=0, stdout=stdout, stderr="")
    with (
        patch("vast.ops.vast_cli_path", return_value="vastai"),
        patch("vast.ops.isolated_cli_environment", return_value={}),
        patch("vast.ops.subprocess.run", return_value=response),
        pytest.raises(OpsError, match="valid JSON"),
    ):
        vast_ops.run_vast("show", "instances")


@pytest.mark.parametrize(
    "stdout",
    [
        "NaN",
        "Infinity",
        "-Infinity",
        '[{"dph_total":NaN}]',
        '{"nested":{"value":Infinity}}',
        '{"nested":{"dph_total":1e400}}',
        '{"nested":{"dph_total":-1e400}}',
        '{"nested":{"value":9223372036854775808}}',
        '{"nested":{"value":-9223372036854775809}}',
    ],
)
def test_provider_json_reader_rejects_nonstandard_numeric_constants(
    stdout: str,
) -> None:
    response = SimpleNamespace(returncode=0, stdout=stdout, stderr="")
    with (
        patch("vast.ops.vast_cli_path", return_value="vastai"),
        patch("vast.ops.isolated_cli_environment", return_value={}),
        patch("vast.ops.subprocess.run", return_value=response),
        pytest.raises(OpsError, match="valid JSON"),
    ):
        vast_ops.run_vast("show", "instances")


def test_failed_provider_command_does_not_reflect_raw_arguments() -> None:
    secret = "provider-only-value-9384"
    public_key = f"ssh-ed25519 {secret} portability"
    response = SimpleNamespace(returncode=1, stdout="", stderr="provider failure")
    with (
        patch("vast.ops.vast_cli_path", return_value="vastai"),
        patch("vast.ops.isolated_cli_environment", return_value={}),
        patch("vast.ops.subprocess.run", return_value=response),
        patch("vast.ops.load_secrets", return_value={}),
        pytest.raises(OpsError, match="provider command failed") as raised,
    ):
        vast_ops.run_vast(
            "attach",
            "ssh",
            "1234",
            public_key,
            "--api-key",
            secret,
        )

    assert secret not in str(raised.value)
    assert public_key not in str(raised.value)


def test_provider_subprocess_exception_does_not_reflect_raw_arguments() -> None:
    secret = "provider-only-value-9384"
    public_key = f"ssh-ed25519 {secret} portability"
    timeout = subprocess.TimeoutExpired(
        ["vastai", "attach", "ssh", "1234", public_key],
        180,
    )
    with (
        patch("vast.ops.vast_cli_path", return_value="vastai"),
        patch("vast.ops.isolated_cli_environment", return_value={}),
        patch("vast.ops.subprocess.run", side_effect=timeout),
        pytest.raises(OpsError, match="command execution failed") as raised,
    ):
        vast_ops.run_vast("attach", "ssh", "1234", public_key)

    assert secret not in str(raised.value)
    assert public_key not in str(raised.value)


def test_provider_json_reader_rejects_oversized_or_deep_response() -> None:
    responses = [
        SimpleNamespace(returncode=0, stdout='{"value":"' + "x" * 100 + '"}', stderr=""),
        SimpleNamespace(returncode=0, stdout="[" * 30 + "0" + "]" * 30, stderr=""),
    ]
    for response in responses:
        with (
            patch("vast.ops.vast_cli_path", return_value="vastai"),
            patch("vast.ops.isolated_cli_environment", return_value={}),
            patch("vast.ops._MAX_PROVIDER_JSON_BYTES", 64),
            patch("vast.ops.subprocess.run", return_value=response),
            pytest.raises(OpsError, match="malformed or oversized"),
        ):
            vast_ops.run_vast("show", "instances")


def test_provider_json_reader_accepts_a_large_valid_offer_market() -> None:
    offers = [
        {
            "id": index + 1,
            **{f"field_{field}": field for field in range(62)},
        }
        for index in range(130)
    ]
    stdout = json.dumps(offers, separators=(",", ":"))
    assert len(stdout.encode("utf-8")) < vast_ops._MAX_PROVIDER_JSON_BYTES
    assert not vast_ops._json_structure_within_diagnostic_limits(stdout)
    response = SimpleNamespace(returncode=0, stdout=stdout, stderr="")
    with (
        patch("vast.ops.vast_cli_path", return_value="vastai"),
        patch("vast.ops.isolated_cli_environment", return_value={}),
        patch("vast.ops.subprocess.run", return_value=response),
    ):
        assert vast_ops.run_vast("search", "offers") == offers


def test_remote_command_freezes_python_hash_seed(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FinishedProcess:
        pid = 123
        returncode = 0

        @staticmethod
        def poll() -> int:
            return 0

    def fake_popen(*args: object, **kwargs: object) -> FinishedProcess:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FinishedProcess()

    with (
        patch.dict(
            os.environ,
            {
                "CONTAINER_API_KEY": "provider-container-secret",
                "VAST_API_KEY": "provider-account-secret",
                "VASTAI_API_KEY": "provider-legacy-secret",
            },
        ),
        patch("vast.remote_runner.subprocess.Popen", side_effect=fake_popen),
    ):
        assert (
            run_command(
                ["python", "scripts/check_scaffold.py"],
                step=0,
                repo=tmp_path,
                log_path=tmp_path / "run.log",
                deadline=time.monotonic() + 10,
                events=EventWriter(tmp_path / "events.jsonl", "test-run"),
            )
            == 0
        )

    environment = captured["kwargs"]["env"]  # type: ignore[index]
    assert environment["PYTHONHASHSEED"] == "0"  # type: ignore[index]
    assert "CONTAINER_API_KEY" not in environment  # type: ignore[operator]
    assert "VAST_API_KEY" not in environment  # type: ignore[operator]
    assert "VASTAI_API_KEY" not in environment  # type: ignore[operator]


def test_runtime_details_records_portability_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    details = runtime_details(require_cuda=False)
    assert details["python_full_version"]
    assert details["python_implementation"]
    assert details["platform"]
    assert details["locale_encoding"]
    assert details["filesystem_encoding"]
    assert details["python_hash_seed"] == "0"
    assert "numpy" in details


def test_local_artifact_extraction_enforces_unpacked_byte_ceiling(
    tmp_path: Path,
) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"x" * 11)
    archive = tmp_path / "artifacts.tar.gz"
    with tarfile.open(archive, "w:gz", format=tarfile.USTAR_FORMAT) as bundle:
        bundle.add(payload, arcname="payload.bin")
    with pytest.raises(OpsError, match="unpacked-byte limit"):
        vast_ops.safe_extract_tar(
            archive,
            tmp_path / "unpacked",
            max_unpacked_bytes=10,
        )


def test_local_artifact_inspection_rejects_member_flood_before_extraction(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "member-flood.tar.gz"
    with tarfile.open(archive, "w:gz", format=tarfile.USTAR_FORMAT) as bundle:
        for name in ("first", "second", "third"):
            member = tarfile.TarInfo(name)
            member.size = 0
            bundle.addfile(member)
    destination = tmp_path / "unpacked"
    with (
        patch("vast.ops.MAX_ARTIFACT_FILE_COUNT", 2),
        pytest.raises(OpsError, match="file-count limit"),
    ):
        vast_ops.safe_extract_tar(
            archive,
            destination,
            max_unpacked_bytes=10,
        )
    assert not destination.exists()


def test_local_artifact_inspection_rejects_pax_metadata_before_extraction(
    tmp_path: Path,
) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"x")
    archive = tmp_path / "pax-metadata.tar.gz"
    with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT) as bundle:
        member = bundle.gettarinfo(str(payload), arcname="payload.bin")
        member.pax_headers = {"comment": "x" * (128 * 1024)}
        with payload.open("rb") as stream:
            bundle.addfile(member, stream)
    destination = tmp_path / "unpacked"

    with pytest.raises(OpsError, match="ustar|regular"):
        vast_ops.safe_extract_tar(
            archive,
            destination,
            max_unpacked_bytes=1024 * 1024,
        )

    assert not destination.exists()


def test_valid_compressible_ustar_body_is_streamed_and_extracted(tmp_path: Path) -> None:
    payload = tmp_path / "large.bin"
    payload.write_bytes(b"\0" * (8 * 1024 * 1024))
    archive = tmp_path / "large-ustar.tar.gz"
    with tarfile.open(archive, "w:gz", format=tarfile.USTAR_FORMAT) as bundle:
        bundle.add(payload, arcname="large.bin")
    destination = tmp_path / "unpacked"

    vast_ops.safe_extract_tar(
        archive,
        destination,
        max_unpacked_bytes=9 * 1024 * 1024,
        expected_files={"large.bin"},
    )

    assert (destination / "large.bin").stat().st_size == 8 * 1024 * 1024


def test_successful_pull_requires_exact_integer_zero_manifest_exit_code(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    payload = tmp_path / "payload.txt"
    payload.write_text("x", encoding="utf-8")
    archive = tmp_path / "source-artifacts.tar.gz"
    with tarfile.open(archive, "w:gz", format=tarfile.USTAR_FORMAT) as bundle:
        bundle.add(payload, arcname="payload.txt")
    artifact_manifest = tmp_path / "source-artifact-manifest.json"
    artifact_manifest.write_text(
        json.dumps(
            {
                "format_version": 1,
                "run_id": manifest.run_id,
                "source_commit": manifest.source_commit,
                "exit_code": False,
                "missing_declared_paths": [],
                "files": [
                    {
                        "path": "payload.txt",
                        "size": payload.stat().st_size,
                        "sha256": vast_ops.sha256_file(payload),
                    }
                ],
                "archive_sha256": vast_ops.sha256_file(archive),
            }
        ),
        encoding="utf-8",
    )
    record = {
        "run_id": manifest.run_id,
        "instance_id": 1234,
        "source_commit": manifest.source_commit,
        "manifest": manifest.__dict__,
    }

    def download(
        _connection: object,
        _remote: object,
        local: Path,
        **_kwargs: object,
    ) -> bool:
        local.parent.mkdir(parents=True, exist_ok=True)
        if local.name == "artifact_manifest.json":
            local.write_bytes(artifact_manifest.read_bytes())
        elif local.name == "artifacts.tar.gz":
            local.write_bytes(archive.read_bytes())
        else:
            local.write_text("", encoding="utf-8")
        return True

    with (
        patch("vast.ops.ssh_url", return_value=("root", "host", 22)),
        patch("vast.ops.scp_download", side_effect=download),
        pytest.raises(OpsError, match="exact integer exit_code"),
    ):
        vast_ops.pull_and_verify(record, destination=tmp_path / "downloaded")


def _ordinary_destroy_record(manifest: RunManifest, instance_id: int = 1234) -> dict:
    return {
        "run_id": manifest.run_id,
        "created_at": utc_now(),
        "dph": manifest.max_dph,
        "instance_id": instance_id,
        "status": "running",
        "manifest": manifest.__dict__,
    }


def test_cleanup_reserve_covers_bounded_destroy_and_followup_window() -> None:
    reserve = cleanup_reserve_hours(60)
    assert reserve >= 0.3
    assert cleanup_reserve_hours(600) > reserve


def test_cleanup_reserve_is_exact_sum_of_declared_call_slots() -> None:
    poll = 60
    handoff = (
        vast_ops.SUPERVISOR_LAUNCH_MAX_ATTEMPTS
        * vast_ops.SUPERVISOR_HANDSHAKE_TIMEOUT_SECONDS
        + (vast_ops.SUPERVISOR_LAUNCH_MAX_ATTEMPTS - 1)
        * vast_ops.SUPERVISOR_LAUNCH_RETRY_DELAY_SECONDS
    )
    expected_seconds = (
        vast_ops.HARD_LIMIT_RECOVERY_RESERVE_SECONDS
        + vast_ops.PURCHASE_FINAL_SCAN_RESERVE_SECONDS
        + destroy_attempt_budget_seconds()
        + destroy_reconciliation_lease_seconds(poll)
        + handoff
        + vast_ops.REMOTE_WATCHDOG_SAFETY_SECONDS
        + vast_ops.LOCAL_WATCHDOG_ORDERING_MARGIN_SECONDS
    )
    assert cleanup_reserve_hours(poll) * 3600 == pytest.approx(expected_seconds)


def test_destroy_budget_includes_exact_refresh_and_state_lock_slots() -> None:
    expected = (
        vast_ops.DESTROY_VERIFY_MAX_ATTEMPTS
        * vast_ops.DESTROY_API_TIMEOUT_SECONDS
        + vast_ops.DESTROY_VERIFY_MAX_ATTEMPTS
        * vast_ops.STATE_LOCK_TIMEOUT_SECONDS
        + vast_ops.DESTROY_MAX_ATTEMPTS
        * vast_ops.DESTROY_API_TIMEOUT_SECONDS
        + (vast_ops.DESTROY_MAX_ATTEMPTS - 1)
        * vast_ops.DESTROY_RETRY_DELAY_SECONDS
        + vast_ops.DESTROY_VERIFY_MAX_ATTEMPTS
        * vast_ops.DESTROY_API_TIMEOUT_SECONDS
        + (vast_ops.DESTROY_VERIFY_MAX_ATTEMPTS - 1)
        * vast_ops.DESTROY_VERIFY_RETRY_DELAY_SECONDS
    )
    assert destroy_attempt_budget_seconds() == expected


def test_campaign_schedule_orders_every_boundary_from_manifest_ceiling() -> None:
    manifest = adoption_manifest()
    started = "2026-09-01T00:00:00Z"
    schedule = campaign_deadline_schedule(
        manifest, campaign_started_at=started
    )
    start = datetime.fromisoformat(started.replace("Z", "+00:00"))
    provider = datetime.fromisoformat(
        schedule["provider_deadline_at"].replace("Z", "+00:00")
    )
    watchdog = datetime.fromisoformat(
        schedule["remote_watchdog_deadline_at"].replace("Z", "+00:00")
    )
    controller = datetime.fromisoformat(
        schedule["controller_deadline_at"].replace("Z", "+00:00")
    )
    cleanup = datetime.fromisoformat(
        schedule["cleanup_start_at"].replace("Z", "+00:00")
    )
    recovery = datetime.fromisoformat(
        schedule["recovery_deadline_at"].replace("Z", "+00:00")
    )

    # The $1 cap at the approved $0.50/h ceiling and the two-hour TTL agree.
    assert (provider - start).total_seconds() == 2 * 3600
    assert (provider - watchdog).total_seconds() == 120
    assert (watchdog - controller).total_seconds() == 10
    assert (recovery - cleanup).total_seconds() == 300
    assert cleanup < recovery < controller < watchdog < provider


def test_campaign_schedule_subtracts_sunk_cost_without_using_cheaper_quote() -> None:
    manifest = adoption_manifest()
    started = (datetime.now(UTC) - timedelta(minutes=5)).isoformat().replace(
        "+00:00", "Z"
    )
    before = datetime.now(UTC)
    schedule = campaign_deadline_schedule(
        manifest,
        campaign_started_at=started,
        campaign_accrued_cost_usd=0.25,
    )
    after = datetime.now(UTC)
    provider = datetime.fromisoformat(
        schedule["provider_deadline_at"].replace("Z", "+00:00")
    )
    expected_remaining = (manifest.max_total_usd - 0.25) / manifest.max_dph
    assert before + timedelta(hours=expected_remaining) <= provider
    assert provider <= after + timedelta(hours=expected_remaining)


def test_destroy_requires_live_api_absence_before_terminal_state(tmp_path: Path) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    state_path.write_text(json.dumps([_ordinary_destroy_record(manifest)]), encoding="utf-8")
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.run_vast", side_effect=["destroyed", []]) as api,
        patch("vast.ops.launch_detached_destroy_reconciliation", return_value=44),
    ):
        destroy_instance(1234, manifest.run_id, "test")
    assert api.call_args_list == [
        (("destroy", "instance", 1234, "-y"), {"raw": False, "timeout": 180}),
        (("show", "instances"), {"timeout": 180}),
    ]
    updated = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert updated["status"] == "destroy_followup_pending"
    assert updated["destroy_verified_absent_at"]
    assert updated["cleanup_required"] is True


def test_duplicate_cleanup_records_cannot_bypass_lifecycle_authority(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    record = _ordinary_destroy_record(manifest)
    state_path = tmp_path / "instances.json"
    state_path.write_text(json.dumps([record, dict(record)]), encoding="utf-8")
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.run_vast") as provider,
        pytest.raises(OpsError, match="expected one tracked record"),
    ):
        destroy_instance(1234, manifest.run_id, "test")

    provider.assert_not_called()


def test_process_death_after_absence_cannot_leave_false_clean_terminal_state(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    state_path.write_text(
        json.dumps([_ordinary_destroy_record(manifest)]), encoding="utf-8"
    )
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.run_vast", side_effect=["destroyed", []]),
        patch(
            "vast.ops._launch_destroy_reconciliation_with_retries",
            side_effect=KeyboardInterrupt("simulated parent death"),
        ),
        pytest.raises(KeyboardInterrupt, match="simulated parent death"),
    ):
        destroy_instance(1234, manifest.run_id, "test")
    updated = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert updated["status"] == "destroy_followup_pending"
    assert updated["cleanup_required"] is True
    assert updated["destroy_verified_absent_at"]


def test_destroy_mutation_reserves_a_full_absence_query_slot(tmp_path: Path) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    state_path.write_text(
        json.dumps([_ordinary_destroy_record(manifest)]), encoding="utf-8"
    )
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.time.monotonic", return_value=0.0),
        patch("vast.ops.run_vast", side_effect=["ok", []]) as api,
        patch("vast.ops.launch_detached_destroy_reconciliation", return_value=44),
    ):
        destroy_instance(
            1234,
            manifest.run_id,
            "test",
            deadline_monotonic=200.0,
        )

    assert api.call_args_list[0].kwargs["timeout"] == 20
    assert api.call_args_list[1].kwargs["timeout"] == 180


def test_deadline_timeout_floors_and_refuses_subsecond_calls() -> None:
    from vast.ops import _deadline_timeout_seconds

    with patch("vast.ops.time.monotonic", return_value=100.0):
        assert _deadline_timeout_seconds(101.9, default=180) == 1
        with pytest.raises(OpsError, match="deadline has elapsed"):
            _deadline_timeout_seconds(100.9, default=180)


def test_destroy_retries_when_acknowledged_instance_remains_live(tmp_path: Path) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    state_path.write_text(json.dumps([_ordinary_destroy_record(manifest)]), encoding="utf-8")
    live = [{"id": 1234}]
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.run_vast", side_effect=["ok", live, "ok", []]) as api,
        patch("vast.ops.time.sleep"),
        patch("vast.ops.launch_detached_destroy_reconciliation", return_value=44),
    ):
        destroy_instance(1234, manifest.run_id, "test")
    assert [call.args[:4] for call in api.call_args_list] == [
        ("destroy", "instance", 1234, "-y"),
        ("show", "instances"),
        ("destroy", "instance", 1234, "-y"),
        ("show", "instances"),
    ]
    updated = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert updated["status"] == "destroy_followup_pending"
    assert updated["cleanup_required"] is True


def test_destroy_event_failure_cannot_stop_next_provider_retry(tmp_path: Path) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    state_path.write_text(
        json.dumps([_ordinary_destroy_record(manifest)]), encoding="utf-8"
    )
    live = [{"id": 1234}]
    real_append = vast_ops.append_local_event

    def fail_positive_event(run_id: str, event: str, **fields: object) -> None:
        if event == "destroy_absence_not_verified":
            raise OSError("event replica unavailable")
        real_append(run_id, event, **fields)

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.run_vast", side_effect=["ok", live, "ok", []]) as api,
        patch("vast.ops.append_local_event", side_effect=fail_positive_event),
        patch("vast.ops.time.sleep"),
        patch("vast.ops.launch_detached_destroy_reconciliation", return_value=44),
    ):
        destroy_instance(1234, manifest.run_id, "test")

    assert api.call_count == 4


def test_destroy_secondary_state_failure_cannot_stop_next_provider_retry(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    state_path.write_text(
        json.dumps([_ordinary_destroy_record(manifest)]), encoding="utf-8"
    )
    live = [{"id": 1234}]
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.run_vast", side_effect=["ok", live, "ok", []]) as api,
        patch(
            "vast.ops._mark_cleanup_required",
            side_effect=OSError("secondary state unavailable"),
        ),
        patch("vast.ops.time.sleep"),
        patch("vast.ops.DESTROY_MAX_ATTEMPTS", 2),
        patch("vast.ops.DESTROY_VERIFY_MAX_ATTEMPTS", 2),
        patch("vast.ops.launch_detached_destroy_reconciliation", return_value=44),
        pytest.raises(OpsError, match="could not verify destruction"),
    ):
        destroy_instance(1234, manifest.run_id, "test")

    assert api.call_count == 4
    updated = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert updated["cleanup_state_integrity_failed"] is True
    assert updated["cleanup_required"] is True
    assert updated["status"] != "destroyed"


def test_destroy_live_positive_advances_cleanup_generation(tmp_path: Path) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    state_path.write_text(
        json.dumps([_ordinary_destroy_record(manifest)]), encoding="utf-8"
    )
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.run_vast", side_effect=["ok", [{"id": 1234}]]),
        patch("vast.ops.DESTROY_MAX_ATTEMPTS", 1),
        patch("vast.ops.DESTROY_VERIFY_MAX_ATTEMPTS", 1),
        patch("vast.ops.launch_detached_destroy_reconciliation", return_value=44),
        pytest.raises(OpsError, match="could not verify destruction"),
    ):
        destroy_instance(1234, manifest.run_id, "test")
    updated = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert updated["cleanup_required"] is True
    assert updated["cleanup_generation"] >= 3


def test_competing_live_positive_fences_older_absence_terminal_write(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    state_path.write_text(
        json.dumps([_ordinary_destroy_record(manifest)]), encoding="utf-8"
    )
    a_destroy_started = threading.Event()
    b_destroy_started = threading.Event()
    a_query_started = threading.Event()
    b_positive_recorded = threading.Event()
    errors: list[BaseException] = []

    def provider(*args: object, **_kwargs: object) -> object:
        name = threading.current_thread().name
        if args[:2] == ("destroy", "instance"):
            if name == "destroy-a":
                a_destroy_started.set()
                assert b_destroy_started.wait(timeout=2)
            else:
                b_destroy_started.set()
            return "ok"
        if name == "destroy-a":
            a_query_started.set()
            assert b_positive_recorded.wait(timeout=2)
            return []
        assert a_query_started.wait(timeout=2)
        return [{"id": 1234}]

    original_mark = vast_ops._mark_cleanup_required

    def mark_and_signal(*args: object, **kwargs: object) -> None:
        original_mark(*args, **kwargs)
        if threading.current_thread().name == "destroy-b":
            b_positive_recorded.set()

    def cleanup() -> None:
        try:
            destroy_instance(
                1234,
                manifest.run_id,
                "competing cleanup",
                launch_supervisor=False,
            )
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.run_vast", side_effect=provider),
        patch("vast.ops._mark_cleanup_required", side_effect=mark_and_signal),
        patch("vast.ops.DESTROY_MAX_ATTEMPTS", 1),
        patch("vast.ops.DESTROY_VERIFY_MAX_ATTEMPTS", 1),
    ):
        first = threading.Thread(target=cleanup, name="destroy-a")
        second = threading.Thread(target=cleanup, name="destroy-b")
        first.start()
        assert a_destroy_started.wait(timeout=2)
        second.start()
        first.join(timeout=5)
        second.join(timeout=5)
    assert not first.is_alive() and not second.is_alive()
    assert len(errors) == 2
    assert all(isinstance(error, OpsError) for error in errors)
    updated = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert updated["status"] == "cleanup_required"
    assert updated["cleanup_required"] is True


def test_destroy_query_failure_persists_cleanup_and_starts_supervisor(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    state_path.write_text(json.dumps([_ordinary_destroy_record(manifest)]), encoding="utf-8")
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.run_vast", side_effect=OpsError("API unavailable")),
        patch("vast.ops.time.sleep"),
        patch("vast.ops.DESTROY_MAX_ATTEMPTS", 1),
        patch("vast.ops.DESTROY_VERIFY_MAX_ATTEMPTS", 1),
        patch("vast.ops.launch_detached_destroy_reconciliation", return_value=44) as supervisor,
        pytest.raises(OpsError, match="could not verify destruction"),
    ):
        destroy_instance(1234, manifest.run_id, "test")
    updated = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert updated["status"] == "cleanup_required"
    assert updated["cleanup_required"] is True
    supervisor.assert_called_once_with(manifest.run_id)


def test_destroy_failure_redacts_provider_and_supervisor_errors(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    event_path = tmp_path / "events.jsonl"
    state_path.write_text(
        json.dumps([_ordinary_destroy_record(manifest)]), encoding="utf-8"
    )
    secret = "destroy-path-secret"

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.EVENT_FILE", event_path),
        patch("vast.ops.load_secrets", return_value={"VAST_API_KEY": secret}),
        patch(
            "vast.ops.run_vast",
            side_effect=OpsError(f"provider api_key={secret}"),
        ),
        patch("vast.ops.DESTROY_MAX_ATTEMPTS", 1),
        patch("vast.ops.DESTROY_VERIFY_MAX_ATTEMPTS", 1),
        patch(
            "vast.ops._launch_destroy_reconciliation_with_retries",
            side_effect=OpsError(f"supervisor api_key={secret}"),
        ),
        pytest.raises(OpsError) as raised,
    ):
        destroy_instance(1234, manifest.run_id, "test")

    assert secret not in str(raised.value)
    assert secret not in state_path.read_text(encoding="utf-8")
    assert secret not in event_path.read_text(encoding="utf-8")
    assert "[DIAGNOSTIC WITHHELD]" in state_path.read_text(encoding="utf-8")


def test_destroy_terminal_write_cannot_clear_newer_cleanup_evidence(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    state_path.write_text(
        json.dumps([_ordinary_destroy_record(manifest)]), encoding="utf-8"
    )

    provider_calls = [0]

    def destroy_then_absence(*_args: object, **_kwargs: object) -> object:
        provider_calls[0] += 1
        if provider_calls[0] == 1:
            return "destroyed"
        vast_ops._mark_cleanup_required(
            1234, manifest.run_id, "concurrent provider-positive observation"
        )
        return []

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.run_vast", side_effect=destroy_then_absence),
        patch("vast.ops.DESTROY_MAX_ATTEMPTS", 1),
        patch("vast.ops.DESTROY_VERIFY_MAX_ATTEMPTS", 1),
        patch("vast.ops.launch_detached_destroy_reconciliation", return_value=44),
        pytest.raises(OpsError, match="could not verify destruction"),
    ):
        destroy_instance(1234, manifest.run_id, "test")
    updated = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert updated["status"] == "cleanup_required"
    assert updated["cleanup_required"] is True
    assert updated["cleanup_generation"] >= 2


def test_nonterminal_purchase_cleanup_waits_for_inflight_bootstrap_barrier(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    state_path.write_text(
        json.dumps(
            [
                {
                    **_ordinary_destroy_record(manifest),
                    "status": "purchase_reconciled_cleanup_pending",
                    "cleanup_required": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    started = threading.Event()
    finished = threading.Event()
    errors: list[BaseException] = []

    def cleanup() -> None:
        started.set()
        try:
            destroy_instance(
                1234,
                manifest.run_id,
                "purchase_outcome_reconciled",
                launch_supervisor=False,
                mark_terminal=False,
            )
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)
        finally:
            finished.set()

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.run_vast", side_effect=["destroyed", []]) as provider,
    ):
        with vast_ops._lifecycle_transition_lock(manifest.run_id):
            worker = threading.Thread(target=cleanup)
            worker.start()
            assert started.wait(timeout=2)
            time.sleep(0.1)
            assert not finished.is_set()
            provider.assert_not_called()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert errors == []
    assert provider.call_count == 2
    updated = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert updated["cleanup_required"] is True
    assert updated["status"] == "cleanup_in_progress"


def test_destroy_reconciliation_catches_reappearing_instance(tmp_path: Path) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    state_path.write_text(
        json.dumps(
            [
                {
                    **_ordinary_destroy_record(manifest),
                    "status": "destroyed",
                    "cleanup_required": False,
                }
            ]
        ),
        encoding="utf-8",
    )
    live = [{"id": 1234}]
    virtual_now = [0.0]

    def monotonic() -> float:
        return virtual_now[0]

    def advance(seconds: float) -> None:
        virtual_now[0] += seconds

    responses = iter([[], live, "ok", [], [], []])

    def fake_vast(*_: object, **__: object) -> object:
        return next(responses, [])

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.run_vast", side_effect=fake_vast),
        patch("vast.ops.time.sleep", side_effect=advance),
        patch("vast.ops.time.monotonic", side_effect=monotonic),
        patch("vast.ops.DESTROY_RECONCILIATION_MIN_SECONDS", 0),
        patch("vast.ops.DESTROY_RECONCILIATION_POLL_SECONDS", 1),
        patch("vast.ops.DESTROY_API_TIMEOUT_SECONDS", 1),
    ):
        assert _destroy_reconciliation(manifest.run_id) == 0
    updated = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert updated["status"] == "destroyed"
    assert updated["cleanup_required"] is False
    assert updated["destroy_reconciliation_verified_at"]


def test_destroy_reconciliation_final_absence_cannot_clear_newer_positive(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    state_path.write_text(
        json.dumps(
            [
                {
                    **_ordinary_destroy_record(manifest),
                    "status": "destroyed",
                    "cleanup_required": False,
                }
            ]
        ),
        encoding="utf-8",
    )
    virtual_now = [0.0]

    def monotonic() -> float:
        return virtual_now[0]

    def advance(seconds: float) -> None:
        virtual_now[0] += seconds

    complete = vast_ops._complete_destroy_reconciliation_if_current

    def competing_completion(
        run_id: str, *, expected_cleanup_generation: int
    ) -> bool:
        vast_ops._mark_cleanup_required(
            1234, run_id, "provider-positive observation after final absence"
        )
        return complete(
            run_id, expected_cleanup_generation=expected_cleanup_generation
        )

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.run_vast", return_value=[]),
        patch("vast.ops.time.sleep", side_effect=advance),
        patch("vast.ops.time.monotonic", side_effect=monotonic),
        patch("vast.ops.DESTROY_RECONCILIATION_MIN_SECONDS", 0),
        patch("vast.ops.DESTROY_RECONCILIATION_POLL_SECONDS", 1),
        patch("vast.ops.DESTROY_API_TIMEOUT_SECONDS", 1),
        patch(
            "vast.ops._complete_destroy_reconciliation_if_current",
            side_effect=competing_completion,
        ),
    ):
        assert _destroy_reconciliation(manifest.run_id) == 2
    updated = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert updated["status"] == "cleanup_required"
    assert updated["cleanup_required"] is True


def test_destroy_reconciliation_stale_expiry_preserves_newer_lifecycle_state(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    event_path = tmp_path / "events.jsonl"
    state_path.write_text(
        json.dumps(
            [
                {
                    **_ordinary_destroy_record(manifest),
                    "status": "destroy_followup_pending",
                    "cleanup_required": True,
                    "cleanup_generation": 4,
                    "destroy_reconciliation_status": "running",
                }
            ]
        ),
        encoding="utf-8",
    )
    virtual_now = [0.0]

    def monotonic() -> float:
        return virtual_now[0]

    def advance(seconds: float) -> None:
        virtual_now[0] += seconds

    complete = vast_ops._complete_destroy_reconciliation_if_current

    def competing_completion(
        run_id: str, *, expected_cleanup_generation: int
    ) -> bool:
        vast_ops.update_record_fields(
            run_id,
            status="cleanup_in_progress",
            cleanup_required=True,
            cleanup_error="newer provider-positive observation",
            destroy_reconciliation_status="newer_cleanup_owner",
            destroy_reconciliation_diagnostic="newer lifecycle evidence",
            destroy_reconciliation_verified_at="newer-verification",
        )
        return complete(
            run_id, expected_cleanup_generation=expected_cleanup_generation
        )

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops.EVENT_FILE", event_path),
        patch("vast.ops.run_vast", return_value=[]),
        patch("vast.ops.time.sleep", side_effect=advance),
        patch("vast.ops.time.monotonic", side_effect=monotonic),
        patch("vast.ops.DESTROY_RECONCILIATION_MIN_SECONDS", 0),
        patch("vast.ops.DESTROY_RECONCILIATION_POLL_SECONDS", 1),
        patch("vast.ops.DESTROY_API_TIMEOUT_SECONDS", 1),
        patch(
            "vast.ops._complete_destroy_reconciliation_if_current",
            side_effect=competing_completion,
        ),
    ):
        assert _destroy_reconciliation(manifest.run_id) == 2

    updated = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert updated["cleanup_generation"] == 5
    assert updated["status"] == "cleanup_in_progress"
    assert updated["cleanup_required"] is True
    assert updated["cleanup_error"] == "newer provider-positive observation"
    assert updated["destroy_reconciliation_status"] == "newer_cleanup_owner"
    assert (
        updated["destroy_reconciliation_diagnostic"]
        == "newer lifecycle evidence"
    )
    assert updated["destroy_reconciliation_verified_at"] == "newer-verification"
    assert "destroy_reconciliation_expired_at" not in updated

    events = [
        json.loads(line)
        for line in event_path.read_text(encoding="utf-8").splitlines()
    ]
    event_names = [event["event"] for event in events]
    assert "destroy_reconciliation_terminal_write_fenced" in event_names
    assert (
        "destroy_reconciliation_expiry_suppressed_after_terminal_refusal"
        in event_names
    )
    assert "destroy_reconciliation_expired" not in event_names


def test_destroy_reconciliation_integrity_refusal_preserves_lifecycle_state(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    event_path = tmp_path / "events.jsonl"
    record = {
        **_ordinary_destroy_record(manifest),
        "status": "destroy_followup_pending",
        "cleanup_required": True,
        "cleanup_generation": 4,
        "cleanup_error": "authoritative state persistence failed",
        "cleanup_state_integrity_failed": True,
        "cleanup_state_integrity_error": "newer diagnostic must survive",
        "destroy_reconciliation_status": "integrity_failure_owned",
        "destroy_reconciliation_diagnostic": "preserve this diagnostic",
        "destroy_reconciliation_verified_at": "preserve-this-timestamp",
    }
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    virtual_now = [0.0]

    def monotonic() -> float:
        return virtual_now[0]

    def advance(seconds: float) -> None:
        virtual_now[0] += seconds

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops.EVENT_FILE", event_path),
        patch("vast.ops.run_vast", return_value=[]),
        patch("vast.ops.time.sleep", side_effect=advance),
        patch("vast.ops.time.monotonic", side_effect=monotonic),
        patch("vast.ops.DESTROY_RECONCILIATION_MIN_SECONDS", 0),
        patch("vast.ops.DESTROY_RECONCILIATION_POLL_SECONDS", 1),
        patch("vast.ops.DESTROY_API_TIMEOUT_SECONDS", 1),
    ):
        assert _destroy_reconciliation(manifest.run_id) == 2

    updated = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert updated == record

    events = [
        json.loads(line)
        for line in event_path.read_text(encoding="utf-8").splitlines()
    ]
    event_names = [event["event"] for event in events]
    assert "destroy_reconciliation_terminal_write_fenced" in event_names
    assert (
        "destroy_reconciliation_expiry_suppressed_after_terminal_refusal"
        in event_names
    )
    assert "destroy_reconciliation_expired" not in event_names


def test_cleanup_state_integrity_failure_forbids_terminal_clear(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    record = {
        **_ordinary_destroy_record(manifest),
        "status": "destroy_followup_pending",
        "cleanup_required": True,
        "cleanup_generation": 4,
        "cleanup_state_integrity_failed": True,
    }
    state_path = tmp_path / "instances.json"
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
    ):
        assert not vast_ops._complete_destroy_reconciliation_if_current(
            manifest.run_id,
            expected_cleanup_generation=4,
        )

    updated = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert updated["cleanup_required"] is True
    assert updated["status"] == "destroy_followup_pending"


def test_destroy_reconciliation_state_diagnostic_failure_cannot_precede_destroy(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    state_path.write_text(
        json.dumps(
            [
                {
                    **_ordinary_destroy_record(manifest),
                    "status": "cleanup_required",
                    "cleanup_required": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    virtual_now = [0.0]

    def monotonic() -> float:
        return virtual_now[0]

    def advance(seconds: float) -> None:
        virtual_now[0] += seconds

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.run_vast", return_value=[{"id": 1234}]),
        patch("vast.ops._cleanup_generation", side_effect=OSError("state unavailable")),
        patch(
            "vast.ops._mark_cleanup_required",
            side_effect=OSError("state unavailable"),
        ),
        patch("vast.ops.destroy_instance", side_effect=OpsError("retry pending")) as destroy,
        patch("vast.ops.time.sleep", side_effect=advance),
        patch("vast.ops.time.monotonic", side_effect=monotonic),
        patch("vast.ops.DESTROY_RECONCILIATION_MIN_SECONDS", 0),
        patch("vast.ops.DESTROY_RECONCILIATION_POLL_SECONDS", 1),
        patch("vast.ops.DESTROY_API_TIMEOUT_SECONDS", 1),
    ):
        assert _destroy_reconciliation(manifest.run_id) == 2

    destroy.assert_called()


def test_destroy_supervisor_spawn_failure_uses_bounded_attached_fallback(
    tmp_path: Path,
) -> None:
    from vast.ops import _launch_destroy_reconciliation_with_retries

    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    state_path.write_text(
        json.dumps([_ordinary_destroy_record(manifest)]), encoding="utf-8"
    )
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch(
            "vast.ops.launch_detached_destroy_reconciliation",
            side_effect=OpsError("spawn unavailable"),
        ) as launch,
        patch("vast.ops._destroy_reconciliation", return_value=0) as attached,
        patch("vast.ops.SUPERVISOR_LAUNCH_RETRY_DELAY_SECONDS", 0),
    ):
        assert (
            _launch_destroy_reconciliation_with_retries(
                manifest.run_id, instance_id=1234
            )
            == os.getpid()
        )
    assert launch.call_count == vast_ops.SUPERVISOR_LAUNCH_MAX_ATTEMPTS
    attached.assert_called_once_with(manifest.run_id)


def test_destroy_supervisor_event_failure_cannot_block_attached_fallback(
    tmp_path: Path,
) -> None:
    from vast.ops import _launch_destroy_reconciliation_with_retries

    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    state_path.write_text(
        json.dumps([_ordinary_destroy_record(manifest)]), encoding="utf-8"
    )
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch(
            "vast.ops.launch_detached_destroy_reconciliation",
            side_effect=OpsError("spawn unavailable"),
        ) as launch,
        patch("vast.ops.append_local_event", side_effect=OSError("event unavailable")),
        patch("vast.ops._destroy_reconciliation", return_value=0) as attached,
        patch("vast.ops.SUPERVISOR_LAUNCH_RETRY_DELAY_SECONDS", 0),
    ):
        assert (
            _launch_destroy_reconciliation_with_retries(
                manifest.run_id, instance_id=1234
            )
            == os.getpid()
        )

    assert launch.call_count == vast_ops.SUPERVISOR_LAUNCH_MAX_ATTEMPTS
    attached.assert_called_once_with(manifest.run_id)


def test_supervisor_lock_uses_os_lease_before_pid_metadata(tmp_path: Path) -> None:
    lock = tmp_path / "supervisor.lock"
    with patch("vast.ops.process_is_alive", return_value=True):
        _claim_supervisor_lock(lock, "test")
    assert _supervisor_lock_owner_pid(lock) == os.getpid()
    _release_supervisor_lock(lock)
    assert _supervisor_lock_owner_pid(lock) is None


@pytest.mark.skipif(os.name != "nt", reason="exercises Windows byte-range locking")
def test_losing_windows_supervisor_claim_cannot_corrupt_owner_metadata(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "supervisor.lock"
    source = (
        "import sys,time\n"
        "from pathlib import Path\n"
        "from vast.ops import _claim_supervisor_lock\n"
        "_claim_supervisor_lock(Path(sys.argv[1]), 'child')\n"
        "print('READY', flush=True)\n"
        "time.sleep(30)\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", source, str(lock)],
        cwd=vast_ops.ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "READY"
        with lock.open("rb") as stream:
            stream.seek(1)
            before = stream.read()
        with pytest.raises(OpsError, match="already supervised"):
            _claim_supervisor_lock(lock, "contender")
        with lock.open("rb") as stream:
            stream.seek(1)
            assert stream.read() == before
        assert _supervisor_lock_owner_pid(lock) == child.pid
    finally:
        child.terminate()
        child.wait(timeout=10)


def test_attached_monitor_registers_the_actual_lease_owner(tmp_path: Path) -> None:
    with (
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops._claim_supervisor_lock"),
        patch("vast.ops._release_supervisor_lock"),
        patch("vast.ops._monitor_run", return_value=0),
        patch("vast.ops.update_record_fields") as update,
    ):
        assert monitor_run("attached-test") == 0
    update.assert_called_once()
    assert update.call_args.args == ("attached-test",)
    assert update.call_args.kwargs["monitor_pid"] == os.getpid()
    assert update.call_args.kwargs["monitor_registered_at"]


def test_unknown_single_purchase_starts_reconciliation_supervisor(tmp_path: Path) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    args = SimpleNamespace(manifest=tmp_path / "run.json", offer=None, detach=True)
    offer = {
        "id": 11,
        "machine_id": 101,
        "dph_total": 0.3,
    }
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.load_manifest", return_value=manifest),
        patch(
            "vast.ops.create_source_bundle",
            return_value=(tmp_path / "source.bundle", tmp_path / "run.json"),
        ),
        patch("vast.ops.load_machine_blacklist", return_value=set()),
        patch("vast.ops.active_machine_ids", return_value=set()),
        patch("vast.ops.run_vast", return_value=[]),
        patch("vast.ops.search_offers", return_value=[offer]),
        patch("vast.ops.create_instance", side_effect=OpsError("create timeout")),
        patch("vast.ops.launch_detached_purchase_reconciliation", return_value=9001) as supervisor,
        pytest.raises(OpsError, match="create timeout"),
    ):
        command_launch(args)
    supervisor.assert_called_once_with(manifest.run_id)
    assert json.loads(state_path.read_text(encoding="utf-8"))[0]["status"] == (
        "purchase_outcome_unknown"
    )


def test_purchase_reconciliation_destroys_delayed_labeled_instance(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    record = {
        "run_id": manifest.run_id,
        "expected_label": f"quant-experiments-{manifest.run_id}",
        "offer_id": 11,
        "machine_id": 101,
        "status": "purchase_outcome_unknown",
        "late_create_reconciliation_deadline_at": utc_now(),
        "manifest": manifest.__dict__,
    }
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    live = [
        {
            "id": 777,
            "label": record["expected_label"],
            "ask_contract_id": record["offer_id"],
            "machine_id": record["machine_id"],
        }
    ]
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops.run_vast", side_effect=[live, []]) as api,
        patch("vast.ops.destroy_instance") as destroy,
        patch("vast.ops.PURCHASE_RECONCILIATION_TIMEOUT_SECONDS", 0),
    ):
        assert _purchase_reconciliation(manifest.run_id) == 2
    # The primary lease is expired, but its explicitly reserved final scan still
    # catches and destroys the exact delayed contract. Provider closure remains
    # unprovable, so the record stays fail-closed for manual reconciliation.
    assert api.call_count == 2
    assert all(call.args == ("show", "instances") for call in api.call_args_list)
    assert all(
        1 <= call.kwargs["timeout"] <= 180 for call in api.call_args_list
    )
    destroy.assert_called_once()
    assert destroy.call_args.args == (
        777,
        manifest.run_id,
        "purchase_outcome_reconciled",
    )
    assert destroy.call_args.kwargs["launch_supervisor"] is False
    assert destroy.call_args.kwargs["mark_terminal"] is False


def test_purchase_reconciliation_event_failure_cannot_precede_destroy(
    tmp_path: Path,
) -> None:
    match = {
        "id": 777,
        "label": "quant-experiments-uncertain-create",
        "ask_contract_id": 11,
        "machine_id": 101,
    }
    destroyed_ids: set[int] = set()
    state_path = tmp_path / "instances.json"
    state_path.write_text(
        json.dumps(
            [
                {
                    "run_id": "uncertain-create",
                    "status": "purchase_pending",
                    "expected_label": match["label"],
                    "offer_id": 11,
                    "machine_id": 101,
                }
            ]
        ),
        encoding="utf-8",
    )
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.append_local_event", side_effect=OSError("event unavailable")),
        patch("vast.ops.destroy_instance") as destroy,
    ):
        observed, all_destroyed = vast_ops._drain_reconcilable_purchase_instances(
            "uncertain-create",
            [match],
            reason="purchase_outcome_reconciled",
            deadline_monotonic=time.monotonic() + 60,
            destroyed_ids=destroyed_ids,
        )

    assert observed is True
    assert all_destroyed is True
    destroy.assert_called_once()
    assert destroyed_ids == {777}


def test_exact_purchase_cleanup_reaches_provider_when_claim_write_fails(
    tmp_path: Path,
) -> None:
    run_id = "uncertain-create"
    match = {
        "id": 777,
        "label": f"quant-experiments-{run_id}",
        "ask_contract_id": 11,
        "machine_id": 101,
    }
    record = {
        "run_id": run_id,
        "status": "purchase_outcome_unknown",
        "expected_label": match["label"],
        "offer_id": 11,
        "machine_id": 101,
    }
    state_path = tmp_path / "instances.json"
    event_path = tmp_path / "events.jsonl"
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    destroyed_ids: set[int] = set()

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.EVENT_FILE", event_path),
        patch(
            "vast.ops._claim_cleanup_authority",
            side_effect=OSError("state write unavailable"),
        ),
        patch("vast.ops.run_vast", side_effect=[[match], "destroyed", []]) as provider,
    ):
        observed, all_destroyed = vast_ops._drain_reconcilable_purchase_instances(
            run_id,
            [match],
            reason="purchase_outcome_reconciled",
            deadline_monotonic=time.monotonic() + 600,
            destroyed_ids=destroyed_ids,
        )

    assert (observed, all_destroyed) == (True, True)
    assert destroyed_ids == {777}
    assert [call.args for call in provider.call_args_list] == [
        ("show", "instances"),
        ("destroy", "instance", 777, "-y"),
        ("show", "instances"),
    ]
    assert "exact_purchase_cleanup_claim_write_failed" in event_path.read_text(
        encoding="utf-8"
    )


def test_unfenced_exact_purchase_cleanup_cannot_bypass_duplicate_state_fence(
    tmp_path: Path,
) -> None:
    run_id = "duplicate-uncertain-create"
    match = {
        "id": 777,
        "label": f"quant-experiments-{run_id}",
        "ask_contract_id": 11,
        "machine_id": 101,
    }
    record = {
        "run_id": run_id,
        "status": "purchase_outcome_unknown",
        "expected_label": match["label"],
        "offer_id": 11,
        "machine_id": 101,
    }
    state_path = tmp_path / "instances.json"
    state_path.write_text(json.dumps([record, dict(record)]), encoding="utf-8")
    destroyed_ids: set[int] = set()

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.run_vast") as provider,
    ):
        observed, all_destroyed = vast_ops._drain_reconcilable_purchase_instances(
            run_id,
            [match],
            reason="purchase_outcome_reconciled",
            deadline_monotonic=time.monotonic() + 600,
            destroyed_ids=destroyed_ids,
        )

    assert (observed, all_destroyed) == (True, False)
    assert destroyed_ids == set()
    provider.assert_not_called()


@pytest.mark.parametrize(
    "record_mutation",
    [
        {"candidates": []},
        {"lifecycle_origin": "existing_instance_adoption"},
        {"lifecycle_origin": "malformed"},
    ],
)
def test_unfenced_purchase_cleanup_rechecks_origin_after_claim_write_failure(
    tmp_path: Path, record_mutation: dict[str, object]
) -> None:
    run_id = "origin-drift"
    match = {
        "id": 777,
        "label": f"quant-experiments-{run_id}",
        "ask_contract_id": 11,
        "machine_id": 101,
    }
    record = {
        "run_id": run_id,
        "status": "purchase_outcome_unknown",
        "expected_label": match["label"],
        "offer_id": 11,
        "machine_id": 101,
    }
    state_path = tmp_path / "instances.json"
    state_path.write_text(json.dumps([record]), encoding="utf-8")

    def mutate_then_fail(*_args: object, **_kwargs: object) -> None:
        state_path.write_text(
            json.dumps([{**record, **record_mutation}]), encoding="utf-8"
        )
        raise OSError("state write unavailable")

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops._claim_cleanup_authority", side_effect=mutate_then_fail),
        patch("vast.ops.run_vast") as provider,
    ):
        observed, all_destroyed = vast_ops._drain_reconcilable_purchase_instances(
            run_id,
            [match],
            reason="purchase_outcome_reconciled",
            deadline_monotonic=time.monotonic() + 600,
            destroyed_ids=set(),
        )

    assert (observed, all_destroyed) == (True, False)
    provider.assert_not_called()


def test_unfenced_purchase_cleanup_rejects_conflicting_offer_aliases(
    tmp_path: Path,
) -> None:
    run_id = "conflicting-offer"
    match = {
        "id": 777,
        "label": f"quant-experiments-{run_id}",
        "ask_contract_id": 11,
        "offer_id": 999,
        "machine_id": 101,
    }
    record = {
        "run_id": run_id,
        "status": "purchase_outcome_unknown",
        "expected_label": match["label"],
        "offer_id": 11,
        "machine_id": 101,
    }
    state_path = tmp_path / "instances.json"
    state_path.write_text(json.dumps([record]), encoding="utf-8")

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.run_vast") as provider,
    ):
        observed, all_destroyed = vast_ops._drain_reconcilable_purchase_instances(
            run_id,
            [match],
            reason="purchase_outcome_reconciled",
            deadline_monotonic=time.monotonic() + 600,
            destroyed_ids=set(),
        )

    assert (observed, all_destroyed) == (True, False)
    provider.assert_not_called()


def test_unfenced_purchase_cleanup_rejects_cross_run_expected_label(
    tmp_path: Path,
) -> None:
    run_id = "right-run"
    match = {
        "id": 777,
        "label": "quant-experiments-wrong-run",
        "ask_contract_id": 11,
        "machine_id": 101,
    }
    record = {
        "run_id": run_id,
        "status": "purchase_outcome_unknown",
        "expected_label": match["label"],
        "offer_id": 11,
        "machine_id": 101,
    }
    state_path = tmp_path / "instances.json"
    state_path.write_text(json.dumps([record]), encoding="utf-8")

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.run_vast") as provider,
    ):
        observed, all_destroyed = vast_ops._drain_reconcilable_purchase_instances(
            run_id,
            [match],
            reason="purchase_outcome_reconciled",
            deadline_monotonic=time.monotonic() + 600,
            destroyed_ids=set(),
        )

    assert (observed, all_destroyed) == (True, False)
    provider.assert_not_called()


def test_unfenced_purchase_cleanup_does_not_redestroy_reused_numeric_id(
    tmp_path: Path,
) -> None:
    run_id = "numeric-id-reuse"
    match = {
        "id": 777,
        "label": f"quant-experiments-{run_id}",
        "ask_contract_id": 11,
        "machine_id": 101,
    }
    record = {
        "run_id": run_id,
        "status": "purchase_outcome_unknown",
        "expected_label": match["label"],
        "offer_id": 11,
        "machine_id": 101,
    }
    drifted = {
        "id": 777,
        "label": "another-owner",
        "ask_contract_id": 999,
        "machine_id": 202,
    }
    state_path = tmp_path / "instances.json"
    event_path = tmp_path / "events.jsonl"
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    destroyed_ids: set[int] = set()

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.EVENT_FILE", event_path),
        patch(
            "vast.ops._claim_cleanup_authority",
            side_effect=OSError("state write unavailable"),
        ),
        patch(
            "vast.ops.run_vast",
            side_effect=[[match], "destroyed", [drifted]],
        ) as provider,
    ):
        observed, all_destroyed = vast_ops._drain_reconcilable_purchase_instances(
            run_id,
            [match],
            reason="purchase_outcome_reconciled",
            deadline_monotonic=time.monotonic() + 600,
            destroyed_ids=destroyed_ids,
        )

    assert (observed, all_destroyed) == (True, True)
    assert destroyed_ids == {777}
    assert [call.args for call in provider.call_args_list] == [
        ("show", "instances"),
        ("destroy", "instance", 777, "-y"),
        ("show", "instances"),
    ]
    assert "exact_purchase_identity_drift_after_destroy" in event_path.read_text(
        encoding="utf-8"
    )


def test_unfenced_purchase_cleanup_refreshes_after_uncertain_absence_query(
    tmp_path: Path,
) -> None:
    run_id = "refresh-after-query-error"
    match = {
        "id": 777,
        "label": f"quant-experiments-{run_id}",
        "ask_contract_id": 11,
        "machine_id": 101,
    }
    drifted = {
        "id": 777,
        "label": "new-owner",
        "ask_contract_id": 99,
        "machine_id": 202,
    }
    record = {
        "run_id": run_id,
        "status": "purchase_outcome_unknown",
        "expected_label": match["label"],
        "offer_id": 11,
        "machine_id": 101,
    }
    state_path = tmp_path / "instances.json"
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops._claim_cleanup_authority", side_effect=OSError("write failed")),
        patch(
            "vast.ops.run_vast",
            side_effect=[
                [match],
                "destroyed",
                OSError("absence query unavailable"),
                [drifted],
            ],
        ) as provider,
        patch("vast.ops._bounded_sleep"),
    ):
        vast_ops.destroy_instance(
            777,
            run_id,
            "purchase_outcome_reconciled",
            launch_supervisor=False,
            mark_terminal=False,
            deadline_monotonic=time.monotonic() + 600,
            unfenced_exact_purchase_match=match,
        )

    assert [call.args for call in provider.call_args_list].count(
        ("destroy", "instance", 777, "-y")
    ) == 1
    assert [call.args for call in provider.call_args_list] == [
        ("show", "instances"),
        ("destroy", "instance", 777, "-y"),
        ("show", "instances"),
        ("show", "instances"),
    ]


def test_unfenced_purchase_cleanup_refreshes_after_uncertain_destroy_outcome(
    tmp_path: Path,
) -> None:
    run_id = "refresh-after-destroy-timeout"
    match = {
        "id": 777,
        "label": f"quant-experiments-{run_id}",
        "ask_contract_id": 11,
        "machine_id": 101,
    }
    drifted = {
        "id": 777,
        "label": "new-owner",
        "ask_contract_id": 99,
        "machine_id": 202,
    }
    record = {
        "run_id": run_id,
        "status": "purchase_outcome_unknown",
        "expected_label": match["label"],
        "offer_id": 11,
        "machine_id": 101,
    }
    state_path = tmp_path / "instances.json"
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops._claim_cleanup_authority", side_effect=OSError("write failed")),
        patch(
            "vast.ops.run_vast",
            side_effect=[[match], TimeoutError("destroy outcome unknown"), [drifted]],
        ) as provider,
        patch("vast.ops._bounded_sleep"),
    ):
        vast_ops.destroy_instance(
            777,
            run_id,
            "purchase_outcome_reconciled",
            launch_supervisor=False,
            mark_terminal=False,
            deadline_monotonic=time.monotonic() + 600,
            unfenced_exact_purchase_match=match,
        )

    assert [call.args for call in provider.call_args_list] == [
        ("show", "instances"),
        ("destroy", "instance", 777, "-y"),
        ("show", "instances"),
    ]


def test_unfenced_destroy_serializes_final_state_proof_with_mutation(
    tmp_path: Path,
) -> None:
    run_id = "serialized-exact-destroy"
    match = {
        "id": 777,
        "label": f"quant-experiments-{run_id}",
        "ask_contract_id": 11,
        "machine_id": 101,
    }
    record = {
        "run_id": run_id,
        "status": "purchase_outcome_unknown",
        "expected_label": match["label"],
        "offer_id": 11,
        "machine_id": 101,
    }
    state_path = tmp_path / "instances.json"
    state_lock = tmp_path / "state.lock"
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    show_returned = threading.Event()
    writer_go = threading.Event()
    writer_attempted = threading.Event()
    order: list[str] = []
    show_count = 0

    def provider(*arguments: object, **_kwargs: object) -> object:
        nonlocal show_count
        if arguments[:2] == ("show", "instances"):
            show_count += 1
            order.append(f"show_{show_count}")
            if show_count == 1:
                show_returned.set()
                return [match]
            return []
        order.append("destroy")
        return "destroyed"

    def writer() -> None:
        assert writer_go.wait(timeout=2)
        order.append("writer_attempt")
        writer_attempted.set()
        with vast_ops.state_file_lock():
            order.append("writer_acquired")
            records = vast_ops._load_state_unlocked()
            records[0]["lifecycle_origin"] = "existing_instance_adoption"
            vast_ops._save_state_unlocked(records)

    real_unlocked_check = vast_ops._require_unfenced_purchase_cleanup_authority_unlocked
    signaled = False

    def checked_boundary(*arguments: object, **kwargs: object) -> dict[str, object]:
        nonlocal signaled
        result = real_unlocked_check(*arguments, **kwargs)
        if show_returned.is_set() and not signaled:
            signaled = True
            order.append("final_check_complete")
            writer_go.set()
            assert writer_attempted.wait(timeout=2)
            time.sleep(0.05)
        return result

    worker = threading.Thread(target=writer)
    worker.start()
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", state_lock),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops._claim_cleanup_authority", side_effect=OSError("write failed")),
        patch("vast.ops.run_vast", side_effect=provider),
        patch(
            "vast.ops._require_unfenced_purchase_cleanup_authority_unlocked",
            side_effect=checked_boundary,
        ),
        patch("vast.ops._bounded_sleep"),
    ):
        try:
            vast_ops.destroy_instance(
                777,
                run_id,
                "purchase_outcome_reconciled",
                launch_supervisor=False,
                mark_terminal=False,
                deadline_monotonic=time.monotonic() + 600,
                unfenced_exact_purchase_match=match,
            )
        except OpsError:
            # A writer that wins immediately after the destroy is allowed to
            # fence terminal interpretation, but never the mutation boundary.
            pass
        worker.join(timeout=2)

    assert not worker.is_alive()
    assert order.index("destroy") < order.index("writer_acquired")
    assert "show_2" in order
    assert order.count("destroy") == 1


def test_purchase_sentinel_replica_failures_cannot_block_exact_destroy(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    record = {
        "run_id": manifest.run_id,
        "expected_label": f"quant-experiments-{manifest.run_id}",
        "offer_id": 11,
        "machine_id": 101,
        "status": "purchase_outcome_unknown",
        "late_create_reconciliation_deadline_at": utc_now(),
        "manifest": manifest.__dict__,
    }
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    live = [
        {
            "id": 777,
            "label": record["expected_label"],
            "ask_contract_id": 11,
            "machine_id": 101,
        }
    ]
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops.append_local_event", side_effect=OSError("event unavailable")),
        patch("vast.ops._best_effort_update_record_fields", return_value=False),
        patch("vast.ops.run_vast", side_effect=[live, []]),
        patch("vast.ops.destroy_instance") as destroy,
        patch("vast.ops.PURCHASE_RECONCILIATION_TIMEOUT_SECONDS", 0),
    ):
        assert _purchase_reconciliation(manifest.run_id) == 2

    destroy.assert_called_once()


def test_purchase_reconciliation_recovers_stale_supervisor_lock(tmp_path: Path) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    record = {
        "run_id": manifest.run_id,
        "created_at": utc_now(),
        "expected_label": f"quant-experiments-{manifest.run_id}",
        "dph": 0.3,
        "status": "purchase_outcome_unknown",
        "late_create_reconciliation_deadline_at": utc_now(),
        "manifest": manifest.__dict__,
    }
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    monitor_dir = tmp_path / "monitors"
    monitor_dir.mkdir()
    (monitor_dir / f"purchase_reconcile_{manifest.run_id}.lock").write_text(
        "999999", encoding="utf-8"
    )
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.MONITOR_DIR", monitor_dir),
        patch("vast.ops.run_vast", return_value=[]),
        patch("vast.ops.process_is_alive", return_value=False),
        patch("vast.ops.PURCHASE_RECONCILIATION_TIMEOUT_SECONDS", 0),
    ):
        assert _purchase_reconciliation(manifest.run_id) == 2
    # Supervisor lock files retain a stable inode after release; stale PID
    # metadata cannot block a new OS-lock-first owner.
    assert (monitor_dir / f"purchase_reconcile_{manifest.run_id}.lock").is_file()


def test_purchase_reconciliation_does_not_destroy_label_collision(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    label = f"quant-experiments-{manifest.run_id}"
    state_path.write_text(
        json.dumps(
            [
                {
                    "run_id": manifest.run_id,
                    "created_at": utc_now(),
                    "expected_label": label,
                    "offer_id": 11,
                    "machine_id": 101,
                    "status": "purchase_outcome_unknown",
                    "late_create_reconciliation_deadline_at": utc_now(),
                    "manifest": manifest.__dict__,
                }
            ]
        ),
        encoding="utf-8",
    )
    live = [{"id": 777, "label": label, "ask_contract_id": 99, "machine_id": 202}]
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops.run_vast", return_value=live),
        patch("vast.ops.destroy_instance") as destroy,
        patch("vast.ops.PURCHASE_RECONCILIATION_TIMEOUT_SECONDS", 0),
    ):
        assert _purchase_reconciliation(manifest.run_id) == 2
        updated = load_state()[0]
    destroy.assert_not_called()
    assert updated["cleanup_required"] is True


def test_purchase_reconciliation_destroys_match_visible_only_in_late_watch(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    record = {
        "run_id": manifest.run_id,
        "created_at": utc_now(),
        "expected_label": f"quant-experiments-{manifest.run_id}",
        "offer_id": 11,
        "machine_id": 101,
        "status": "purchase_outcome_unknown",
        "manifest": manifest.__dict__,
    }
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    late = {
        "id": 777,
        "label": record["expected_label"],
        "ask_contract_id": record["offer_id"],
        "machine_id": record["machine_id"],
    }
    virtual_now = [0.0]

    def monotonic() -> float:
        return virtual_now[0]

    def advance(seconds: float) -> None:
        virtual_now[0] += seconds

    wall_now = datetime.now(UTC)
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch(
            "vast.ops._purchase_reconciliation_deadline",
            return_value=wall_now,
        ),
        patch(
            "vast.ops._late_create_reconciliation_deadline",
            return_value=wall_now + timedelta(seconds=3),
        ),
        patch("vast.ops.PURCHASE_FINAL_SCAN_RESERVE_SECONDS", 1),
        patch("vast.ops.LATE_CREATE_RECONCILIATION_POLL_SECONDS", 1),
        patch("vast.ops.time.monotonic", side_effect=monotonic),
        patch("vast.ops.time.sleep", side_effect=advance),
        patch("vast.ops.run_vast", side_effect=[[], [], [late], []]),
        patch("vast.ops.destroy_instance") as destroy,
    ):
        assert _purchase_reconciliation(manifest.run_id) == 2

    destroy.assert_called_once()
    assert destroy.call_args.args == (
        777,
        manifest.run_id,
        "late_purchase_outcome_reconciled",
    )
    assert destroy.call_args.kwargs["launch_supervisor"] is False
    assert destroy.call_args.kwargs["mark_terminal"] is False
    assert destroy.call_args.kwargs["deadline_monotonic"] <= 3
    updated = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert updated["status"] == "late_create_reconciliation_expired"
    assert updated["manual_reconciliation_required"] is True
    assert updated["cap_control_failed"] is True


def test_late_create_deadline_is_persisted_and_never_renewed(
    tmp_path: Path,
) -> None:
    from vast.ops import (
        _late_create_reconciliation_deadline,
        _purchase_reconciliation_deadline,
    )

    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    state_path.write_text(
        json.dumps(
            [
                {
                    "run_id": manifest.run_id,
                    "created_at": utc_now(),
                    "manifest": manifest.__dict__,
                    "status": "purchase_outcome_unknown",
                }
            ]
        ),
        encoding="utf-8",
    )
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.LATE_CREATE_RECONCILIATION_SECONDS", 3600),
    ):
        record = load_state()[0]
        primary = _purchase_reconciliation_deadline(record, manifest)
        first = _late_create_reconciliation_deadline(load_state()[0], manifest)
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.LATE_CREATE_RECONCILIATION_SECONDS", 99 * 3600),
    ):
        second = _late_create_reconciliation_deadline(load_state()[0], manifest)

    assert (first - primary).total_seconds() == pytest.approx(3600)
    assert second == first


def test_manual_restart_after_late_expiry_gets_one_cleanup_only_scan(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    expired = (datetime.now(UTC) - timedelta(minutes=1)).isoformat().replace(
        "+00:00", "Z"
    )
    live = {
        "id": 777,
        "label": f"quant-experiments-{manifest.run_id}",
        "ask_contract_id": 11,
        "machine_id": 101,
    }
    state_path = tmp_path / "instances.json"
    state_path.write_text(
        json.dumps(
            [
                {
                    "run_id": manifest.run_id,
                    "created_at": utc_now(),
                    "expected_label": live["label"],
                    "offer_id": 11,
                    "machine_id": 101,
                    "status": "late_create_reconciliation_expired",
                    "purchase_reconciliation_status": "late_watch_expired",
                    "purchase_reconciliation_deadline_at": expired,
                    "late_create_reconciliation_deadline_at": expired,
                    "controller_deadline_at": expired,
                    "cleanup_required": True,
                    "manifest": manifest.__dict__,
                }
            ]
        ),
        encoding="utf-8",
    )
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.run_vast", return_value=[live]) as api,
        patch("vast.ops.destroy_instance") as destroy,
    ):
        assert _purchase_reconciliation(manifest.run_id) == 2

    api.assert_called_once()
    destroy.assert_called_once()
    assert destroy.call_args.args[2] == "late_purchase_outcome_reconciled"
    updated = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert updated["cap_control_failed"] is True
    assert updated["manual_reconciliation_required"] is True


def test_race_cleanup_recovers_stale_supervisor_lock(tmp_path: Path) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    record = {
        "run_id": manifest.run_id,
        "created_at": utc_now(),
        "campaign_started_at": utc_now(),
        "campaign_cap_usd": manifest.max_total_usd,
        "manifest": manifest.__dict__,
        "status": "provisioning_cleanup_pending",
        "candidates": [],
    }
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    monitor_dir = tmp_path / "monitors"
    monitor_dir.mkdir()
    (monitor_dir / f"race_cleanup_{manifest.run_id}.lock").write_text("999999", encoding="utf-8")
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.MONITOR_DIR", monitor_dir),
        patch("vast.ops.run_vast", return_value=[]),
        patch("vast.ops.process_is_alive", return_value=False),
    ):
        assert _race_cleanup_pending(manifest.run_id) == 0
    assert (monitor_dir / f"race_cleanup_{manifest.run_id}.lock").is_file()


def test_concurrent_artifact_packaging_uses_unique_temporary_names(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    ops_dir = tmp_path / "ops"
    (repo / "artifacts" / "run").mkdir(parents=True)
    (repo / "artifacts" / "run" / "result.json").write_text("{}", encoding="utf-8")
    manifest = {
        "run_id": "packaging_test",
        "source_commit": "a" * 40,
        "artifact_paths": ["artifacts/run"],
    }
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            package_artifacts(
                manifest,
                repo=repo,
                ops_dir=ops_dir,
                exit_code=0,
                events=EventWriter(ops_dir / "events.jsonl", "packaging_test"),
            )
        except BaseException as error:  # pragma: no cover - assertion below reports it
            errors.append(error)

    import threading

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert (ops_dir / "artifacts.tar.gz").is_file()
    assert (ops_dir / "artifact_manifest.json").is_file()
    assert not list(ops_dir.glob("*.tmp"))
    extracted = tmp_path / "verified"
    vast_ops.safe_extract_tar(
        ops_dir / "artifacts.tar.gz",
        extracted,
        max_unpacked_bytes=1024,
        expected_files={"artifacts/run/result.json"},
    )
    assert (extracted / "artifacts" / "run" / "result.json").is_file()


def test_known_hosts_option_quotes_paths_for_openssh_config_parser(
    tmp_path: Path,
) -> None:
    known_hosts = tmp_path / "Python Projects" / "known_hosts"
    with patch("vast.ops.KNOWN_HOSTS", known_hosts):
        assert known_hosts_option() == (f'UserKnownHostsFile="{known_hosts.as_posix()}"')


def test_provider_ssh_url_has_a_strict_safe_operand_grammar() -> None:
    assert vast_ops.parse_ssh_url("ssh://root@ssh4.vast.ai:22001") == (
        "root",
        "ssh4.vast.ai",
        22001,
    )
    assert vast_ops.ssh_base(("root", "host.example", 22))[-2:] == [
        "--",
        "root@host.example",
    ]


@pytest.mark.parametrize(
    "value",
    [
        "ssh://-oProxyCommand=whoami@host.example:22",
        "ssh://root@-oProxyCommand=whoami:22",
        "ssh://root@host example:22",
        "ssh://root@host.example.:22",
        "ssh://root@host..example:22",
        "ssh://root@host_example:22",
        "ssh://root@host.example:0",
        "ssh://root@host.example:65536",
        "ssh://root@host.example:-1",
        "ssh://root@host.example:\N{ARABIC-INDIC DIGIT TWO}\N{ARABIC-INDIC DIGIT TWO}",
        "ssh://root\n@host.example:22",
        "ssh://root@host.example:22\n-oProxyCommand=whoami",
        "ssh://r\N{FULLWIDTH LATIN SMALL LETTER O}ot@host.example:22",
    ],
)
def test_provider_ssh_url_rejects_option_injection_and_invalid_endpoints(
    value: str,
) -> None:
    with pytest.raises(OpsError, match="SSH"):
        vast_ops.parse_ssh_url(value)


@pytest.mark.parametrize(
    "connection",
    [
        ("-oProxyCommand=whoami", "host.example", 22),
        ("root", "-oProxyCommand=whoami", 22),
        ("root", "host example", 22),
        ("root", "host.example", 0),
        ("root", "host.example", 65536),
        ("root", "host.example", True),
        ("root", "host.example", 22.0),
        ("root", "host.example", "22"),
    ],
)
def test_every_ssh_and_scp_command_boundary_revalidates_provider_operands(
    tmp_path: Path,
    connection: tuple[object, object, object],
) -> None:
    local = tmp_path / "input.txt"
    local.write_text("payload", encoding="utf-8")
    with patch("vast.ops.subprocess.run") as run:
        with pytest.raises(OpsError, match="SSH"):
            vast_ops.remote_exec(connection, "true")  # type: ignore[arg-type]
        with pytest.raises(OpsError, match="SSH"):
            scp_upload(connection, (local,), "/workspace/")  # type: ignore[arg-type]
        with pytest.raises(OpsError, match="SSH"):
            scp_download(  # type: ignore[arg-type]
                connection,
                "/workspace/result.json",
                tmp_path / "result.json",
            )
    run.assert_not_called()


def test_scp_upload_and_download_use_the_exact_ssh_identity(tmp_path: Path) -> None:
    private_key = tmp_path / "id_ed25519"
    private_key.write_bytes(b"private")
    local = tmp_path / "input.txt"
    local.write_text("payload", encoding="utf-8")
    calls: list[tuple[list[object], dict[str, object]]] = []

    def fake_run(command: list[object], **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with (
        patch("vast.ops.SSH_PRIVATE_KEY", private_key),
        patch("vast.ops.subprocess.run", side_effect=fake_run),
    ):
        scp_upload(("root", "host", 2222), (local,), "/workspace/")
        assert scp_download(
            ("root", "host", 2222),
            "/workspace/result.json",
            tmp_path / "result.json",
            timeout_seconds=7,
        )

    assert all(
        command[command.index("-i") + 1] == str(private_key)
        and command[command.index("-o") + 1] == "BatchMode=yes"
        for command, _kwargs in calls
    )
    assert all("IdentitiesOnly=yes" in command for command, _kwargs in calls)
    upload_command = calls[0][0]
    download_command = calls[1][0]
    assert upload_command[upload_command.index("--") + 1 :] == [
        str(local),
        "root@host:/workspace/",
    ]
    assert download_command[download_command.index("--") + 1 :] == [
        "root@host:/workspace/result.json",
        str(tmp_path / "result.json"),
    ]
    assert calls[-1][1]["timeout"] == 7


def test_created_instance_ssh_probe_requires_watchdog_armed_markers() -> None:
    response = SimpleNamespace(returncode=0, stdout="SSH_READY\n", stderr="")
    with (
        patch("vast.ops.attach_local_ssh_key"),
        patch("vast.ops.ssh_url", return_value=("root", "host", 22)),
        patch("vast.ops.remote_exec", return_value=response) as execute,
    ):
        assert wait_for_ssh(
            1234,
            require_watchdog_armed=True,
            expected_watchdog_deadline_epoch=1_756_608_000.0,
        ) == (
            "root",
            "host",
            22,
        )
    command = execute.call_args.args[1]
    match = re.search(r"b64decode\('([A-Za-z0-9+/=]+)'\)", command)
    assert match is not None
    probe = base64.b64decode(match.group(1)).decode("ascii")
    assert "watchdog_armed.json" in probe
    assert "EXPECTED_INSTANCE = '1234'" in probe
    assert "EXPECTED_DEADLINE = 1756608000.0" in probe
    assert 'fields[0] in {"Z", "X", "x"}' in probe
    assert "proc_start_ticks" in probe
    assert "cmdline_sha256" in probe
    assert "time.sleep(1.0)" in probe
    assert "bootstrap_state" in probe


def test_state_read_modify_write_lock_is_reentrant_and_cross_process_file_backed(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "instances.state.lock"
    with patch("vast.ops.STATE_LOCK_FILE", lock_path):
        with state_file_lock():
            assert lock_path.is_file()
            with state_file_lock():
                assert lock_path.is_file()
        assert lock_path.is_file()


def test_state_lock_timeout_never_unlocks_an_unowned_lease(tmp_path: Path) -> None:
    lock_path = tmp_path / "instances.state.lock"
    with (
        patch("vast.ops.STATE_LOCK_FILE", lock_path),
        patch("vast.ops.STATE_LOCK_TIMEOUT_SECONDS", 0),
    ):
        if os.name == "nt":
            import msvcrt

            with patch.object(
                msvcrt,
                "locking",
                side_effect=PermissionError(errno.EACCES, "busy"),
            ) as locking, pytest.raises(
                OpsError, match="timed out acquiring"
            ), state_file_lock():
                pass
        else:
            import fcntl

            with patch.object(
                fcntl,
                "flock",
                side_effect=BlockingIOError(errno.EAGAIN, "busy"),
            ) as locking, pytest.raises(
                OpsError, match="timed out acquiring"
            ), state_file_lock():
                pass
    # A second call would be an invalid unlock attempt from the non-owner.
    assert locking.call_count == 1


def test_unknown_supervisor_lock_io_error_cannot_attest_stale_pid(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "monitor.lock"
    lock_path.write_bytes(b"\0" + b"4243")
    process = SimpleNamespace(pid=4243, poll=lambda: None)
    if os.name == "nt":
        import msvcrt

        lock_failure = patch.object(
            msvcrt, "locking", side_effect=OSError(5, "simulated I/O failure")
        )
    else:
        import fcntl

        lock_failure = patch.object(
            fcntl, "flock", side_effect=OSError(5, "simulated I/O failure")
        )
    with lock_failure, pytest.raises(OpsError, match="lock probe failed"):
        vast_ops._wait_for_supervisor_lease(
            lock_path, process, "test supervisor"
        )


def test_supervisor_handshake_retries_transient_pid_metadata_gap(
    tmp_path: Path,
) -> None:
    process = SimpleNamespace(pid=4243, poll=lambda: None)
    with (
        patch(
            "vast.ops._supervisor_lock_owner_pid",
            side_effect=[
                OpsError("held supervisor lock has invalid owner metadata"),
                4243,
                4243,
            ],
        ),
        patch("vast.ops._bounded_sleep"),
    ):
        assert (
            vast_ops._wait_for_supervisor_lease(
                tmp_path / "monitor.lock", process, "test supervisor"
            )
            == 4243
        )


def test_supervisor_handshake_rejects_exit_between_stable_samples(
    tmp_path: Path,
) -> None:
    process = SimpleNamespace(pid=4243, poll=lambda: next(exit_states))
    exit_states = iter([None, 7])
    with (
        patch("vast.ops._supervisor_lock_owner_pid", return_value=4243),
        patch("vast.ops._bounded_sleep"),
        pytest.raises(OpsError, match="stable OS-lease handoff"),
    ):
        vast_ops._wait_for_supervisor_lease(
            tmp_path / "monitor.lock", process, "test supervisor"
        )


def test_local_ssh_key_attachment_matches_explicit_identity(tmp_path: Path) -> None:
    private_key = tmp_path / "id_ed25519"
    public_key = private_key.with_suffix(".pub")
    private_key.write_bytes(b"private-key-placeholder")
    public_key.write_text("ssh-ed25519 AAAATEST profiler\n", encoding="utf-8")
    with (
        patch("vast.ops.SSH_PRIVATE_KEY", private_key),
        patch("vast.ops.SSH_PUBLIC_KEY", public_key),
        patch("vast.ops.run_vast", return_value="SSH key added") as run,
    ):
        assert local_ssh_public_key() == "ssh-ed25519 AAAATEST profiler"
        attach_local_ssh_key(1234)

    run.assert_called_once_with(
        "attach",
        "ssh",
        1234,
        "ssh-ed25519 AAAATEST profiler",
        raw=False,
        timeout=180,
    )


def test_require_local_ssh_identity_checks_private_key_readability(
    tmp_path: Path,
) -> None:
    private_key = tmp_path / "id_ed25519"
    public_key = private_key.with_suffix(".pub")
    private_key.write_bytes(b"private-key-placeholder")
    public_key.write_text("ssh-ed25519 AAAATEST profiler\n", encoding="utf-8")

    with (
        patch("vast.ops.SSH_PRIVATE_KEY", private_key),
        patch("vast.ops.SSH_PUBLIC_KEY", public_key),
    ):
        require_local_ssh_identity()


def test_require_local_ssh_identity_rejects_missing_private_key(
    tmp_path: Path,
) -> None:
    private_key = tmp_path / "id_ed25519"
    public_key = private_key.with_suffix(".pub")
    public_key.write_text("ssh-ed25519 AAAATEST profiler\n", encoding="utf-8")

    with (
        patch("vast.ops.SSH_PRIVATE_KEY", private_key),
        patch("vast.ops.SSH_PUBLIC_KEY", public_key),
        pytest.raises(OpsError, match="private key is unreadable"),
    ):
        require_local_ssh_identity()


def test_local_ssh_public_key_rejects_private_material(tmp_path: Path) -> None:
    public_key = tmp_path / "id_ed25519.pub"
    private_header = "-----BEGIN " + "OPENSSH PRIVATE KEY-----"
    public_key.write_text(private_header, encoding="utf-8")
    with (
        patch("vast.ops.SSH_PUBLIC_KEY", public_key),
        pytest.raises(OpsError, match="malformed"),
    ):
        local_ssh_public_key()


def test_search_offers_enforces_consumer_cpu_and_exact_geolocation() -> None:
    offers = [
        {
            "id": 1,
            "machine_id": 101,
            "cpu_name": "AMD Ryzen 7 7800X3D",
            "geolocation": "South Korea, KR",
        },
        {
            "id": 2,
            "machine_id": 102,
            "cpu_name": "AMD EPYC 7452",
            "geolocation": "South Korea, KR",
        },
        {
            "id": 3,
            "machine_id": 103,
            "cpu_name": "AMD Ryzen 9 7900X",
            "geolocation": "Alberta, CA",
        },
    ]

    with patch("vast.ops.run_vast", return_value=offers):
        result = search_offers(
            "RTX_5090",
            0.5,
            disk_gb=30,
            min_cuda=12.8,
            min_inet_down_mbps=100,
            require_consumer_cpu=True,
            required_geolocation="South Korea, KR",
        )

    assert [offer["id"] for offer in result] == [1]


def test_search_offers_country_filter_accepts_geolocation_suffix() -> None:
    offers = [
        {
            "id": 1,
            "machine_id": 101,
            "cpu_name": "AMD Ryzen 7 7800X3D",
            "geolocation": "Alberta, CA",
        },
        {
            "id": 2,
            "machine_id": 102,
            "cpu_name": "AMD Ryzen 9 7900X",
            "geolocation": "Texas, US",
        },
    ]
    with patch("vast.ops.run_vast", return_value=offers):
        result = search_offers(
            "RTX_5090",
            0.5,
            disk_gb=30,
            min_cuda=12.8,
            min_inet_down_mbps=100,
            required_country="CA",
        )
    assert [offer["id"] for offer in result] == [1]


def test_pick_offer_is_cost_first_and_sees_all_candidates() -> None:
    offers = [
        {
            "id": 1,
            "machine_id": 101,
            "dph_total": 0.41,
            "cpu_cores_effective": 8,
            "cpu_ram": 16 * 1024,
            "cpu_name": "AMD Ryzen 7 7700X",
        },
        {
            "id": 2,
            "machine_id": 102,
            "dph_total": 0.29,
            "cpu_cores_effective": 4,
            "cpu_ram": 16 * 1024,
            "cpu_name": "AMD Ryzen 5 5600X",
        },
    ]
    assert pick_offer(offers)["id"] == 2


@pytest.mark.parametrize(
    "offer",
    [
        {"id": 11, "machine_id": 101, "dph_total": float("nan")},
        {"id": 11, "machine_id": None, "dph_total": 0.3},
        {"id": None, "machine_id": 101, "dph_total": 0.3},
        {"id": 1.0, "machine_id": 101, "dph_total": 0.3},
        {"id": 11, "machine_id": 101.0, "dph_total": 0.3},
        {"id": "11", "machine_id": 101, "dph_total": 0.3},
        {"id": 11, "machine_id": "101", "dph_total": 0.3},
        {"id": float("nan"), "machine_id": 101, "dph_total": 0.3},
        {"id": 11, "machine_id": float("nan"), "dph_total": 0.3},
    ],
)
def test_purchase_rejects_nonfinite_or_missing_offer_identity(
    offer: dict[str, object],
) -> None:
    with pytest.raises(OpsError, match="selected offer"):
        validate_offer_for_purchase(offer, adoption_manifest())


@pytest.mark.parametrize("dph", [0, -0.01, float("nan"), float("inf")])
def test_post_purchase_rejects_nonpositive_or_nonfinite_price(dph: float) -> None:
    with pytest.raises(OpsError, match=r"price|dph_total"):
        validate_adoption_instance(
            adoption_instance(dph_total=dph),
            adoption_manifest(),
            tracked_records=[],
        )


def test_search_offers_does_not_truncate_before_selection() -> None:
    offers = [{"id": index, "machine_id": index, "dph_total": 0.2} for index in range(1, 30)]
    with patch("vast.ops.run_vast", return_value=offers):
        result = search_offers(
            "RTX_5090",
            0.5,
            disk_gb=30,
            min_cuda=12.8,
            min_inet_down_mbps=100,
            limit=1,
        )
    assert len(result) == len(offers)


def test_manifest_rejects_remote_input_artifacts() -> None:
    raw = {
        "format_version": 1,
        "run_id": "sealed_boundary_probe",
        "description": "Boundary contract probe",
        "source_commit": "a" * 40,
        "command": ["python", "scripts/check_scaffold.py"],
        "artifact_paths": ["artifacts/sealed_boundary_probe"],
        "gpu_name": "RTX_5090",
        "max_dph": 0.5,
        "max_total_usd": 1.0,
        "ttl_hours": 1.0,
        "failure_grace_minutes": 10,
        "disk_gb": 30,
        "poll_seconds": 60,
        "seed_policy": "No training seeds",
        "interaction_budget": "One contract probe",
        "checkpoint_cadence": "Final output only",
        "intervention_suite": "No scientific evaluation",
        "input_artifacts": {"artifacts/remote/test.npz": "a" * 64},
    }
    with pytest.raises(OpsError, match="jobs download their own data"):
        RunManifest.from_dict(raw)


def test_remote_runner_rejects_tampered_input_artifact_manifest(
    tmp_path: Path,
) -> None:
    manifest = {"input_artifacts": {"artifacts/remote/test.npz": "a" * 64}}
    with pytest.raises(ValueError, match="jobs download their own data"):
        verify_input_artifacts(tmp_path, manifest)


def test_search_offers_requests_exact_multi_gpu_contract() -> None:
    with patch("vast.ops.run_vast", return_value=[]) as run:
        search_offers(
            "RTX_5090",
            2.0,
            num_gpus=4,
            disk_gb=30,
            min_cuda=12.8,
            min_inet_down_mbps=100,
        )
    assert "num_gpus=4" in run.call_args.args[2]


def test_search_offers_passes_requested_storage_allocation() -> None:
    with patch("vast.ops.run_vast", return_value=[]) as run:
        search_offers(
            "RTX_5090",
            2.0,
            disk_gb=20,
            min_cuda=12.8,
            min_inet_down_mbps=100,
        )

    assert run.call_args.args[0:3] == (
        "search",
        "offers",
        "gpu_name=RTX_5090 num_gpus=1 rentable=true verified=true "
        "dph<=2.0 disk_space>=20 cuda_max_good>=12.8 inet_down>=100 "
        "cpu_cores_effective>=4 cpu_ram>=16 direct_port_count>=1",
    )
    assert run.call_args.args[3:] == ("--storage", 20, "-o", "dph")


def test_verify_created_instance_rejects_wrong_geolocation() -> None:
    instance = {
        "id": 42,
        "label": "quant-experiments-adopt_seed0",
        "dph_total": 0.4,
        "num_gpus": 1,
        "gpu_name": "RTX 5090",
        "disk_space": 40,
        "cuda_max_good": 12.8,
        "inet_down": 500,
        "cpu_cores_effective": 8,
        "cpu_ram": 32 * 1024,
        "direct_port_count": 1,
        "machine_id": 101,
        "verification": "verified",
        "cpu_name": "AMD Ryzen 9 7900X",
        "geolocation": "Alberta, CA",
    }
    manifest = SimpleNamespace(
        max_dph=0.5,
        num_gpus=1,
        gpu_name="RTX 5090",
        disk_gb=30,
        min_cuda=12.8,
        min_inet_down_mbps=100,
        min_cpu_cores_effective=4,
        min_cpu_ram_gb=16,
        require_consumer_cpu=True,
        required_geolocation="South Korea, KR",
    )

    with (
        patch("vast.ops.run_vast", return_value=[instance]),
        pytest.raises(OpsError, match="does not match required"),
    ):
        verify_created_instance(
            42,
            manifest,
            expected_label="quant-experiments-adopt_seed0",
        )


def test_purchased_live_contract_requires_exact_offer_identity() -> None:
    instance = adoption_instance()
    manifest = replace(adoption_manifest(), required_geolocation=None)
    # No offer identity reported and no create-response binding: refuse.
    with pytest.raises(OpsError, match="unambiguous ask/offer contract ID"):
        _validate_live_contract_constraints(
            instance, manifest, expected_offer={"id": 11, "machine_id": 999}
        )
    # A null identity field is the same as an absent one.
    with pytest.raises(OpsError, match="unambiguous ask/offer contract ID"):
        _validate_live_contract_constraints(
            {**instance, "ask_contract_id": None},
            manifest,
            expected_offer={"id": 11, "machine_id": 999},
        )

    instance["ask_contract_id"] = 12
    with pytest.raises(OpsError, match="does not match selected offer"):
        _validate_live_contract_constraints(
            instance, manifest, expected_offer={"id": 11, "machine_id": 999}
        )


def test_purchased_live_contract_rejects_ambiguous_offer_identities() -> None:
    instance = adoption_instance(ask_contract_id=11, offer_id=12)
    manifest = replace(adoption_manifest(), required_geolocation=None)
    with pytest.raises(OpsError, match="ambiguous offer identities"):
        _validate_live_contract_constraints(
            instance, manifest, expected_offer={"id": 11, "machine_id": 999}
        )


@pytest.mark.parametrize("value", [1.0, "1", True, 0, None])
def test_live_gpu_count_requires_exact_integer(value: object) -> None:
    instance = adoption_instance(num_gpus=value)
    with pytest.raises(OpsError, match="num_gpus"):
        _validate_live_contract_constraints(instance, adoption_manifest())


def test_race_cleanup_handles_missing_persisted_instance_without_int_none(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    record = {
        "run_id": manifest.run_id,
        "manifest": manifest.__dict__,
        "status": "provisioning_cleanup_pending",
        "candidates": [
            {
                "candidate_id": "adopt_seed0.candidate1",
                "instance_id": None,
                "discovered_instance_ids": [],
                "status": "cleanup_pending",
            }
        ],
    }
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    lock = threading.Lock()
    with patch("vast.ops.STATE_FILE", state_path):
        assert _race_cleanup_losers(manifest, record, lock, -1) == []


def test_purchase_supervisor_child_state_survives_parent_registration_race(
    tmp_path: Path,
) -> None:
    """A child update must not be clobbered by the post-spawn PID update."""

    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    monitor_dir = tmp_path / "monitors"
    state_path.write_text(
        json.dumps(
            [
                {
                    "run_id": manifest.run_id,
                    "created_at": utc_now(),
                    "campaign_started_at": utc_now(),
                    "manifest": manifest.__dict__,
                    "status": "purchase_pending",
                    "expected_label": "quant-experiments-adopt_seed0",
                }
            ]
        ),
        encoding="utf-8",
    )

    class ChildProcess:
        pid = 4242

    def fake_popen(*_: object, **__: object) -> ChildProcess:
        # Simulate the detached child observing a lost parent and persisting a
        # cleanup transition before the launcher receives the child PID.
        from vast.ops import update_record_fields

        update_record_fields(
            manifest.run_id,
            status="purchase_reconciled_cleanup_pending",
            cleanup_required=True,
        )
        return ChildProcess()

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.MONITOR_DIR", monitor_dir),
        patch("vast.ops.subprocess.Popen", side_effect=fake_popen),
        patch("vast.ops._wait_for_supervisor_lease", return_value=4242),
    ):
        from vast.ops import launch_detached_purchase_reconciliation

        assert launch_detached_purchase_reconciliation(manifest.run_id) == 4242
        updated = load_state()[0]
        assert updated["status"] == "purchase_reconciled_cleanup_pending"
        assert updated["cleanup_required"] is True
        assert updated["purchase_reconciliation_pid"] == 4242


def test_purchase_supervisor_requires_child_os_lease(tmp_path: Path) -> None:
    from vast.ops import launch_detached_purchase_reconciliation

    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    state_path.write_text(
        json.dumps(
            [
                {
                    "run_id": manifest.run_id,
                    "created_at": utc_now(),
                    "campaign_started_at": utc_now(),
                    "manifest": manifest.__dict__,
                    "status": "purchase_pending",
                    "expected_label": "quant-experiments-adopt_seed0",
                }
            ]
        ),
        encoding="utf-8",
    )

    class ExitedChild:
        pid = 4242

        @staticmethod
        def poll() -> int:
            return 7

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops._supervisor_lock_owner_pid", return_value=None),
        patch("vast.ops.subprocess.Popen", return_value=ExitedChild()),
        pytest.raises(OpsError, match="handshake failed"),
    ):
        launch_detached_purchase_reconciliation(manifest.run_id)

    updated = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert updated["purchase_reconciliation_status"] == "launch_failed"


def test_command_launch_never_purchases_when_supervisor_handshake_fails(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    args = SimpleNamespace(manifest=tmp_path / "run.json", offer=None, detach=True)
    offer = {"id": 11, "machine_id": 101, "dph_total": 0.3}
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.load_manifest", return_value=manifest),
        patch(
            "vast.ops.create_source_bundle",
            return_value=(tmp_path / "source.bundle", tmp_path / "run.json"),
        ),
        patch("vast.ops.load_machine_blacklist", return_value=set()),
        patch("vast.ops.active_machine_ids", return_value=set()),
        patch("vast.ops.run_vast", return_value=[]),
        patch("vast.ops.search_offers", return_value=[offer]),
        patch(
            "vast.ops.launch_detached_purchase_reconciliation",
            side_effect=OpsError("child exited before OS lease"),
        ),
        patch("vast.ops.create_instance") as create,
        pytest.raises(OpsError, match="no instance purchase was attempted"),
    ):
        command_launch(args)
    create.assert_not_called()


def test_unverified_create_response_id_never_gains_destroy_or_bootstrap_authority(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    args = SimpleNamespace(manifest=tmp_path / "run.json", offer=None, detach=True)
    offer = {"id": 11, "machine_id": 101, "dph_total": 0.3}
    wrong_contract = adoption_instance(
        id=1234,
        label="unrelated-contract",
        machine_id=101,
        ask_contract_id=11,
    )
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.load_manifest", return_value=manifest),
        patch(
            "vast.ops.create_source_bundle",
            return_value=(tmp_path / "source.bundle", tmp_path / "run.json"),
        ),
        patch("vast.ops.load_machine_blacklist", return_value=set()),
        patch("vast.ops.active_machine_ids", return_value=set()),
        patch("vast.ops.search_offers", return_value=[offer]),
        patch("vast.ops.run_vast", side_effect=[[], [wrong_contract]]),
        patch("vast.ops.launch_detached_purchase_reconciliation", return_value=9001),
        patch("vast.ops.create_instance", return_value=1234),
        patch("vast.ops.destroy_instance") as destroy,
        patch("vast.ops.wait_for_ssh") as wait,
        patch("vast.ops.start_remote_run") as bootstrap,
        pytest.raises(OpsError, match="label"),
    ):
        command_launch(args)

    record = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert record["instance_id"] is None
    assert record["create_response_instance_id"] == 1234
    assert record["status"] == "purchase_outcome_unknown"
    assert record["cleanup_required"] is True
    destroy.assert_not_called()
    wait.assert_not_called()
    bootstrap.assert_not_called()


def test_command_launch_persists_redacted_live_validation_failure_before_cleanup(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    event_path = tmp_path / "events.jsonl"
    args = SimpleNamespace(manifest=tmp_path / "run.json", offer=None, detach=True)
    offer = {"id": 11, "machine_id": 101, "dph_total": 0.3}
    secret = "sentinel-secret-value"
    error = (
        "live instance price $0.44/h exceeds cap $0.40/h "
        f"api_key={secret}&request=1"
    )

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.EVENT_FILE", event_path),
        patch("vast.ops.load_manifest", return_value=manifest),
        patch(
            "vast.ops.create_source_bundle",
            return_value=(tmp_path / "source.bundle", tmp_path / "run.json"),
        ),
        patch("vast.ops.load_machine_blacklist", return_value=set()),
        patch("vast.ops.active_machine_ids", return_value=set()),
        patch("vast.ops.run_vast", return_value=[]),
        patch("vast.ops.load_secrets", return_value={"VAST_API_KEY": secret}),
        patch("vast.ops.search_offers", return_value=[offer]),
        patch("vast.ops.launch_detached_purchase_reconciliation", return_value=9001),
        patch("vast.ops.create_instance", return_value=1234),
        patch("vast.ops.verify_created_instance", side_effect=OpsError(error)),
        patch("vast.ops.destroy_instance") as destroy,
        pytest.raises(OpsError, match="live instance price"),
    ):
        command_launch(args)

    record = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert secret not in json.dumps(record)
    assert record["setup_error"] == "[DIAGNOSTIC WITHHELD]"
    assert record["instance_id"] is None
    assert record["create_response_instance_id"] == 1234
    assert record["status"] == "purchase_outcome_unknown"
    assert record["cleanup_required"] is True
    assert record["setup_error_at"].endswith("Z")
    events = [
        json.loads(line)
        for line in event_path.read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["event"] == "provisioning_setup_failed"
    assert events[-1]["instance_id"] is None
    assert secret not in json.dumps(events[-1])
    assert events[-1] == record["setup_failure_event"]
    destroy.assert_not_called()


def test_command_launch_cleanup_survives_setup_diagnostic_failure(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    event_path = tmp_path / "events.jsonl"
    args = SimpleNamespace(manifest=tmp_path / "run.json", offer=None, detach=True)
    offer = {"id": 11, "machine_id": 101, "dph_total": 0.3}

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.EVENT_FILE", event_path),
        patch("vast.ops.load_manifest", return_value=manifest),
        patch(
            "vast.ops.create_source_bundle",
            return_value=(tmp_path / "source.bundle", tmp_path / "run.json"),
        ),
        patch("vast.ops.load_machine_blacklist", return_value=set()),
        patch("vast.ops.active_machine_ids", return_value=set()),
        patch("vast.ops.run_vast", return_value=[]),
        patch("vast.ops.search_offers", return_value=[offer]),
        patch("vast.ops.launch_detached_purchase_reconciliation", return_value=9001),
        patch("vast.ops.create_instance", return_value=1234),
        patch(
            "vast.ops.verify_created_instance",
            side_effect=OpsError("live validation failed"),
        ),
        patch(
            "vast.ops._record_provisioning_setup_failure",
            side_effect=OSError("diagnostic disk unavailable"),
        ),
        patch("vast.ops.destroy_instance") as destroy,
        pytest.raises(OpsError, match="live validation failed"),
    ):
        command_launch(args)

    destroy.assert_not_called()
    record = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert record["instance_id"] is None
    assert record["create_response_instance_id"] == 1234
    assert record["status"] == "purchase_outcome_unknown"
    assert record["cleanup_required"] is True
    assert record["setup_error_persistence_error"] == "diagnostic disk unavailable"
    events = [
        json.loads(line)
        for line in event_path.read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["event"] == "provisioning_setup_persistence_failed"


def test_command_launch_cleanup_survives_setup_event_append_failure(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    event_path = tmp_path / "events.jsonl"
    args = SimpleNamespace(manifest=tmp_path / "run.json", offer=None, detach=True)
    offer = {"id": 11, "machine_id": 101, "dph_total": 0.3}
    real_append = vast_ops.append_local_event

    def append_except_setup(run_id: str, event: str, **fields: object) -> None:
        if event == "provisioning_setup_failed":
            raise OSError("setup event disk unavailable")
        real_append(run_id, event, **fields)

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.EVENT_FILE", event_path),
        patch("vast.ops.load_manifest", return_value=manifest),
        patch(
            "vast.ops.create_source_bundle",
            return_value=(tmp_path / "source.bundle", tmp_path / "run.json"),
        ),
        patch("vast.ops.load_machine_blacklist", return_value=set()),
        patch("vast.ops.active_machine_ids", return_value=set()),
        patch("vast.ops.run_vast", return_value=[]),
        patch("vast.ops.search_offers", return_value=[offer]),
        patch("vast.ops.launch_detached_purchase_reconciliation", return_value=9001),
        patch("vast.ops.create_instance", return_value=1234),
        patch(
            "vast.ops.verify_created_instance",
            side_effect=OpsError("original live validation failure"),
        ),
        patch("vast.ops.append_local_event", side_effect=append_except_setup),
        patch("vast.ops.destroy_instance") as destroy,
        pytest.raises(OpsError, match="original live validation failure"),
    ):
        command_launch(args)

    destroy.assert_not_called()
    record = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert record["instance_id"] is None
    assert record["create_response_instance_id"] == 1234
    assert record["status"] == "purchase_outcome_unknown"
    assert record["cleanup_required"] is True
    assert record["setup_error"] == "original live validation failure"
    assert record["setup_error_persistence_error"] == "setup event disk unavailable"


def test_command_launch_cleanup_fallback_survives_secondary_state_failure(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    event_path = tmp_path / "events.jsonl"
    args = SimpleNamespace(manifest=tmp_path / "run.json", offer=None, detach=True)
    offer = {"id": 11, "machine_id": 101, "dph_total": 0.3}
    original = OpsError("original live validation failure")

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.EVENT_FILE", event_path),
        patch("vast.ops.load_manifest", return_value=manifest),
        patch(
            "vast.ops.create_source_bundle",
            return_value=(tmp_path / "source.bundle", tmp_path / "run.json"),
        ),
        patch("vast.ops.load_machine_blacklist", return_value=set()),
        patch("vast.ops.active_machine_ids", return_value=set()),
        patch("vast.ops.run_vast", return_value=[]),
        patch("vast.ops.search_offers", return_value=[offer]),
        patch("vast.ops.launch_detached_purchase_reconciliation", return_value=9001),
        patch("vast.ops.create_instance", return_value=1234),
        patch(
            "vast.ops.verify_created_instance",
            return_value={"id": 1234, "dph_total": 0.3},
        ),
        patch("vast.ops.wait_for_ssh", side_effect=original),
        patch("vast.ops.destroy_instance", side_effect=OSError("destroy event failed")),
        patch("vast.ops.update_record_fields", side_effect=OSError("state unavailable")),
        patch("vast.ops.launch_detached_monitor") as monitor,
        pytest.raises(OpsError, match="original live validation failure") as raised,
    ):
        command_launch(args)

    assert raised.value is original
    monitor.assert_called_once_with(manifest.run_id)
    events = [
        json.loads(line)
        for line in event_path.read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["event"] == "provisioning_cleanup_pending"
    assert events[-1]["error"] == "destroy event failed"


def test_command_launch_actual_destroy_event_failure_preserves_original_and_fallback(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    event_path = tmp_path / "events.jsonl"
    args = SimpleNamespace(manifest=tmp_path / "run.json", offer=None, detach=True)
    offer = {"id": 11, "machine_id": 101, "dph_total": 0.3}
    original = OpsError("original live validation failure")
    real_append = vast_ops.append_local_event
    destroy_attempts = 0

    def provider(*arguments: object, **_kwargs: object) -> object:
        nonlocal destroy_attempts
        if arguments[:2] == ("show", "instances"):
            return []
        if arguments[:2] == ("destroy", "instance"):
            destroy_attempts += 1
            raise OpsError("provider destroy failed")
        raise AssertionError(f"unexpected provider call: {arguments!r}")

    def append_except_destroy_retry(
        run_id: str, event: str, **fields: object
    ) -> None:
        if event == "destroy_retry":
            raise OSError("destroy event append unavailable")
        real_append(run_id, event, **fields)

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops.EVENT_FILE", event_path),
        patch("vast.ops.load_manifest", return_value=manifest),
        patch(
            "vast.ops.create_source_bundle",
            return_value=(tmp_path / "source.bundle", tmp_path / "run.json"),
        ),
        patch("vast.ops.load_machine_blacklist", return_value=set()),
        patch("vast.ops.active_machine_ids", return_value=set()),
        patch("vast.ops.run_vast", side_effect=provider),
        patch("vast.ops.search_offers", return_value=[offer]),
        patch("vast.ops.launch_detached_purchase_reconciliation", return_value=9001),
        patch("vast.ops.create_instance", return_value=1234),
        patch(
            "vast.ops.verify_created_instance",
            return_value={"id": 1234, "dph_total": 0.3},
        ),
        patch("vast.ops.wait_for_ssh", side_effect=original),
        patch("vast.ops.append_local_event", side_effect=append_except_destroy_retry),
        patch("vast.ops.DESTROY_MAX_ATTEMPTS", 1),
        patch("vast.ops.DESTROY_VERIFY_MAX_ATTEMPTS", 1),
        patch(
            "vast.ops._launch_destroy_reconciliation_with_retries",
            side_effect=OpsError("destroy supervisor unavailable"),
        ),
        patch("vast.ops.launch_detached_monitor") as monitor,
        pytest.raises(OpsError) as raised,
    ):
        command_launch(args)

    assert raised.value is original
    assert destroy_attempts == 1
    monitor.assert_called_once_with(manifest.run_id)
    record = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert record["cleanup_required"] is True


def test_command_launch_actual_destroy_state_failure_preserves_original_and_fallback(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    args = SimpleNamespace(manifest=tmp_path / "run.json", offer=None, detach=True)
    offer = {"id": 11, "machine_id": 101, "dph_total": 0.3}
    original = OpsError("original live validation failure")

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.load_manifest", return_value=manifest),
        patch(
            "vast.ops.create_source_bundle",
            return_value=(tmp_path / "source.bundle", tmp_path / "run.json"),
        ),
        patch("vast.ops.load_machine_blacklist", return_value=set()),
        patch("vast.ops.active_machine_ids", return_value=set()),
        patch("vast.ops.run_vast", return_value=[]),
        patch("vast.ops.search_offers", return_value=[offer]),
        patch("vast.ops.launch_detached_purchase_reconciliation", return_value=9001),
        patch("vast.ops.create_instance", return_value=1234),
        patch(
            "vast.ops.verify_created_instance",
            return_value={"id": 1234, "dph_total": 0.3},
        ),
        patch("vast.ops.wait_for_ssh", side_effect=original),
        patch(
            "vast.ops._claim_cleanup_authority",
            side_effect=OSError("cleanup state unavailable"),
        ),
        patch("vast.ops.launch_detached_monitor") as monitor,
        pytest.raises(OpsError) as raised,
    ):
        command_launch(args)

    assert raised.value is original
    monitor.assert_called_once_with(manifest.run_id)


def test_command_launch_preserves_create_error_when_unknown_outcome_write_fails(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    event_path = tmp_path / "events.jsonl"
    args = SimpleNamespace(manifest=tmp_path / "run.json", offer=None, detach=True)
    offer = {"id": 11, "machine_id": 101, "dph_total": 0.3}
    original = OpsError("original create outcome unknown")
    real_append = vast_ops.append_local_event

    def append_except_unknown_outcome(
        run_id: str, event: str, **fields: object
    ) -> None:
        if event == "purchase_outcome_persistence_failed":
            raise OSError("unknown-outcome event unavailable")
        real_append(run_id, event, **fields)

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.EVENT_FILE", event_path),
        patch("vast.ops.load_manifest", return_value=manifest),
        patch(
            "vast.ops.create_source_bundle",
            return_value=(tmp_path / "source.bundle", tmp_path / "run.json"),
        ),
        patch("vast.ops.load_machine_blacklist", return_value=set()),
        patch("vast.ops.active_machine_ids", return_value=set()),
        patch("vast.ops.run_vast", return_value=[]),
        patch("vast.ops.search_offers", return_value=[offer]),
        patch("vast.ops.launch_detached_purchase_reconciliation", return_value=9001),
        patch("vast.ops.create_instance", side_effect=original),
        patch(
            "vast.ops._provisioning_write_if_active",
            side_effect=OSError("unknown-outcome state unavailable"),
        ),
        patch("vast.ops.append_local_event", side_effect=append_except_unknown_outcome),
        patch("vast.ops.destroy_instance") as destroy,
        patch("vast.ops.launch_detached_monitor") as monitor,
        pytest.raises(OpsError, match="original create outcome unknown") as raised,
    ):
        command_launch(args)

    assert raised.value is original
    destroy.assert_not_called()
    monitor.assert_not_called()
    record = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert record["status"] == "purchase_pending"
    assert (
        record["purchase_outcome_persistence_error"]
        == "unknown-outcome state unavailable"
    )
    assert "purchase_outcome_persistence_failed" not in event_path.read_text(
        encoding="utf-8"
    )


def test_command_launch_redacts_cleanup_and_monitor_failures(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    event_path = tmp_path / "events.jsonl"
    args = SimpleNamespace(manifest=tmp_path / "run.json", offer=None, detach=True)
    offer = {"id": 11, "machine_id": 101, "dph_total": 0.3}
    secret = "secondary-secret-value"

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.EVENT_FILE", event_path),
        patch("vast.ops.load_manifest", return_value=manifest),
        patch(
            "vast.ops.create_source_bundle",
            return_value=(tmp_path / "source.bundle", tmp_path / "run.json"),
        ),
        patch("vast.ops.load_machine_blacklist", return_value=set()),
        patch("vast.ops.active_machine_ids", return_value=set()),
        patch("vast.ops.run_vast", return_value=[]),
        patch("vast.ops.load_secrets", return_value={"VAST_API_KEY": secret}),
        patch("vast.ops.search_offers", return_value=[offer]),
        patch("vast.ops.launch_detached_purchase_reconciliation", return_value=9001),
        patch("vast.ops.create_instance", return_value=1234),
        patch(
            "vast.ops.verify_created_instance",
            return_value={"id": 1234, "dph_total": 0.3},
        ),
        patch(
            "vast.ops.wait_for_ssh",
            side_effect=OpsError("original live validation failure"),
        ),
        patch(
            "vast.ops.destroy_instance",
            side_effect=OSError(f"cleanup api_key={secret}"),
        ),
        patch(
            "vast.ops.launch_detached_monitor",
            side_effect=OSError(f"monitor api_key={secret}"),
        ),
        pytest.raises(OpsError, match="original live validation failure"),
    ):
        command_launch(args)

    state_text = state_path.read_text(encoding="utf-8")
    event_text = event_path.read_text(encoding="utf-8")
    assert secret not in state_text
    assert secret not in event_text
    record = json.loads(state_text)[0]
    assert record["provisioning_cleanup_error"] == "[DIAGNOSTIC WITHHELD]"
    assert record["provisioning_cleanup_monitor_error"] == (
        "[DIAGNOSTIC WITHHELD]"
    )


def test_command_launch_cleanup_survives_redaction_failure(tmp_path: Path) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    args = SimpleNamespace(manifest=tmp_path / "run.json", offer=None, detach=True)
    offer = {"id": 11, "machine_id": 101, "dph_total": 0.3}

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.load_manifest", return_value=manifest),
        patch(
            "vast.ops.create_source_bundle",
            return_value=(tmp_path / "source.bundle", tmp_path / "run.json"),
        ),
        patch("vast.ops.load_machine_blacklist", return_value=set()),
        patch("vast.ops.active_machine_ids", return_value=set()),
        patch("vast.ops.run_vast", return_value=[]),
        patch("vast.ops.search_offers", return_value=[offer]),
        patch("vast.ops.launch_detached_purchase_reconciliation", return_value=9001),
        patch("vast.ops.create_instance", return_value=1234),
        patch(
            "vast.ops.verify_created_instance",
            return_value={"id": 1234, "dph_total": 0.3},
        ),
        patch(
            "vast.ops.wait_for_ssh",
            side_effect=OpsError("original live validation failure"),
        ),
        patch("vast.ops.redact_secrets", side_effect=OSError("secret file unreadable")),
        patch("vast.ops.destroy_instance") as destroy,
        pytest.raises(OpsError, match="original live validation failure"),
    ):
        command_launch(args)

    destroy.assert_called_once()
    record = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert record["setup_error"] == "[DIAGNOSTIC WITHHELD]"


def test_setup_failure_evidence_does_not_resurrect_cleanup_state(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    event_path = tmp_path / "events.jsonl"
    state_path.write_text(
        json.dumps(
            [
                {
                    "run_id": manifest.run_id,
                    "status": "cleanup_in_progress",
                    "cleanup_required": True,
                    "cleanup_generation": 3,
                }
            ]
        ),
        encoding="utf-8",
    )

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.EVENT_FILE", event_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.load_secrets", return_value={}),
    ):
        updated = _record_provisioning_setup_failure(
            manifest.run_id,
            instance_id=1234,
            error=OpsError("live validation failed"),
        )

    assert updated["status"] == "cleanup_in_progress"
    assert updated["cleanup_required"] is True
    assert updated["cleanup_generation"] == 3
    assert updated["setup_error"] == "live validation failed"


def test_command_launch_cannot_start_after_cleanup_wins_provisioning_race(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    args = SimpleNamespace(manifest=tmp_path / "run.json", offer=None, detach=True)
    offer = {"id": 11, "machine_id": 101, "dph_total": 0.3}

    def cleanup_during_ssh(*_args: object, **_kwargs: object) -> tuple[str, str, int]:
        vast_ops._mark_cleanup_required(
            1234, manifest.run_id, "explicit cleanup won before bootstrap"
        )
        return ("root", "host", 22)

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.load_manifest", return_value=manifest),
        patch(
            "vast.ops.create_source_bundle",
            return_value=(tmp_path / "source.bundle", tmp_path / "run.json"),
        ),
        patch("vast.ops.load_machine_blacklist", return_value=set()),
        patch("vast.ops.active_machine_ids", return_value=set()),
        patch("vast.ops.run_vast", return_value=[]),
        patch("vast.ops.search_offers", return_value=[offer]),
        patch("vast.ops.launch_detached_purchase_reconciliation", return_value=9001),
        patch("vast.ops.create_instance", return_value=1234),
        patch(
            "vast.ops.verify_created_instance",
            return_value={"id": 1234, "dph_total": 0.3},
        ),
        patch("vast.ops.wait_for_ssh", side_effect=cleanup_during_ssh),
        patch("vast.ops.start_remote_run") as start,
        patch("vast.ops.destroy_instance"),
        pytest.raises(OpsError, match="lost lifecycle authority"),
    ):
        command_launch(args)
    start.assert_not_called()
    updated = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert updated["status"] == "cleanup_required"
    assert updated["cleanup_required"] is True


def test_command_launch_cannot_resurrect_cleanup_that_arrives_during_bootstrap(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    args = SimpleNamespace(manifest=tmp_path / "run.json", offer=None, detach=True)
    offer = {"id": 11, "machine_id": 101, "dph_total": 0.3}

    def cleanup_during_start(*_args: object, **_kwargs: object) -> None:
        vast_ops._mark_cleanup_required(
            1234, manifest.run_id, "cleanup arrived during remote bootstrap"
        )

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.load_manifest", return_value=manifest),
        patch(
            "vast.ops.create_source_bundle",
            return_value=(tmp_path / "source.bundle", tmp_path / "run.json"),
        ),
        patch("vast.ops.load_machine_blacklist", return_value=set()),
        patch("vast.ops.active_machine_ids", return_value=set()),
        patch("vast.ops.run_vast", return_value=[]),
        patch("vast.ops.search_offers", return_value=[offer]),
        patch("vast.ops.launch_detached_purchase_reconciliation", return_value=9001),
        patch("vast.ops.create_instance", return_value=1234),
        patch(
            "vast.ops.verify_created_instance",
            return_value={"id": 1234, "dph_total": 0.3},
        ),
        patch("vast.ops.wait_for_ssh", return_value=("root", "host", 22)),
        patch("vast.ops.start_remote_run", side_effect=cleanup_during_start) as start,
        patch("vast.ops.destroy_instance"),
        pytest.raises(OpsError, match="lost lifecycle authority"),
    ):
        command_launch(args)
    start.assert_called_once()
    updated = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert updated["status"] == "cleanup_required"
    assert updated["cleanup_required"] is True


def test_purchase_supervisor_transfers_running_remote_job_to_monitor(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    state_path.write_text(
        json.dumps(
            [
                {
                    "run_id": manifest.run_id,
                    "created_at": utc_now(),
                    "campaign_started_at": utc_now(),
                    "manifest": manifest.__dict__,
                    "status": "running",
                    "instance_id": 1234,
                    "provisioning_owner_pid": 4241,
                }
            ]
        ),
        encoding="utf-8",
    )
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops.process_is_alive", return_value=False),
        patch("vast.ops.launch_detached_monitor", return_value=4243) as monitor,
    ):
        from vast.ops import _purchase_reconciliation

        assert _purchase_reconciliation(manifest.run_id) == 0
    monitor.assert_called_once_with(manifest.run_id)


def test_purchase_supervisor_waits_for_monitor_registration_while_owner_alive(
    tmp_path: Path,
) -> None:
    """Owner liveness alone is not a completed running-to-monitor handoff."""

    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    deadline = (datetime.now(UTC) + timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    state_path.write_text(
        json.dumps(
            [
                {
                    "run_id": manifest.run_id,
                    "created_at": utc_now(),
                    "campaign_started_at": utc_now(),
                    "purchase_reconciliation_deadline_at": deadline,
                    "manifest": manifest.__dict__,
                    "status": "running",
                    "instance_id": 1234,
                    "provisioning_owner_pid": 4241,
                    "monitor_pid": None,
                }
            ]
        ),
        encoding="utf-8",
    )
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops.process_is_alive", return_value=True),
        # The first poll has no monitor lease.  The second observes the
        # registered monitor's held OS lock; bare PID liveness is not used.
        patch("vast.ops._supervisor_lock_owner_pid", side_effect=[None, 4243]),
        patch("vast.ops.PURCHASE_RECONCILIATION_POLL_SECONDS", 0),
        patch("vast.ops.launch_detached_monitor") as launch,
        patch("vast.ops.time.sleep"),
    ):
        assert _purchase_reconciliation(manifest.run_id) == 0
    launch.assert_not_called()


def test_purchase_supervisor_retries_failed_monitor_takeover_by_destroying_at_deadline(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    expired = (datetime.now(UTC) - timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
    state_path.write_text(
        json.dumps(
            [
                {
                    "run_id": manifest.run_id,
                    "created_at": utc_now(),
                    "campaign_started_at": utc_now(),
                    "purchase_reconciliation_deadline_at": expired,
                    "manifest": manifest.__dict__,
                    "status": "running",
                    "instance_id": 1234,
                    "provisioning_owner_pid": 4241,
                }
            ]
        ),
        encoding="utf-8",
    )
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops.process_is_alive", return_value=False),
        patch(
            "vast.ops.launch_detached_monitor",
            side_effect=OpsError("spawn unavailable"),
        ),
        patch("vast.ops.destroy_instance") as destroy,
    ):
        assert _purchase_reconciliation(manifest.run_id) == 0
    destroy.assert_called_once()
    assert destroy.call_args.args == (
        1234,
        manifest.run_id,
        "monitor_takeover_failed_at_reconciliation_deadline",
    )
    assert destroy.call_args.kwargs["deadline_monotonic"] > 0


def test_monitor_does_not_reverse_terminal_destroyed_record() -> None:
    manifest = adoption_manifest()
    record = {
        "run_id": manifest.run_id,
        "instance_id": 1234,
        "dph": 0.3,
        "manifest": manifest.__dict__,
        "status": "destroyed",
    }
    with (
        patch("vast.ops.get_record", return_value=record),
        patch("vast.ops.run_vast") as api,
        patch("vast.ops.destroy_instance") as destroy,
    ):
        assert _monitor_run(manifest.run_id) == 0
    api.assert_not_called()
    destroy.assert_not_called()


def test_monitor_state_merge_is_fenced_after_cleanup_transition(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    state_path.write_text(
        json.dumps(
            [
                {
                    "run_id": manifest.run_id,
                    "instance_id": 1234,
                    "manifest": manifest.__dict__,
                    "status": "provisioning_cleanup_pending",
                    "cleanup_required": True,
                    "cleanup_error": "destroy retry pending",
                }
            ]
        ),
        encoding="utf-8",
    )
    with patch("vast.ops.STATE_FILE", state_path):
        wrote, current = _monitor_write_if_active(
            manifest.run_id,
            status="running",
            last_event_signature="stale",
            event="remote_state_change",
            event_fields={"remote": {"event": "run_succeeded"}},
        )
    assert wrote is False
    assert current["status"] == "provisioning_cleanup_pending"
    assert current["cleanup_required"] is True
    assert current["cleanup_error"] == "destroy retry pending"
    assert json.loads(state_path.read_text(encoding="utf-8"))[0]["status"] == (
        "provisioning_cleanup_pending"
    )


def test_reconcile_does_not_spawn_duplicate_live_ordinary_monitor(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    state_path.write_text(
        json.dumps(
            [
                {
                    "run_id": manifest.run_id,
                    "instance_id": 1234,
                    "expected_label": f"quant-experiments-{manifest.run_id}",
                    "manifest": manifest.__dict__,
                    "status": "running",
                    "monitor_pid": 4243,
                }
            ]
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(restart_monitors=True)
    live = [{"id": 1234, "label": f"quant-experiments-{manifest.run_id}"}]
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.run_vast", return_value=live),
        patch("vast.ops.process_is_alive", return_value=True),
        patch("vast.ops._supervisor_lock_owner_pid", return_value=4243),
        patch("vast.ops.launch_detached_monitor") as launch,
    ):
        assert command_reconcile(args) == 0
    launch.assert_not_called()


def test_precreate_label_check_rejects_live_collision() -> None:
    from vast.ops import assert_live_label_unused

    with (
        patch(
            "vast.ops.run_vast",
            return_value=[{"id": 77, "label": "quant-experiments-collision"}],
        ),
        pytest.raises(OpsError, match="already present"),
    ):
        assert_live_label_unused("quant-experiments-collision")


def adoption_manifest() -> RunManifest:
    return RunManifest.from_dict(
        {
            "format_version": 1,
            "run_id": "adopt_seed0",
            "description": "mocked existing-instance adoption",
            "source_commit": "a" * 40,
            "command": ["python", "scripts/train_v0.py", "--generations", "1"],
            "artifact_paths": ["artifacts/adopt_seed0"],
            "gpu_name": "RTX_5090",
            "max_dph": 0.5,
            "max_total_usd": 1.0,
            "ttl_hours": 2.0,
            "failure_grace_minutes": 0,
            "disk_gb": 30,
            "poll_seconds": 60,
            "seed_policy": "seed 0",
            "interaction_budget": "one mocked generation",
            "checkpoint_cadence": "final checkpoint",
            "intervention_suite": "none",
            "min_cuda": 12.8,
            "min_inet_down_mbps": 100,
            "min_cpu_cores_effective": 4,
            "min_cpu_ram_gb": 16,
            "require_consumer_cpu": True,
            "required_geolocation": "South Korea, KR",
        }
    )


def _artifact_pull_fixture(
    tmp_path: Path,
    *,
    exit_code: object = 0,
    manifest_updates: dict[str, object] | None = None,
    extra_archive_member: bool = False,
    extra_manifest_member: bool = False,
    raw_manifest_text: str | None = None,
) -> tuple[dict[str, object], Path, object, list[str]]:
    manifest = adoption_manifest()
    source = tmp_path / "pull-source"
    source.mkdir()
    payload = source / "payload.txt"
    payload.write_text("public receipt", encoding="utf-8")
    artifact_path = "artifacts/adopt_seed0/payload.txt"
    archive = source / "artifacts.tar.gz"
    with tarfile.open(archive, "w:gz", format=tarfile.USTAR_FORMAT) as bundle:
        bundle.add(payload, arcname=artifact_path)
        if extra_archive_member:
            private = source / "private.bin"
            private.write_bytes(b"must not be recovered")
            bundle.add(private, arcname="private.bin")
    artifact_manifest = {
        "format_version": 1,
        "run_id": manifest.run_id,
        "source_commit": manifest.source_commit,
        "exit_code": exit_code,
        "missing_declared_paths": [],
        "files": [
            {
                "path": artifact_path,
                "size": payload.stat().st_size,
                "sha256": vast_ops.sha256_file(payload),
            }
        ],
        "archive_sha256": vast_ops.sha256_file(archive),
    }
    if extra_manifest_member:
        private = source / "private.bin"
        if not private.is_file():
            raise AssertionError("extra manifest member requires extra archive member")
        artifact_manifest["files"].append(
            {
                "path": "private.bin",
                "size": private.stat().st_size,
                "sha256": vast_ops.sha256_file(private),
            }
        )
    if manifest_updates:
        artifact_manifest.update(manifest_updates)
    artifact_manifest_path = source / "artifact_manifest.json"
    artifact_manifest_path.write_text(
        raw_manifest_text
        if raw_manifest_text is not None
        else json.dumps(artifact_manifest),
        encoding="utf-8",
    )
    record: dict[str, object] = {
        "run_id": manifest.run_id,
        "instance_id": 1234,
        "source_commit": manifest.source_commit,
        "manifest": manifest.__dict__,
    }
    destination = tmp_path / "downloaded"
    calls: list[str] = []

    def download(
        _connection: object,
        _remote: object,
        local: Path,
        **_kwargs: object,
    ) -> bool:
        calls.append(local.name)
        local.parent.mkdir(parents=True, exist_ok=True)
        if local.name == "artifact_manifest.json":
            local.write_bytes(artifact_manifest_path.read_bytes())
        elif local.name == "artifacts.tar.gz":
            local.write_bytes(archive.read_bytes())
        else:
            local.write_text("", encoding="utf-8")
        return True

    return record, destination, download, calls


def test_pull_rejects_manifest_and_archive_member_outside_declared_artifacts(
    tmp_path: Path,
) -> None:
    record, destination, download, _calls = _artifact_pull_fixture(
        tmp_path,
        extra_archive_member=True,
        extra_manifest_member=True,
    )
    with (
        patch("vast.ops.ssh_url", return_value=("root", "host", 22)),
        patch("vast.ops.scp_download", side_effect=download),
        pytest.raises(OpsError, match="outside declared artifacts"),
    ):
        vast_ops.pull_and_verify(record, destination=destination)

    assert not (destination / "pull_verified.json").exists()
    assert not (destination / "pull_recovered.json").exists()
    assert not (destination / "extracted").exists()


def test_best_effort_recovery_receipt_cannot_be_reused_as_strict_success(
    tmp_path: Path,
) -> None:
    record, destination, download, calls = _artifact_pull_fixture(
        tmp_path, exit_code=1
    )
    with (
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.ssh_url", return_value=("root", "host", 22)),
        patch("vast.ops.scp_download", side_effect=download),
    ):
        assert vast_ops.pull_and_verify(
            record, destination=destination, best_effort=True
        )
        with pytest.raises(OpsError, match="exact integer exit_code zero"):
            vast_ops.pull_and_verify(record, destination=destination)

    assert len(calls) == 4
    assert set(calls) == {"artifact_manifest.json", "artifacts.tar.gz"}
    assert not (destination / "pull_verified.json").exists()
    recovery = json.loads(
        (destination / "pull_recovered.json").read_text(encoding="utf-8")
    )
    assert recovery["verification_mode"] == "best_effort_recovery"
    assert recovery["artifact_exit_code"] == 1


@pytest.mark.parametrize("exit_code", [False, "0", None])
def test_best_effort_pull_rejects_malformed_exit_code_without_any_receipt(
    tmp_path: Path, exit_code: object
) -> None:
    record, destination, download, _calls = _artifact_pull_fixture(
        tmp_path, exit_code=exit_code
    )
    with (
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.ssh_url", return_value=("root", "host", 22)),
        patch("vast.ops.scp_download", side_effect=download),
        pytest.raises(OpsError, match="exact integer exit_code"),
    ):
        vast_ops.pull_and_verify(record, destination=destination, best_effort=True)

    assert not (destination / "pull_verified.json").exists()
    assert not (destination / "pull_recovered.json").exists()


def test_valid_strict_pull_receipt_is_bound_and_can_cache_hit(tmp_path: Path) -> None:
    record, destination, download, calls = _artifact_pull_fixture(tmp_path)
    with (
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.ssh_url", return_value=("root", "host", 22)) as connection,
        patch("vast.ops.scp_download", side_effect=download),
    ):
        assert vast_ops.pull_and_verify(record, destination=destination)
        assert vast_ops.pull_and_verify(record, destination=destination)

    assert len(calls) == 2
    assert set(calls) == {"artifact_manifest.json", "artifacts.tar.gz"}
    connection.assert_called_once()
    receipt = json.loads(
        (destination / "pull_verified.json").read_text(encoding="utf-8")
    )
    assert receipt["receipt_version"] == 1
    assert receipt["verification_mode"] == "strict_success"
    assert type(receipt["artifact_exit_code"]) is int
    assert receipt["artifact_exit_code"] == 0


@pytest.mark.parametrize(
    "receipt_mutation",
    [
        {},
        {"verification_mode": "best_effort_recovery"},
        {"artifact_exit_code": True},
        {"run_id": "another-run"},
    ],
)
def test_legacy_or_binding_mismatched_pull_receipt_cannot_short_circuit(
    tmp_path: Path, receipt_mutation: dict[str, object]
) -> None:
    record, destination, _download, _calls = _artifact_pull_fixture(tmp_path)
    destination.mkdir(parents=True)
    (destination / "pull_verified.json").write_text(
        json.dumps(receipt_mutation), encoding="utf-8"
    )
    with (
        patch("vast.ops.ssh_url", side_effect=OpsError("cache rejected")) as connection,
        pytest.raises(OpsError, match="cache rejected"),
    ):
        vast_ops.pull_and_verify(record, destination=destination)
    connection.assert_called_once()


def test_pull_rejects_duplicate_artifact_manifest_keys(tmp_path: Path) -> None:
    manifest = adoption_manifest()
    raw = (
        '{"format_version":1,"run_id":"'
        + manifest.run_id
        + '","source_commit":"'
        + manifest.source_commit
        + '","exit_code":1,"exit_code":0,"missing_declared_paths":[],"files":[],'
        '"archive_sha256":"'
        + "0" * 64
        + '"}'
    )
    record, destination, download, _calls = _artifact_pull_fixture(
        tmp_path, raw_manifest_text=raw
    )
    with (
        patch("vast.ops.ssh_url", return_value=("root", "host", 22)),
        patch("vast.ops.scp_download", side_effect=download),
        pytest.raises(OpsError, match="manifest is malformed"),
    ):
        vast_ops.pull_and_verify(record, destination=destination)


@pytest.mark.parametrize("missing_value", [False, 0, {}, ""])
def test_pull_requires_exact_missing_declared_paths_list(
    tmp_path: Path, missing_value: object
) -> None:
    record, destination, download, _calls = _artifact_pull_fixture(
        tmp_path, manifest_updates={"missing_declared_paths": missing_value}
    )
    with (
        patch("vast.ops.ssh_url", return_value=("root", "host", 22)),
        patch("vast.ops.scp_download", side_effect=download),
        pytest.raises(OpsError, match="missing-path list is malformed"),
    ):
        vast_ops.pull_and_verify(record, destination=destination)


def test_pull_rejects_undeclared_archive_member_before_extraction(
    tmp_path: Path,
) -> None:
    record, destination, download, _calls = _artifact_pull_fixture(
        tmp_path, extra_archive_member=True
    )
    with (
        patch("vast.ops.ssh_url", return_value=("root", "host", 22)),
        patch("vast.ops.scp_download", side_effect=download),
        pytest.raises(OpsError, match="exactly match"),
    ):
        vast_ops.pull_and_verify(record, destination=destination)

    assert not (destination / "extracted" / "private.bin").exists()


def test_manifest_accepts_and_validates_multi_gpu_count() -> None:
    assert replace(adoption_manifest(), num_gpus=4).num_gpus == 4
    with pytest.raises(OpsError, match="num_gpus"):
        RunManifest.from_dict({**adoption_manifest().__dict__, "num_gpus": 0})


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_dph", True),
        ("max_total_usd", float("nan")),
        ("ttl_hours", float("inf")),
        ("disk_gb", 30.5),
        ("poll_seconds", "60"),
        ("min_cuda", float("-inf")),
        ("num_gpus", False),
    ],
)
def test_manifest_numeric_fields_reject_bool_non_number_and_nonfinite(
    field: str, value: object
) -> None:
    with pytest.raises(OpsError, match=field):
        RunManifest.from_dict({**adoption_manifest().__dict__, field: value})


def adoption_instance(*, age_minutes: float = 5, **updates: object) -> dict[str, object]:
    result: dict[str, object] = {
        "id": 1234,
        "actual_status": "running",
        "label": None,
        "gpu_name": "RTX 5090",
        "num_gpus": 1,
        "dph_total": 0.4,
        "disk_space": 40,
        "cuda_max_good": 12.8,
        "inet_down": 500,
        "cpu_cores_effective": 8,
        "cpu_ram": 32 * 1024,
        "cpu_name": "AMD Ryzen 9 7900X",
        "direct_port_count": 1,
        "geolocation": "South Korea, KR",
        "machine_id": 999,
        "verification": "verified",
        "start_date": (datetime.now(UTC) - timedelta(minutes=age_minutes)).timestamp(),
    }
    result.update(updates)
    return result


def test_adoption_derives_identity_from_manifest_and_instance() -> None:
    manifest = adoption_manifest()
    require_adoption_approval(manifest, instance_id=1234)
    with pytest.raises(OpsError, match="positive"):
        require_adoption_approval(manifest, instance_id=0)


def test_adoption_parser_only_requires_manifest_and_instance() -> None:
    args = build_parser().parse_args(
        [
            "adopt-existing",
            "run.json",
            "--instance",
            "1234",
        ]
    )
    assert args.instance == 1234
    assert not hasattr(args, "confirm_run")
    assert not hasattr(args, "approve_max_usd")
    assert not hasattr(args, "confirm_instance")
    assert not hasattr(args, "confirm_source_commit")


@pytest.mark.parametrize(
    ("function", "flag", "args", "message"),
    [
        (
            command_race,
            "ENABLE_CANDIDATE_RACE",
            SimpleNamespace(candidates=2, manifest=Path("unused"), detach=True),
            "candidate race is disabled",
        ),
        (
            command_adopt_existing,
            "ENABLE_EXISTING_INSTANCE_ADOPTION",
            SimpleNamespace(instance=1234, manifest=Path("unused"), detach=True),
            "adopt-existing is disabled",
        ),
    ],
)
def test_excluded_lifecycle_modes_fail_before_manifest_or_provider_access(
    monkeypatch: pytest.MonkeyPatch,
    function: object,
    flag: str,
    args: SimpleNamespace,
    message: str,
) -> None:
    monkeypatch.setattr(vast_ops, flag, False)
    with (
        patch("vast.ops.load_manifest") as load,
        pytest.raises(OpsError, match=message),
    ):
        function(args)  # type: ignore[operator]
    load.assert_not_called()


def test_launch_parser_uses_manifest_contract_without_redundant_confirmations() -> None:
    args = build_parser().parse_args(["launch", "run.json"])
    assert args.manifest == Path("run.json")
    assert not hasattr(args, "confirm_run")
    assert not hasattr(args, "approve_max_usd")


def test_remote_watchdog_deadline_uses_precreate_cap_and_ttl() -> None:
    manifest = adoption_manifest()
    started = "2026-08-31T00:00:00Z"
    deadline = remote_watchdog_deadline_epoch(
        {"id": 11, "machine_id": 101, "dph_total": 0.25},
        manifest,
        campaign_started_at=started,
    )
    expected = datetime.fromisoformat(started.replace("Z", "+00:00")).timestamp()
    # max_total_usd / dph_total is four hours, so the two-hour TTL wins;
    # the watchdog is armed two minutes before that hard boundary.
    assert deadline == expected + (2 * 3600) - 120


def test_remote_watchdog_deadline_uses_manifest_ceiling_not_cheaper_offer() -> None:
    manifest = replace(adoption_manifest(), ttl_hours=10.0, max_total_usd=1.0)
    started = "2026-08-31T00:00:00Z"
    deadline = remote_watchdog_deadline_epoch(
        {"id": 11, "machine_id": 101, "dph_total": 0.01},
        manifest,
        campaign_started_at=started,
    )
    expected = datetime.fromisoformat(started.replace("Z", "+00:00")).timestamp()
    # The manifest ceiling is $0.50/h, so the spend boundary is two hours;
    # the cheaper quoted offer must not extend the in-container kill switch.
    assert deadline == expected + (2 * 3600) - 120


def test_remote_watchdog_payload_is_deterministic_and_provider_scoped() -> None:
    command = remote_watchdog_onstart_command(1_756_608_000.0)
    repeat = remote_watchdog_onstart_command(1_756_608_000.0)
    assert command == repeat
    match = re.search(r"b64decode\('([A-Za-z0-9+/=]+)'\)", command)
    assert match is not None
    payload = base64.b64decode(match.group(1)).decode("ascii")
    assert "https://console.vast.ai/api/v0/instances/" in payload
    assert 'os.environ.get("CONTAINER_ID"' in payload
    assert 'os.environ.get("CONTAINER_API_KEY"' in payload
    assert 'method="DELETE"' in payload
    assert "Vast container kill-switch credentials are unavailable" in payload
    assert "MONOTONIC_DEADLINE" in payload
    assert '"schema": "qe-vast-watchdog-ready-v1"' in payload
    assert '"instance_id": INSTANCE_ID' in payload
    assert '"proc_start_ticks": start_ticks' in payload
    assert '"cmdline_sha256": cmdline_sha256' in payload
    assert "os.replace(temporary, ARMED_PATH)" in payload
    assert "VAST_API_KEY" not in command
    assert "account-secret-that-must-not-cross" not in command
    assert "nohup python3 -c" in command


def test_remote_watchdog_launcher_fails_closed_before_ready() -> None:
    command = remote_watchdog_onstart_command(1_756_608_000.0)
    assert "command -v python3 >/dev/null 2>&1 || exit 125" in command
    assert "qe-vast-watchdog-v1" in command
    assert "watchdog_pid" not in command
    assert "kill -0" not in command
    assert "echo WATCHDOG_ARMED" not in command
    assert "watchdog_armed.json" in command


def test_create_instance_embeds_watchdog_in_onstart_command() -> None:
    manifest = adoption_manifest()
    with patch(
        "vast.ops.run_vast",
        return_value={"success": True, "new_contract": 201},
    ) as run:
        assert (
            create_instance(
                {"id": 11, "machine_id": 101, "dph_total": 0.25},
                manifest,
                campaign_started_at="2026-08-31T00:00:00Z",
            )
            == 201
        )
    args = list(run.call_args.args)
    onstart = args[args.index("--onstart-cmd") + 1]
    assert "set -eu; mkdir -p" in onstart
    assert "nohup python3 -c" in onstart
    assert "CONTAINER_API_KEY" not in onstart
    assert "watchdog_armed.json" in onstart
    assert "qe-vast-watchdog-v1" in onstart
    assert "echo INSTANCE_READY" in onstart
    assert onstart.index("qe-vast-watchdog-v1") < onstart.index(
        "echo INSTANCE_READY"
    )
    assert "&;" not in onstart


@pytest.mark.parametrize("success", [1, "true", [True], {"value": True}])
def test_create_instance_requires_exact_boolean_success(success: object) -> None:
    manifest = adoption_manifest()
    with (
        patch(
            "vast.ops.run_vast",
            return_value={"success": success, "new_contract": 201},
        ),
        pytest.raises(OpsError, match="malformed or negative"),
    ):
        create_instance(
            {"id": 11, "machine_id": 101, "dph_total": 0.25},
            manifest,
            campaign_started_at="2026-08-31T00:00:00Z",
        )


@pytest.mark.parametrize("live_label", [None, "another-contract"])
def test_created_instance_must_match_the_exact_requested_label(
    live_label: object,
) -> None:
    manifest = adoption_manifest()
    instance = adoption_instance(id=42, label=live_label)
    with (
        patch("vast.ops.run_vast", return_value=[instance]),
        pytest.raises(OpsError, match="label"),
    ):
        verify_created_instance(
            42,
            manifest,
            expected_label="quant-experiments-adopt_seed0",
        )


def test_search_parser_accepts_neutral_adapter_cpu_and_country_constraints() -> None:
    args = build_parser().parse_args(
        [
            "search",
            "--gpu",
            "RTX_3060",
            "--max-dph",
            "0.25",
            "--require-consumer-cpu",
            "--country",
            "CA",
        ]
    )
    assert args.require_consumer_cpu is True
    assert args.country == "CA"


def test_race_parser_rejects_more_than_three_candidates_before_purchase() -> None:
    args = SimpleNamespace(manifest=Path("run.json"), candidates=4, detach=True)
    with pytest.raises(OpsError, match=r"candidates must be in \[1, 3\]"):
        command_race(args)


def test_race_rejects_a_nonpositive_precleanup_window(tmp_path: Path) -> None:
    manifest = replace(adoption_manifest(), ttl_hours=0.01)
    args = SimpleNamespace(manifest=tmp_path / "run.json", candidates=1, detach=True)
    offer = {"id": 11, "machine_id": 101, "dph_total": 0.3}
    with (
        patch("vast.ops.load_manifest", return_value=manifest),
        patch(
            "vast.ops.create_source_bundle",
            return_value=(tmp_path / "source.bundle", tmp_path / "resolved.json"),
        ),
        patch("vast.ops.load_machine_blacklist", return_value=set()),
        patch("vast.ops.active_machine_ids", return_value=set()),
        patch("vast.ops.search_offers", return_value=[offer]),
        patch("vast.ops.launch_detached_race_watchdog", return_value=901),
        pytest.raises(OpsError, match="no positive race window"),
    ):
        command_race(args)


def test_race_requires_winner_cleanup_reserve_before_training(tmp_path: Path) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    args = SimpleNamespace(manifest=tmp_path / "run.json", candidates=1, detach=True)
    offer = {"id": 11, "machine_id": 101, "dph_total": 0.3}
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.load_manifest", return_value=manifest),
        patch(
            "vast.ops.create_source_bundle",
            return_value=(tmp_path / "source.bundle", tmp_path / "resolved.json"),
        ),
        patch("vast.ops.load_machine_blacklist", return_value=set()),
        patch("vast.ops.active_machine_ids", return_value=set()),
        patch("vast.ops.search_offers", return_value=[offer]),
        patch("vast.ops.launch_detached_race_watchdog", return_value=901),
        patch("vast.ops.create_instance", return_value=201),
        patch("vast.ops.verify_created_instance", return_value={"dph_total": 0.3}),
        patch(
            "vast.ops.wait_for_ssh",
            return_value=("root", "ready", 22),
        ),
        patch("vast.ops.verify_cuda_readiness", return_value={"cuda_ready": True}),
        patch("vast.ops._race_candidate_cost", return_value=0.97),
        patch("vast.ops.destroy_instance") as destroy,
        patch("vast.ops.start_remote_run") as start,
        pytest.raises(OpsError, match="cleanup reserve"),
    ):
        command_race(args)

    start.assert_not_called()
    destroy.assert_called_once_with(201, manifest.run_id, "race_spend_cap")


def test_race_starts_only_first_ready_candidate_and_destroys_losers(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    manifest_path = tmp_path / "run.json"
    state_path = tmp_path / "instances.json"
    event_path = tmp_path / "events.jsonl"
    offers = [
        {"id": 11, "machine_id": 101, "dph_total": 0.30},
        {"id": 12, "machine_id": 102, "dph_total": 0.31},
        {"id": 13, "machine_id": 103, "dph_total": 0.32},
    ]
    instance_by_offer = {11: 201, 12: 202, 13: 203}
    connection_by_instance = {
        201: ("root", "slow-a", 22),
        202: ("root", "fast", 22),
        203: ("root", "slow-b", 22),
    }
    args = SimpleNamespace(manifest=manifest_path, candidates=3, detach=True)

    def fake_create(offer: dict[str, object], *_: object, **__: object) -> int:
        return instance_by_offer[int(offer["id"])]

    def fake_verify(instance_id: int, _manifest: RunManifest, **_: object) -> dict[str, object]:
        return {"dph_total": 0.3, "machine_id": instance_id}

    def fake_cuda(connection: tuple[str, str, int], _manifest: RunManifest) -> dict[str, object]:
        # Candidate 2 is deliberately the first readiness check to finish.
        if connection[1] != "fast":
            import time as _time

            # Leave enough separation to remain deterministic when the full
            # repository suite has heavily loaded the local scheduler.
            _time.sleep(0.2)
        return {"cuda_ready": True, "report": connection[1]}

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.EVENT_FILE", event_path),
        patch("vast.ops.load_manifest", return_value=manifest),
        patch(
            "vast.ops.create_source_bundle",
            return_value=(tmp_path / "source.bundle", tmp_path / "resolved.json"),
        ),
        patch("vast.ops.load_machine_blacklist", return_value=set()),
        patch("vast.ops.active_machine_ids", return_value=set()),
        patch("vast.ops.search_offers", return_value=offers),
        patch("vast.ops.launch_detached_race_watchdog", return_value=901),
        patch("vast.ops.create_instance", side_effect=fake_create),
        patch("vast.ops.verify_created_instance", side_effect=fake_verify),
        patch(
            "vast.ops.wait_for_ssh",
            side_effect=lambda instance_id, **_: connection_by_instance[instance_id],
        ),
        patch("vast.ops.verify_cuda_readiness", side_effect=fake_cuda),
        patch("vast.ops.destroy_instance") as destroy,
        patch("vast.ops.start_remote_run") as start,
        patch("vast.ops.start_requested_monitor", return_value=0),
    ):
        assert command_race(args) == 0
        record = load_state()[0]

    assert start.call_count == 1
    assert start.call_args.args[0] == connection_by_instance[202]
    destroyed_ids = {call.args[0] for call in destroy.call_args_list}
    assert destroyed_ids == {201, 203}
    assert record["instance_id"] == 202
    assert record["status"] == "running"
    assert record["campaign_cap_usd"] == manifest.max_total_usd
    assert {candidate["label"] for candidate in record["candidates"]} == {
        "quant-experiments-adopt_seed0-candidate1",
        "quant-experiments-adopt_seed0-candidate2",
        "quant-experiments-adopt_seed0-candidate3",
    }


def test_race_no_winner_surfaces_live_candidates_and_starts_cleanup_supervisor(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    args = SimpleNamespace(manifest=tmp_path / "run.json", candidates=2, detach=True)
    offers = [
        {"id": 11, "machine_id": 101, "dph_total": 0.30},
        {"id": 12, "machine_id": 102, "dph_total": 0.31},
    ]
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.load_manifest", return_value=manifest),
        patch(
            "vast.ops.create_source_bundle",
            return_value=(tmp_path / "source.bundle", tmp_path / "resolved.json"),
        ),
        patch("vast.ops.load_machine_blacklist", return_value=set()),
        patch("vast.ops.active_machine_ids", return_value=set()),
        patch("vast.ops.search_offers", return_value=offers),
        patch("vast.ops.launch_detached_race_watchdog", return_value=901),
        patch("vast.ops.create_instance", side_effect=[301, 302]),
        patch(
            "vast.ops.verify_created_instance",
            side_effect=lambda instance_id, _manifest, **_: {"dph_total": 0.3},
        ),
        patch(
            "vast.ops.wait_for_ssh",
            side_effect=lambda instance_id, **_: ("root", f"host-{instance_id}", 22),
        ),
        patch(
            "vast.ops.verify_cuda_readiness",
            side_effect=OpsError("CUDA readiness failed"),
        ),
        patch("vast.ops.destroy_instance", side_effect=OpsError("destroy unavailable")),
        patch("vast.ops.launch_detached_race_cleanup", return_value=77) as supervisor,
        patch("vast.ops.start_remote_run") as start,
    ):
        with pytest.raises(OpsError, match="live candidates"):
            command_race(args)
        record = load_state()[0]

    start.assert_not_called()
    supervisor.assert_called_once_with(manifest.run_id)
    assert record["status"] == "provisioning_cleanup_pending"
    assert {candidate["instance_id"] for candidate in record["candidates"]} == {
        301,
        302,
    }


def test_race_winner_cleanup_failure_supervises_entire_candidate_ledger(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    args = SimpleNamespace(manifest=tmp_path / "run.json", candidates=2, detach=True)
    offers = [
        {"id": 11, "machine_id": 101, "dph_total": 0.30},
        {"id": 12, "machine_id": 102, "dph_total": 0.31},
    ]

    def fake_cuda(connection: tuple[str, str, int], _manifest: RunManifest) -> dict[str, object]:
        if connection[1] == "host-302":
            import time as _time

            _time.sleep(0.03)
        return {"cuda_ready": True}

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.load_manifest", return_value=manifest),
        patch(
            "vast.ops.create_source_bundle",
            return_value=(tmp_path / "source.bundle", tmp_path / "resolved.json"),
        ),
        patch("vast.ops.load_machine_blacklist", return_value=set()),
        patch("vast.ops.active_machine_ids", return_value=set()),
        patch("vast.ops.search_offers", return_value=offers),
        patch("vast.ops.launch_detached_race_watchdog", return_value=901),
        patch("vast.ops.create_instance", side_effect=[301, 302]),
        patch(
            "vast.ops.verify_created_instance",
            side_effect=lambda instance_id, _manifest, **_: {"dph_total": 0.3},
        ),
        patch(
            "vast.ops.wait_for_ssh",
            side_effect=lambda instance_id, **_: ("root", f"host-{instance_id}", 22),
        ),
        patch("vast.ops.verify_cuda_readiness", side_effect=fake_cuda),
        patch("vast.ops.destroy_instance", side_effect=OpsError("destroy unavailable")),
        patch("vast.ops.launch_detached_race_cleanup", return_value=88) as supervisor,
        patch("vast.ops.start_remote_run") as start,
    ):
        with pytest.raises(OpsError, match="winner destruction failed"):
            command_race(args)
        record = load_state()[0]

    start.assert_not_called()
    supervisor.assert_called_once_with(manifest.run_id)
    assert record["status"] == "provisioning_cleanup_pending"
    assert record["instance_id"] in {301, 302}
    assert record["winner_cleanup_error"] == "destroy unavailable"


def test_race_unknown_create_outcome_triggers_reconcilable_cleanup(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    args = SimpleNamespace(manifest=tmp_path / "run.json", candidates=1, detach=True)
    offer = {"id": 11, "machine_id": 101, "dph_total": 0.3}
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.load_manifest", return_value=manifest),
        patch(
            "vast.ops.create_source_bundle",
            return_value=(tmp_path / "source.bundle", tmp_path / "resolved.json"),
        ),
        patch("vast.ops.load_machine_blacklist", return_value=set()),
        patch("vast.ops.active_machine_ids", return_value=set()),
        patch("vast.ops.search_offers", return_value=[offer]),
        patch("vast.ops.launch_detached_race_watchdog", return_value=901),
        patch("vast.ops.create_instance", side_effect=OpsError("create timeout")),
        patch("vast.ops.launch_detached_race_cleanup", return_value=902) as cleanup,
    ):
        with pytest.raises(OpsError, match="unknown candidates=1"):
            command_race(args)
        record = load_state()[0]

    cleanup.assert_called_once_with(manifest.run_id)
    assert record["status"] == "provisioning_cleanup_pending"
    assert record["candidates"][0]["status"] == "purchase_outcome_unknown"


def test_race_cleanup_marks_absent_unknown_purchase_terminal(tmp_path: Path) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    record = {
        "run_id": manifest.run_id,
        "created_at": utc_now(),
        "campaign_started_at": utc_now(),
        "campaign_cap_usd": manifest.max_total_usd,
        "manifest": manifest.__dict__,
        "candidates": [
            {
                "candidate_id": "adopt_seed0.candidate1",
                "label": "quant-experiments-adopt_seed0-candidate1",
                "status": "purchase_outcome_unknown",
                "instance_id": None,
                "dph": 0.3,
                "unknown_since": (datetime.now(UTC) - timedelta(seconds=200))
                .isoformat()
                .replace("+00:00", "Z"),
                "live_miss_count": 1,
            }
        ],
        "status": "provisioning_cleanup_pending",
    }
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops.run_vast", return_value=[]),
    ):
        assert _race_cleanup_pending(manifest.run_id) == 0
        updated = load_state()[0]
    assert updated["status"] == "race_failed"
    assert updated["candidates"][0]["status"] == "absent_from_live_api"


def test_stale_supervisor_pid_metadata_does_not_block_os_lock(tmp_path: Path) -> None:
    monitor_dir = tmp_path / "monitors"
    monitor_dir.mkdir()
    lock = monitor_dir / "race_cleanup_adopt_seed0.lock"
    lock.write_text(str(os.getpid()), encoding="utf-8")
    with (
        patch("vast.ops.MONITOR_DIR", monitor_dir),
    ):
        # PID metadata can be stale or reused. The kernel-held lock, acquired
        # before interpreting metadata, is the only ownership authority.
        from vast.ops import _claim_supervisor_lock, _release_supervisor_lock

        _claim_supervisor_lock(lock, "test supervisor")
        _release_supervisor_lock(lock)
    assert lock.is_file()


def test_reconcile_marks_interrupted_race_cleanup_required_and_restarts_supervisor(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    record = {
        "run_id": manifest.run_id,
        "created_at": utc_now(),
        "campaign_started_at": utc_now(),
        "coordinator_pid": 999999,
        "manifest": manifest.__dict__,
        "status": "race_pending",
        "candidates": [
            {
                "candidate_id": "adopt_seed0.candidate1",
                "label": "quant-experiments-adopt_seed0-candidate1",
                "status": "purchase_outcome_unknown",
                "instance_id": None,
                "dph": 0.3,
            }
        ],
    }
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    args = SimpleNamespace(restart_monitors=True)
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.run_vast", return_value=[]),
        patch("vast.ops.process_is_alive", return_value=False),
        patch("vast.ops.launch_detached_race_cleanup", return_value=777) as cleanup,
    ):
        assert command_reconcile(args) == 0
        updated = load_state()[0]
    cleanup.assert_called_once_with(manifest.run_id)
    assert updated["status"] == "race_cleanup_required"


def test_reconcile_reopens_locally_destroyed_but_provider_live_race_candidate(
    tmp_path: Path,
) -> None:
    """A delayed destroy response must remain cleanup work, never orphaned."""

    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    candidate_label = "quant-experiments-adopt_seed0-candidate1"
    record = {
        "run_id": manifest.run_id,
        "created_at": utc_now(),
        "campaign_started_at": utc_now(),
        "manifest": manifest.__dict__,
        "status": "race_failed",
        "candidates": [
            {
                "candidate_id": "adopt_seed0.candidate1",
                "label": candidate_label,
                "instance_id": 301,
                "status": "destroyed",
                "dph": 0.3,
            }
        ],
    }
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    args = SimpleNamespace(restart_monitors=True)
    live = [{"id": 301, "label": candidate_label}]
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.run_vast", return_value=live),
        patch("vast.ops.process_is_alive", return_value=False),
        patch("vast.ops.launch_detached_race_cleanup", return_value=778) as cleanup,
    ):
        assert command_reconcile(args) == 0
        updated = load_state()[0]

    cleanup.assert_called_once_with(manifest.run_id)
    assert updated["status"] == "race_cleanup_required"
    assert updated["cleanup_required"] is True
    assert updated["candidates"][0]["status"] == "cleanup_pending"
    assert updated["candidates"][0]["instance_id"] == 301


def test_reconcile_transient_absence_keeps_cleanup_and_reopens_on_reappearance(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    lock_path = tmp_path / "instances.state.lock"
    record = {
        "run_id": manifest.run_id,
        "instance_id": 1234,
        "created_at": utc_now(),
        "campaign_started_at": utc_now(),
        "manifest": manifest.__dict__,
        "status": "running",
    }
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    args = SimpleNamespace(restart_monitors=True)
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", lock_path),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops.run_vast", return_value=[]),
        patch(
            "vast.ops.launch_detached_destroy_reconciliation", return_value=77
        ) as verifier,
    ):
        assert command_reconcile(args) == 0
    absent = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert absent["status"] == "emergency_termination_unverified"
    assert absent["cleanup_required"] is True
    verifier.assert_called_once_with(manifest.run_id)

    live = [{"id": 1234, "label": f"quant-experiments-{manifest.run_id}"}]
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", lock_path),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops.run_vast", return_value=live),
        patch(
            "vast.ops.launch_detached_destroy_reconciliation", return_value=88
        ) as verifier,
    ):
        assert command_reconcile(args) == 0
    reopened = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert reopened["status"] == "cleanup_required"
    assert reopened["cleanup_required"] is True
    verifier.assert_called_once_with(manifest.run_id)


def test_reconcile_ignores_absence_snapshot_while_monitor_lease_is_held(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    monitor_dir = tmp_path / "monitors"
    record = {
        "run_id": manifest.run_id,
        "instance_id": 1234,
        "created_at": utc_now(),
        "campaign_started_at": utc_now(),
        "manifest": manifest.__dict__,
        "status": "running",
    }
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.MONITOR_DIR", monitor_dir),
    ):
        lease = monitor_dir / f"{manifest.run_id}.lock"
        _claim_supervisor_lock(lease, "healthy test monitor")
        try:
            with patch("vast.ops.run_vast", return_value=[]):
                assert command_reconcile(SimpleNamespace(restart_monitors=True)) == 0
        finally:
            _release_supervisor_lock(lease)
    updated = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert updated["status"] == "running"
    assert not updated.get("cleanup_required", False)


def test_reconcile_ignores_provider_snapshot_when_local_generation_changes(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    state_path.write_text(
        json.dumps(
            [
                {
                    "run_id": manifest.run_id,
                    "instance_id": 1234,
                    "created_at": utc_now(),
                    "campaign_started_at": utc_now(),
                    "manifest": manifest.__dict__,
                    "status": "provisioning",
                }
            ]
        ),
        encoding="utf-8",
    )

    def stale_empty_snapshot(*_args: object, **_kwargs: object) -> list[object]:
        vast_ops.update_record_fields(manifest.run_id, status="running")
        return []

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops.run_vast", side_effect=stale_empty_snapshot),
    ):
        assert command_reconcile(SimpleNamespace(restart_monitors=False)) == 0
    updated = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert updated["status"] == "running"
    assert not updated.get("cleanup_required", False)


def test_expired_late_create_is_destroyed_by_one_manual_reconcile(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    expired = (datetime.now(UTC) - timedelta(minutes=1)).isoformat().replace(
        "+00:00", "Z"
    )
    label = f"quant-experiments-{manifest.run_id}"
    live = [
        {"id": 777, "label": label, "ask_contract_id": 11, "machine_id": 101}
    ]
    state_path = tmp_path / "instances.json"
    state_path.write_text(
        json.dumps(
            [
                {
                    "run_id": manifest.run_id,
                    "instance_id": None,
                    "expected_label": label,
                    "offer_id": 11,
                    "machine_id": 101,
                    "created_at": expired,
                    "campaign_started_at": expired,
                    "purchase_reconciliation_deadline_at": expired,
                    "late_create_reconciliation_deadline_at": expired,
                    "controller_deadline_at": expired,
                    "manifest": manifest.__dict__,
                    "status": "late_create_reconciliation_expired",
                    "purchase_reconciliation_status": "late_watch_expired",
                    "cleanup_required": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    attempted_deadlines: list[float] = []
    successful_deadlines: list[float] = []

    def bounded_destroy(*_args: object, **kwargs: object) -> None:
        deadline = float(kwargs["deadline_monotonic"])
        attempted_deadlines.append(deadline)
        if deadline - time.monotonic() < 1:
            raise OpsError("expired controller deadline")
        successful_deadlines.append(deadline)

    def run_attached_reconciler(_run_id: str) -> int:
        assert _purchase_reconciliation(manifest.run_id) == 2
        return 909

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops.run_vast", side_effect=[live, live, live]),
        patch("vast.ops.destroy_instance", side_effect=bounded_destroy),
        patch(
            "vast.ops.launch_detached_purchase_reconciliation",
            side_effect=run_attached_reconciler,
        ),
    ):
        assert command_reconcile(SimpleNamespace(restart_monitors=True)) == 0
    assert len(attempted_deadlines) == 1
    assert len(successful_deadlines) == 1
    updated = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert updated["cap_control_failed"] is True
    assert updated["status"] == "late_create_reconciliation_expired"


def test_post_cap_absence_reconcile_gets_explicit_bounded_verification_lease(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    expired = (datetime.now(UTC) - timedelta(minutes=1)).isoformat().replace(
        "+00:00", "Z"
    )
    state_path = tmp_path / "instances.json"
    state_path.write_text(
        json.dumps(
            [
                {
                    "run_id": manifest.run_id,
                    "instance_id": 1234,
                    "created_at": expired,
                    "campaign_started_at": expired,
                    "controller_deadline_at": expired,
                    "manifest": manifest.__dict__,
                    "status": "cleanup_required",
                    "cleanup_required": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops.run_vast", return_value=[]),
        patch(
            "vast.ops.launch_detached_destroy_reconciliation", return_value=77
        ) as verifier,
    ):
        assert command_reconcile(SimpleNamespace(restart_monitors=True)) == 0
    updated = json.loads(state_path.read_text(encoding="utf-8"))[0]
    verifier.assert_called_once_with(manifest.run_id)
    assert updated["cap_control_failed"] is True
    manual_deadline = datetime.fromisoformat(
        updated["manual_destroy_reconciliation_deadline_at"].replace("Z", "+00:00")
    )
    assert manual_deadline > datetime.now(UTC)


def test_reconcile_restarts_crashed_destroy_followup_before_terminal_closure(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    state_path.write_text(
        json.dumps(
            [
                {
                    "run_id": manifest.run_id,
                    "instance_id": 1234,
                    "created_at": utc_now(),
                    "campaign_started_at": utc_now(),
                    "manifest": manifest.__dict__,
                    "status": "destroyed",
                    "destroy_reconciliation_status": "running",
                    "cleanup_required": False,
                }
            ]
        ),
        encoding="utf-8",
    )
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops.run_vast", return_value=[]),
        patch(
            "vast.ops.launch_detached_destroy_reconciliation", return_value=77
        ) as verifier,
    ):
        assert command_reconcile(SimpleNamespace(restart_monitors=True)) == 0
    updated = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert updated["status"] == "emergency_termination_unverified"
    assert updated["cleanup_required"] is True
    verifier.assert_called_once_with(manifest.run_id)


def test_reconcile_routes_new_exact_id_after_destroy_to_purchase_cleanup(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A delayed duplicate must not be hidden by the old ID's absence."""

    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    label = f"quant-experiments-{manifest.run_id}"
    record = {
        "run_id": manifest.run_id,
        "instance_id": 1234,
        "expected_label": label,
        "offer_id": 11,
        "machine_id": 101,
        "created_at": utc_now(),
        "campaign_started_at": utc_now(),
        "manifest": manifest.__dict__,
        "status": "destroyed",
        "destroy_reconciliation_status": "complete",
        "cleanup_required": False,
    }
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    late_duplicate = {
        "id": 777,
        "label": label,
        "ask_contract_id": 11,
        "machine_id": 101,
    }
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops.run_vast", return_value=[late_duplicate]),
        patch(
            "vast.ops.launch_detached_purchase_reconciliation", return_value=77
        ) as purchase_verifier,
        patch("vast.ops.launch_detached_destroy_reconciliation") as destroy_verifier,
    ):
        assert command_reconcile(SimpleNamespace(restart_monitors=True)) == 0

    updated = json.loads(state_path.read_text(encoding="utf-8"))[0]
    report = json.loads(capsys.readouterr().out)
    assert updated["status"] == "purchase_reconciled_cleanup_pending"
    assert updated["cleanup_required"] is True
    assert updated["instance_id"] == 1234
    assert updated["cleanup_generation"] > 0
    assert updated["reconciliation_exact_match_ids"] == [777]
    assert report["purchase_cleanup_matches"] == [
        {
            "run_id": manifest.run_id,
            "persisted_instance_id": 1234,
            "exact_live_instance_ids": [777],
            "invalid_live_instance_id": False,
        }
    ]
    assert report["untracked_live"] == []
    purchase_verifier.assert_called_once_with(manifest.run_id)
    destroy_verifier.assert_not_called()


def test_reconcile_exact_live_id_survives_malformed_persisted_id(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    label = f"quant-experiments-{manifest.run_id}"
    record = {
        "run_id": manifest.run_id,
        "instance_id": "not-an-instance-id",
        "expected_label": label,
        "offer_id": 11,
        "machine_id": 101,
        "manifest": manifest.__dict__,
        "status": "destroyed",
        "cleanup_required": False,
    }
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    live = [
        {"id": 777, "label": label, "ask_contract_id": 11, "machine_id": 101}
    ]
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops.run_vast", return_value=live),
        patch(
            "vast.ops.launch_detached_purchase_reconciliation", return_value=77
        ) as purchase_verifier,
        patch("vast.ops.launch_detached_destroy_reconciliation") as destroy_verifier,
    ):
        assert command_reconcile(SimpleNamespace(restart_monitors=True)) == 0

    updated = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert updated["status"] == "purchase_reconciled_cleanup_pending"
    assert updated["cleanup_required"] is True
    assert updated["reconciliation_exact_match_ids"] == [777]
    assert "persisted instance ID" in updated["persisted_instance_id_error"]
    purchase_verifier.assert_called_once_with(manifest.run_id)
    destroy_verifier.assert_not_called()


@pytest.mark.parametrize(
    "live_dph",
    ["NaN", "Infinity", "-Infinity", "1e400", True, False, 0, -0.1, {}, []],
)
def test_reconcile_never_persists_an_invalid_provider_price_or_loses_cleanup(
    tmp_path: Path,
    live_dph: object,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    label = f"quant-experiments-{manifest.run_id}"
    record = {
        "run_id": manifest.run_id,
        "instance_id": None,
        "expected_label": label,
        "offer_id": 11,
        "machine_id": 101,
        "dph": 0.3,
        "manifest": manifest.__dict__,
        "status": "purchase_outcome_unknown",
        "cleanup_required": False,
    }
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    live = [
        {
            "id": 777,
            "label": label,
            "ask_contract_id": 11,
            "machine_id": 101,
            "dph_total": live_dph,
        }
    ]
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops.run_vast", return_value=live),
        patch(
            "vast.ops.launch_detached_purchase_reconciliation", return_value=77
        ) as purchase_verifier,
        patch("vast.ops.launch_detached_destroy_reconciliation") as destroy_verifier,
    ):
        assert command_reconcile(SimpleNamespace(restart_monitors=True)) == 0

    updated = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert updated["instance_id"] == 777
    assert updated["dph"] == 0.3
    assert updated["provider_dph_invalid"] is True
    assert updated["status"] == "purchase_reconciled_cleanup_pending"
    assert updated["cleanup_required"] is True
    assert updated["manual_reconciliation_required"] is True
    assert "invalid hourly price" in updated["reconciliation_error"]
    purchase_verifier.assert_called_once_with(manifest.run_id)
    destroy_verifier.assert_not_called()


def test_reconcile_exact_multi_match_with_invalid_id_stays_fail_closed(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    label = f"quant-experiments-{manifest.run_id}"
    record = {
        "run_id": manifest.run_id,
        "instance_id": 1234,
        "expected_label": label,
        "offer_id": 11,
        "machine_id": 101,
        "manifest": manifest.__dict__,
        "status": "destroyed",
        "cleanup_required": False,
    }
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    exact = {"label": label, "ask_contract_id": 11, "machine_id": 101}
    live = [{"id": 777, **exact}, {"id": "invalid", **exact}]
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops.run_vast", return_value=live),
        patch(
            "vast.ops.launch_detached_purchase_reconciliation", return_value=77
        ) as purchase_verifier,
        patch("vast.ops.launch_detached_destroy_reconciliation") as destroy_verifier,
        pytest.raises(OpsError, match="exact positive integer"),
    ):
        command_reconcile(SimpleNamespace(restart_monitors=True))

    assert json.loads(state_path.read_text(encoding="utf-8"))[0] == record
    purchase_verifier.assert_not_called()
    destroy_verifier.assert_not_called()


def test_adoption_accounts_from_original_start_and_refuses_depleted_reserve() -> None:
    manifest = adoption_manifest()
    with pytest.raises(OpsError, match="exhausted the approved runtime"):
        validate_adoption_instance(
            adoption_instance(age_minutes=116),
            manifest,
            tracked_records=[],
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"gpu_name": "RTX 4090"}, "GPU"),
        ({"dph_total": 0.51}, "exceeds cap"),
        ({"cpu_name": "AMD EPYC 7452"}, "server CPU"),
        ({"geolocation": "Alberta, CA"}, "geolocation"),
        ({"cpu_ram": 8 * 1024}, "CPU RAM"),
        ({"label": "some-other-job"}, "existing label"),
    ],
)
def test_adoption_rejects_live_contract_mismatches(
    updates: dict[str, object], message: str
) -> None:
    with pytest.raises(OpsError, match=message):
        validate_adoption_instance(
            adoption_instance(**updates),
            adoption_manifest(),
            tracked_records=[],
        )


def test_adoption_rejects_an_instance_already_in_local_lifecycle() -> None:
    with pytest.raises(OpsError, match="already tracked"):
        validate_adoption_instance(
            adoption_instance(),
            adoption_manifest(),
            tracked_records=[{"run_id": "other", "instance_id": 1234}],
        )


def test_append_record_atomically_rejects_duplicate_instance_identity(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "instances.json"
    state_path.write_text(
        json.dumps([{"run_id": "first", "instance_id": 1234}]),
        encoding="utf-8",
    )
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        pytest.raises(OpsError, match="already tracked by run 'first'"),
    ):
        vast_ops.append_record({"run_id": "second", "instance_id": 1234})


def test_adoption_monitor_does_not_destroy_while_provisioning_owner_is_alive(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    started = utc_now()
    record = {
        "run_id": manifest.run_id,
        "instance_id": 1234,
        "created_at": started,
        "campaign_started_at": started,
        "manifest": manifest.__dict__,
        "lifecycle_origin": "existing_instance_adoption",
        "provisioning_owner_pid": 4321,
        "status": "adoption_pending",
    }
    state_path = tmp_path / "instances.json"
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.process_is_alive", return_value=True),
        patch("vast.ops.destroy_instance") as destroy,
    ):
        assert _monitor_run(manifest.run_id) == 0
    destroy.assert_not_called()


def test_adoption_requires_exact_multi_gpu_count() -> None:
    with pytest.raises(OpsError, match="GPU count"):
        validate_adoption_instance(
            adoption_instance(num_gpus=1),
            replace(adoption_manifest(), num_gpus=4),
            tracked_records=[],
        )


def test_blank_host_preflight_is_read_only_and_rejects_workloads() -> None:
    response = SimpleNamespace(
        returncode=0,
        stdout=json.dumps(
            {
                "idle": False,
                "occupied_paths": [],
                "project_processes": [],
                "gpu_processes": ["55, python"],
                "errors": [],
            }
        ),
        stderr="",
    )
    with (
        patch("vast.ops.remote_exec", return_value=response) as execute,
        pytest.raises(OpsError, match="not blank and idle"),
    ):
        verify_adoption_host_idle(("root", "host", 22))
    command = execute.call_args.args[1]
    assert "mkdir" not in command
    assert "rm " not in command
    assert "nvidia-smi" in command


@pytest.mark.parametrize(
    "stdout",
    [
        '{"idle":true,"idle":false}',
        "[]",
        "[" * 30 + "0" + "]" * 30,
        json.dumps(
            {
                "idle": False,
                "project_processes": ["password=provider-only-token"],
            }
        ),
    ],
)
def test_blank_host_preflight_rejects_untrusted_control_payload_without_echo(
    stdout: str,
) -> None:
    response = SimpleNamespace(returncode=0, stdout=stdout, stderr="")
    with (
        patch("vast.ops.remote_exec", return_value=response) as execute,
        pytest.raises(OpsError) as raised,
    ):
        verify_adoption_host_idle(("root", "host", 22))

    assert "provider-only-token" not in str(raised.value)


def test_adoption_label_is_applied_and_verified_without_creating() -> None:
    labeled = adoption_instance(label="quant-experiments-adopt_seed0")
    calls: list[tuple[object, ...]] = []

    def fake_vast(*arguments: object, **_: object) -> object:
        calls.append(arguments)
        assert arguments[:2] != ("create", "instance")
        if arguments[:2] == ("label", "instance"):
            return "success"
        return [labeled]

    with patch("vast.ops.run_vast", side_effect=fake_vast):
        label = label_adopted_instance(1234, "adopt_seed0")

    assert label == "quant-experiments-adopt_seed0"
    assert calls == [
        ("label", "instance", 1234, "quant-experiments-adopt_seed0"),
        ("show", "instances"),
    ]


def test_successful_adoption_never_creates_and_enters_normal_monitor_state(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    manifest_path = tmp_path / "run.json"
    manifest_path.write_text("{}", encoding="utf-8")
    state_path = tmp_path / "instances.json"
    event_path = tmp_path / "events.jsonl"
    bundle = tmp_path / "source.bundle"
    resolved = tmp_path / "run_manifest.json"
    args = SimpleNamespace(
        manifest=manifest_path,
        instance=1234,
        detach=True,
    )
    vast_calls: list[tuple[object, ...]] = []

    def fake_vast(*arguments: object, **_: object) -> object:
        vast_calls.append(arguments)
        assert arguments[:2] != ("create", "instance")
        return [adoption_instance()]

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.EVENT_FILE", event_path),
        patch("vast.ops.load_manifest", return_value=manifest),
        patch("vast.ops.create_source_bundle", return_value=(bundle, resolved)),
        patch("vast.ops.run_vast", side_effect=fake_vast),
        patch("vast.ops.load_machine_blacklist", return_value=set()),
        patch("vast.ops.wait_for_ssh", return_value=("root", "host", 22)),
        patch("vast.ops.verify_adoption_host_idle", return_value={"idle": True}),
        patch(
            "vast.ops.label_adopted_instance",
            return_value="quant-experiments-adopt_seed0",
        ),
        patch("vast.ops.start_remote_run") as start,
        patch("vast.ops.start_requested_monitor", return_value=0) as monitor,
    ):
        assert command_adopt_existing(args) == 0
        record = load_state()[0]

    assert vast_calls == [("show", "instances")]
    start.assert_called_once()
    monitor.assert_called_once_with(manifest.run_id, True)
    assert record["status"] == "running"
    assert record["instance_id"] == 1234
    assert record["lifecycle_origin"] == "existing_instance_adoption"
    assert record["created_at"] == record["rental_started_at"]
    assert record["accrued_cost_at_adoption_usd"] > 0
    assert record["observed_label"] == "quant-experiments-adopt_seed0"


def test_retained_debug_session_survives_a_diagnostic_failure(tmp_path: Path) -> None:
    manifest = replace(adoption_manifest(), success_policy="keep_alive_after_verify")
    record = {
        "run_id": manifest.run_id,
        "instance_id": 1234,
        "status": "running",
    }
    state_path = tmp_path / "instances.json"
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.pull_and_verify", return_value=True) as pull,
    ):
        assert retain_failed_debug_session(record, manifest, 1234)

    pull.assert_called_once_with(
        record, best_effort=True, deadline_monotonic=None
    )
    updated = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert updated["status"] == "session_ready"
    assert updated["failure_artifacts_verified"] is True
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["event"] == "session_ready_after_failure"


def test_retained_failure_cannot_resurrect_concurrent_cleanup(tmp_path: Path) -> None:
    from vast.ops import update_record_fields

    manifest = replace(adoption_manifest(), success_policy="keep_alive_after_verify")
    record = {
        "run_id": manifest.run_id,
        "instance_id": 1234,
        "status": "running",
    }
    state_path = tmp_path / "instances.json"
    event_path = tmp_path / "events.jsonl"
    state_path.write_text(json.dumps([record]), encoding="utf-8")

    def cleanup_wins(*_: object, **__: object) -> bool:
        update_record_fields(
            manifest.run_id,
            status="cleanup_required",
            cleanup_required=True,
        )
        return True

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.EVENT_FILE", event_path),
        patch("vast.ops.pull_and_verify", side_effect=cleanup_wins),
    ):
        assert retain_failed_debug_session(record, manifest, 1234)

    updated = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert updated["status"] == "cleanup_required"
    assert "session_ready_at" not in updated
    assert not event_path.exists()


def test_verified_success_cannot_resurrect_destroyed_record(tmp_path: Path) -> None:
    from vast.ops import handle_verified_success

    manifest = replace(adoption_manifest(), success_policy="keep_alive_after_verify")
    stale = {
        "run_id": manifest.run_id,
        "instance_id": 1234,
        "status": "running",
    }
    state_path = tmp_path / "instances.json"
    event_path = tmp_path / "events.jsonl"
    state_path.write_text(
        json.dumps([{**stale, "status": "destroyed", "cleanup_required": False}]),
        encoding="utf-8",
    )
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.EVENT_FILE", event_path),
    ):
        assert not handle_verified_success(stale, manifest, 1234)

    updated = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert updated["status"] == "destroyed"
    assert "session_ready_at" not in updated
    assert not event_path.exists()


def test_monitor_retries_cleanup_when_no_cleanup_lease_exists(tmp_path: Path) -> None:
    manifest = adoption_manifest()
    record = {
        **_ordinary_destroy_record(manifest),
        "cleanup_required": True,
        "status": "cleanup_required",
    }
    state_path = tmp_path / "instances.json"
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops._supervisor_lock_owner_pid", return_value=None),
        patch("vast.ops.destroy_instance") as destroy,
    ):
        assert _monitor_run(manifest.run_id) == 2
    destroy.assert_called_once()
    assert destroy.call_args.args == (
        1234,
        manifest.run_id,
        "cleanup_required_retry",
    )


def test_monitor_cleanup_fallback_contains_non_ops_destroy_failure(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    record = {
        **_ordinary_destroy_record(manifest),
        "cleanup_required": True,
        "status": "cleanup_required",
    }
    state_path = tmp_path / "instances.json"
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops._supervisor_lock_owner_pid", return_value=None),
        patch(
            "vast.ops.destroy_instance",
            side_effect=OSError("destroy event unavailable"),
        ) as destroy,
    ):
        assert _monitor_run(manifest.run_id) == 2

    destroy.assert_called_once()


def test_monitor_hands_cleanup_to_confirmed_os_lease(tmp_path: Path) -> None:
    manifest = adoption_manifest()
    record = {
        **_ordinary_destroy_record(manifest),
        "cleanup_required": True,
        "status": "cleanup_required",
    }
    state_path = tmp_path / "instances.json"
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops._supervisor_lock_owner_pid", return_value=77),
        patch("vast.ops.destroy_instance") as destroy,
    ):
        assert _monitor_run(manifest.run_id) == 2
    destroy.assert_not_called()


def test_adoption_persists_before_remote_mutation_and_cleans_setup_failure(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    args = SimpleNamespace(
        manifest=tmp_path / "run.json",
        instance=1234,
        detach=False,
    )

    def verify_persisted_before_label(*_: object) -> str:
        records = json.loads(state_path.read_text(encoding="utf-8"))
        assert records[0]["status"] == "adoption_pending"
        return "quant-experiments-adopt_seed0"

    def fail_after_persistence(*_: object) -> None:
        records = json.loads(state_path.read_text(encoding="utf-8"))
        assert records[0]["status"] == "adoption_provisioning"
        raise OpsError("bootstrap failed")

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.load_manifest", return_value=manifest),
        patch(
            "vast.ops.create_source_bundle",
            return_value=(tmp_path / "source.bundle", tmp_path / "run.json"),
        ),
        patch("vast.ops.run_vast", return_value=[adoption_instance()]) as vast_call,
        patch("vast.ops.load_machine_blacklist", return_value=set()),
        patch("vast.ops.wait_for_ssh", return_value=("root", "host", 22)),
        patch("vast.ops.verify_adoption_host_idle", return_value={"idle": True}),
        patch(
            "vast.ops.label_adopted_instance",
            side_effect=verify_persisted_before_label,
        ),
        patch("vast.ops.start_remote_run", side_effect=fail_after_persistence) as start,
        patch("vast.ops.destroy_instance") as destroy,
        pytest.raises(OpsError, match="bootstrap failed"),
    ):
        command_adopt_existing(args)

    assert start.call_count == 1
    destroy.assert_called_once_with(1234, manifest.run_id, "adoption_setup_failed")
    assert all(call.args[:2] != ("create", "instance") for call in vast_call.mock_calls)


def test_adoption_cleanup_failure_leaves_reconcilable_monitored_record(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    args = SimpleNamespace(
        manifest=tmp_path / "run.json",
        instance=1234,
        detach=False,
    )
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.load_manifest", return_value=manifest),
        patch(
            "vast.ops.create_source_bundle",
            return_value=(tmp_path / "source.bundle", tmp_path / "run.json"),
        ),
        patch("vast.ops.run_vast", return_value=[adoption_instance()]),
        patch("vast.ops.load_machine_blacklist", return_value=set()),
        patch("vast.ops.wait_for_ssh", return_value=("root", "host", 22)),
        patch("vast.ops.verify_adoption_host_idle", return_value={"idle": True}),
        patch(
            "vast.ops.label_adopted_instance",
            return_value="quant-experiments-adopt_seed0",
        ),
        patch("vast.ops.start_remote_run", side_effect=OpsError("bootstrap failed")),
        patch("vast.ops.destroy_instance", side_effect=OpsError("destroy failed")),
        patch("vast.ops.launch_detached_monitor", return_value=77) as monitor,
    ):
        with pytest.raises(OpsError, match="bootstrap failed"):
            command_adopt_existing(args)
        record = load_state()[0]

    assert record["status"] == "provisioning_cleanup_pending"
    assert record["instance_id"] == 1234
    assert record["cleanup_error"] == "destroy failed"
    monitor.assert_called_once_with(manifest.run_id)


def test_restarted_monitor_cleans_an_interrupted_adoption(tmp_path: Path) -> None:
    manifest = adoption_manifest()
    record = {
        "run_id": manifest.run_id,
        "instance_id": 1234,
        "dph": 0.4,
        "created_at": (datetime.now(UTC) - timedelta(minutes=5)).isoformat().replace("+00:00", "Z"),
        "source_commit": manifest.source_commit,
        "manifest": manifest.__dict__,
        "lifecycle_origin": "existing_instance_adoption",
        "status": "adoption_pending",
    }
    state_path = tmp_path / "instances.json"
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.get_record", return_value=record),
        patch("vast.ops.destroy_instance") as destroy,
    ):
        assert _monitor_run(manifest.run_id) == 2
    destroy.assert_called_once()
    assert destroy.call_args.args == (
        1234,
        manifest.run_id,
        "adoption_setup_interrupted",
    )
    assert destroy.call_args.kwargs["deadline_monotonic"] > 0


def test_hard_ttl_snapshots_before_pull_and_destroy(tmp_path: Path) -> None:
    manifest = replace(adoption_manifest(), ttl_hours=1.0)
    started = (datetime.now(UTC) - timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    record = {
        "run_id": manifest.run_id,
        "instance_id": 1234,
        "dph": 0.4,
        "created_at": started,
        "campaign_started_at": started,
        "source_commit": manifest.source_commit,
        "manifest": manifest.__dict__,
        "status": "running",
    }
    order: list[str] = []
    recovery_deadlines: list[float] = []
    state_path = tmp_path / "instances.json"
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.get_record", return_value=record),
        patch(
            "vast.ops.request_remote_session_snapshot",
            side_effect=lambda _, **kwargs: (
                recovery_deadlines.append(kwargs["deadline_monotonic"]),
                order.append("snapshot"),
            )[-1],
        ),
        patch(
            "vast.ops.pull_and_verify",
            side_effect=lambda *_args, **kwargs: (
                recovery_deadlines.append(kwargs["deadline_monotonic"]),
                order.append("pull"),
                True,
            )[-1],
        ),
            patch(
                "vast.ops.destroy_instance",
                side_effect=lambda *_args, **_kwargs: order.append("destroy"),
            ),
    ):
        assert _monitor_run(manifest.run_id) == 2

    assert order == ["snapshot", "pull", "destroy"]
    assert len(recovery_deadlines) == 2
    assert recovery_deadlines[0] == recovery_deadlines[1]


def test_hard_limit_destroy_failure_hands_off_without_unbounded_monitor_retry(
    tmp_path: Path,
) -> None:
    manifest = replace(adoption_manifest(), ttl_hours=1.0)
    started = (datetime.now(UTC) - timedelta(hours=2)).isoformat().replace(
        "+00:00", "Z"
    )
    record = {
        "run_id": manifest.run_id,
        "instance_id": 1234,
        "dph": 0.4,
        "created_at": started,
        "campaign_started_at": started,
        "source_commit": manifest.source_commit,
        "manifest": manifest.__dict__,
        "status": "running",
    }
    state_path = tmp_path / "instances.json"
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.get_record", return_value=record),
        patch("vast.ops.request_remote_session_snapshot"),
        patch("vast.ops.pull_and_verify", return_value=False),
        patch("vast.ops.destroy_instance", side_effect=OpsError("API unavailable")),
        patch(
            "vast.ops.launch_detached_destroy_reconciliation", return_value=77
        ) as handoff,
    ):
        assert _monitor_run(manifest.run_id) == 2
    # destroy_instance owns the bounded detached+attached handoff contract; the
    # monitor does not start an independent unbounded post-deadline retry.
    handoff.assert_not_called()


def test_hard_limit_diagnostic_failures_cannot_precede_destroy(
    tmp_path: Path,
) -> None:
    manifest = replace(adoption_manifest(), ttl_hours=1.0)
    started = (datetime.now(UTC) - timedelta(hours=2)).isoformat().replace(
        "+00:00", "Z"
    )
    record = {
        "run_id": manifest.run_id,
        "instance_id": 1234,
        "dph": 0.4,
        "created_at": started,
        "campaign_started_at": started,
        "source_commit": manifest.source_commit,
        "manifest": manifest.__dict__,
        "status": "running",
    }
    state_path = tmp_path / "instances.json"
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.get_record", return_value=record),
        patch(
            "vast.ops.request_remote_session_snapshot",
            side_effect=OSError("snapshot unavailable"),
        ),
        patch(
            "vast.ops.pull_and_verify",
            side_effect=OSError("recovery unavailable"),
        ),
        patch("vast.ops._monitor_event_if_active", side_effect=OSError("event unavailable")),
        patch("vast.ops._best_effort_update_record_fields", return_value=False),
        patch("vast.ops.destroy_instance") as destroy,
    ):
        assert _monitor_run(manifest.run_id) == 2

    destroy.assert_called_once()


def test_monitor_records_unverified_emergency_termination_and_reconciles(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    started = utc_now()
    record = {
        "run_id": manifest.run_id,
        "instance_id": 1234,
        "dph": 0.4,
        "created_at": started,
        "campaign_started_at": started,
        "source_commit": manifest.source_commit,
        "manifest": manifest.__dict__,
        "status": "running",
    }
    state_path = tmp_path / "instances.json"
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.get_record", return_value=record),
        patch("vast.ops.run_vast", return_value=[]),
        patch(
            "vast.ops._launch_destroy_reconciliation_with_retries", return_value=77
        ) as handoff,
    ):
        assert _monitor_run(manifest.run_id) == 2
    updated = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert updated["status"] == "emergency_termination_unverified"
    assert updated["cleanup_required"] is True
    handoff.assert_called_once()
    assert handoff.call_args.args == (manifest.run_id,)
    assert handoff.call_args.kwargs["instance_id"] == 1234
    assert handoff.call_args.kwargs["deadline_monotonic"] > time.monotonic()


def test_monitor_event_replica_failure_cannot_block_emergency_reconciliation(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    started = utc_now()
    record = {
        "run_id": manifest.run_id,
        "instance_id": 1234,
        "dph": 0.4,
        "created_at": started,
        "campaign_started_at": started,
        "source_commit": manifest.source_commit,
        "manifest": manifest.__dict__,
        "status": "running",
    }
    state_path = tmp_path / "instances.json"
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.get_record", return_value=record),
        patch("vast.ops.run_vast", return_value=[]),
        patch("vast.ops.append_local_event", side_effect=OSError("event unavailable")),
        patch(
            "vast.ops._launch_destroy_reconciliation_with_retries", return_value=77
        ) as handoff,
    ):
        assert _monitor_run(manifest.run_id) == 2

    updated = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert updated["cleanup_required"] is True
    handoff.assert_called_once()


def test_monitor_provider_disappearance_state_failure_still_hands_off(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    started = utc_now()
    record = {
        "run_id": manifest.run_id,
        "instance_id": 1234,
        "dph": 0.4,
        "created_at": started,
        "campaign_started_at": started,
        "source_commit": manifest.source_commit,
        "manifest": manifest.__dict__,
        "status": "running",
    }
    state_path = tmp_path / "instances.json"
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.get_record", return_value=record),
        patch("vast.ops.run_vast", return_value=[]),
        patch(
            "vast.ops._monitor_write_if_active",
            side_effect=OSError("state unavailable"),
        ),
        patch("vast.ops._best_effort_update_record_fields", return_value=False),
        patch(
            "vast.ops._launch_destroy_reconciliation_with_retries", return_value=77
        ) as handoff,
    ):
        assert _monitor_run(manifest.run_id) == 2

    handoff.assert_called_once()


def test_monitor_diagnostics_redact_state_events_and_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = adoption_manifest()
    started = utc_now()
    record = {
        "run_id": manifest.run_id,
        "instance_id": 1234,
        "dph": 0.4,
        "created_at": started,
        "campaign_started_at": started,
        "source_commit": manifest.source_commit,
        "manifest": manifest.__dict__,
        "status": "running",
    }
    state_path = tmp_path / "instances.json"
    event_path = tmp_path / "events.jsonl"
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    secret = "monitor-diagnostic-secret"
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.EVENT_FILE", event_path),
        patch("vast.ops.load_secrets", return_value={"VAST_API_KEY": secret}),
        patch(
            "vast.ops.run_vast",
            side_effect=OpsError(f"live api_key={secret}"),
        ),
        patch(
            "vast.ops.ssh_url",
            side_effect=OSError(f"transport api_key={secret}"),
        ),
        patch("vast.ops._bounded_sleep", side_effect=KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        _monitor_run(manifest.run_id)

    persisted = state_path.read_text(encoding="utf-8") + event_path.read_text(
        encoding="utf-8"
    )
    captured = capsys.readouterr()
    assert secret not in persisted
    assert secret not in captured.out
    assert secret not in captured.err
    assert "[DIAGNOSTIC WITHHELD]" in persisted


def test_monitor_recursively_redacts_remote_event_before_any_local_sink(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = adoption_manifest()
    started = utc_now()
    record = {
        "run_id": manifest.run_id,
        "instance_id": 1234,
        "dph": 0.4,
        "created_at": started,
        "campaign_started_at": started,
        "source_commit": manifest.source_commit,
        "manifest": manifest.__dict__,
        "status": "running",
    }
    state_path = tmp_path / "instances.json"
    event_path = tmp_path / "events.jsonl"
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    secret = "remote-event-secret"
    remote_only_token = "remote-only-container-token"
    remote_event = {
        "event": "runner_exception",
        "error": f"remote api_key={secret}",
        "nested": {
            f"secret-key-{secret}": [
                secret,
                {"url": f"https://example.invalid/?api_key={secret}"},
            ],
            "CONTAINER_API_KEY": remote_only_token,
            "authorization": f"Bearer {remote_only_token}",
        },
    }

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.EVENT_FILE", event_path),
        patch("vast.ops.load_secrets", return_value={"VAST_API_KEY": secret}),
        patch("vast.ops.run_vast", return_value=[{"id": 1234}]),
        patch("vast.ops.ssh_url", return_value=("root", "host", 22)),
        # Bypass the reader's first sanitization pass to prove that the monitor
        # itself treats its input as untrusted before output or persistence.
        patch("vast.ops.read_remote_events", return_value=[remote_event]),
        patch("vast.ops._bounded_sleep", side_effect=KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        _monitor_run(manifest.run_id)

    state_text = state_path.read_text(encoding="utf-8")
    event_text = event_path.read_text(encoding="utf-8")
    captured = capsys.readouterr()
    combined = state_text + event_text + captured.out + captured.err
    assert secret not in combined
    assert remote_only_token not in combined
    persisted = json.loads(state_text)[0]["last_remote_event"]
    assert persisted == {
        "event": "remote_event_withheld",
        "diagnostic": "[DIAGNOSTIC WITHHELD]",
    }


def test_monitor_withholds_remote_event_when_redaction_itself_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = adoption_manifest()
    started = utc_now()
    record = {
        "run_id": manifest.run_id,
        "instance_id": 1234,
        "dph": 0.4,
        "created_at": started,
        "campaign_started_at": started,
        "source_commit": manifest.source_commit,
        "manifest": manifest.__dict__,
        "status": "running",
    }
    state_path = tmp_path / "instances.json"
    event_path = tmp_path / "events.jsonl"
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    secret = "redaction-failure-secret"

    def fail_only_on_secret(value: str) -> str:
        if secret in value:
            raise OSError("redactor unavailable for sensitive text")
        return value

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.EVENT_FILE", event_path),
        patch("vast.ops.redact_secrets", side_effect=fail_only_on_secret),
        patch("vast.ops.run_vast", return_value=[{"id": 1234}]),
        patch("vast.ops.ssh_url", return_value=("root", "host", 22)),
        patch(
            "vast.ops.read_remote_events",
            return_value=[
                {
                    "event": "artifacts_ready",
                    "exit_code": 0,
                    "error": f"remote api_key={secret}",
                    "nested": [secret],
                },
                # Both control-authoritative markers would normally open local
                # artifact verification. One unsafe sibling field must withhold
                # the complete batch instead of retaining a partial success.
                {"event": "run_succeeded"},
            ],
        ),
        patch("vast.ops.pull_and_verify") as verify,
        patch("vast.ops._bounded_sleep", side_effect=KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        _monitor_run(manifest.run_id)

    captured = capsys.readouterr()
    combined = (
        state_path.read_text(encoding="utf-8")
        + event_path.read_text(encoding="utf-8")
        + captured.out
        + captured.err
    )
    assert secret not in combined
    assert "[DIAGNOSTIC WITHHELD]" in combined
    persisted = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert persisted["last_remote_event"] == {
        "event": "remote_event_withheld",
        "diagnostic": "[DIAGNOSTIC WITHHELD]",
    }
    verify.assert_not_called()


def test_monitor_output_failure_cannot_block_sanitized_remote_state_write(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    started = utc_now()
    record = {
        "run_id": manifest.run_id,
        "instance_id": 1234,
        "dph": 0.4,
        "created_at": started,
        "campaign_started_at": started,
        "source_commit": manifest.source_commit,
        "manifest": manifest.__dict__,
        "status": "running",
    }
    state_path = tmp_path / "instances.json"
    event_path = tmp_path / "events.jsonl"
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    secret = "print-failure-secret"

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.EVENT_FILE", event_path),
        patch("vast.ops.load_secrets", return_value={"VAST_API_KEY": secret}),
        patch("vast.ops.run_vast", return_value=[{"id": 1234}]),
        patch("vast.ops.ssh_url", return_value=("root", "host", 22)),
        patch(
            "vast.ops.read_remote_events",
            return_value=[
                {
                    "event": "runner_exception",
                    "error": f"remote api_key={secret}",
                }
            ],
        ),
        patch("builtins.print", side_effect=OSError("stdout unavailable")),
        patch("vast.ops._bounded_sleep", side_effect=KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        _monitor_run(manifest.run_id)

    persisted = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert persisted["last_remote_event"] == {
        "event": "remote_event_withheld",
        "diagnostic": "[DIAGNOSTIC WITHHELD]",
    }
    combined = state_path.read_text(encoding="utf-8") + event_path.read_text(
        encoding="utf-8"
    )
    assert secret not in combined
    assert "[DIAGNOSTIC WITHHELD]" in combined
    assert "remote_state_change" in combined


@pytest.mark.parametrize(
    "remote_timestamp", [["invalid"], "1970-01-01T00:00:00Z"]
)
def test_monitor_remote_failure_timestamp_cannot_shorten_local_grace(
    tmp_path: Path, remote_timestamp: object
) -> None:
    manifest = replace(adoption_manifest(), failure_grace_minutes=10)
    started = utc_now()
    record = {
        "run_id": manifest.run_id,
        "instance_id": 1234,
        "dph": 0.4,
        "created_at": started,
        "campaign_started_at": started,
        "source_commit": manifest.source_commit,
        "manifest": manifest.__dict__,
        "status": "running",
    }
    state_path = tmp_path / "instances.json"
    event_path = tmp_path / "events.jsonl"
    state_path.write_text(json.dumps([record]), encoding="utf-8")

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.EVENT_FILE", event_path),
        patch("vast.ops.run_vast", return_value=[{"id": 1234}]),
        patch("vast.ops.ssh_url", return_value=("root", "host", 22)),
        patch(
            "vast.ops.read_remote_events",
            return_value=[
                {
                    "event": "run_failed",
                    "run_id": manifest.run_id,
                    "timestamp": remote_timestamp,
                }
            ],
        ),
        patch("vast.ops.pull_and_verify", return_value=False),
        patch("vast.ops.destroy_instance") as destroy,
        patch("vast.ops._bounded_sleep", side_effect=KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        _monitor_run(manifest.run_id)

    destroy.assert_not_called()
    persisted = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert persisted["status"] == "running"


def test_monitor_remote_failure_timestamp_cannot_extend_zero_local_grace(
    tmp_path: Path,
) -> None:
    manifest = replace(adoption_manifest(), failure_grace_minutes=0)
    started = utc_now()
    record = {
        "run_id": manifest.run_id,
        "instance_id": 1234,
        "dph": 0.4,
        "created_at": started,
        "campaign_started_at": started,
        "source_commit": manifest.source_commit,
        "manifest": manifest.__dict__,
        "status": "running",
    }
    state_path = tmp_path / "instances.json"
    event_path = tmp_path / "events.jsonl"
    state_path.write_text(json.dumps([record]), encoding="utf-8")

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.EVENT_FILE", event_path),
        patch("vast.ops.run_vast", return_value=[{"id": 1234}]),
        patch("vast.ops.ssh_url", return_value=("root", "host", 22)),
        patch(
            "vast.ops.read_remote_events",
            return_value=[
                {
                    "event": "run_failed",
                    "run_id": manifest.run_id,
                    "timestamp": "2999-01-01T00:00:00Z",
                }
            ],
        ),
        patch("vast.ops.pull_and_verify", return_value=False),
        patch("vast.ops.destroy_instance") as destroy,
    ):
        assert _monitor_run(manifest.run_id) == 1

    destroy.assert_called_once()
    assert destroy.call_args.args == (1234, manifest.run_id, "failure_grace_elapsed")
    assert isinstance(destroy.call_args.kwargs["deadline_monotonic"], float)


def test_monitor_rejects_boolean_artifact_success_exit_code(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    started = utc_now()
    record = {
        "run_id": manifest.run_id,
        "instance_id": 1234,
        "dph": 0.4,
        "created_at": started,
        "campaign_started_at": started,
        "source_commit": manifest.source_commit,
        "manifest": manifest.__dict__,
        "status": "running",
    }
    state_path = tmp_path / "instances.json"
    state_path.write_text(json.dumps([record]), encoding="utf-8")

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.run_vast", return_value=[{"id": 1234}]),
        patch("vast.ops.ssh_url", return_value=("root", "host", 22)),
        patch(
            "vast.ops.read_remote_events",
            return_value=[
                {
                    "event": "artifacts_ready",
                    "run_id": manifest.run_id,
                    "exit_code": False,
                },
                {"event": "run_succeeded", "run_id": manifest.run_id},
            ],
        ),
        patch("vast.ops.pull_and_verify") as verify,
        patch("vast.ops._bounded_sleep", side_effect=KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        _monitor_run(manifest.run_id)

    verify.assert_not_called()


def test_stage_monitor_rejects_float_artifact_success_exit_code() -> None:
    manifest = adoption_manifest()
    stage = {
        "stage_id": "typed-exit-stage",
        "parent_run_id": manifest.run_id,
        "instance_id": 1234,
        "manifest": manifest.__dict__,
        "remote_ops": "/workspace/stage/ops",
        "status": "stage_running",
    }
    parent = {
        "run_id": manifest.run_id,
        "instance_id": 1234,
        "manifest": manifest.__dict__,
        "status": "session_ready",
    }

    with (
        patch("vast.ops.get_stage_record", return_value=stage),
        patch("vast.ops.get_record", return_value=parent),
        patch("vast.ops.replace_stage_record"),
        patch("vast.ops.append_local_event"),
        patch("vast.ops.run_vast", return_value=[{"id": 1234}]),
        patch("vast.ops.ssh_url", return_value=("root", "host", 22)),
        patch(
            "vast.ops.read_remote_events",
            return_value=[
                {
                    "event": "artifacts_ready",
                    "run_id": stage["stage_id"],
                    "exit_code": 0.0,
                },
                {"event": "run_succeeded", "run_id": stage["stage_id"]},
            ],
        ),
        patch("vast.ops.pull_and_verify") as verify,
        patch("vast.ops.time.sleep", side_effect=KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        vast_ops._monitor_session_stage(stage["stage_id"])

    verify.assert_not_called()


def test_ordinary_monitor_does_not_promote_contradictory_terminal_history(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    record = {
        "run_id": manifest.run_id,
        "instance_id": 1234,
        "dph": 0.4,
        "created_at": utc_now(),
        "campaign_started_at": utc_now(),
        "source_commit": manifest.source_commit,
        "manifest": manifest.__dict__,
        "status": "running",
    }
    state_path = tmp_path / "instances.json"
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    events = [
        {
            "event": "artifacts_ready",
            "run_id": manifest.run_id,
            "exit_code": 0,
        },
        {"event": "run_succeeded", "run_id": manifest.run_id},
        {"event": "run_failed", "run_id": manifest.run_id, "timestamp": utc_now()},
    ]
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.run_vast", return_value=[{"id": 1234}]),
        patch("vast.ops.ssh_url", return_value=("root", "host", 22)),
        patch("vast.ops.read_remote_events", return_value=events),
        patch("vast.ops.retain_failed_debug_session", return_value=True),
        patch("vast.ops.pull_and_verify") as verify,
        patch("vast.ops.handle_verified_success") as handle_success,
        patch("vast.ops._bounded_sleep", side_effect=KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        _monitor_run(manifest.run_id)

    verify.assert_not_called()
    handle_success.assert_not_called()


def test_stage_monitor_treats_contradictory_terminal_history_only_as_failure(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    stage_id = "contradictory-stage"
    stage = {
        "stage_id": stage_id,
        "run_id": stage_id,
        "parent_run_id": manifest.run_id,
        "instance_id": 1234,
        "source_commit": manifest.source_commit,
        "manifest": {**manifest.__dict__, "run_id": stage_id},
        "remote_ops": "/workspace/stage/ops",
        "status": "stage_running",
    }
    parent = {
        "run_id": manifest.run_id,
        "instance_id": 1234,
        "manifest": manifest.__dict__,
        "status": "session_ready",
    }
    events = [
        {"event": "artifacts_ready", "run_id": stage_id, "exit_code": 0},
        {"event": "run_succeeded", "run_id": stage_id},
        {"event": "run_failed", "run_id": stage_id, "timestamp": utc_now()},
    ]
    with (
        patch("vast.ops.get_stage_record", return_value=stage),
        patch("vast.ops.get_record", return_value=parent),
        patch("vast.ops.replace_stage_record"),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.run_vast", return_value=[{"id": 1234}]),
        patch("vast.ops.ssh_url", return_value=("root", "host", 22)),
        patch("vast.ops.read_remote_events", return_value=events),
        patch("vast.ops.pull_and_verify", return_value=False) as verify,
    ):
        assert vast_ops._monitor_session_stage(stage_id) == 1

    verify.assert_called_once()
    assert verify.call_args.kwargs["best_effort"] is True
    assert stage["status"] == "stage_failed"
    assert stage["failure_artifacts_verified"] is False


def test_stage_success_control_survives_broken_append_only_event_replica() -> None:
    manifest = adoption_manifest()
    stage_id = "event-replica-stage"
    stage = {
        "stage_id": stage_id,
        "run_id": stage_id,
        "parent_run_id": manifest.run_id,
        "instance_id": 1234,
        "source_commit": manifest.source_commit,
        "manifest": {**manifest.__dict__, "run_id": stage_id},
        "remote_ops": "/workspace/stage/ops",
        "status": "stage_running",
    }
    parent = {
        "run_id": manifest.run_id,
        "instance_id": 1234,
        "manifest": manifest.__dict__,
        "status": "session_ready",
    }
    events = [
        {"event": "artifacts_ready", "run_id": stage_id, "exit_code": 0},
        {"event": "run_succeeded", "run_id": stage_id},
    ]
    with (
        patch("vast.ops.get_stage_record", return_value=stage),
        patch("vast.ops.get_record", return_value=parent),
        patch("vast.ops.replace_stage_record") as persist_stage,
        patch("vast.ops.append_local_event", side_effect=OSError("replica unavailable")),
        patch("vast.ops.run_vast", return_value=[{"id": 1234}]),
        patch("vast.ops.ssh_url", return_value=("root", "host", 22)),
        patch("vast.ops.read_remote_events", return_value=events),
        patch("vast.ops.pull_and_verify", return_value=True),
    ):
        assert vast_ops._monitor_session_stage(stage_id) == 0

    assert stage["status"] == "stage_ready"
    assert persist_stage.call_count >= 2


@pytest.mark.parametrize(
    "provider_snapshot",
    [
        [{"id": True}],
        [{"id": 1234}, {"id": 1234}],
    ],
)
def test_stage_monitor_retries_malformed_provider_snapshot_without_absence_decision(
    tmp_path: Path, provider_snapshot: list[dict[str, object]]
) -> None:
    manifest = adoption_manifest()
    stage_id = "malformed-live-stage"
    stage = {
        "stage_id": stage_id,
        "parent_run_id": manifest.run_id,
        "instance_id": 1234,
        "manifest": {**manifest.__dict__, "run_id": stage_id},
        "remote_ops": "/workspace/stage/ops",
        "status": "stage_running",
    }
    parent = {
        "run_id": manifest.run_id,
        "instance_id": 1234,
        "manifest": manifest.__dict__,
        "status": "session_ready",
    }
    with (
        patch("vast.ops.get_stage_record", return_value=stage),
        patch("vast.ops.get_record", return_value=parent),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.run_vast", return_value=provider_snapshot),
        patch("vast.ops.replace_stage_record") as persist_stage,
        patch("vast.ops.time.sleep", side_effect=KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        vast_ops._monitor_session_stage(stage_id)

    assert stage["status"] == "stage_running"
    persist_stage.assert_not_called()
    assert "stage_live_query_error" in (tmp_path / "events.jsonl").read_text(
        encoding="utf-8"
    )


def test_stage_monitor_keeps_malformed_success_artifact_non_authoritative(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    stage_id = "malformed-success-artifact"
    stage = {
        "stage_id": stage_id,
        "run_id": stage_id,
        "parent_run_id": manifest.run_id,
        "instance_id": 1234,
        "source_commit": manifest.source_commit,
        "manifest": {**manifest.__dict__, "run_id": stage_id},
        "remote_ops": "/workspace/stage/ops",
        "status": "stage_running",
    }
    parent = {
        "run_id": manifest.run_id,
        "instance_id": 1234,
        "manifest": manifest.__dict__,
        "status": "session_ready",
    }
    events = [
        {"event": "artifacts_ready", "run_id": stage_id, "exit_code": 0},
        {"event": "run_succeeded", "run_id": stage_id},
    ]
    with (
        patch("vast.ops.get_stage_record", return_value=stage),
        patch("vast.ops.get_record", return_value=parent),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.replace_stage_record"),
        patch("vast.ops.run_vast", return_value=[{"id": 1234}]),
        patch("vast.ops.ssh_url", return_value=("root", "host", 22)),
        patch("vast.ops.read_remote_events", return_value=events),
        patch(
            "vast.ops.pull_and_verify",
            side_effect=OpsError("duplicate artifact manifest key"),
        ),
        patch("vast.ops.time.sleep", side_effect=KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        vast_ops._monitor_session_stage(stage_id)

    assert stage["status"] == "stage_running"
    assert "stage_success_verification_retry" in (
        tmp_path / "events.jsonl"
    ).read_text(encoding="utf-8")


def test_stage_monitor_persists_failure_when_artifact_recovery_is_malformed(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    stage_id = "malformed-failure-artifact"
    stage = {
        "stage_id": stage_id,
        "run_id": stage_id,
        "parent_run_id": manifest.run_id,
        "instance_id": 1234,
        "source_commit": manifest.source_commit,
        "manifest": {**manifest.__dict__, "run_id": stage_id},
        "remote_ops": "/workspace/stage/ops",
        "status": "stage_running",
    }
    parent = {
        "run_id": manifest.run_id,
        "instance_id": 1234,
        "manifest": manifest.__dict__,
        "status": "session_ready",
    }
    with (
        patch("vast.ops.get_stage_record", return_value=stage),
        patch("vast.ops.get_record", return_value=parent),
        patch("vast.ops.EVENT_FILE", tmp_path / "events.jsonl"),
        patch("vast.ops.replace_stage_record") as persist_stage,
        patch("vast.ops.run_vast", return_value=[{"id": 1234}]),
        patch("vast.ops.ssh_url", return_value=("root", "host", 22)),
        patch(
            "vast.ops.read_remote_events",
            return_value=[{"event": "run_failed", "run_id": stage_id}],
        ),
        patch(
            "vast.ops.pull_and_verify",
            side_effect=OpsError("oversized artifact manifest"),
        ),
    ):
        assert vast_ops._monitor_session_stage(stage_id) == 1

    assert stage["status"] == "stage_failed"
    assert stage["failure_artifacts_verified"] is False
    assert persist_stage.call_count >= 2
    event_text = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    assert "stage_failure_recovery_error" in event_text
    assert "stage_failed" in event_text


def test_remote_event_reader_redacts_nested_payload_at_ingress() -> None:
    secret = "reader-ingress-secret"
    remote_only_token = "reader-remote-only-token"
    response = SimpleNamespace(
        returncode=0,
        stdout=(
            json.dumps(
                {
                    "event": "runner_exception",
                    "nested": {
                        "error": f"remote api_key={secret}",
                        "password": remote_only_token,
                        "authorization": f"Bearer {remote_only_token}",
                    },
                }
            )
            + "\n"
        )
    )

    with (
        patch("vast.ops.load_secrets", return_value={"VAST_API_KEY": secret}),
        patch("vast.ops.remote_exec", return_value=response) as execute,
    ):
        events = vast_ops.read_remote_events(("root", "host", 22))

    encoded = json.dumps(events, sort_keys=True)
    assert secret not in encoded
    assert remote_only_token not in encoded
    assert events == [
        {
            "event": "remote_event_withheld",
            "diagnostic": "[DIAGNOSTIC WITHHELD]",
        }
    ]
    remote_command = execute.call_args.args[1]
    assert "cat " not in remote_command
    assert "O_NOFOLLOW" in remote_command


@pytest.mark.parametrize("returncode", [17, True, 0.0, None])
def test_remote_event_reader_failure_cannot_return_control_authority(
    returncode: object,
) -> None:
    response = SimpleNamespace(
        returncode=returncode,
        stdout=(
            json.dumps(
                {"event": "artifacts_ready", "run_id": "run", "exit_code": 0}
            )
            + "\n"
            + json.dumps({"event": "run_succeeded", "run_id": "run"})
            + "\n"
        ),
        stderr="reader failed",
    )
    with patch("vast.ops.remote_exec", return_value=response):
        events = vast_ops.read_remote_events(("root", "host", 22))
    assert events == [
        {
            "event": "remote_event_withheld",
            "diagnostic": "[DIAGNOSTIC WITHHELD]",
        }
    ]


@pytest.mark.parametrize(
    "event",
    [{}, {"event": None}, {"event": []}, {"event": {}}, {"event": ""}],
)
def test_remote_event_sanitizer_withholds_malformed_event_name(
    event: dict[str, object],
) -> None:
    assert vast_ops._sanitized_remote_events([event]) == [
        {
            "event": "remote_event_withheld",
            "diagnostic": "[DIAGNOSTIC WITHHELD]",
        }
    ]


@pytest.mark.parametrize(
    "raw_events",
    [
        [
            {
                "event": "ａｒｔｉｆａｃｔｓ＿ｒｅａｄｙ",
                "run_id": "bound-run",
                "exit_code": 0,
            },
            {"event": "ｒｕｎ＿ｓｕｃｃｅｅｄ", "run_id": "bound-run"},
        ],
        [
            {"ｅｖｅｎｔ": "artifacts_ready", "run_id": "bound-run", "exit_code": 0},
            {"ｅｖｅｎｔ": "run_succeeded", "run_id": "bound-run"},
        ],
        [
            {
                "event": "artifacts_ready",
                "ｒｕｎ＿ｉｄ": "bound-run",
                "exit_code": 0,
            },
            {"event": "run_succeeded", "ｒｕｎ＿ｉｄ": "bound-run"},
        ],
        [
            {
                "event": "artifacts_ready",
                "run_id": "ｂｏｕｎｄ－ｒｕｎ",
                "exit_code": 0,
            },
            {"event": "run_succeeded", "run_id": "ｂｏｕｎｄ－ｒｕｎ"},
        ],
        [
            {
                "event": "artifacts_ready",
                "run_id": "bound-run",
                "ｅｘｉｔ＿ｃｏｄｅ": 0,
            },
            {"event": "run_succeeded", "run_id": "bound-run"},
        ],
    ],
)
def test_remote_event_sanitization_cannot_create_success_authority(
    raw_events: list[dict[str, object]],
) -> None:
    safe = vast_ops._sanitized_remote_events(raw_events)

    assert safe == [
        {
            "event": "remote_event_withheld",
            "diagnostic": "[DIAGNOSTIC WITHHELD]",
        }
    ]
    assert vast_ops._strict_remote_success_evidence(safe, "bound-run") is None


@pytest.mark.parametrize("rewrite_key", [False, True])
def test_remote_redactor_cannot_create_success_authority(rewrite_key: bool) -> None:
    def authority_creating_redactor(value: str) -> str:
        if rewrite_key and value == "ordinary_field":
            return "exit_code"
        if not rewrite_key and value == "heartbeat":
            return "run_succeeded"
        return value

    raw_ready = {
        "event": "artifacts_ready",
        "run_id": "bound-run",
        "exit_code": 0,
    }
    raw_other = {
        "event": "heartbeat",
        "run_id": "bound-run",
        "ordinary_field": 0,
    }
    with patch("vast.ops.redact_secrets", side_effect=authority_creating_redactor):
        safe = vast_ops._sanitized_remote_events([raw_ready, raw_other])

    assert safe == [
        {
            "event": "remote_event_withheld",
            "diagnostic": "[DIAGNOSTIC WITHHELD]",
        }
    ]
    assert vast_ops._strict_remote_success_evidence(safe, "bound-run") is None


def test_remote_failure_authority_requires_a_canonical_exact_run_binding() -> None:
    missing = vast_ops._sanitized_remote_events([{"event": "run_failed"}])
    mismatched = vast_ops._sanitized_remote_events(
        [{"event": "run_failed", "run_id": "other-run", "exit_code": 1}]
    )
    exact = vast_ops._sanitized_remote_events(
        [{"event": "run_failed", "run_id": "bound-run", "exit_code": 1}]
    )

    assert missing == [
        {
            "event": "remote_event_withheld",
            "diagnostic": "[DIAGNOSTIC WITHHELD]",
        }
    ]
    assert not vast_ops._is_bound_remote_failure_event(mismatched[-1], "bound-run")
    assert vast_ops._is_bound_remote_failure_event(exact[-1], "bound-run")


def test_remote_withheld_sentinel_cannot_precede_success_authority() -> None:
    safe = vast_ops._sanitized_remote_events(
        [
            {
                "event": "remote_event_withheld",
                "run_id": "bound-run",
                "diagnostic": "[DIAGNOSTIC WITHHELD]",
            },
            {
                "event": "artifacts_ready",
                "run_id": "bound-run",
                "exit_code": 0,
            },
            {"event": "run_succeeded", "run_id": "bound-run"},
        ]
    )

    assert safe == [
        {
            "event": "remote_event_withheld",
            "diagnostic": "[DIAGNOSTIC WITHHELD]",
        }
    ]
    assert vast_ops._strict_remote_success_evidence(safe, "bound-run") is None


def test_remote_success_authority_requires_one_ordered_run_bound_terminal_pair() -> None:
    run_id = "bound-run"
    valid = [
        {"event": "artifacts_ready", "run_id": run_id, "exit_code": 0},
        {"event": "run_succeeded", "run_id": run_id},
    ]
    assert vast_ops._strict_remote_success_evidence(valid, run_id) == (
        valid[0],
        valid[1],
    )
    invalid_histories = [
        [*valid, {"event": "run_failed", "run_id": run_id}],
        [*valid, {"event": "heartbeat", "run_id": run_id}],
        [valid[0], {"event": "run_succeeded", "run_id": "other"}],
        [
            {"event": "artifacts_ready", "run_id": "other", "exit_code": 0},
            valid[1],
        ],
        [
            {"event": "artifacts_ready", "run_id": run_id, "exit_code": True},
            valid[1],
        ],
        [
            {"event": "run_failed", "run_id": run_id},
            *valid,
        ],
        [
            {"event": "runner_exception", "run_id": run_id},
            *valid,
        ],
        [
            {"event": "artifact_packaging_failed", "run_id": run_id},
            *valid,
        ],
        [
            {"event": "ttl_reached", "run_id": run_id},
            *valid,
        ],
        [
            {
                "event": "command_finished",
                "run_id": run_id,
                "exit_code": 17,
            },
            *valid,
        ],
        [
            {"event": "heartbeat", "run_id": "other-run"},
            *valid,
        ],
        [
            {
                "event": "remote_event_withheld",
                "run_id": run_id,
                "diagnostic": "[DIAGNOSTIC WITHHELD]",
            },
            *valid,
        ],
    ]
    assert all(
        vast_ops._strict_remote_success_evidence(history, run_id) is None
        for history in invalid_histories
    )


@pytest.mark.parametrize(
    "invalid_line",
    [
        "not-json",
        "[]",
        '{"event":"heartbeat","event":"run_succeeded"}',
        '{"event":"heartbeat","nested":{"key":1,"key":2}}',
    ],
)
def test_remote_event_reader_withholds_complete_batch_on_malformed_line(
    invalid_line: str,
) -> None:
    response = SimpleNamespace(
        returncode=0,
        stdout="\n".join(
            [
                '{"event":"artifacts_ready","exit_code":0}',
                invalid_line,
                '{"event":"run_succeeded"}',
            ]
        )
    )

    with patch("vast.ops.remote_exec", return_value=response):
        events = vast_ops.read_remote_events(("root", "host", 22))

    assert events == [
        {
            "event": "remote_event_withheld",
            "diagnostic": "[DIAGNOSTIC WITHHELD]",
        }
    ]


def test_remote_event_reader_withholds_oversized_raw_batch() -> None:
    response = SimpleNamespace(
        returncode=0,
        stdout=json.dumps({"event": "run_succeeded", "detail": "x" * 100})
    )

    with (
        patch("vast.ops._MAX_REMOTE_EVENT_BYTES", 32),
        patch("vast.ops.remote_exec", return_value=response),
    ):
        events = vast_ops.read_remote_events(("root", "host", 22))

    assert events == [
        {
            "event": "remote_event_withheld",
            "diagnostic": "[DIAGNOSTIC WITHHELD]",
        }
    ]


def test_remote_event_reader_withholds_line_and_node_amplification() -> None:
    too_many_lines = "\n".join(
        json.dumps({"event": "heartbeat", "index": index})
        for index in range(vast_ops._MAX_DIAGNOSTIC_JSON_ITEMS + 1)
    )
    too_many_nodes = json.dumps(
        {
            "event": "heartbeat",
            "nested": [0] * (vast_ops._MAX_DIAGNOSTIC_JSON_NODES + 1),
        }
    )
    withheld = [
        {
            "event": "remote_event_withheld",
            "diagnostic": "[DIAGNOSTIC WITHHELD]",
        }
    ]

    for stdout in (too_many_lines, too_many_nodes):
        with patch(
            "vast.ops.remote_exec",
            return_value=SimpleNamespace(returncode=0, stdout=stdout),
        ):
            assert vast_ops.read_remote_events(("root", "host", 22)) == withheld


@pytest.mark.parametrize(
    "diagnostic",
    [
        "--api-key provider-only-token",
        "API Key: provider-only-token",
        "ＡＰＩ Ｋｅｙ: provider-only-token",
        "Private Key: provider-only-token",
        "SSH Key: provider-only-token",
        "Identity File: provider-only-token",
        "VAST_API_KEY provider-only-token",
        "password provider-only-token",
        "Authorization: Basic provider-only-token",
        "Authorization=Token provider-only-token",
        "Set-Cookie: provider-only-token",
        "private_key:\n-----BEGIN "
        + "PRIVATE KEY-----\nprovider-only-token\n-----END PRIVATE KEY-----",
        "-----BEGIN "
        + "OPENSSH PRIVATE KEY-----\nprovider-only-token\n-----END OPENSSH PRIVATE KEY-----",
        "ssh_key=provider-only-token",
        "identity_file=provider-only-token",
        "pass\x1b[31mword=provider-only-token",
        "pass\u200bword=provider-only-token",
        "pass\ufffdword=provider-only-token",
        "\x1b]52;c;provider-only-token\x07",
        r'vastai failed: {"pass\u0077ord":"provider-only-token"}',
        r'prefix {\"password\":\"provider-only-token\"}',
        "password%3Dprovider-only-token",
        "api%5Fkey=provider-only-token",
        "pass\nword=provider-only-token",
        "api\r\nkey:\r\nprovider-only-token",
        "private\vkey provider-only-token",
        "password\n=\nprovider-only-token",
        "Cookie: session=alpha; provider-only-token",
        "Set-Cookie: first=alpha, provider-only-token",
        "Authorization: Bearer alpha,provider-only-token",
        "password=alpha;provider-only-token",
        "https://user:provider-only-token@example.invalid/path",
        "//user:provider-only-token@example.invalid/path",
        "https://provider-only-token@example.invalid/path",
        "//provider-only-token@example.invalid/path",
        "PWD=provider-only-token",
        "AccountKey=provider-only-token",
        "Ocp-Apim-Subscription-Key: provider-only-token",
        "bindpw=provider-only-token",
        "MYSQL_PWD=provider-only-token",
        "DB_PWD=provider-only-token",
        "LDAP_BIND_PW=provider-only-token",
        "DB_PASS=provider-only-token",
        "Basic provider-only-token",
        "form user[password]=provider-only-token",
        "credentials[api_key]=provider-only-token",
        "config[signing_key]=provider-only-token",
        "<password>provider-only-token</password>",
        "<api_key>provider-only-token</api_key>",
        "${PASSWORD}=provider-only-token",
        "password|provider-only-token",
        "PAT=provider-only-token",
        "jwt=provider-only-token",
        "sessionId=provider-only-token",
        "JSESSIONID=provider-only-token",
        "webhook_url=https://hooks.slack.com/services/provider-only-token",
        "PuTTY-User-Key-File-3: ssh-rsa\nPrivate-Lines: 1\nprovider-only-token",
    ],
)
def test_diagnostic_redaction_covers_unknown_assignment_encodings(
    diagnostic: str,
) -> None:
    with patch("vast.ops.load_secrets", return_value={}):
        safe = vast_ops._redacted_diagnostic_text(diagnostic)

    assert "provider-only-token" not in safe
    assert "[REDACTED]" in safe or safe == "[DIAGNOSTIC WITHHELD]"


@pytest.mark.parametrize(
    "payload",
    [
        {"argv": ["tool", "--api-key", "provider-only-value-9384"]},
        {"nested": ["--password", "provider-only-value-9384"]},
        {"nested": ["Authorization", "provider-only-value-9384"]},
        {"nested": ["Cookie", "provider-only-value-9384"]},
        {"pair": {"name": "Authorization", "value": "provider-only-value-9384"}},
        {"pair": {"key": "api_key", "value": "provider-only-value-9384"}},
        {"pair": {"field": "password", "data": "provider-only-value-9384"}},
        {"pair": {"parameter": "api_key", "value": "provider-only-value-9384"}},
        {
            "pair": {
                "environment": "VAST_API_KEY",
                "data": "provider-only-value-9384",
            }
        },
        {"pair": {"key_name": "api_key", "key_value": "provider-only-value-9384"}},
        {
            "pair": {
                "optionName": "--password",
                "optionValue": "provider-only-value-9384",
            }
        },
        {"pair": {"kind": "Authorization", "payload": "provider-only-value-9384"}},
        {
            "pair": {
                "scheme": "Basic",
                "value": "provider-only-value-9384",
            }
        },
        {"argv": ["tool", "--access-key-id", "provider-only-value-9384"]},
    ],
)
def test_structured_sensitive_pairs_withhold_the_complete_remote_batch(
    payload: dict[str, object],
) -> None:
    with patch("vast.ops.load_secrets", return_value={}):
        safe = vast_ops._sanitized_remote_events(
            [
                {"event": "heartbeat", **payload},
                {"event": "artifacts_ready", "run_id": "adopt_seed0"},
                {"event": "run_succeeded", "run_id": "adopt_seed0"},
            ]
        )

    assert safe == [
        {
            "event": "remote_event_withheld",
            "diagnostic": "[DIAGNOSTIC WITHHELD]",
        }
    ]
    assert "provider-only-value-9384" not in json.dumps(safe)


@pytest.mark.parametrize(
    "payload",
    [
        {"TLS_KEY": "provider-only-leak-991"},
        {"tlsKey": "provider-only-leak-991"},
        {"signingKey": "provider-only-leak-991"},
        {"encryptionKey": "provider-only-leak-991"},
        {"masterKey": "provider-only-leak-991"},
        {"clientKey": "provider-only-leak-991"},
        {"fernetKey": "provider-only-leak-991"},
        {"TLSKEY": "provider-only-leak-991"},
        {"SIGNINGKEY": "provider-only-leak-991"},
        {"ENCRYPTIONKEY": "provider-only-leak-991"},
        {"MASTERKEY": "provider-only-leak-991"},
        {"CLIENTKEY": "provider-only-leak-991"},
        {"FERNETKEY": "provider-only-leak-991"},
        {"tlskey": "provider-only-leak-991"},
        {"signingkey": "provider-only-leak-991"},
        {"masterkey": "provider-only-leak-991"},
        {"sensitive": "provider-only-leak-991"},
        {"sensitiveValue": "provider-only-leak-991"},
        {"sensitive_data": "provider-only-leak-991"},
        {"confidential": "provider-only-leak-991"},
        {"PAT": "provider-only-leak-991"},
        {"jwt": "provider-only-leak-991"},
        {"sessionId": "provider-only-leak-991"},
        {
            "webhook_url": (
                "https://hooks.slack.com/services/provider-only-leak-991"
            )
        },
        {
            "pair": {
                "name": "signing_key",
                "value": "provider-only-leak-991",
            }
        },
        {
            "pair": {
                "name": "tlsKey",
                "value": "provider-only-leak-991",
            }
        },
        {
            "pair": {
                "name": "encryptionKey",
                "value": "provider-only-leak-991",
            }
        },
        {
            "pair": {
                "name": "TLSKEY",
                "value": "provider-only-leak-991",
            }
        },
        {
            "pair": {
                "name": "signingkey",
                "value": "provider-only-leak-991",
            }
        },
        {"signature": "provider-only-leak-991"},
        {
            "url": (
                "https://example.invalid/blob?sv=1&sp=r&"
                "sig=provider-only-leak-991"
            )
        },
        {"detail": "SharedAccessSignature sr=x&sig=provider-only-leak-991"},
    ],
)
def test_remote_signed_credentials_do_not_survive_a_success_sandwich(
    payload: dict[str, object],
) -> None:
    events = [
        {"event": "runner_exception", "run_id": "bound-run", **payload},
        {
            "event": "artifacts_ready",
            "run_id": "bound-run",
            "exit_code": 0,
        },
        {"event": "run_succeeded", "run_id": "bound-run"},
    ]
    with patch("vast.ops.load_secrets", return_value={}):
        safe = vast_ops._sanitized_remote_events(events)

    assert "provider-only-leak-991" not in json.dumps(safe, sort_keys=True)


@pytest.mark.parametrize("benign", ["compass", "monkey", "hockey", "turkey"])
def test_terminal_key_detection_does_not_match_benign_word_suffixes(
    benign: str,
) -> None:
    assert not vast_ops._is_sensitive_diagnostic_key(benign)


def test_diagnostic_redaction_is_linear_on_maximum_no_match_string() -> None:
    diagnostic = "a" * vast_ops._MAX_DIAGNOSTIC_JSON_STRING_CHARACTERS
    started = time.perf_counter()
    with patch("vast.ops.load_secrets", return_value={}):
        assert vast_ops._redacted_diagnostic_text(diagnostic) == diagnostic
    assert time.perf_counter() - started < 2.0


def test_sensitive_remote_key_name_is_replaced_not_reflected() -> None:
    secret = "provider-only-token"
    event = {
        "event": "runner_exception",
        "run_id": "safe-run",
        "nested": {f"api_key_{secret}": "arbitrary"},
    }
    with patch("vast.ops.load_secrets", return_value={}):
        safe = vast_ops._sanitized_remote_events([event])

    encoded = json.dumps(safe, sort_keys=True)
    assert secret not in encoded
    assert safe[0]["nested"] == {"[REDACTED KEY]": "[REDACTED]"}


@pytest.mark.parametrize(
    "sensitive_key",
    [
        "API Key",
        "api.key",
        "Private Key",
        "SSH Key",
        "Identity File",
        "Ｐａｓｓｗｏｒｄ",
        "ｐｒｉｖａｔｅ＿ｋｅｙ",
    ],
)
def test_sensitive_remote_mapping_key_separator_variants_are_hidden(
    sensitive_key: str,
) -> None:
    secret = "provider-only-token"
    with patch("vast.ops.load_secrets", return_value={}):
        safe = vast_ops._sanitized_remote_events(
            [
                {
                    "event": "runner_exception",
                    "nested": {sensitive_key: secret},
                }
            ]
        )

    encoded = json.dumps(safe, sort_keys=True)
    assert secret not in encoded
    assert safe == [
        {
            "event": "remote_event_withheld",
            "diagnostic": "[DIAGNOSTIC WITHHELD]",
        }
    ]


def test_remote_event_sanitizer_withholds_post_redaction_expansion() -> None:
    def expanding_redactor(value: str) -> str:
        return "x" * 100 if value == "run_succeeded" else value

    with (
        patch("vast.ops._MAX_DIAGNOSTIC_JSON_STRING_CHARACTERS", 32),
        patch("vast.ops.redact_secrets", side_effect=expanding_redactor),
    ):
        events = vast_ops._sanitized_remote_events(
            [{"event": "run_succeeded"}]
        )

    assert events == [
        {
            "event": "remote_event_withheld",
            "diagnostic": "[DIAGNOSTIC WITHHELD]",
        }
    ]


def test_train_tail_redacts_remote_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "train-tail-secret"
    args = SimpleNamespace(run_id="tail-run", train_tail=10)
    with (
        patch(
            "vast.ops.get_record",
            return_value={"run_id": "tail-run", "instance_id": 1234},
        ),
        patch("vast.ops.ssh_url", return_value=("root", "host", 22)),
        patch(
            "vast.ops.remote_exec",
            return_value=SimpleNamespace(
                returncode=0, stdout=f"remote api_key={secret}\n"
            ),
        ) as execute,
        patch("vast.ops.load_secrets", return_value={"VAST_API_KEY": secret}),
    ):
        assert vast_ops.command_logs(args) == 0

    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert captured.out == "[DIAGNOSTIC WITHHELD]"
    remote_command = execute.call_args.args[1]
    assert "O_NOFOLLOW" in remote_command
    assert str(vast_ops._MAX_MANUAL_DIAGNOSTIC_BYTES) in remote_command


@pytest.mark.parametrize("stage_diagnostics", [False, True])
def test_diagnose_commands_redact_successful_remote_snapshot(
    capsys: pytest.CaptureFixture[str], stage_diagnostics: bool
) -> None:
    manifest = adoption_manifest()
    secret = "diagnostic-snapshot-secret"
    remote_only_token = "diagnostic-remote-only-token"
    response = SimpleNamespace(
        returncode=0,
        stdout=json.dumps(
            {
                "event": "diagnostics_snapshot",
                "nested": {
                    "error": f"remote api_key={secret}",
                    "access_token": remote_only_token,
                },
            }
        ),
        stderr="",
    )
    if stage_diagnostics:
        stage = {
            "stage_id": "diagnostic-stage",
            "parent_run_id": manifest.run_id,
            "instance_id": 1234,
            "manifest": manifest.__dict__,
            "remote_repo": "/workspace/stage/repo",
            "status": "stage_running",
        }
        args = SimpleNamespace(stage_id=stage["stage_id"], metrics_tail=10)
        record_patch = patch("vast.ops.get_stage_record", return_value=stage)
        command = vast_ops.command_diagnose_stage
    else:
        record = {
            "run_id": manifest.run_id,
            "instance_id": 1234,
            "manifest": manifest.__dict__,
            "status": "running",
        }
        args = SimpleNamespace(run_id=manifest.run_id, metrics_tail=10)
        record_patch = patch("vast.ops.get_record", return_value=record)
        command = vast_ops.command_diagnose

    with (
        record_patch,
        patch("vast.ops.ssh_url", return_value=("root", "host", 22)),
        patch("vast.ops.remote_exec", return_value=response) as execute,
        patch("vast.ops.load_secrets", return_value={"VAST_API_KEY": secret}),
    ):
        assert command(args) == 0

    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert remote_only_token not in captured.out + captured.err
    assert "[DIAGNOSTIC WITHHELD]" in captured.out
    remote_command = execute.call_args.args[1]
    assert "O_NOFOLLOW" in remote_command
    assert str(vast_ops._MAX_MANUAL_METRIC_TAIL_BYTES) in remote_command


def test_session_stage_start_error_is_redacted_before_persistence(
    tmp_path: Path,
) -> None:
    parent_manifest = replace(
        adoption_manifest(), success_policy="keep_alive_after_verify"
    )
    stage_manifest = replace(
        parent_manifest,
        run_id="redacted-stage-start",
        description="redacted stage start failure",
        ttl_hours=1.0,
    )
    parent = {
        "run_id": parent_manifest.run_id,
        "instance_id": 1234,
        "manifest": parent_manifest.__dict__,
        "status": "session_ready",
        "created_at": utc_now(),
        "dph": 0.1,
    }
    secret = "stage-start-secret"
    stage_path = tmp_path / "stages.json"
    args = SimpleNamespace(
        parent_run=parent_manifest.run_id,
        confirm_instance="1234",
        confirm_parent=parent_manifest.run_id,
        manifest=tmp_path / "stage.json",
        confirm_stage=stage_manifest.run_id,
        confirm_source_commit=stage_manifest.source_commit,
    )

    with (
        patch("vast.ops.STAGE_STATE_FILE", stage_path),
        patch("vast.ops.get_record", return_value=parent),
        patch("vast.ops.load_manifest", return_value=stage_manifest),
        patch("vast.ops.elapsed_and_cost", return_value=(0.0, 0.0)),
        patch("vast.ops.cleanup_trigger_hours", return_value=5.0),
        patch("vast.ops.cleanup_reserve_hours", return_value=0.5),
        patch("vast.ops.run_vast", return_value=[{"id": 1234}]),
        patch(
            "vast.ops.create_source_bundle",
            return_value=(tmp_path / "source.bundle", tmp_path / "resolved.json"),
        ),
        patch("vast.ops.ssh_url", return_value=("root", "host", 22)),
        patch(
            "vast.ops.start_remote_session_stage",
            side_effect=OpsError(f"remote api_key={secret}"),
        ),
        patch("vast.ops.load_secrets", return_value={"VAST_API_KEY": secret}),
        pytest.raises(OpsError),
    ):
        vast_ops.command_session_stage(args)

    persisted = stage_path.read_text(encoding="utf-8")
    assert secret not in persisted
    assert "[DIAGNOSTIC WITHHELD]" in persisted


@pytest.mark.parametrize("provider_id", [True, 1234.9, "1234"])
def test_session_stage_requires_exact_provider_parent_identity(
    tmp_path: Path, provider_id: object
) -> None:
    parent_manifest = replace(
        adoption_manifest(), success_policy="keep_alive_after_verify"
    )
    stage_manifest = replace(
        parent_manifest,
        run_id="exact-stage-parent",
        ttl_hours=1.0,
    )
    parent = {
        "run_id": parent_manifest.run_id,
        "instance_id": 1234,
        "manifest": parent_manifest.__dict__,
        "status": "session_ready",
        "created_at": utc_now(),
        "dph": 0.1,
    }
    args = SimpleNamespace(
        parent_run=parent_manifest.run_id,
        confirm_instance="1234",
        confirm_parent=parent_manifest.run_id,
        manifest=tmp_path / "stage.json",
        confirm_stage=stage_manifest.run_id,
        confirm_source_commit=stage_manifest.source_commit,
    )
    with (
        patch("vast.ops.STAGE_STATE_FILE", tmp_path / "stages.json"),
        patch("vast.ops.get_record", return_value=parent),
        patch("vast.ops.load_manifest", return_value=stage_manifest),
        patch("vast.ops.elapsed_and_cost", return_value=(0.0, 0.0)),
        patch("vast.ops.cleanup_trigger_hours", return_value=5.0),
        patch("vast.ops.cleanup_reserve_hours", return_value=0.5),
        patch("vast.ops.run_vast", return_value=[{"id": provider_id}]),
        patch("vast.ops.create_source_bundle") as create_bundle,
        pytest.raises(OpsError, match="exact positive integer"),
    ):
        vast_ops.command_session_stage(args)

    create_bundle.assert_not_called()


def test_stage_monitor_sanitizes_remote_event_before_every_local_sink(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = adoption_manifest()
    parent_run_id = manifest.run_id
    stage_id = "stage-secret-boundary"
    parent = {
        "run_id": parent_run_id,
        "instance_id": 1234,
        "manifest": manifest.__dict__,
        "status": "session_ready",
    }
    stage = {
        "stage_id": stage_id,
        "parent_run_id": parent_run_id,
        "manifest": manifest.__dict__,
        "remote_ops": "/workspace/stage/ops",
        "status": "running",
    }
    state_path = tmp_path / "instances.json"
    stage_path = tmp_path / "stages.json"
    event_path = tmp_path / "events.jsonl"
    state_path.write_text(json.dumps([parent]), encoding="utf-8")
    stage_path.write_text(json.dumps([stage]), encoding="utf-8")
    secret = "stage-remote-secret"
    remote_only_token = "stage-remote-only-token"

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.STAGE_STATE_FILE", stage_path),
        patch("vast.ops.EVENT_FILE", event_path),
        patch("vast.ops.load_secrets", return_value={"VAST_API_KEY": secret}),
        patch("vast.ops.run_vast", return_value=[{"id": 1234}]),
        patch("vast.ops.ssh_url", return_value=("root", "host", 22)),
        patch(
            "vast.ops.read_remote_events",
            return_value=[
                {
                    "event": "runner_exception",
                    "nested": {
                        "error": f"remote api_key={secret}",
                        "credential": remote_only_token,
                    },
                }
            ],
        ),
        patch("vast.ops.time.sleep", side_effect=KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        vast_ops._monitor_session_stage(stage_id)

    captured = capsys.readouterr()
    combined = (
        stage_path.read_text(encoding="utf-8")
        + event_path.read_text(encoding="utf-8")
        + captured.out
        + captured.err
    )
    assert secret not in combined
    assert remote_only_token not in combined
    assert "[DIAGNOSTIC WITHHELD]" in combined
    persisted = json.loads(stage_path.read_text(encoding="utf-8"))[0]
    assert persisted["last_remote_event"] == {
        "event": "remote_event_withheld",
        "diagnostic": "[DIAGNOSTIC WITHHELD]",
    }


@pytest.mark.parametrize(
    ("failure_site", "expected_event"),
    [
        ("provider", "stage_live_query_error"),
        ("transport", "stage_transport_error"),
    ],
)
def test_stage_monitor_redacts_provider_and_transport_errors(
    tmp_path: Path, failure_site: str, expected_event: str
) -> None:
    manifest = adoption_manifest()
    stage_id = f"stage-{failure_site}-error"
    parent = {
        "run_id": manifest.run_id,
        "instance_id": 1234,
        "manifest": manifest.__dict__,
        "status": "session_ready",
    }
    stage = {
        "stage_id": stage_id,
        "parent_run_id": manifest.run_id,
        "manifest": manifest.__dict__,
        "remote_ops": "/workspace/stage/ops",
        "status": "running",
    }
    state_path = tmp_path / "instances.json"
    stage_path = tmp_path / "stages.json"
    event_path = tmp_path / "events.jsonl"
    state_path.write_text(json.dumps([parent]), encoding="utf-8")
    stage_path.write_text(json.dumps([stage]), encoding="utf-8")
    secret = "stage-diagnostic-secret"

    def provider(*_args: object, **_kwargs: object) -> list[dict[str, int]]:
        if failure_site == "provider":
            raise OpsError(f"provider api_key={secret}")
        return [{"id": 1234}]

    def connection(*_args: object, **_kwargs: object) -> tuple[str, str, int]:
        if failure_site == "transport":
            raise OSError(f"transport api_key={secret}")
        return ("root", "host", 22)

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.STAGE_STATE_FILE", stage_path),
        patch("vast.ops.EVENT_FILE", event_path),
        patch("vast.ops.load_secrets", return_value={"VAST_API_KEY": secret}),
        patch("vast.ops.run_vast", side_effect=provider),
        patch("vast.ops.ssh_url", side_effect=connection),
        patch("vast.ops.time.sleep", side_effect=KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        vast_ops._monitor_session_stage(stage_id)

    event_text = event_path.read_text(encoding="utf-8")
    assert secret not in event_text
    assert "[DIAGNOSTIC WITHHELD]" in event_text
    assert expected_event in event_text


def test_main_redacts_operational_exception_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "top-level-secret"

    def fail(_args: object) -> int:
        raise OpsError(f"provider api_key={secret}")

    parser = SimpleNamespace(
        parse_args=lambda: SimpleNamespace(function=fail),
    )
    with (
        patch("vast.ops.build_parser", return_value=parser),
        patch("vast.ops.load_secrets", return_value={"VAST_API_KEY": secret}),
        pytest.raises(SystemExit) as raised,
    ):
        vast_ops.main()

    assert raised.value.code == 2
    captured = capsys.readouterr()
    assert secret not in captured.err
    assert "[DIAGNOSTIC WITHHELD]" in captured.err


def test_status_redacts_live_query_failure_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "status-query-secret"
    with (
        patch("vast.ops.run_vast", side_effect=OpsError(f"api_key={secret}")),
        patch("vast.ops.load_secrets", return_value={"VAST_API_KEY": secret}),
        patch("vast.ops.load_state", return_value=[]),
    ):
        assert vast_ops.command_status(SimpleNamespace()) == 0
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert "[DIAGNOSTIC WITHHELD]" in captured.err


@pytest.mark.parametrize("snapshot", [{}, "", 0, False, None])
def test_launch_label_guard_rejects_every_falsey_nonlist_snapshot(
    snapshot: object,
) -> None:
    with (
        patch("vast.ops.run_vast", return_value=snapshot),
        pytest.raises(OpsError, match="did not return a list"),
    ):
        vast_ops.assert_live_label_unused("quant-experiments-new-run")


def test_launch_label_guard_accepts_a_real_empty_snapshot() -> None:
    with patch("vast.ops.run_vast", return_value=[]):
        vast_ops.assert_live_label_unused("quant-experiments-new-run")


@pytest.mark.parametrize(
    "snapshot",
    [
        [{}],
        ["malformed"],
        [{"id": "1", "label": None}],
        [{"id": 1}],
        [{"id": 1, "label": None}, {"id": 1, "label": "other"}],
    ],
)
def test_launch_label_guard_rejects_malformed_or_duplicate_rows(
    snapshot: object,
) -> None:
    with (
        patch("vast.ops.run_vast", return_value=snapshot),
        pytest.raises(OpsError),
    ):
        vast_ops.assert_live_label_unused("quant-experiments-new-run")


def test_created_instance_verification_rejects_duplicate_provider_ids() -> None:
    manifest = adoption_manifest()
    with (
        patch(
            "vast.ops.run_vast",
            return_value=[
                {"id": 42, "label": "quant-experiments-adopt_seed0"},
                {"id": 42, "label": "quant-experiments-adopt_seed0"},
            ],
        ),
        pytest.raises(OpsError, match="duplicate instance IDs"),
    ):
        verify_created_instance(
            42,
            manifest,
            expected_label="quant-experiments-adopt_seed0",
        )


def test_destroy_orphan_rejects_quarantined_create_response_id_without_mutation(
) -> None:
    args = SimpleNamespace(
        instance=1234,
        confirm_label="quant-experiments-manual-orphan",
    )
    with (
        patch(
            "vast.ops.load_state",
            return_value=[
                {
                    "run_id": "different-run",
                    "instance_id": None,
                    "create_response_instance_id": 1234,
                }
            ],
        ),
        patch("vast.ops.load_stage_state", return_value=[]),
        patch("vast.ops.run_vast") as provider,
        patch("vast.ops.destroy_instance") as destroy,
        pytest.raises(OpsError, match="referenced by local lifecycle state"),
    ):
        vast_ops.command_destroy_orphan(args)

    provider.assert_not_called()
    destroy.assert_not_called()


def test_destroy_orphan_continues_to_reject_tracked_instance() -> None:
    args = SimpleNamespace(
        instance=1234,
        confirm_label="quant-experiments-manual-orphan",
    )
    with (
        patch(
            "vast.ops.load_state",
            return_value=[{"run_id": "different-run", "instance_id": 1234}],
        ),
        patch("vast.ops.load_stage_state", return_value=[]),
        patch("vast.ops.run_vast") as provider,
        patch("vast.ops.destroy_instance") as destroy,
        pytest.raises(OpsError, match="referenced by local lifecycle state"),
    ):
        vast_ops.command_destroy_orphan(args)

    provider.assert_not_called()
    destroy.assert_not_called()


@pytest.mark.parametrize(
    ("records", "stages"),
    [
        ([{"run_id": "race", "winner_instance_id": 1234}], []),
        (
            [
                {
                    "run_id": "race",
                    "reconciliation_exact_match_ids": [1234],
                }
            ],
            [],
        ),
        (
            [
                {
                    "run_id": "race",
                    "candidates": [{"instance_id": 1234}],
                }
            ],
            [],
        ),
        (
            [
                {
                    "run_id": "race",
                    "candidates": [
                        {"create_response_instance_id": 1234}
                    ],
                }
            ],
            [],
        ),
        (
            [
                {
                    "run_id": "race",
                    "candidates": [
                        {"discovered_instance_ids": [1234]}
                    ],
                }
            ],
            [],
        ),
        ([], [{"stage_id": "stage", "instance_id": 1234}]),
    ],
    ids=(
        "race-winner",
        "reconciliation-exact-match",
        "race-candidate",
        "race-create-response",
        "race-discovered",
        "session-stage",
    ),
)
def test_destroy_orphan_rejects_every_other_direct_lifecycle_identity(
    records: list[dict[str, object]], stages: list[dict[str, object]]
) -> None:
    args = SimpleNamespace(
        instance=1234,
        confirm_label="quant-experiments-manual-orphan",
    )
    with (
        patch("vast.ops.load_state", return_value=records),
        patch("vast.ops.load_stage_state", return_value=stages),
        patch("vast.ops.run_vast") as provider,
        patch("vast.ops.destroy_instance") as destroy,
        pytest.raises(OpsError, match="referenced by local"),
    ):
        vast_ops.command_destroy_orphan(args)

    provider.assert_not_called()
    destroy.assert_not_called()


@pytest.mark.parametrize(
    ("records", "stages"),
    [
        ([{"run_id": "manual-orphan", "instance_id": None}], []),
        (
            [
                {
                    "run_id": "other",
                    "expected_label": "quant-experiments-manual-orphan",
                }
            ],
            [],
        ),
        (
            [
                {
                    "run_id": "race",
                    "candidates": [
                        {"label": "quant-experiments-manual-orphan"}
                    ],
                }
            ],
            [],
        ),
        ([], [{"parent_run_id": "manual-orphan"}]),
    ],
    ids=("run-id", "expected-label", "race-label", "session-stage-parent"),
)
def test_destroy_orphan_rejects_any_local_label_owner(
    records: list[dict[str, object]], stages: list[dict[str, object]]
) -> None:
    args = SimpleNamespace(
        instance=1234,
        confirm_label="quant-experiments-manual-orphan",
    )
    with (
        patch("vast.ops.load_state", return_value=records),
        patch("vast.ops.load_stage_state", return_value=stages),
        patch("vast.ops.run_vast") as provider,
        patch("vast.ops.destroy_instance") as destroy,
        pytest.raises(OpsError, match="label belongs to local"),
    ):
        vast_ops.command_destroy_orphan(args)

    provider.assert_not_called()
    destroy.assert_not_called()


@pytest.mark.parametrize(
    ("records", "stages"),
    [
        ([{"run_id": 1234}], []),
        ([{"run_id": "other", "expected_label": 1234}], []),
        (
            [{"run_id": "race", "candidates": [{"label": 1234}]}],
            [],
        ),
        (
            [
                {
                    "run_id": "race",
                    "reconciliation_exact_match_ids": "1234",
                }
            ],
            [],
        ),
        ([], [{"stage_id": 1234}]),
        ([], [{"parent_run_id": 1234}]),
    ],
    ids=(
        "run-id",
        "expected-label",
        "candidate-label",
        "reconciliation-identities",
        "stage-id",
        "stage-parent",
    ),
)
def test_destroy_orphan_fails_closed_on_malformed_local_ownership_state(
    records: list[dict[str, object]], stages: list[dict[str, object]]
) -> None:
    args = SimpleNamespace(
        instance=1234,
        confirm_label="quant-experiments-manual-orphan",
    )
    with (
        patch("vast.ops.load_state", return_value=records),
        patch("vast.ops.load_stage_state", return_value=stages),
        patch("vast.ops.run_vast") as provider,
        patch("vast.ops.destroy_instance") as destroy,
        pytest.raises(OpsError, match="malformed"),
    ):
        vast_ops.command_destroy_orphan(args)

    provider.assert_not_called()
    destroy.assert_not_called()


def test_destroy_orphan_rechecks_ownership_after_live_query() -> None:
    label = "quant-experiments-manual-orphan"
    args = SimpleNamespace(instance=1234, confirm_label=label)
    with (
        patch(
            "vast.ops.load_state",
            side_effect=[
                [],
                [
                    {
                        "run_id": "late-owner",
                        "create_response_instance_id": 1234,
                    }
                ],
            ],
        ),
        patch("vast.ops.load_stage_state", return_value=[]),
        patch(
            "vast.ops.run_vast",
            return_value=[{"id": 1234, "label": label}],
        ) as provider,
        patch("vast.ops.destroy_instance") as destroy,
        pytest.raises(OpsError, match="referenced by local lifecycle state"),
    ):
        vast_ops.command_destroy_orphan(args)

    provider.assert_called_once_with("show", "instances")
    destroy.assert_not_called()


def test_destroy_orphan_allows_genuinely_unrecorded_exact_project_label() -> None:
    label = "quant-experiments-manual-orphan"
    args = SimpleNamespace(instance=1234, confirm_label=label)
    with (
        patch("vast.ops.load_state", return_value=[]),
        patch("vast.ops.load_stage_state", return_value=[]),
        patch(
            "vast.ops.run_vast",
            return_value=[{"id": 1234, "label": label}],
        ) as provider,
        patch("vast.ops.destroy_instance") as destroy,
    ):
        assert vast_ops.command_destroy_orphan(args) == 0

    provider.assert_called_once_with("show", "instances")
    destroy.assert_called_once_with(
        1234,
        "manual-orphan",
        "explicit_orphan_cleanup",
        untracked_orphan_label=label,
    )


def test_destroy_orphan_final_barrier_blocks_late_lifecycle_owner(
    tmp_path: Path,
) -> None:
    label = "quant-experiments-manual-orphan"
    state_path = tmp_path / "instances.json"
    stage_path = tmp_path / "stages.json"
    event_path = tmp_path / "events.jsonl"
    state_path.write_text(
        json.dumps(
            [
                {
                    "run_id": "late-owner",
                    "instance_id": None,
                    "create_response_instance_id": 1234,
                }
            ]
        ),
        encoding="utf-8",
    )
    stage_path.write_text("[]", encoding="utf-8")
    provider_calls: list[tuple[object, ...]] = []

    def provider(*arguments: object, **_: object) -> list[dict[str, object]]:
        provider_calls.append(arguments)
        return [{"id": 1234, "label": label}]

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.STAGE_STATE_FILE", stage_path),
        patch("vast.ops.EVENT_FILE", event_path),
        patch("vast.ops.run_vast", side_effect=provider),
        patch("vast.ops.DESTROY_MAX_ATTEMPTS", 1),
        patch("vast.ops.DESTROY_VERIFY_MAX_ATTEMPTS", 1),
        pytest.raises(OpsError, match="referenced by local lifecycle state"),
    ):
        destroy_instance(
            1234,
            "manual-orphan",
            "explicit_orphan_cleanup",
            launch_supervisor=False,
            untracked_orphan_label=label,
        )

    assert not any(call[:2] == ("destroy", "instance") for call in provider_calls)


def test_reconcile_rejects_duplicate_provider_ids_before_state_merge(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "instances.json"
    state_path.write_text("[]", encoding="utf-8")
    live = [
        {"id": 42, "label": "first"},
        {"id": 42, "label": "second"},
    ]
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.run_vast", return_value=live),
        pytest.raises(OpsError, match="duplicate instance IDs"),
    ):
        command_reconcile(SimpleNamespace(restart_monitors=False))


def test_ordinary_monitor_retries_duplicate_provider_snapshot(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    started = utc_now()
    record = {
        "run_id": manifest.run_id,
        "instance_id": 1234,
        "dph": 0.1,
        "created_at": started,
        "campaign_started_at": started,
        "source_commit": manifest.source_commit,
        "manifest": manifest.__dict__,
        "status": "running",
    }
    state_path = tmp_path / "instances.json"
    event_path = tmp_path / "events.jsonl"
    state_path.write_text(json.dumps([record]), encoding="utf-8")
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.EVENT_FILE", event_path),
        patch(
            "vast.ops.run_vast",
            return_value=[{"id": 1234}, {"id": 1234}],
        ),
        patch("vast.ops._bounded_sleep", side_effect=KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        _monitor_run(manifest.run_id)

    assert json.loads(state_path.read_text(encoding="utf-8"))[0]["status"] == "running"
    assert "live_query_error" in event_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("command_name", ["status", "search", "reconcile"])
def test_successful_provider_reports_redact_remote_fields(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    command_name: str,
) -> None:
    secret = "provider-report-secret"
    if command_name == "status":
        live = [{"id": 1234, "actual_status": f"api_key={secret}"}]
        with (
            patch("vast.ops.run_vast", return_value=live),
            patch(
                "vast.ops.load_state",
                return_value=[
                    {"run_id": "status-run", "instance_id": 1234, "status": "running"}
                ],
            ),
            patch("vast.ops.elapsed_and_cost", return_value=(0.0, 0.0)),
            patch("vast.ops.load_secrets", return_value={}),
        ):
            vast_ops.command_status(SimpleNamespace())
    elif command_name == "search":
        offer = {
            "id": 1,
            "machine_id": 2,
            "gpu_name": f"api_key={secret}",
            "num_gpus": 1,
            "dph_total": 0.1,
            "reliability2": 0.99,
            "inet_down": 1000,
            "cpu_name": "consumer",
            "cpu_cores_effective": 8,
            "cpu_ram": 32768,
            "geolocation": "CA",
        }
        args = SimpleNamespace(
            gpu="RTX_3060_Ti",
            max_dph=0.1,
            num_gpus=1,
            disk_gb=20,
            min_cuda=12.8,
            min_inet=100,
            min_cpu_cores=4,
            min_cpu_ram_gb=16,
            require_consumer_cpu=False,
            country="CA",
            avoid_machine=[],
            limit=5,
        )
        with (
            patch("vast.ops.load_machine_blacklist", return_value=set()),
            patch("vast.ops.active_machine_ids", return_value=set()),
            patch("vast.ops.search_offers", return_value=[offer]),
            patch("vast.ops.load_secrets", return_value={}),
        ):
            vast_ops.command_search(args)
    else:
        state_path = tmp_path / "instances.json"
        state_path.write_text("[]", encoding="utf-8")
        live = [{"id": 1, "label": f"api_key={secret}", "dph_total": 0.1}]
        with (
            patch("vast.ops.STATE_FILE", state_path),
            patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
            patch("vast.ops.run_vast", return_value=live),
            patch("vast.ops.load_secrets", return_value={}),
        ):
            command_reconcile(SimpleNamespace(restart_monitors=False))

    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err


def test_raw_and_expanding_diagnostics_are_withheld_within_string_bound() -> None:
    with patch("vast.ops._MAX_DIAGNOSTIC_JSON_STRING_CHARACTERS", 32):
        assert vast_ops._redacted_exception_text(OpsError("x" * 33)) == (
            "[DIAGNOSTIC WITHHELD]"
        )
        with patch("vast.ops.redact_secrets", return_value="y" * 33):
            assert vast_ops._redacted_exception_text(OpsError("short")) == (
                "[DIAGNOSTIC WITHHELD]"
            )
            assert vast_ops._redacted_diagnostic_text("short") == (
                "[DIAGNOSTIC WITHHELD]"
            )


@pytest.mark.parametrize(
    "text",
    [
        '{"CONTAINER_API_KEY":"provider-only-token"}',
        '{"nested":{"password" : "provider-only-token"}}',
        "Authorization: Bearer provider-only-token",
        "access_token=provider-only-token",
        "credential: provider-only-token",
    ],
)
def test_free_text_redaction_covers_unknown_remote_credentials(text: str) -> None:
    with patch("vast.ops.load_secrets", return_value={}):
        safe = vast_ops._redacted_diagnostic_text(text)
    assert "provider-only-token" not in safe
    assert "[REDACTED]" in safe or safe == "[DIAGNOSTIC WITHHELD]"


def test_adoption_setup_cleanup_and_monitor_errors_are_redacted_before_sinks(
    tmp_path: Path,
) -> None:
    manifest = adoption_manifest()
    secret = "adoption-lifecycle-secret"
    state_path = tmp_path / "instances.json"
    event_path = tmp_path / "events.jsonl"
    args = SimpleNamespace(
        manifest=tmp_path / "run.json",
        instance=1234,
        detach=False,
    )
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.EVENT_FILE", event_path),
        patch("vast.ops.load_manifest", return_value=manifest),
        patch(
            "vast.ops.create_source_bundle",
            return_value=(tmp_path / "source.bundle", tmp_path / "run.json"),
        ),
        patch("vast.ops.run_vast", return_value=[adoption_instance()]),
        patch("vast.ops.load_machine_blacklist", return_value=set()),
        patch("vast.ops.wait_for_ssh", return_value=("root", "host", 22)),
        patch("vast.ops.verify_adoption_host_idle", return_value={"idle": True}),
        patch(
            "vast.ops.label_adopted_instance",
            return_value="quant-experiments-adopt_seed0",
        ),
        patch(
            "vast.ops.start_remote_run",
            side_effect=OpsError(f"setup api_key={secret}"),
        ),
        patch(
            "vast.ops.destroy_instance",
            side_effect=OpsError(f"cleanup api_key={secret}"),
        ),
        patch(
            "vast.ops.launch_detached_monitor",
            side_effect=OpsError(f"monitor api_key={secret}"),
        ),
        patch("vast.ops.load_secrets", return_value={"VAST_API_KEY": secret}),
        pytest.raises(OpsError, match="setup"),
    ):
        command_adopt_existing(args)

    combined = state_path.read_text(encoding="utf-8") + event_path.read_text(
        encoding="utf-8"
    )
    assert secret not in combined
    assert combined.count("[DIAGNOSTIC WITHHELD]") >= 3
    record = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert record["status"] == "provisioning_cleanup_pending"


def test_main_still_exits_two_when_stderr_is_unavailable() -> None:
    class BrokenStderr:
        def write(self, _value: str) -> int:
            raise OSError("stderr unavailable")

        def flush(self) -> None:
            raise OSError("stderr unavailable")

    def fail(_args: object) -> int:
        raise OSError("operational failure")

    parser = SimpleNamespace(
        parse_args=lambda: SimpleNamespace(function=fail),
    )
    with (
        patch("vast.ops.build_parser", return_value=parser),
        patch("vast.ops.sys.stderr", BrokenStderr()),
        pytest.raises(SystemExit) as raised,
    ):
        vast_ops.main()

    assert raised.value.code == 2


def test_detached_monitor_diagnostics_cannot_block_attached_fallback() -> None:
    secret = "monitor-launch-secret"
    with (
        patch(
            "vast.ops.launch_detached_monitor",
            side_effect=OSError(f"launch api_key={secret}"),
        ),
        patch("vast.ops.load_secrets", return_value={"VAST_API_KEY": secret}),
        patch("vast.ops.append_local_event", side_effect=OSError("event unavailable")),
        patch("builtins.print", side_effect=OSError("stdout unavailable")),
        patch("vast.ops.monitor_run", return_value=7) as attached,
    ):
        assert vast_ops.start_requested_monitor("run", True) == 7

    attached.assert_called_once_with("run")


def test_reconcile_live_instance_past_controller_deadline_still_destroys(
    tmp_path: Path,
) -> None:
    """Regression: reconcile must not hand a live, past-deadline instance to a
    destroy-reconciliation supervisor whose deadline has already expired."""

    manifest = adoption_manifest()
    state_path = tmp_path / "instances.json"
    label = f"quant-experiments-{manifest.run_id}"
    past = "2020-01-01T00:00:00Z"
    state_path.write_text(
        json.dumps(
            [
                {
                    "run_id": manifest.run_id,
                    "instance_id": 1234,
                    "expected_label": label,
                    "created_at": past,
                    "campaign_started_at": past,
                    "controller_deadline_at": past,
                    "manifest": manifest.__dict__,
                    "status": "cleanup_required",
                    "cleanup_required": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    live_row = {"id": 1234, "label": label, "actual_status": "running"}
    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops.run_vast", return_value=[live_row]),
        patch(
            "vast.ops.launch_detached_destroy_reconciliation", return_value=77
        ) as verifier,
    ):
        assert command_reconcile(SimpleNamespace(restart_monitors=True)) == 0
    updated = json.loads(state_path.read_text(encoding="utf-8"))[0]
    verifier.assert_called_once_with(manifest.run_id)
    manual = vast_ops.parse_utc(updated["manual_destroy_reconciliation_deadline_at"])
    assert manual > datetime.now(UTC)
    assert updated["cap_control_failed"] is True

    calls = {"destroy": 0}

    def fake_present(*_args: object, **_kwargs: object) -> bool:
        return calls["destroy"] == 0

    def fake_destroy(*_args: object, **_kwargs: object) -> None:
        calls["destroy"] += 1

    with (
        patch("vast.ops.STATE_FILE", state_path),
        patch("vast.ops.STATE_LOCK_FILE", tmp_path / "state.lock"),
        patch("vast.ops.MONITOR_DIR", tmp_path / "monitors"),
        patch("vast.ops._claim_supervisor_lock", side_effect=lambda p, *_a, **_k: p),
        patch("vast.ops._release_supervisor_lock"),
        patch("vast.ops._live_instance_is_present", side_effect=fake_present),
        patch("vast.ops.destroy_instance", side_effect=fake_destroy),
        patch("vast.ops._bounded_sleep", side_effect=_StopLoop),
    ):
        # The supervisor holds its whole follow-up lease by design; stop after
        # the first loop iteration -- we only need proof that it ran and
        # attempted the destroy instead of exiting on an expired deadline.
        with pytest.raises(_StopLoop):
            vast_ops._destroy_reconciliation(manifest.run_id)
    assert calls["destroy"] == 1


class _StopLoop(Exception):
    pass


def test_live_row_not_running_flags_stopped_and_exited_instances() -> None:
    assert vast_ops._live_row_not_running({"actual_status": "exited"})
    assert vast_ops._live_row_not_running({"actual_status": "stopped"})
    assert vast_ops._live_row_not_running(
        {"actual_status": "running", "intended_status": "stopped"}
    )
    assert not vast_ops._live_row_not_running({"actual_status": "running"})
    assert not vast_ops._live_row_not_running({"actual_status": "loading"})


def test_manifest_stall_minutes_bounds() -> None:
    base = adoption_manifest()
    assert base.stall_minutes == 45
    replace(base, stall_minutes=0).validate()
    replace(base, stall_minutes=120).validate()
    for bad in (5, 2000, -1, True):
        with pytest.raises(OpsError, match="stall_minutes"):
            replace(base, stall_minutes=bad).validate()


def test_manifest_only_requires_seed_and_checkpoint_notes() -> None:
    base = adoption_manifest()
    replace(base, interaction_budget="", intervention_suite="").validate()
    with pytest.raises(OpsError, match="seed_policy and checkpoint_cadence"):
        replace(base, seed_policy=" ").validate()


def test_load_secrets_tolerates_bom_quotes_and_export(tmp_path: Path) -> None:
    secrets = tmp_path / "secrets.env"
    secrets.write_bytes(
        "\ufeffVAST_API_KEY=\"abc123\"\nexport OTHER='x y'\n# c\n".encode("utf-8")
    )
    with patch("vast.ops.SECRETS_FILE", secrets):
        loaded = vast_ops.load_secrets()
    assert loaded == {"VAST_API_KEY": "abc123", "OTHER": "x y"}


def test_start_remote_run_anchors_workload_stop_to_campaign_schedule(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(json.dumps({"run_id": "r"}), encoding="utf-8")
    bundle = tmp_path / "source.bundle"
    bundle.write_bytes(b"x")
    with (
        patch("vast.ops.remote_exec"),
        patch("vast.ops.scp_upload"),
    ):
        try:
            vast_ops.start_remote_run(
                ("root", "h", 22),
                bundle,
                manifest_path,
                workload_stop_at="2030-01-01T00:00:00Z",
            )
        except Exception:
            pass  # later remote steps are irrelevant here
    written = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = datetime(2030, 1, 1, tzinfo=UTC).timestamp() - (
        vast_ops.REMOTE_PACKAGING_MARGIN_SECONDS
    )
    assert written["workload_stop_at_epoch"] == expected


def test_peek_paths_are_restricted_to_experiment_outputs() -> None:
    assert vast_ops._peek_remote_path("train.log").endswith("/train.log")
    assert vast_ops._peek_remote_path("artifacts/run1/metrics.json").endswith(
        "/quant-experiments/artifacts/run1/metrics.json"
    )
    assert vast_ops._peek_remote_path("").endswith("/quant-experiments/artifacts")
    for bad in ("/etc/passwd", "vast/secrets.env", "artifacts/../.git", ".."):
        with pytest.raises(OpsError, match="peek paths"):
            vast_ops._peek_remote_path(bad)


def test_peek_is_read_only_and_bounded() -> None:
    seen: list[str] = []

    def fake_exec(_connection: object, command: str, **_kwargs: object) -> object:
        seen.append(command)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    with (
        patch("vast.ops.get_record", return_value={"instance_id": 5}),
        patch("vast.ops.record_instance_id", return_value=5),
        patch("vast.ops.ssh_url", return_value=("root", "h", 22)),
        patch("vast.ops.remote_exec", side_effect=fake_exec),
    ):
        assert (
            vast_ops.command_peek(
                SimpleNamespace(run_id="r", path="artifacts/x", tail=50)
            )
            == 0
        )
    assert "tail -n 50" in seen[0] and "head -c" in seen[0]
    for verb in ("rm ", "mv ", "kill", "> "):
        assert verb not in seen[0]


# Vast's live `show instances` row carries no offer/ask identity at all
# (observed 2026-11-02, instance 58748479). The create response binds the
# purchased offer to the instance ID instead.

def _real_shaped_vast_row(**updates: object) -> dict[str, object]:
    row = adoption_instance()
    for key in ("ask_contract_id", "offer_id", "actual_offer_id", "contract_offer_id"):
        row.pop(key, None)
    row.update({"bundle_id": 5, "host_id": 7, "is_bid": False, "template_id": 1})
    row.update(updates)
    return row


def test_unreported_offer_identity_is_bound_by_create_response_id() -> None:
    manifest = replace(adoption_manifest(), required_geolocation=None)
    offer = {"id": 11, "machine_id": 999}
    row = _real_shaped_vast_row()
    _validate_live_contract_constraints(
        row, manifest, expected_offer=offer, create_response_instance_id=1234
    )
    with pytest.raises(OpsError, match="does not match the create-response instance ID"):
        _validate_live_contract_constraints(
            row, manifest, expected_offer=offer, create_response_instance_id=4321
        )


def test_reported_offer_identity_must_still_match_even_with_binding() -> None:
    manifest = replace(adoption_manifest(), required_geolocation=None)
    row = _real_shaped_vast_row(ask_contract_id=12)
    with pytest.raises(OpsError, match="does not match selected offer"):
        _validate_live_contract_constraints(
            row,
            manifest,
            expected_offer={"id": 11, "machine_id": 999},
            create_response_instance_id=1234,
        )


def test_verify_created_instance_accepts_real_shaped_row() -> None:
    manifest = replace(adoption_manifest(), required_geolocation=None)
    label = "quant-experiments-adopt_seed0"
    row = _real_shaped_vast_row(label=label)
    with patch("vast.ops.run_vast", return_value=[row]):
        live = vast_ops.verify_created_instance(
            1234,
            manifest,
            expected_label=label,
            expected_offer={"id": 11, "machine_id": 999},
        )
    assert live["id"] == 1234


def _purchase_record(**updates: object) -> dict[str, object]:
    record: dict[str, object] = {
        "run_id": "adopt_seed0",
        "expected_label": "quant-experiments-adopt_seed0",
        "offer_id": 11,
        "machine_id": 999,
        "create_response_instance_id": 1234,
    }
    record.update(updates)
    return record


def test_purchase_reconciliation_binds_real_shaped_row_by_create_response_id() -> None:
    row = _real_shaped_vast_row(label="quant-experiments-adopt_seed0")
    assert vast_ops._is_reconcilable_purchase_instance(_purchase_record(), row)
    # Without a persisted create-response ID a label alone must not authorize destroy.
    assert not vast_ops._is_reconcilable_purchase_instance(
        _purchase_record(create_response_instance_id=None), row
    )
    # A different instance carrying the same label is not ours.
    assert not vast_ops._is_reconcilable_purchase_instance(
        _purchase_record(), {**row, "id": 4321}
    )


def _offer(i: int, dph: float, reliability: float | None = None, **extra: object) -> dict[str, object]:
    o: dict[str, object] = {
        "id": i, "machine_id": 100 + i, "dph_total": dph,
        "cpu_cores_effective": 8, "cpu_ram": 32 * 1024, "cpu_name": "AMD Ryzen 7 7700X",
    }
    if reliability is not None:
        o["reliability2"] = reliability
    o.update(extra)
    return o


def test_pick_offer_prefers_reliable_host_slightly_above_floor() -> None:
    offers = [
        _offer(1, 0.30, 0.90),   # floor price, flaky host
        _offer(2, 0.33, 0.995),  # +10%, reliable -> chosen
        _offer(3, 0.35, 0.99),   # +16.7%, a bit less reliable
        _offer(4, 0.40, 0.999),  # +33%, outside the 20% band
    ]
    assert pick_offer(offers)["id"] == 2


def test_pick_offer_band_zero_is_cheapest_first() -> None:
    offers = [_offer(1, 0.30, 0.90), _offer(2, 0.33, 0.995)]
    assert pick_offer(offers, price_band=0.0)["id"] == 1


def test_pick_offer_price_breaks_reliability_ties_and_missing_reliability_ranks_last() -> None:
    offers = [_offer(1, 0.31, 0.99), _offer(2, 0.30, 0.99), _offer(3, 0.30)]
    assert pick_offer(offers)["id"] == 2
