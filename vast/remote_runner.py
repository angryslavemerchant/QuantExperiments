"""Deterministic runner used inside a rented Vast.ai instance.

The account API key never reaches this process. It runs only the commands in the
uploaded manifest, writes append-only JSONL events, enforces the manifest TTL,
and packages declared artifacts for local verification.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import hashlib
import json
import locale
import math
import os
import platform
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows hosts.
    fcntl = None


DEFAULT_MAX_ARTIFACT_ARCHIVE_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_ARTIFACT_UNPACKED_BYTES = 256 * 1024 * 1024
MAX_ARTIFACT_FILE_COUNT = 10_000


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EventWriter:
    def __init__(self, path: Path, run_id: str) -> None:
        self.path = path
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **fields: Any) -> None:
        record = {
            "timestamp": utc_now(),
            "run_id": self.run_id,
            "event": event,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"manifest contains non-finite JSON number: {value}")


def _manifest_number(value: Any, field: str, *, integer: bool = False) -> int | float:
    """Validate a manifest number before any safety calculation consumes it."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        kind = "integer" if integer else "number"
        raise ValueError(f"{field} must be a finite {kind}")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field} must be a finite number")
    if integer and not isinstance(value, int):
        raise ValueError(f"{field} must be a finite integer")
    return int(value) if integer else converted


def _validate_command(command: Any, field: str) -> None:
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(item, str) and item for item in command)
        or command[0] not in {"python", "python3"}
    ):
        raise ValueError(f"{field} must be a non-empty Python argument array")


def read_manifest(path: Path) -> dict[str, Any]:
    raw = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite_json_constant,
    )
    if not isinstance(raw, dict):
        raise ValueError("manifest root must be an object")
    required = {
        "run_id",
        "source_commit",
        "command",
        "artifact_paths",
        "ttl_hours",
        "max_dph",
        "max_total_usd",
        "poll_seconds",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"manifest is missing: {', '.join(missing)}")
    if not isinstance(raw.get("run_id"), str) or not raw["run_id"]:
        raise ValueError("run_id must be a non-empty string")
    if not isinstance(raw.get("source_commit"), str) or not raw["source_commit"]:
        raise ValueError("source_commit must be a non-empty string")
    _validate_command(raw["command"], "command")
    post_commands = raw.get("post_commands", [])
    if not isinstance(post_commands, list):
        raise ValueError("post_commands must be a list of Python argument arrays")
    for index, command in enumerate(post_commands):
        _validate_command(command, f"post_commands[{index}]")

    # Keep this schema explicit.  These values either determine spend/TTL or are
    # later converted to ints/floats for identity, hardware, and verification
    # checks.  Rejecting booleans and non-finite values prevents Python's permissive
    # numeric conversions from changing the safety contract.
    if "format_version" in raw:
        _manifest_number(raw["format_version"], "format_version", integer=True)
    ttl_hours = float(_manifest_number(raw["ttl_hours"], "ttl_hours"))
    if ttl_hours <= 0 or ttl_hours > 168:
        raise ValueError("ttl_hours is outside the remote safety range")
    max_dph = float(_manifest_number(raw["max_dph"], "max_dph"))
    max_total_usd = float(_manifest_number(raw["max_total_usd"], "max_total_usd"))
    if max_dph <= 0 or max_total_usd <= 0:
        raise ValueError("remote price and total caps must be positive")
    _manifest_number(raw["poll_seconds"], "poll_seconds", integer=True)
    for field in (
        "num_gpus",
        "failure_grace_minutes",
        "disk_gb",
        "min_inet_down_mbps",
        "min_cpu_cores_effective",
        "min_cpu_ram_gb",
        "max_artifact_archive_bytes",
        "max_artifact_unpacked_bytes",
    ):
        if field in raw:
            _manifest_number(raw[field], field, integer=True)
    for field in ("min_cuda",):
        if field in raw:
            _manifest_number(raw[field], field)
    max_archive = int(
        raw.get(
            "max_artifact_archive_bytes",
            DEFAULT_MAX_ARTIFACT_ARCHIVE_BYTES,
        )
    )
    max_unpacked = int(
        raw.get(
            "max_artifact_unpacked_bytes",
            DEFAULT_MAX_ARTIFACT_UNPACKED_BYTES,
        )
    )
    if not 1 <= max_archive <= 4 * 1024**3:
        raise ValueError("max_artifact_archive_bytes is outside the safety range")
    if not max_archive <= max_unpacked <= 16 * 1024**3:
        raise ValueError("max_artifact_unpacked_bytes is outside the safety range")
    avoid_machine_ids = raw.get("avoid_machine_ids", [])
    if not isinstance(avoid_machine_ids, list):
        raise ValueError("avoid_machine_ids must be a list of finite integers")
    for index, value in enumerate(avoid_machine_ids):
        _manifest_number(value, f"avoid_machine_ids[{index}]", integer=True)
    if "require_cuda" in raw and not isinstance(raw["require_cuda"], bool):
        raise ValueError("require_cuda must be a boolean")
    if not isinstance(raw["artifact_paths"], list) or not raw["artifact_paths"]:
        raise ValueError("artifact_paths must be a non-empty list")
    if raw.get("success_policy", "destroy_after_verify") not in {
        "destroy_after_verify",
        "keep_alive_after_verify",
    }:
        raise ValueError("invalid success_policy")
    for value in raw["artifact_paths"]:
        if not isinstance(value, str) or not value:
            raise ValueError(f"unsafe artifact path: {value!r}")
        relative = PurePosixPath(value.replace("\\", "/"))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or relative.parts[0] != "artifacts"
        ):
            raise ValueError(f"unsafe artifact path: {value!r}")
    return raw


def verify_source_commit(repo: Path, expected: str) -> None:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    actual = process.stdout.strip()
    if process.returncode != 0 or actual != expected:
        raise ValueError(
            f"remote source commit mismatch: expected {expected}, found {actual or 'unknown'}"
        )


def verify_input_artifacts(repo: Path, manifest: dict[str, Any]) -> None:
    artifacts = manifest.get("input_artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("input_artifacts must map paths to SHA-256 digests")
    if artifacts:
        raise ValueError(
            "QuantExperiments remote input artifacts are disabled; "
            "jobs download their own data at runtime"
        )
    resolved_repo = repo.resolve()
    for value, expected in artifacts.items():
        relative = PurePosixPath(str(value).replace("\\", "/"))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or len(relative.parts) < 3
            or relative.parts[:2] != ("artifacts", "remote")
        ):
            raise ValueError(f"unsafe input artifact path: {value!r}")
        local = repo.joinpath(*relative.parts).resolve()
        if resolved_repo not in local.parents:
            raise ValueError(f"unsafe input artifact target: {value!r}")
        if not local.is_file() or sha256_file(local) != expected:
            raise ValueError(f"input artifact verification failed: {value}")


def runtime_details(require_cuda: bool, num_gpus: int = 1) -> dict[str, Any]:
    details: dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_full_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "locale_encoding": locale.getpreferredencoding(False),
        "filesystem_encoding": sys.getfilesystemencoding(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "cuda_required": require_cuda,
    }
    try:
        import numpy
    except ImportError:
        details["numpy"] = None
    else:
        details["numpy"] = numpy.__version__
    if not require_cuda:
        return details
    import torch

    details["torch"] = torch.__version__
    details["cuda_available"] = torch.cuda.is_available()
    if not details["cuda_available"]:
        raise RuntimeError("manifest requires CUDA, but torch.cuda.is_available() is false")
    details["cuda_device_count"] = torch.cuda.device_count()
    if details["cuda_device_count"] < num_gpus:
        raise RuntimeError(
            f"manifest requires {num_gpus} CUDA devices, but Torch sees "
            f"{details['cuda_device_count']}"
        )
    details["cuda_device"] = torch.cuda.get_device_name(0)
    return details


def terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)


def run_command(
    command: list[str],
    *,
    step: int,
    repo: Path,
    log_path: Path,
    deadline: float,
    events: EventWriter,
) -> int:
    resolved = [sys.executable, *command[1:]]
    events.write("command_started", step=step, argv=command)
    environment = os.environ.copy()
    # Vast's host watchdog may need its instance-scoped credential, but the
    # research workload never does. Do not hand provider control-plane secrets
    # to arbitrary experiment code or its logs/artifacts.
    for key in ("CONTAINER_API_KEY", "VAST_API_KEY", "VASTAI_API_KEY"):
        environment.pop(key, None)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    # Make the checked-out repository importable without packaging it.
    existing_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{repo}{os.pathsep}{existing_path}" if existing_path else str(repo)
    )
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            resolved,
            cwd=repo,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        next_heartbeat = time.monotonic()
        while process.poll() is None:
            now = time.monotonic()
            if now >= deadline:
                events.write("ttl_reached", step=step, pid=process.pid)
                terminate_process(process)
                return 124
            if now >= next_heartbeat:
                events.write(
                    "heartbeat",
                    step=step,
                    pid=process.pid,
                    log_bytes=log_path.stat().st_size if log_path.exists() else 0,
                )
                next_heartbeat = now + 300
            time.sleep(min(1, max(0.1, deadline - now)))
    return_code = int(process.returncode or 0)
    events.write("command_finished", step=step, exit_code=return_code)
    return return_code


def collect_artifact_files(repo: Path, artifact_paths: list[str]) -> tuple[list[Path], list[str]]:
    files: set[Path] = set()
    missing: list[str] = []
    resolved_repo = repo.resolve()
    for raw_path in artifact_paths:
        relative = PurePosixPath(raw_path.replace("\\", "/"))
        target = repo.joinpath(*relative.parts)
        resolved_target = target.resolve()
        if resolved_repo not in resolved_target.parents:
            raise ValueError(f"artifact escapes repository: {raw_path!r}")
        if not target.exists():
            missing.append(raw_path)
            continue
        candidates = [target] if target.is_file() else target.rglob("*")
        for candidate in candidates:
            if candidate.is_symlink():
                raise ValueError(f"artifact contains a symbolic link: {candidate}")
            if candidate.is_file():
                files.add(candidate)
    return sorted(files), missing


@contextlib.contextmanager
def _artifact_packaging_lock(ops_dir: Path):
    """Serialize publication with an OS lock that is released on process exit.

    The lock file is intentionally retained.  Its inode is stable across a
    crash, while the advisory lock itself is released by the operating system;
    deleting a lock path during cleanup would reintroduce a replacement race.
    """

    ops_dir.mkdir(parents=True, exist_ok=True)
    lock_path = ops_dir / ".artifact-packaging.lock"
    deadline = time.monotonic() + 120
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    acquired = False
    try:
        # msvcrt.locking requires a byte to exist.  This also makes the lock
        # file self-describing when an interrupted old runner left it empty.
        if fcntl is None and os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        while not acquired:
            if fcntl is not None:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as error:
                    if error.errno not in {errno.EACCES, errno.EAGAIN}:
                        raise
                else:
                    acquired = True
            else:  # pragma: no cover - exercised on Windows hosts.
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                try:
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                except OSError:
                    pass
                else:
                    acquired = True
            if acquired:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("timed out acquiring artifact packaging lock")
            time.sleep(0.05)
        yield
    finally:
        try:
            if acquired and fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            elif acquired:
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        finally:
            os.close(descriptor)


def package_artifacts(
    manifest: dict[str, Any],
    *,
    repo: Path,
    ops_dir: Path,
    exit_code: int,
    events: EventWriter,
) -> None:
    with _artifact_packaging_lock(ops_dir):
        _package_artifacts_unlocked(
            manifest,
            repo=repo,
            ops_dir=ops_dir,
            exit_code=exit_code,
            events=events,
        )


def _package_artifacts_unlocked(
    manifest: dict[str, Any],
    *,
    repo: Path,
    ops_dir: Path,
    exit_code: int,
    events: EventWriter,
) -> None:
    ops_dir.mkdir(parents=True, exist_ok=True)
    files, missing = collect_artifact_files(repo, manifest["artifact_paths"])
    entries = [
        {
            "path": path.relative_to(repo).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    if len(entries) > MAX_ARTIFACT_FILE_COUNT:
        raise ValueError("artifact file count exceeds the manifest limit")
    max_unpacked = int(
        manifest.get(
            "max_artifact_unpacked_bytes",
            DEFAULT_MAX_ARTIFACT_UNPACKED_BYTES,
        )
    )
    if sum(int(entry["size"]) for entry in entries) > max_unpacked:
        raise ValueError("artifact bytes exceed the unpacked manifest limit")
    archive = ops_dir / "artifacts.tar.gz"
    temporary_archive_handle, temporary_archive_name = tempfile.mkstemp(
        prefix=".artifacts-", suffix=".tar.gz.tmp", dir=ops_dir
    )
    os.close(temporary_archive_handle)
    temporary_archive = Path(temporary_archive_name)
    try:
        # The local verifier deliberately accepts only bounded POSIX ustar.
        # Python's default PAX writer can emit variable-sized metadata records
        # before an ordinary file body, so keep the producer/consumer format
        # contract explicit here.
        with tarfile.open(
            temporary_archive, "w:gz", format=tarfile.USTAR_FORMAT
        ) as tar:
            for path in files:
                tar.add(path, arcname=path.relative_to(repo).as_posix(), recursive=False)
        max_archive = int(
            manifest.get(
                "max_artifact_archive_bytes",
                DEFAULT_MAX_ARTIFACT_ARCHIVE_BYTES,
            )
        )
        if temporary_archive.stat().st_size > max_archive:
            raise ValueError("artifact archive exceeds the compressed manifest limit")
        temporary_archive.replace(archive)
    finally:
        temporary_archive.unlink(missing_ok=True)
    artifact_manifest = {
        "format_version": 1,
        "created_at": utc_now(),
        "run_id": manifest["run_id"],
        "source_commit": manifest["source_commit"],
        "exit_code": exit_code,
        "missing_declared_paths": missing,
        "files": entries,
        "archive_sha256": sha256_file(archive),
    }
    temporary_manifest_handle, temporary_manifest_name = tempfile.mkstemp(
        prefix=".artifact-manifest-", suffix=".json.tmp", dir=ops_dir
    )
    os.close(temporary_manifest_handle)
    temporary_manifest = Path(temporary_manifest_name)
    try:
        temporary_manifest.write_text(
            json.dumps(artifact_manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary_manifest.replace(ops_dir / "artifact_manifest.json")
    finally:
        temporary_manifest.unlink(missing_ok=True)
    events.write(
        "artifacts_ready",
        exit_code=exit_code,
        file_count=len(entries),
        missing_declared_paths=missing,
    )


def execute(manifest_path: Path, ops_dir: Path, repo: Path) -> int:
    manifest = read_manifest(manifest_path)
    # Child workloads and the startup provenance record use one explicit hash
    # seed.  (The interpreter's own hash table is already initialized, so this
    # is an audit/provenance setting as well as a child-process setting.)
    os.environ["PYTHONHASHSEED"] = "0"
    events = EventWriter(ops_dir / "events.jsonl", str(manifest["run_id"]))
    log_path = ops_dir / "train.log"
    verify_source_commit(repo, str(manifest["source_commit"]))
    verify_input_artifacts(repo, manifest)
    environment = runtime_details(
        bool(manifest.get("require_cuda", True)),
        int(manifest.get("num_gpus", 1)),
    )
    reserve_hours = max(0.1, (2 * int(manifest["poll_seconds"])) / 3600)
    effective_ttl_hours = max(
        0.0,
        min(
            float(manifest["ttl_hours"]),
            float(manifest["max_total_usd"]) / float(manifest["max_dph"]),
        )
        - reserve_hours,
    )
    if effective_ttl_hours <= 0:
        raise ValueError("no workload time remains after the cleanup reserve")
    deadline = time.monotonic() + effective_ttl_hours * 3600
    stop_epoch = manifest.get("workload_stop_at_epoch")
    if stop_epoch is not None:
        # The local controller's campaign schedule is anchored at purchase
        # time; provisioning already consumed part of it.
        remaining = float(stop_epoch) - time.time()
        if remaining <= 0:
            raise ValueError("campaign workload window already elapsed")
        deadline = min(deadline, time.monotonic() + remaining)
    commands = [manifest["command"], *manifest.get("post_commands", [])]
    events.write(
        "run_started",
        source_commit=manifest["source_commit"],
        command_count=len(commands),
        ttl_hours=float(manifest["ttl_hours"]),
        effective_workload_ttl_hours=effective_ttl_hours,
        environment=environment,
    )
    exit_code = 0
    try:
        for step, command in enumerate(commands):
            exit_code = run_command(
                command,
                step=step,
                repo=repo,
                log_path=log_path,
                deadline=deadline,
                events=events,
            )
            if exit_code != 0:
                break
        package_artifacts(
            manifest,
            repo=repo,
            ops_dir=ops_dir,
            exit_code=exit_code,
            events=events,
        )
    except Exception as error:
        exit_code = exit_code or 1
        events.write("runner_exception", error=f"{type(error).__name__}: {error}")
        try:
            package_artifacts(
                manifest,
                repo=repo,
                ops_dir=ops_dir,
                exit_code=exit_code,
                events=events,
            )
        except Exception as packaging_error:
            events.write(
                "artifact_packaging_failed",
                error=f"{type(packaging_error).__name__}: {packaging_error}",
            )
    if exit_code == 0:
        events.write("run_succeeded")
    else:
        events.write("run_failed", exit_code=exit_code)
    return exit_code


def snapshot(manifest_path: Path, ops_dir: Path, repo: Path) -> int:
    """Package the current declared research-session artifacts without rerunning."""

    manifest = read_manifest(manifest_path)
    events = EventWriter(ops_dir / "events.jsonl", str(manifest["run_id"]))
    verify_source_commit(repo, str(manifest["source_commit"]))
    verify_input_artifacts(repo, manifest)
    events.write("session_snapshot_started")
    package_artifacts(
        manifest,
        repo=repo,
        ops_dir=ops_dir,
        exit_code=0,
        events=events,
    )
    events.write("session_snapshot_ready")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ops-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--snapshot-only", action="store_true")
    args = parser.parse_args()
    try:
        status = (
            snapshot(args.manifest, args.ops_dir, args.repo)
            if args.snapshot_only
            else execute(args.manifest, args.ops_dir, args.repo)
        )
    except Exception as error:
        # Bootstrap also records pre-run failures. This fallback ensures an invalid
        # manifest is visible without assuming a usable run ID.
        args.ops_dir.mkdir(parents=True, exist_ok=True)
        with (args.ops_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "timestamp": utc_now(),
                        "run_id": "unknown",
                        "event": "bootstrap_failed",
                        "error": f"{type(error).__name__}: {error}",
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        status = 2
    raise SystemExit(status)


if __name__ == "__main__":
    main()
