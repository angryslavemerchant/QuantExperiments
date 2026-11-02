"""Guarded local orchestrator for QuantExperiments runs on Vast.ai.

This is the supported lifecycle entry point. It keeps the account API key local,
uploads an exact clean Git commit as a bundle, monitors machine-readable remote
events, verifies pulled artifacts, and enforces approved time and cost limits.

The limits are conservative client-side guardrails, not provider-enforced
guarantees.  Every unresolved control-plane outcome remains explicit and enters
cleanup-only reconciliation; it is never reported as cap compliance.

No command rents hardware except ``launch``. Candidate racing and adoption are
fail-closed in this release pending their separate lifecycle audits.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import errno
import gzip
import hashlib
import io
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import unicodedata
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
VAST_DIR = ROOT / "vast"
RUNTIME_DIR = ROOT / ".vast"
STATE_FILE = RUNTIME_DIR / "instances.json"
STATE_LOCK_FILE = RUNTIME_DIR / "instances.state.lock"
STAGE_STATE_FILE = RUNTIME_DIR / "stages.json"
EVENT_FILE = RUNTIME_DIR / "events.jsonl"
KNOWN_HOSTS = RUNTIME_DIR / "known_hosts"
# Stop the remote workload this long before the local cleanup boundary so it
# can write its final checkpoint and artifact archive before recovery begins.
REMOTE_PACKAGING_MARGIN_SECONDS = 300
SECRETS_FILE = VAST_DIR / "secrets.env"
STAGING_DIR = RUNTIME_DIR / "staging"
MONITOR_DIR = RUNTIME_DIR / "monitors"
REMOTE_ROOT = "/workspace"
REMOTE_REPO = f"{REMOTE_ROOT}/quant-experiments"
REMOTE_OPS = f"{REMOTE_ROOT}/quant_experiments_ops"
REMOTE_STAGES = f"{REMOTE_ROOT}/quant_experiments_stages"
# Official "PyTorch (Vast)" template recorded in the legacy launcher. Vast's
# automatic tag selector currently asks RTX 5090 hosts for a nonexistent CUDA
# 13.0 build, so reproduce the template with its compatible CUDA 12.8 image
# pinned while retaining the template's ports and environment.
OFFICIAL_PYTORCH_TEMPLATE_HASH = "2ad6d615db5927a06fef0c9cd51d77c4"
DEFAULT_IMAGE = "vastai/pytorch:cuda-12.8.1-auto"
DEFAULT_TEMPLATE_ENV = (
    "-p 1111:1111 -p 6006:6006 -p 8080:8080 -p 8384:8384 -p 10100:10100 "
    "-p 10200:10200 -p 72299:72299 "
    '-e OPEN_BUTTON_PORT="1111" -e OPEN_BUTTON_TOKEN="1" -e JUPYTER_DIR="/" '
    '-e DATA_DIRECTORY="/workspace/" '
    '-e PORTAL_CONFIG="localhost:1111:11111:/:Instance Portal|'
    "localhost:8080:18080:/:Jupyter|"
    "localhost:8080:8080:/terminals/1:Jupyter Terminal|"
    "localhost:8384:18384:/:Syncthing|"
    'localhost:6006:16006:/:Tensorboard"'
)
MACHINE_BLACKLIST_FILE = VAST_DIR / "blacklist.json"

# Modest host defaults.  Jobs download their own market data at runtime, so
# data-heavy manifests should raise CPU, RAM, disk, and min_inet_down_mbps
# explicitly (e.g. 8 cores, 48 GB, 500 Mbps).
DEFAULT_MIN_CPU_CORES = 4
DEFAULT_MIN_CPU_RAM_GB = 16
DEFAULT_SSH_READY_TIMEOUT_SECONDS = 180
# Ordinary launches wait longer: the vastai/pytorch image pull plus container
# boot routinely exceeds 3 minutes (run s01_qwen_seed0_20261101b, 2026-11-02,
# failed with sshd still refusing connections at 180 s). Still bounded by the
# run's cleanup deadline.
LAUNCH_SSH_READY_TIMEOUT_SECONDS = 600
# Offer selection considers offers up to this fraction above the cheapest
# eligible price and picks the most reliable among them (see pick_offer).
OFFER_PRICE_BAND = 0.20
# Hard per-GPU hourly ceilings applied before the band (user preference,
# 2026-11-02). Keys are normalized GPU names (lowercase alphanumerics).
OFFER_MAX_DPH_BY_GPU = {"rtx4090": 0.50, "rtx5090": 0.70}
_MAX_PROVIDER_JSON_BYTES = 4 * 1024 * 1024
_MAX_PROVIDER_JSON_DEPTH = 24
# Provider market snapshots legitimately contain many scalar fields. Bound
# decoded structure in proportion to the already-enforced byte ceiling instead
# of reusing the much smaller free-form diagnostic budget.
_MAX_PROVIDER_JSON_NODES = _MAX_PROVIDER_JSON_BYTES // 4
_MAX_REMOTE_PREFLIGHT_BYTES = 256 * 1024
PURCHASE_RECONCILIATION_TIMEOUT_SECONDS = 180
PURCHASE_RECONCILIATION_POLL_SECONDS = 10
PURCHASE_RECONCILIATION_MIN_POLL_SECONDS = 1
# Vast's CLI does not provide a provider-side destroy deadline.  Keep the
# teardown budget explicit and conservative: a destroy command may consume its
# full subprocess timeout on each retry, followed by an equally expensive
# live-API absence check.  The follow-up scan allowance is intentionally part of
# the campaign reserve because a successful API acknowledgement is not proof
# that a delayed contract has disappeared.
DESTROY_API_TIMEOUT_SECONDS = 180
DESTROY_MAX_ATTEMPTS = 3
DESTROY_RETRY_DELAY_SECONDS = 5
DESTROY_VERIFY_MAX_ATTEMPTS = 3
DESTROY_VERIFY_RETRY_DELAY_SECONDS = 5
DESTROY_FOLLOWUP_ABSENT_SCANS = 2
# Don't start a follow-up provider query with less than this much lease left:
# a query bounded to a few seconds times out, records a spurious error, and the
# lease then expires 'unverified' despite many clean absence scans (runs
# s01_qwen_seed0_20261101a and ...c, 2026-11-02).
DESTROY_FOLLOWUP_MIN_QUERY_SECONDS = 30
DESTROY_RECONCILIATION_MIN_SECONDS = 60
DESTROY_RECONCILIATION_POLL_SECONDS = 10
# An uncertain create gets one provider query after its primary lease.  This
# query is charged to the predeclared cleanup reserve, so it cannot begin at the
# mathematical hard boundary with a fresh 180-second subprocess timeout.
PURCHASE_FINAL_SCAN_RESERVE_SECONDS = DESTROY_API_TIMEOUT_SECONDS
# Vast can acknowledge a create only after its ordinary reconciliation window.
# The same already-detached supervisor therefore remains as a sparse cleanup-
# only sentinel.  The deadline is anchored to the original primary deadline and
# cannot be renewed by restarting the process.
LATE_CREATE_RECONCILIATION_SECONDS = 24 * 3600
# These broader lifecycle modes require separate campaign-specific proofs.
# The current release is intentionally one new instance with no race/adoption.
ENABLE_CANDIDATE_RACE = False
ENABLE_EXISTING_INSTANCE_ADOPTION = False
LATE_CREATE_RECONCILIATION_POLL_SECONDS = 300
SUPERVISOR_LAUNCH_MAX_ATTEMPTS = 3
SUPERVISOR_LAUNCH_RETRY_DELAY_SECONDS = 1
SUPERVISOR_HANDSHAKE_TIMEOUT_SECONDS = 10
SUPERVISOR_HANDSHAKE_POLL_SECONDS = 0.05
# Hard-limit recovery is strictly best effort.  It gets its own bounded slice
# before the separately-accounted destruction window; long SSH/SCP defaults
# must never consume the teardown reserve.
HARD_LIMIT_RECOVERY_RESERVE_SECONDS = 300
LOCAL_ARTIFACT_VERIFICATION_RESERVE_SECONDS = 30
DEFAULT_MAX_ARTIFACT_ARCHIVE_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_ARTIFACT_UNPACKED_BYTES = 256 * 1024 * 1024
MAX_ARTIFACT_MANIFEST_BYTES = 1024 * 1024
MAX_ARTIFACT_FILE_COUNT = 10_000
# The provider-injected container credential is a last-resort kill switch.  It
# is deliberately armed a little before the campaign's mathematical hard
# boundary so clock/network jitter cannot turn the watchdog into an
# after-the-fact cleanup mechanism.
REMOTE_WATCHDOG_SAFETY_SECONDS = 120
LOCAL_WATCHDOG_ORDERING_MARGIN_SECONDS = 10
REMOTE_WATCHDOG_SLEEP_CHUNK_SECONDS = 15
REMOTE_WATCHDOG_REQUEST_TIMEOUT_SECONDS = 15
SSH_PRIVATE_KEY = Path.home() / ".ssh" / "id_ed25519"
SSH_PUBLIC_KEY = SSH_PRIVATE_KEY.with_suffix(".pub")

RUN_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{2,79}$")
INSTANCE_LABEL_PATTERN = re.compile(
    r"^quant-experiments-[A-Za-z0-9][A-Za-z0-9_.-]{2,99}$", re.ASCII
)
SSH_URL_PATTERN = re.compile(r"^ssh://([^@]+)@([^:]+):([0-9]{1,5})$", re.ASCII)
SSH_USER_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$", re.ASCII)
SSH_HOST_PATTERN = re.compile(r"^[A-Za-z0-9.-]{1,253}$", re.ASCII)


class OpsError(RuntimeError):
    """A guarded lifecycle operation could not proceed safely."""


def _manifest_number(value: object, field: str, *, integer: bool = False) -> float:
    """Validate a manifest number without accepting bools or non-finite values."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OpsError(f"{field} must be a finite number")
    if integer and not isinstance(value, int):
        raise OpsError(f"{field} must be an integer")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise OpsError(f"{field} must be finite")
    return numeric


def hidden_subprocess_options() -> dict[str, int]:
    """Keep local command-line helpers from opening console windows on Windows."""

    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


def _popen_detached(
    command: list[str], popen_options: dict[str, Any]
) -> subprocess.Popen[Any]:
    """Start a supervisor that outlives the launching terminal/agent session.

    On Windows, IDE terminals and agent harnesses often run children inside a
    kill-on-close job object; DETACHED_PROCESS alone does not escape it, so
    closing the session would silently kill the monitor.  Request breakaway
    first and fall back when the job forbids it (ERROR_ACCESS_DENIED).
    """

    if os.name == "nt" and "creationflags" in popen_options:
        breakaway = dict(popen_options)
        breakaway["creationflags"] = (
            int(popen_options["creationflags"])
            | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000)
        )
        try:
            return subprocess.Popen(command, **breakaway)
        except OSError:
            pass
    return subprocess.Popen(command, **popen_options)


def windowless_python_executable() -> str:
    """Prefer the console-free interpreter for detached Windows monitors."""

    if os.name != "nt":
        return sys.executable
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    return str(pythonw) if pythonw.is_file() else sys.executable


@dataclass(frozen=True)
class RunManifest:
    format_version: int
    run_id: str
    description: str
    source_commit: str
    command: list[str]
    artifact_paths: list[str]
    gpu_name: str
    max_dph: float
    max_total_usd: float
    ttl_hours: float
    failure_grace_minutes: int
    disk_gb: int
    poll_seconds: int
    seed_policy: str
    checkpoint_cadence: str
    success_policy: str = "destroy_after_verify"
    data_policy: str = ""
    interaction_budget: str = ""
    intervention_suite: str = ""
    stall_minutes: int = 45
    num_gpus: int = 1
    post_commands: list[list[str]] = field(default_factory=list)
    require_cuda: bool = True
    min_cuda: float = 12.8
    min_inet_down_mbps: int = 100
    min_cpu_cores_effective: int = DEFAULT_MIN_CPU_CORES
    min_cpu_ram_gb: int = DEFAULT_MIN_CPU_RAM_GB
    prefer_consumer_cpu: bool = True
    require_consumer_cpu: bool = False
    required_geolocation: str | None = None
    required_country: str | None = None
    avoid_machine_ids: list[int] = field(default_factory=list)
    input_artifacts: dict[str, str] = field(default_factory=dict)
    max_artifact_archive_bytes: int = DEFAULT_MAX_ARTIFACT_ARCHIVE_BYTES
    max_artifact_unpacked_bytes: int = DEFAULT_MAX_ARTIFACT_UNPACKED_BYTES

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RunManifest:
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise OpsError(f"unknown manifest fields: {', '.join(unknown)}")
        try:
            manifest = cls(**raw)
        except TypeError as error:
            raise OpsError(
                "invalid manifest fields: " + _redacted_exception_text(error)
            ) from error
        manifest.validate()
        return manifest

    def validate(self) -> None:
        _manifest_number(self.format_version, "format_version", integer=True)
        _manifest_number(self.num_gpus, "num_gpus", integer=True)
        _manifest_number(self.max_dph, "max_dph")
        _manifest_number(self.max_total_usd, "max_total_usd")
        _manifest_number(self.ttl_hours, "ttl_hours")
        _manifest_number(self.failure_grace_minutes, "failure_grace_minutes", integer=True)
        _manifest_number(self.disk_gb, "disk_gb", integer=True)
        _manifest_number(self.poll_seconds, "poll_seconds", integer=True)
        _manifest_number(self.min_cuda, "min_cuda")
        _manifest_number(self.min_inet_down_mbps, "min_inet_down_mbps", integer=True)
        _manifest_number(
            self.min_cpu_cores_effective,
            "min_cpu_cores_effective",
            integer=True,
        )
        _manifest_number(self.min_cpu_ram_gb, "min_cpu_ram_gb", integer=True)
        _manifest_number(
            self.max_artifact_archive_bytes,
            "max_artifact_archive_bytes",
            integer=True,
        )
        _manifest_number(
            self.max_artifact_unpacked_bytes,
            "max_artifact_unpacked_bytes",
            integer=True,
        )
        if not isinstance(self.avoid_machine_ids, list):
            raise OpsError("avoid_machine_ids must be a list of positive integers")
        for index, value in enumerate(self.avoid_machine_ids):
            _manifest_number(value, f"avoid_machine_ids[{index}]", integer=True)
        if self.format_version != 1:
            raise OpsError("manifest format_version must be 1")
        if not RUN_ID_PATTERN.fullmatch(self.run_id):
            raise OpsError("run_id must be 3-80 safe filename characters")
        if not self.description.strip():
            raise OpsError("description must be non-empty")
        if not re.fullmatch(r"[0-9a-f]{40}", self.source_commit):
            raise OpsError("source_commit must be an exact 40-character Git SHA")
        if not self.command or not all(
            isinstance(item, str) and item for item in self.command
        ):
            raise OpsError("command must be a non-empty string array")
        if self.command[0] not in {"python", "python3"}:
            raise OpsError("command must begin with python or python3")
        for index, command in enumerate(self.post_commands):
            if (
                not isinstance(command, list)
                or not command
                or not all(isinstance(item, str) and item for item in command)
            ):
                raise OpsError(
                    f"post_commands[{index}] must be a non-empty string array"
                )
            if command[0] not in {"python", "python3"}:
                raise OpsError(
                    f"post_commands[{index}] must begin with python or python3"
                )
        if not self.artifact_paths:
            raise OpsError("artifact_paths must be non-empty")
        for value in self.artifact_paths:
            path = PurePosixPath(value.replace("\\", "/"))
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise OpsError(f"unsafe artifact path: {value!r}")
            if path.parts[0] != "artifacts":
                raise OpsError("artifact paths must stay under artifacts/")
        if not isinstance(self.input_artifacts, dict):
            raise OpsError("input_artifacts must map paths to SHA-256 digests")
        if self.input_artifacts:
            raise OpsError(
                "QuantExperiments remote input artifacts are disabled; "
                "jobs download their own data at runtime"
            )
        for value, digest in self.input_artifacts.items():
            path = PurePosixPath(str(value).replace("\\", "/"))
            if (
                path.is_absolute()
                or ".." in path.parts
                or len(path.parts) < 3
                or path.parts[:2] != ("artifacts", "remote")
            ):
                raise OpsError(
                    "input artifact paths must stay under artifacts/remote/: "
                    f"{value!r}"
                )
            if not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
                raise OpsError(f"invalid input artifact SHA-256 for {value!r}")
        if not self.gpu_name.strip():
            raise OpsError("gpu_name must be non-empty")
        if not isinstance(self.num_gpus, int) or not 1 <= self.num_gpus <= 8:
            raise OpsError("num_gpus must be an integer in [1, 8]")
        if not isinstance(self.require_cuda, bool):
            raise OpsError("require_cuda must be a boolean")
        if self.max_dph <= 0 or self.max_total_usd <= 0 or self.ttl_hours <= 0:
            raise OpsError("price, total-spend, and TTL limits must be positive")
        if self.max_dph > 10 or self.max_total_usd > 100 or self.ttl_hours > 168:
            raise OpsError("manifest exceeds the hard operator safety ceiling")
        if self.failure_grace_minutes < 0 or self.failure_grace_minutes > 240:
            raise OpsError("failure_grace_minutes must be in [0, 240]")
        if self.disk_gb < 10 or self.disk_gb > 500:
            raise OpsError("disk_gb must be in [10, 500]")
        if self.min_inet_down_mbps < 1 or self.min_inet_down_mbps > 100_000:
            raise OpsError("min_inet_down_mbps must be in [1, 100000]")
        if self.min_cpu_cores_effective < 1 or self.min_cpu_cores_effective > 256:
            raise OpsError("min_cpu_cores_effective must be in [1, 256]")
        if self.min_cpu_ram_gb < 1 or self.min_cpu_ram_gb > 4096:
            raise OpsError("min_cpu_ram_gb must be in [1, 4096]")
        if not 1 <= self.max_artifact_archive_bytes <= 4 * 1024**3:
            raise OpsError("max_artifact_archive_bytes must be in [1, 4294967296]")
        if not self.max_artifact_archive_bytes <= self.max_artifact_unpacked_bytes <= 16 * 1024**3:
            raise OpsError(
                "max_artifact_unpacked_bytes must be at least the archive limit "
                "and no more than 17179869184"
            )
        if not isinstance(self.prefer_consumer_cpu, bool):
            raise OpsError("prefer_consumer_cpu must be a boolean")
        if not isinstance(self.require_consumer_cpu, bool):
            raise OpsError("require_consumer_cpu must be a boolean")
        for value, field_name in (
            (self.required_geolocation, "required_geolocation"),
            (self.required_country, "required_country"),
        ):
            if value is not None and (
                not isinstance(value, str)
                or not value.strip()
                or len(value) > 100
            ):
                raise OpsError(
                    f"{field_name} must be null or non-empty text up to 100 characters"
                )
        if not isinstance(self.avoid_machine_ids, list) or any(
            not isinstance(value, int) or value <= 0 for value in self.avoid_machine_ids
        ):
            raise OpsError("avoid_machine_ids must be a list of positive integers")
        if len(set(self.avoid_machine_ids)) != len(self.avoid_machine_ids):
            raise OpsError("avoid_machine_ids must not contain duplicates")
        if (
            not isinstance(self.stall_minutes, int)
            or isinstance(self.stall_minutes, bool)
            or not (self.stall_minutes == 0 or 15 <= self.stall_minutes <= 1440)
        ):
            raise OpsError("stall_minutes must be 0 (disabled) or in [15, 1440]")
        if self.poll_seconds < 15 or self.poll_seconds > 600:
            raise OpsError("poll_seconds must be in [15, 600]")
        if self.success_policy not in {
            "destroy_after_verify",
            "keep_alive_after_verify",
        }:
            raise OpsError(
                "success_policy must be destroy_after_verify or "
                "keep_alive_after_verify"
            )
        reserve_cost = self.max_dph * cleanup_reserve_hours(self.poll_seconds)
        if self.max_total_usd <= reserve_cost:
            raise OpsError("max_total_usd is too small for the cleanup reserve")
        if self.ttl_hours <= cleanup_reserve_hours(self.poll_seconds):
            raise OpsError("ttl_hours is too small for the cleanup reserve")
        required_notes = (self.seed_policy, self.checkpoint_cadence)
        if any(not isinstance(note, str) or not note.strip() for note in required_notes):
            raise OpsError("seed_policy and checkpoint_cadence must be non-empty")
        for note in (self.data_policy, self.interaction_budget, self.intervention_suite):
            if not isinstance(note, str):
                raise OpsError("optional policy notes must be strings")


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


_STATE_LOCK_LOCAL = threading.local()
_LIFECYCLE_LOCK_LOCAL = threading.local()
STATE_LOCK_TIMEOUT_SECONDS = 30.0


def _open_lock_stream(path: Path) -> Any:
    """Open a stable read/write lock inode without append semantics.

    ``a+b`` forces every write to EOF on Windows even after ``seek(0)``. A
    losing byte-range-lock contender could therefore append a NUL to the live
    owner's PID metadata before discovering that the lease was held. Opening
    with ``O_RDWR | O_CREAT`` preserves positional writes and the stable inode.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    return os.fdopen(descriptor, "r+b")


def _prepare_windows_lock_byte(stream: Any) -> None:
    """Best-effort initialize byte zero without writing through a live lease."""

    if os.fstat(stream.fileno()).st_size == 0:
        try:
            stream.seek(0)
            stream.write(b"\0")
            stream.flush()
        except OSError as error:
            # A simultaneous owner may already hold byte zero. The subsequent
            # nonblocking lock attempt is the authoritative contention check.
            if not _lock_is_contended(error):
                raise OpsError("lock-byte initialization failed") from error
    stream.seek(0)


def _lock_is_contended(error: OSError) -> bool:
    """Distinguish an ordinary held lease from an unknown lock I/O failure."""

    return isinstance(error, (BlockingIOError, PermissionError)) and error.errno in {
        errno.EACCES,
        errno.EAGAIN,
    }


@contextmanager
def state_file_lock() -> Iterator[None]:
    """Serialize state read-modify-write transitions across processes.

    The lock is a separate, stable inode: ``atomic_write_json`` may replace the
    state file while the lock is held.  A small thread-local reentrancy guard is
    intentional because existing lifecycle helpers call ``get_record`` and
    ``replace_record`` from already-serialized transitions.
    """

    depth = int(getattr(_STATE_LOCK_LOCAL, "depth", 0))
    if depth:
        _STATE_LOCK_LOCAL.depth = depth + 1
        try:
            yield
        finally:
            _STATE_LOCK_LOCAL.depth = depth
        return
    stream = _open_lock_stream(STATE_LOCK_FILE)
    deadline = time.monotonic() + STATE_LOCK_TIMEOUT_SECONDS
    acquired = False
    try:
        if os.name == "nt":
            import msvcrt

            _prepare_windows_lock_byte(stream)
            while True:
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as error:
                    if not _lock_is_contended(error):
                        raise OpsError("Vast state lock probe failed") from error
                    if time.monotonic() >= deadline:
                        raise OpsError("timed out acquiring Vast state lock") from error
                    time.sleep(0.02)
        else:
            import fcntl

            while True:
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as error:
                    if not _lock_is_contended(error):
                        raise OpsError("Vast state lock probe failed") from error
                    if time.monotonic() >= deadline:
                        raise OpsError("timed out acquiring Vast state lock") from error
                    time.sleep(0.02)
        acquired = True
        _STATE_LOCK_LOCAL.depth = 1
        yield
    finally:
        _STATE_LOCK_LOCAL.depth = 0
        try:
            if acquired and os.name == "nt":
                import msvcrt

                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            elif acquired:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


@contextmanager
def _lifecycle_transition_lock(
    run_id: str, *, deadline_monotonic: float | None = None
) -> Iterator[None]:
    """Serialize remote start authorization against cleanup for one run.

    State compare-and-set fences prevent a stale parent from relabelling a run,
    but a check immediately before a network call still has a time-of-check /
    time-of-use gap. Cleanup first records its authority, then crosses this
    barrier; a bootstrap holds it from its final state check through the remote
    start. Thus a start already in flight finishes before destruction, while a
    later start observes cleanup authority and never begins.
    """

    lock_name = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
    lock_path = MONITOR_DIR / f"lifecycle_{lock_name}.lock"
    held = set(getattr(_LIFECYCLE_LOCK_LOCAL, "held", set()))
    if lock_path in held:
        yield
        return
    stream = _open_lock_stream(lock_path)
    deadline = (
        deadline_monotonic
        if deadline_monotonic is not None
        else time.monotonic() + STATE_LOCK_TIMEOUT_SECONDS
    )
    acquired = False
    try:
        if os.name == "nt":
            import msvcrt

            _prepare_windows_lock_byte(stream)
            while True:
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as error:
                    if not _lock_is_contended(error):
                        raise OpsError(
                            f"lifecycle lock probe failed for run {run_id!r}"
                        ) from error
                    if time.monotonic() >= deadline:
                        raise OpsError(
                            f"timed out acquiring lifecycle lock for run {run_id!r}"
                        ) from error
                    time.sleep(0.02)
        else:
            import fcntl

            while True:
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as error:
                    if not _lock_is_contended(error):
                        raise OpsError(
                            f"lifecycle lock probe failed for run {run_id!r}"
                        ) from error
                    if time.monotonic() >= deadline:
                        raise OpsError(
                            f"timed out acquiring lifecycle lock for run {run_id!r}"
                        ) from error
                    time.sleep(0.02)
        acquired = True
        held.add(lock_path)
        _LIFECYCLE_LOCK_LOCAL.held = held
        yield
    finally:
        held.discard(lock_path)
        _LIFECYCLE_LOCK_LOCAL.held = held
        try:
            if acquired and os.name == "nt":
                import msvcrt

                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            elif acquired:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


def append_local_event(run_id: str, event: str, **fields: Any) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": utc_now(), "run_id": run_id, "event": event, **fields}
    with EVENT_FILE.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def load_secrets() -> dict[str, str]:
    if not SECRETS_FILE.exists():
        raise OpsError(f"missing {SECRETS_FILE}")
    result: dict[str, str] = {}
    # utf-8-sig tolerates the BOM PowerShell 5.1 writes with -Encoding utf8.
    for raw_line in SECRETS_FILE.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            result[key.strip()] = value
    return result


_DIAGNOSTIC_KEY_TOKEN = re.compile(r"(?<![A-Za-z0-9_.-])[A-Za-z0-9_.-]+")


def _redact_sensitive_assignments(value: str) -> str:
    """Withhold free text containing an unknown credential assignment."""

    length = len(value)
    tokens = list(_DIAGNOSTIC_KEY_TOKEN.finditer(value))
    for token_index, match in enumerate(tokens):
        key_end: int | None = None
        for candidate_index in range(token_index, min(token_index + 3, len(tokens))):
            candidate = tokens[candidate_index]
            if candidate_index > token_index:
                gap = value[tokens[candidate_index - 1].end() : candidate.start()]
                if not gap or not all(character.isspace() for character in gap):
                    break
            if _is_sensitive_diagnostic_key(value[match.start() : candidate.end()]):
                key_end = candidate.end()
                break
        if key_end is None:
            continue
        position = key_end
        had_whitespace = False
        had_wrapper = False
        # Web forms and configuration diagnostics commonly spell nested keys
        # as ``user[password]`` or ``config['signing_key']``. Closing syntax is
        # part of the key boundary, not evidence that the following value is
        # safe to retain.
        while position < length:
            if value[position].isspace():
                had_whitespace = True
                position += 1
            elif value[position] in {'"', "'", "]", "}", ")", ">"}:
                had_wrapper = True
                position += 1
            else:
                break
        if position < length and value[position] in ":=|":
            position += 1
            while position < length and value[position].isspace():
                position += 1
        elif not (had_whitespace or had_wrapper):
            continue
        if position >= length:
            continue
        # Free text has no authenticated key/value boundary. A delimiter may
        # separate another cookie, auth token, or key fragment, so partial
        # preservation cannot prove the suffix safe.
        return _DIAGNOSTIC_WITHHELD
    return value


def _contains_uri_userinfo(value: str) -> bool:
    """Return whether bounded text contains a URI authority with userinfo."""

    # Scan URI and protocol-relative authorities in a single forward pass.
    # Advancing past each authority keeps the maximum-size diagnostic path
    # linear even for adversarial strings containing many slash pairs.
    search_start = 0
    while True:
        marker = value.find("//", search_start)
        if marker < 0:
            return False
        authority_start = marker + 2
        authority_end = authority_start
        while (
            authority_end < len(value)
            and value[authority_end] not in "/?# \t\r\n"
        ):
            authority_end += 1
        authority = value[authority_start:authority_end]
        at = authority.find("@")
        if at > 0:
            return True
        search_start = max(authority_end, authority_start)


def redact_secrets(value: str) -> str:
    """Remove configured credential values and common API-key URL forms."""

    if any(
        (ord(character) < 32 and character not in "\n\t")
        or 127 <= ord(character) <= 159
        or unicodedata.category(character) == "Cf"
        or character == "\ufffd"
        for character in value
    ):
        return "[DIAGNOSTIC WITHHELD]"
    value = unicodedata.normalize("NFKC", value)
    if re.search(
        r"(?i)(?:-----BEGIN(?: [A-Z0-9]+)? PRIVATE KEY-----|"
        r"PuTTY-User-Key-File-\d+:)",
        value,
    ):
        return "[DIAGNOSTIC WITHHELD]"
    if _contains_uri_userinfo(value):
        return "[DIAGNOSTIC WITHHELD]"
    # Escaped identifiers can conceal a sensitive key from a textual scanner
    # (for example ``pass\u0077ord`` in provider-prefixed JSON). Raw diagnostic
    # text has no need to preserve that representation, so withhold it whole.
    if re.search(
        r"(?:\\(?:u[0-9a-fA-F]{4}|x[0-9a-fA-F]{2}|[\"'])|%[0-9a-fA-F]{2})",
        value,
    ):
        return "[DIAGNOSTIC WITHHELD]"
    redacted = _redact_sensitive_assignments(value)
    try:
        secret_values = [item for item in load_secrets().values() if item]
    except OpsError:
        secret_values = []
    for secret in sorted(secret_values, key=len, reverse=True):
        redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def _is_sensitive_diagnostic_key(value: str) -> bool:
    normalized = _diagnostic_key_fingerprint(value)
    canonical = unicodedata.normalize("NFKC", value)
    segmented = re.sub(
        r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])",
        "_",
        canonical,
    )
    segments = re.findall(
        r"[a-z0-9]+", segmented.casefold()
    )
    return (
        normalized
        in {
            "basic",
            "bindpw",
            "clientkey",
            "encryptionkey",
            "fernetkey",
            "jsessionid",
            "jwt",
            "masterkey",
            "pat",
            "pwd",
            "sessionid",
            "signingkey",
            "tlskey",
            "webhookurl",
        }
        or (
            bool(segments)
            and segments[-1] in {"key", "pass", "pw", "pwd", "sig"}
        )
        or bool({"confidential", "sensitive"}.intersection(segments))
        or any(
            marker in normalized
            for marker in (
                "apikey",
                "authorization",
                "bearer",
                "token",
                "secret",
                "password",
                "passwd",
                "credential",
                "privatekey",
                "passphrase",
                "sshkey",
                "identityfile",
                "accesskey",
                "accountkey",
                "sessionkey",
                "subscriptionkey",
                "puttyuserkeyfile",
                "privatelines",
                "signature",
                "setcookie",
                "cookie",
                "auth",
            )
        )
    )


def _diagnostic_key_fingerprint(value: str) -> str:
    return re.sub(
        r"[^a-z0-9]+", "", unicodedata.normalize("NFKC", value).casefold()
    )


def _redacted_exception_text(error: Exception) -> str:
    """Return diagnostic text without ever letting redaction block cleanup.

    Reading the ignored credential file is itself fallible.  If either the
    exception's string conversion or secret redaction fails, withhold the raw
    text instead of risking credential persistence or replacing the failure
    that triggered cleanup.
    """

    try:
        raw_error = str(error)
    except Exception:
        return _DIAGNOSTIC_WITHHELD
    return _redacted_diagnostic_text(raw_error)


_DIAGNOSTIC_WITHHELD = "[DIAGNOSTIC WITHHELD]"
_MAX_DIAGNOSTIC_JSON_DEPTH = 24
_MAX_DIAGNOSTIC_JSON_ITEMS = 1024
_MAX_DIAGNOSTIC_JSON_NODES = 8192
_MAX_DIAGNOSTIC_JSON_CHARACTERS = 1024 * 1024
_MAX_DIAGNOSTIC_JSON_STRING_CHARACTERS = 64 * 1024
_MAX_REMOTE_EVENT_BYTES = 256 * 1024
_MAX_MANUAL_DIAGNOSTIC_BYTES = 256 * 1024
_MAX_MANUAL_METRIC_TAIL_BYTES = 64 * 1024
_REMOTE_CONTROL_KEYS = frozenset({"event", "run_id", "exit_code"})
_REMOTE_EVENT_NAMES = frozenset(
    {
        "artifact_packaging_failed",
        "artifacts_ready",
        "bootstrap_failed",
        "command_finished",
        "command_started",
        "heartbeat",
        "remote_event_withheld",
        "run_failed",
        "run_started",
        "run_succeeded",
        "runner_exception",
        "session_snapshot_ready",
        "session_snapshot_started",
        "ttl_reached",
    }
)


class _DiagnosticSanitizationError(Exception):
    """Internal marker for a diagnostic that must be withheld in full."""


def _redacted_diagnostic_text(value: object) -> str:
    """Return redacted text, withholding it completely on any failure."""

    try:
        raw_value = value if isinstance(value, str) else str(value)
    except Exception:
        return _DIAGNOSTIC_WITHHELD
    if len(raw_value) > _MAX_DIAGNOSTIC_JSON_STRING_CHARACTERS:
        return _DIAGNOSTIC_WITHHELD
    stripped = raw_value.strip()
    if stripped.startswith(("{", "[")):
        try:
            parsed = json.loads(raw_value, object_pairs_hook=_unique_json_object)
        except Exception:
            return _DIAGNOSTIC_WITHHELD
        else:
            try:
                safe_parsed = _redacted_json_value_strict(parsed)
                safe_value = json.dumps(safe_parsed, sort_keys=True)
            except Exception:
                return _DIAGNOSTIC_WITHHELD
            if len(safe_value) > _MAX_DIAGNOSTIC_JSON_STRING_CHARACTERS:
                return _DIAGNOSTIC_WITHHELD
            return safe_value
    try:
        safe_value = redact_secrets(raw_value)
    except Exception:
        return _DIAGNOSTIC_WITHHELD
    if (
        not isinstance(safe_value, str)
        or len(safe_value) > _MAX_DIAGNOSTIC_JSON_STRING_CHARACTERS
    ):
        return _DIAGNOSTIC_WITHHELD
    return safe_value


def _redacted_json_value_strict(
    value: Any,
    *,
    _depth: int = 0,
    _budget: dict[str, int] | None = None,
) -> Any:
    """Return a JSON-safe redacted value or reject the entire diagnostic.

    Remote event files are untrusted diagnostic input.  They can contain raw
    exception text from the rented host, including a credential echoed by a
    failing command.  Redact every string, including mapping keys, before the
    value can reach a local sink.  Any traversal, type, size, or redaction
    failure rejects the enclosing event so a partially sanitized success marker
    can never remain control-authoritative.
    """

    try:
        if _budget is None:
            _budget = {
                "nodes": _MAX_DIAGNOSTIC_JSON_NODES,
                "characters": _MAX_DIAGNOSTIC_JSON_CHARACTERS,
            }
        _budget["nodes"] -= 1
        if _budget["nodes"] < 0:
            raise _DiagnosticSanitizationError(
                "diagnostic aggregate node limit exceeded"
            )
        if _depth >= _MAX_DIAGNOSTIC_JSON_DEPTH:
            raise _DiagnosticSanitizationError("diagnostic nesting limit exceeded")
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise _DiagnosticSanitizationError("non-finite diagnostic number")
            return value
        if isinstance(value, str):
            if len(value) > _MAX_DIAGNOSTIC_JSON_STRING_CHARACTERS:
                raise _DiagnosticSanitizationError(
                    "diagnostic string limit exceeded"
                )
            _budget["characters"] -= len(value)
            if _budget["characters"] < 0:
                raise _DiagnosticSanitizationError(
                    "diagnostic aggregate character limit exceeded"
                )
            try:
                safe_value = redact_secrets(value)
            except Exception as error:
                raise _DiagnosticSanitizationError(
                    "diagnostic redaction failed"
                ) from error
            if not isinstance(safe_value, str):
                raise _DiagnosticSanitizationError(
                    "diagnostic redactor returned a non-string"
                )
            if len(safe_value) > _MAX_DIAGNOSTIC_JSON_STRING_CHARACTERS:
                raise _DiagnosticSanitizationError(
                    "redacted diagnostic string limit exceeded"
                )
            expansion = len(safe_value) - len(value)
            if expansion > 0:
                _budget["characters"] -= expansion
                if _budget["characters"] < 0:
                    raise _DiagnosticSanitizationError(
                        "redacted aggregate character limit exceeded"
                    )
            return safe_value
        if isinstance(value, dict):
            try:
                items = list(value.items())
            except Exception as error:
                raise _DiagnosticSanitizationError(
                    "diagnostic mapping traversal failed"
                ) from error
            if len(items) > _MAX_DIAGNOSTIC_JSON_ITEMS:
                raise _DiagnosticSanitizationError(
                    "diagnostic mapping item limit exceeded"
                )
            for _key, item in items:
                if isinstance(item, str) and _is_sensitive_diagnostic_key(item):
                    raise _DiagnosticSanitizationError(
                        "diagnostic contains a standalone sensitive descriptor"
                    )
            safe_mapping: dict[str, Any] = {}
            for key, item in items:
                if not isinstance(key, str):
                    raise _DiagnosticSanitizationError(
                        "diagnostic mapping key is not a string"
                    )
                safe_key = _redacted_json_value_strict(
                    key, _depth=_depth + 1, _budget=_budget
                )
                if _is_sensitive_diagnostic_key(key):
                    safe_key = "[REDACTED KEY]"
                if safe_key in safe_mapping:
                    raise _DiagnosticSanitizationError(
                        "diagnostic keys collide after redaction"
                    )
                if _is_sensitive_diagnostic_key(key):
                    safe_mapping[safe_key] = "[REDACTED]"
                else:
                    safe_mapping[safe_key] = _redacted_json_value_strict(
                        item, _depth=_depth + 1, _budget=_budget
                    )
            return safe_mapping
        if isinstance(value, (list, tuple)):
            try:
                items = list(value)
            except Exception as error:
                raise _DiagnosticSanitizationError(
                    "diagnostic sequence traversal failed"
                ) from error
            if len(items) > _MAX_DIAGNOSTIC_JSON_ITEMS:
                raise _DiagnosticSanitizationError(
                    "diagnostic sequence item limit exceeded"
                )
            if any(
                isinstance(item, str) and _is_sensitive_diagnostic_key(item)
                for item in items
            ):
                raise _DiagnosticSanitizationError(
                    "diagnostic sequence contains a sensitive key or flag"
                )
            return [
                _redacted_json_value_strict(
                    item, _depth=_depth + 1, _budget=_budget
                )
                for item in items
            ]
        raise _DiagnosticSanitizationError("diagnostic value is not JSON-safe")
    except _DiagnosticSanitizationError:
        raise
    except Exception as error:
        raise _DiagnosticSanitizationError(
            "unexpected diagnostic sanitization failure"
        ) from error


def _redacted_json_value(value: Any) -> Any:
    """Return a safe diagnostic value, withholding it in full on failure."""

    try:
        return _redacted_json_value_strict(value)
    except Exception:
        return _DIAGNOSTIC_WITHHELD


def _validate_raw_remote_control_fields(event: Any) -> None:
    """Reject remote control fields before diagnostic normalization.

    Redaction normalizes Unicode to make credential matching conservative.  It
    must never turn a look-alike remote key, event name, or run ID into an
    authoritative ASCII success marker.
    """

    if type(event) is not dict:
        raise _DiagnosticSanitizationError("remote event is not a JSON object")
    for key in event:
        if not isinstance(key, str):
            raise _DiagnosticSanitizationError(
                "remote event mapping key is not a string"
            )
        normalized_key = unicodedata.normalize("NFKC", key)
        if normalized_key in _REMOTE_CONTROL_KEYS and key != normalized_key:
            raise _DiagnosticSanitizationError(
                "remote control key is not canonical ASCII"
            )
    event_name = event.get("event")
    if event_name not in _REMOTE_EVENT_NAMES:
        raise _DiagnosticSanitizationError(
            "remote event name is absent or outside the fixed protocol"
        )
    if event_name != "remote_event_withheld" or "run_id" in event:
        run_id = event.get("run_id")
        if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
            raise _DiagnosticSanitizationError(
                "remote run ID is not canonical ASCII"
            )
    if "exit_code" in event and type(event["exit_code"]) is not int:
        raise _DiagnosticSanitizationError(
            "remote exit code is not an exact integer"
        )


def _sanitized_remote_events(events: Any) -> list[dict[str, Any]]:
    """Sanitize remote events and make malformed input non-authoritative."""

    if not isinstance(events, (list, tuple)):
        return [
            {
                "event": "remote_event_withheld",
                "diagnostic": _DIAGNOSTIC_WITHHELD,
            }
        ]
    if len(events) > _MAX_DIAGNOSTIC_JSON_ITEMS:
        return [
            {
                "event": "remote_event_withheld",
                "diagnostic": _DIAGNOSTIC_WITHHELD,
            }
        ]
    safe_events: list[dict[str, Any]] = []
    budget = {
        "nodes": _MAX_DIAGNOSTIC_JSON_NODES,
        "characters": _MAX_DIAGNOSTIC_JSON_CHARACTERS,
    }
    try:
        for event in events:
            _validate_raw_remote_control_fields(event)
            if event.get("event") == "remote_event_withheld" and (
                len(events) != 1
                or event
                != {
                    "event": "remote_event_withheld",
                    "diagnostic": _DIAGNOSTIC_WITHHELD,
                }
            ):
                raise _DiagnosticSanitizationError(
                    "remote withheld sentinel has an invalid authority shape"
                )
            safe_event = _redacted_json_value_strict(event, _budget=budget)
            if not isinstance(safe_event, dict):
                raise _DiagnosticSanitizationError(
                    "remote event is not a JSON object"
                )
            raw_controls = {
                key: event[key] for key in _REMOTE_CONTROL_KEYS if key in event
            }
            safe_controls = {
                key: safe_event[key]
                for key in _REMOTE_CONTROL_KEYS
                if key in safe_event
            }
            if raw_controls.keys() != safe_controls.keys() or any(
                type(safe_controls[key]) is not type(raw_value)
                or safe_controls[key] != raw_value
                for key, raw_value in raw_controls.items()
            ):
                raise _DiagnosticSanitizationError(
                    "diagnostic sanitization changed remote control authority"
                )
            event_name = safe_event.get("event")
            if not isinstance(event_name, str) or not event_name:
                raise _DiagnosticSanitizationError(
                    "remote event name is absent or malformed"
                )
            safe_events.append(safe_event)
    except Exception:
        return [
            {
                "event": "remote_event_withheld",
                "diagnostic": _DIAGNOSTIC_WITHHELD,
            }
        ]
    return safe_events


def _is_bound_remote_failure_event(event: Any, expected_run_id: str) -> bool:
    return (
        isinstance(event, dict)
        and event.get("event") in {"run_failed", "bootstrap_failed"}
        and event.get("run_id") == expected_run_id
    )


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON keys instead of silently taking the last value."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DiagnosticSanitizationError("duplicate remote JSON key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    """Reject Python's non-standard NaN and infinity JSON extensions."""

    del value
    raise _DiagnosticSanitizationError("non-standard JSON numeric constant")


def _finite_json_float(value: str) -> float:
    """Decode one standard JSON float without accepting exponent overflow."""

    number = float(value)
    if not math.isfinite(number):
        raise _DiagnosticSanitizationError("non-finite JSON number")
    return number


def _bounded_json_int(value: str) -> int:
    """Decode provider integers within the signed 64-bit operational domain."""

    number = int(value)
    if not -(2**63) <= number <= 2**63 - 1:
        raise _DiagnosticSanitizationError("provider JSON integer is out of range")
    return number


def vast_cli_path() -> str:
    interpreter_directory = Path(sys.executable).parent
    candidates = [
        os.environ.get("VAST_CLI"),
        shutil.which("vastai"),
        shutil.which("vastai.exe"),
        str(interpreter_directory / "vastai.exe"),
        str(interpreter_directory / "vastai"),
        # Windows virtualenv/Conda console entry points live under Scripts,
        # while python.exe itself lives at the environment root.
        str(interpreter_directory / "Scripts" / "vastai.exe"),
        str(interpreter_directory / "Scripts" / "vastai"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise OpsError("vastai CLI not found; install it or set VAST_CLI")


def isolated_cli_environment() -> dict[str, str]:
    secrets = load_secrets()
    api_key = secrets.get("VAST_API_KEY")
    if not api_key:
        raise OpsError("VAST_API_KEY is absent or empty in vast/secrets.env")
    env = os.environ.copy()
    xdg_root = RUNTIME_DIR / "xdg"
    locations = {
        "XDG_CONFIG_HOME": xdg_root / "config",
        "XDG_CACHE_HOME": xdg_root / "cache",
        "XDG_STATE_HOME": xdg_root / "state",
    }
    for key, path in locations.items():
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env["VAST_API_KEY"] = api_key
    env["VASTAI_NO_UPDATE_CHECK"] = "1"
    return env


def run_vast(
    *arguments: object,
    raw: bool = True,
    check: bool = True,
    timeout: int = 180,
) -> Any:
    command = [vast_cli_path(), *map(str, arguments)]
    if raw:
        command.append("--raw")
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=isolated_cli_environment(),
            **hidden_subprocess_options(),
        )
    except (OSError, subprocess.SubprocessError) as error:
        # Subprocess exceptions may render the complete argv, including the
        # per-instance SSH public-key operand or future provider credentials.
        raise OpsError("vastai provider command execution failed") from None
    if check and process.returncode != 0:
        safe_stdout = _redacted_diagnostic_text(process.stdout.strip())
        safe_stderr = _redacted_diagnostic_text(process.stderr.strip())
        raise OpsError(f"vastai provider command failed: {safe_stdout} {safe_stderr}")
    if not raw:
        return process.stdout
    stdout = process.stdout
    try:
        stdout_bytes = len(stdout.encode("utf-8"))
    except (AttributeError, UnicodeError):
        stdout_bytes = _MAX_PROVIDER_JSON_BYTES + 1
    if (
        not isinstance(stdout, str)
        or stdout_bytes > _MAX_PROVIDER_JSON_BYTES
        or not _json_structure_within_provider_limits(stdout)
    ):
        raise OpsError("vastai returned a malformed or oversized JSON response")
    try:
        return json.loads(
            stdout,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_float,
            parse_int=_bounded_json_int,
        )
    except (json.JSONDecodeError, _DiagnosticSanitizationError) as error:
        if check:
            safe_stdout = _redacted_diagnostic_text(stdout.strip())
            safe_stderr = _redacted_diagnostic_text(process.stderr.strip())
            raise OpsError(
                "vastai did not return valid JSON: "
                f"stdout={safe_stdout[:1000]!r} stderr={safe_stderr[:1000]!r}"
            ) from error
        return process.stdout


def git_output(*arguments: str, check: bool = True) -> str:
    process = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        **hidden_subprocess_options(),
    )
    if check and process.returncode != 0:
        raise OpsError(
            _redacted_diagnostic_text(
                f"git {' '.join(arguments)} failed: {process.stderr.strip()}"
            )
        )
    return process.stdout.strip()


def git_head() -> str:
    return git_output("rev-parse", "HEAD")


def git_is_clean() -> bool:
    return not git_output("status", "--porcelain", "--untracked-files=all")


def load_manifest(path: Path) -> RunManifest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OpsError(
            "cannot read manifest: " + _redacted_exception_text(error)
        ) from error
    if not isinstance(raw, dict):
        raise OpsError("manifest root must be a JSON object")
    return RunManifest.from_dict(raw)


def require_adoption_approval(manifest: RunManifest, *, instance_id: int) -> None:
    """Validate the sole adoption target; manifest fields are authoritative."""

    del manifest  # The manifest is validated by load_manifest/create_source_bundle.
    _strict_positive_identity(instance_id, "--instance")


def _load_state_unlocked() -> list[dict[str, Any]]:
    value = read_json(STATE_FILE, [])
    if not isinstance(value, list):
        raise OpsError("local Vast state is malformed")
    return value


def load_state() -> list[dict[str, Any]]:
    with state_file_lock():
        return _load_state_unlocked()


def _save_state_unlocked(records: list[dict[str, Any]]) -> None:
    atomic_write_json(STATE_FILE, records)


def save_state(records: list[dict[str, Any]]) -> None:
    with state_file_lock():
        _save_state_unlocked(records)


def get_record(run_id: str) -> dict[str, Any]:
    matches = [record for record in load_state() if record.get("run_id") == run_id]
    if len(matches) != 1:
        raise OpsError(f"expected one tracked record for run {run_id!r}")
    return matches[0]


def _record_cleanup_generation(record: dict[str, Any]) -> int:
    value = record.get("cleanup_generation", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OpsError("persisted cleanup_generation must be a non-negative integer")
    return value


def _record_cleanup_evidence(
    record: dict[str, Any], **updates: Any
) -> dict[str, Any]:
    """Merge a cleanup-positive transition and advance its CAS generation."""

    updates["cleanup_required"] = True
    updates["cleanup_generation"] = _record_cleanup_generation(record) + 1
    record.update(updates)
    return record


def record_instance_id(record: dict[str, Any]) -> int:
    value = record.get("instance_id")
    if value is None:
        raise OpsError(
            f"run {record.get('run_id')!r} has no known instance ID; run reconcile"
        )
    return _strict_positive_identity(value, "persisted instance ID")


def update_record_fields(run_id: str, **updates: Any) -> dict[str, Any]:
    """Merge fields into the current record under the cross-process lock.

    Detached supervisors can begin mutating a record before the process that
    launched them receives its PID.  Callers that retained a pre-launch
    snapshot must therefore never write that snapshot back wholesale.
    """

    with state_file_lock():
        records = _load_state_unlocked()
        matches = [record for record in records if record.get("run_id") == run_id]
        if len(matches) != 1:
            raise OpsError(f"expected one tracked record for run {run_id!r}")
        if updates.get("cleanup_required") is True:
            _record_cleanup_evidence(matches[0], **updates)
        else:
            matches[0].update(updates)
        _save_state_unlocked(records)
        return dict(matches[0])


def _record_provisioning_setup_failure(
    run_id: str,
    *,
    instance_id: int | None,
    error: Exception,
) -> dict[str, Any]:
    """Persist one redacted setup failure without changing lifecycle authority.

    The state record is authoritative and embeds the complete event payload, so
    a crash between its atomic replacement and the append-only event replica
    cannot erase the diagnostic.  Holding the state lock across both writes
    prevents cleanup and provisioning controllers from interleaving them.
    Callers must still continue cleanup if either persistence write fails.
    """

    safe_error = _redacted_exception_text(error)
    observed_at = utc_now()
    evidence = {
        "timestamp": observed_at,
        "run_id": run_id,
        "event": "provisioning_setup_failed",
        "instance_id": instance_id,
        "error": safe_error,
    }
    with state_file_lock():
        records = _load_state_unlocked()
        matches = [record for record in records if record.get("run_id") == run_id]
        if len(matches) != 1:
            raise OpsError(f"expected one tracked record for run {run_id!r}")
        current = matches[0]
        current.update(
            setup_error=safe_error,
            setup_error_at=observed_at,
            setup_failure_event=evidence,
        )
        _save_state_unlocked(records)
        append_local_event(
            run_id,
            "provisioning_setup_failed",
            timestamp=observed_at,
            instance_id=instance_id,
            error=safe_error,
        )
        return dict(current)


def _best_effort_record_secondary_failure(
    run_id: str,
    *,
    error: Exception,
    error_field: str,
    error_at_field: str,
    event: str,
    state_fields: dict[str, Any] | None = None,
    **event_fields: Any,
) -> None:
    """Persist redacted secondary diagnostics without raising an exception."""

    try:
        safe_error = _redacted_exception_text(error)
        observed_at = utc_now()
        try:
            updates = {
                error_field: safe_error,
                error_at_field: observed_at,
            }
            if state_fields:
                updates.update(state_fields)
            update_record_fields(
                run_id,
                **updates,
            )
        except Exception:
            pass
        try:
            append_local_event(
                run_id,
                event,
                timestamp=observed_at,
                error=safe_error,
                **event_fields,
            )
        except Exception:
            pass
    except Exception:
        # This helper is used only while another exception is already driving
        # teardown.  No diagnostic operation may replace that exception.
        pass


def _best_effort_append_local_event(
    run_id: str, event: str, **fields: Any
) -> None:
    """Append redacted non-authoritative evidence without interrupting cleanup."""

    try:
        safe_fields = _redacted_json_value(fields)
        if not isinstance(safe_fields, dict):
            safe_fields = {"diagnostic": _DIAGNOSTIC_WITHHELD}
        append_local_event(run_id, event, **safe_fields)
    except Exception:
        pass


def _best_effort_update_record_fields(run_id: str, **updates: Any) -> bool:
    """Merge non-authoritative or already-fenced evidence without raising."""

    try:
        update_record_fields(run_id, **updates)
    except Exception:
        return False
    return True


def _best_effort_monitor_event_if_active(
    run_id: str, event: str, **fields: Any
) -> None:
    """Emit a redacted monitor diagnostic without delaying cleanup."""

    try:
        _monitor_event_if_active(run_id, event, **fields)
    except Exception:
        pass


def _best_effort_emit_json(
    payload: Any,
    *,
    stream: Any | None = None,
    indent: int | None = None,
) -> None:
    """Emit redacted operator output without interrupting lifecycle ownership."""

    try:
        safe_payload = _redacted_json_value(payload)
        print(
            json.dumps(safe_payload, indent=indent, sort_keys=indent is None),
            file=stream,
            flush=True,
        )
    except Exception:
        pass


def _best_effort_emit_text(
    payload: Any,
    *,
    stream: Any | None = None,
    end: str = "",
) -> None:
    """Emit recursively bounded/redacted text without affecting control flow."""

    try:
        safe_payload = _redacted_json_value(payload)
        if not isinstance(safe_payload, str):
            safe_payload = _DIAGNOSTIC_WITHHELD
        safe_payload = "".join(
            character
            if character in "\n\t"
            or (ord(character) >= 32 and not 127 <= ord(character) <= 159)
            else f"\\x{ord(character):02x}"
            for character in safe_payload
        )
        print(safe_payload, file=stream, end=end, flush=True)
    except Exception:
        pass


def _provisioning_write_if_active(
    run_id: str,
    *,
    allowed_statuses: set[str],
    **updates: Any,
) -> tuple[bool, dict[str, Any]]:
    """Apply one provisioning transition only from its declared live state."""

    with state_file_lock():
        records = _load_state_unlocked()
        matches = [record for record in records if record.get("run_id") == run_id]
        if len(matches) != 1:
            raise OpsError(f"expected one tracked record for run {run_id!r}")
        current = matches[0]
        current_status = str(current.get("status") or "")
        if (
            current_status == "destroyed"
            or bool(current.get("cleanup_required"))
            or current_status not in allowed_statuses
        ):
            return False, dict(current)
        if updates.get("cleanup_required") is True:
            _record_cleanup_evidence(current, **updates)
        else:
            current.update(updates)
        _save_state_unlocked(records)
        return True, dict(current)


def _require_provisioning_transition(
    run_id: str, *, allowed_statuses: set[str], **updates: Any
) -> dict[str, Any]:
    changed, current = _provisioning_write_if_active(
        run_id, allowed_statuses=allowed_statuses, **updates
    )
    if not changed:
        raise OpsError(
            "provisioning lost lifecycle authority before transition; "
            f"current status={current.get('status')!r}, "
            f"cleanup_required={bool(current.get('cleanup_required'))}"
        )
    return current


def _monitor_write_if_active(
    run_id: str,
    *,
    event: str | None = None,
    event_fields: dict[str, Any] | None = None,
    **updates: Any,
) -> tuple[bool, dict[str, Any]]:
    """Merge a monitor update only while the run still permits monitor writes.

    The monitor can race a cleanup supervisor.  A stale ``record`` snapshot must
    never be written back after that supervisor has marked the run destroyed or
    requiring cleanup.  State and its corresponding event are written while the
    state lock is held, so the fence applies to both sides of the transition.
    """

    with state_file_lock():
        records = _load_state_unlocked()
        matches = [record for record in records if record.get("run_id") == run_id]
        if len(matches) != 1:
            raise OpsError(f"expected one tracked record for run {run_id!r}")
        current = matches[0]
        if current.get("status") == "destroyed" or current.get("cleanup_required"):
            return False, dict(current)
        if updates.get("cleanup_required") is True:
            _record_cleanup_evidence(current, **updates)
        else:
            current.update(updates)
        _save_state_unlocked(records)
        if event is not None:
            # State is the authoritative transition.  The append-only event is
            # a replica and must not turn a successfully fenced cleanup state
            # into a control-flow failure.
            _best_effort_append_local_event(
                run_id, event, **(event_fields or {})
            )
        return True, dict(current)


def _monitor_event_if_active(
    run_id: str, event: str, **fields: Any
) -> tuple[bool, dict[str, Any]]:
    """Append a monitor event only if cleanup has not won the state race."""

    return _monitor_write_if_active(
        run_id, event=event, event_fields=fields
    )


def replace_record(updated: dict[str, Any]) -> None:
    with state_file_lock():
        records = _load_state_unlocked()
        records = [
            updated if item.get("run_id") == updated.get("run_id") else item
            for item in records
        ]
        if not any(item.get("run_id") == updated.get("run_id") for item in records):
            records.append(updated)
        _save_state_unlocked(records)


def append_record(record: dict[str, Any]) -> None:
    """Append one new lifecycle record under the same cross-process lock."""

    with state_file_lock():
        records = _load_state_unlocked()
        run_id = record.get("run_id")
        if any(item.get("run_id") == run_id for item in records):
            raise OpsError(f"run_id {run_id!r} is already tracked")
        instance_value = record.get("instance_id")
        if instance_value is not None:
            instance_id = _strict_positive_identity(
                instance_value, "new tracked instance ID"
            )
            for item in records:
                existing_value = item.get("instance_id")
                if existing_value is None:
                    continue
                existing_id = _strict_positive_identity(
                    existing_value, "persisted tracked instance ID"
                )
                if existing_id == instance_id:
                    raise OpsError(
                        f"instance {instance_id} is already tracked by run "
                        f"{item.get('run_id')!r}"
                    )
        records.append(record)
        _save_state_unlocked(records)


def load_stage_state() -> list[dict[str, Any]]:
    value = read_json(STAGE_STATE_FILE, [])
    if not isinstance(value, list):
        raise OpsError("local Vast stage state is malformed")
    return value


def save_stage_state(records: list[dict[str, Any]]) -> None:
    atomic_write_json(STAGE_STATE_FILE, records)


def get_stage_record(stage_id: str) -> dict[str, Any]:
    matches = [
        record for record in load_stage_state() if record.get("stage_id") == stage_id
    ]
    if len(matches) != 1:
        raise OpsError(f"expected one tracked stage record for {stage_id!r}")
    return matches[0]


def replace_stage_record(updated: dict[str, Any]) -> None:
    records = load_stage_state()
    records = [
        updated if item.get("stage_id") == updated.get("stage_id") else item
        for item in records
    ]
    if not any(item.get("stage_id") == updated.get("stage_id") for item in records):
        records.append(updated)
    save_stage_state(records)


def elapsed_and_cost(record: dict[str, Any]) -> tuple[float, float]:
    elapsed_hours = max(
        0.0,
        (datetime.now(UTC) - parse_utc(record["created_at"])).total_seconds()
        / 3600,
    )
    campaign_offset = float(record.get("campaign_accrued_cost_usd", 0.0) or 0.0)
    return elapsed_hours, campaign_offset + elapsed_hours * float(record["dph"])


def elapsed_since(record: dict[str, Any], field: str) -> float:
    """Return hours since a persisted lifecycle timestamp."""

    value = record.get(field) or record["created_at"]
    return max(
        0.0,
        (datetime.now(UTC) - parse_utc(str(value))).total_seconds() / 3600,
    )


def cleanup_reserve_hours(poll_seconds: int) -> float:
    """Reserve the worst-case bounded destroy and verification window.

    A ``destroy`` acknowledgement is deliberately not treated as terminal.
    The operator must still be able to retry the command and query the live API
    enough times to establish absence, with a short follow-up window for a
    delayed/reappearing contract.  This is an operational reserve, not a
    provider guarantee: an API outage can still leave cleanup unresolved and is
    persisted as such by :func:`destroy_instance`.
    """

    try:
        poll = int(poll_seconds)
    except (TypeError, ValueError):
        poll = 0
    if poll < 0:
        poll = 0
    destruction_seconds = destroy_attempt_budget_seconds()
    followup_poll = max(poll, DESTROY_RECONCILIATION_POLL_SECONDS)
    followup_seconds = destroy_reconciliation_lease_seconds(followup_poll)
    handoff_seconds = (
        SUPERVISOR_LAUNCH_MAX_ATTEMPTS * SUPERVISOR_HANDSHAKE_TIMEOUT_SECONDS
        + (SUPERVISOR_LAUNCH_MAX_ATTEMPTS - 1)
        * SUPERVISOR_LAUNCH_RETRY_DELAY_SECONDS
    )
    return max(
        0.1,
        (
            HARD_LIMIT_RECOVERY_RESERVE_SECONDS
            + PURCHASE_FINAL_SCAN_RESERVE_SECONDS
            + destruction_seconds
            + followup_seconds
            + handoff_seconds
            + REMOTE_WATCHDOG_SAFETY_SECONDS
            + LOCAL_WATCHDOG_ORDERING_MARGIN_SECONDS
        )
        / 3600,
    )


def destroy_attempt_budget_seconds() -> float:
    """Return the complete bounded refresh, destroy, and verification budget."""

    # The exact-purchase escape path takes a fresh provider identity snapshot
    # before every possible destructive mutation. Ordinary cleanup does not
    # need these slots, but the shared reserve must cover the most expensive
    # authorized path.
    identity_refresh_seconds = (
        DESTROY_VERIFY_MAX_ATTEMPTS * DESTROY_API_TIMEOUT_SECONDS
    )
    identity_lock_seconds = (
        DESTROY_VERIFY_MAX_ATTEMPTS * STATE_LOCK_TIMEOUT_SECONDS
    )
    destroy_seconds = (
        DESTROY_MAX_ATTEMPTS * DESTROY_API_TIMEOUT_SECONDS
        + (DESTROY_MAX_ATTEMPTS - 1) * DESTROY_RETRY_DELAY_SECONDS
    )
    verify_seconds = (
        DESTROY_VERIFY_MAX_ATTEMPTS * DESTROY_API_TIMEOUT_SECONDS
        + (DESTROY_VERIFY_MAX_ATTEMPTS - 1) * DESTROY_VERIFY_RETRY_DELAY_SECONDS
    )
    return float(
        identity_refresh_seconds
        + identity_lock_seconds
        + destroy_seconds
        + verify_seconds
    )


def destroy_reconciliation_lease_seconds(poll_seconds: int) -> float:
    """Budget one reappearance, one re-destroy, and fresh absence proof."""

    poll = max(0, int(poll_seconds))
    return float(
        max(
            DESTROY_RECONCILIATION_MIN_SECONDS,
            (DESTROY_FOLLOWUP_ABSENT_SCANS + 3)
            * DESTROY_API_TIMEOUT_SECONDS
            + (DESTROY_FOLLOWUP_ABSENT_SCANS + 2) * poll,
        )
    )


def cleanup_trigger_hours(manifest: RunManifest, dph: float) -> float:
    spend_hours = manifest.max_total_usd / dph
    return max(
        0.0,
        min(manifest.ttl_hours, spend_hours)
        - cleanup_reserve_hours(manifest.poll_seconds),
    )


def load_machine_blacklist() -> set[int]:
    """Load the optional local do-not-rent list without mutating it.

    The list is intentionally outside manifests and ignored by Git: it captures
    account-specific host failures, not a reproducible scientific setting.
    """

    if not MACHINE_BLACKLIST_FILE.exists():
        return set()
    try:
        raw = json.loads(MACHINE_BLACKLIST_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OpsError(
            "cannot read machine blacklist: " + _redacted_exception_text(error)
        ) from error
    if not isinstance(raw, list) or any(
        not isinstance(value, int) or value <= 0 for value in raw
    ):
        raise OpsError("machine blacklist must be a JSON list of positive integer IDs")
    return set(raw)


def _is_server_cpu(offer: dict[str, Any]) -> bool:
    name = str(offer.get("cpu_name") or "").lower()
    return "epyc" in name or "xeon" in name


def _matches_country(record: dict[str, Any], required_country: str) -> bool:
    expected = required_country.strip().casefold()
    country = str(record.get("country") or "").strip().casefold()
    geolocation = str(record.get("geolocation") or "").strip()
    geographic_country = geolocation.rsplit(",", 1)[-1].strip().casefold()
    return expected in {country, geographic_country}


def _finite_offer_number(offer: dict[str, Any], key: str) -> float:
    value = offer.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OpsError(f"offer did not report a finite {key}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise OpsError(f"offer reported a non-finite {key}")
    return numeric


def _optional_finite_offer_number(offer: dict[str, Any], key: str) -> float:
    if offer.get(key) is None:
        return 0.0
    try:
        return _finite_offer_number(offer, key)
    except OpsError:
        return 0.0


def _offer_identity_number(offer: dict[str, Any], key: str) -> int:
    value = offer.get(key)
    if isinstance(value, bool):
        return 2**63 - 1
    if not isinstance(value, int) or value <= 0:
        return 2**63 - 1
    return value if value < 2**63 - 1 else 2**63 - 1


def _strict_positive_identity(value: object, field: str) -> int:
    """Return a provider identity only when it is an exact bounded integer.

    Purchase boundaries must not silently turn values such as ``1.0`` or
    ``"1"`` into a different identity.  Keep the sentinel-based helper above
    for read-only filtering/reconciliation, where malformed provider rows are
    ignored; use this fail-closed helper before any create or tracked-ID
    transition.
    """

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OpsError(f"{field} must be an exact positive integer")
    if value >= 2**63 - 1:
        raise OpsError(f"{field} is outside the supported integer range")
    return value


def _optional_strict_identity(value: object, field: str) -> int | None:
    """Validate an optional persisted identity without coercing sentinels."""

    if value is None:
        return None
    return _strict_positive_identity(value, field)


def _require_provider_instance_list(value: Any, context: str) -> list[Any]:
    """Reject every malformed provider snapshot, including falsey non-lists."""

    if not isinstance(value, list):
        raise OpsError(f"{context} did not return a list")
    return value


def _require_provider_instance_rows(
    value: Any,
    context: str,
    *,
    require_label: bool = False,
    require_machine_id: bool = False,
) -> list[dict[str, Any]]:
    """Validate every row before a provider snapshot can prove absence."""

    rows = _require_provider_instance_list(value, context)
    checked: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise OpsError(f"{context} contains a malformed instance entry")
        instance_id = _strict_positive_identity(
            row.get("id"), f"{context} entry {index} ID"
        )
        if instance_id in seen_ids:
            raise OpsError(f"{context} contains duplicate instance IDs")
        seen_ids.add(instance_id)
        if require_label and (
            "label" not in row
            or (row["label"] is not None and not isinstance(row["label"], str))
        ):
            raise OpsError(f"{context} contains a malformed instance label")
        if require_machine_id:
            _strict_positive_identity(
                row.get("machine_id"), f"{context} entry {index} machine ID"
            )
        checked.append(row)
    return checked


def active_machine_ids() -> set[int]:
    """Read physical hosts already rented by this account.

    Avoiding another slot on the same machine is a soft operational protection
    against correlated host/network failures; it is not a scientific criterion.
    """

    live = _require_provider_instance_rows(
        run_vast("show", "instances"),
        "Vast instance listing",
        require_machine_id=True,
    )
    return {
        _strict_positive_identity(instance["machine_id"], "live machine ID")
        for instance in live
    }


def assert_live_label_unused(label: str) -> None:
    """Fail closed if the exact ordinary-run label is already live."""

    if not label:
        raise OpsError("ordinary launch requires a non-empty expected label")
    live = _require_provider_instance_rows(
        run_vast("show", "instances"),
        "Vast instance listing",
        require_label=True,
    )
    if any(item.get("label") == label for item in live):
        raise OpsError(
            f"expected label {label!r} is already present on a live Vast instance"
        )


def search_offers(
    gpu_name: str,
    max_dph: float,
    *,
    num_gpus: int = 1,
    disk_gb: int,
    min_cuda: float,
    min_inet_down_mbps: int,
    min_cpu_cores_effective: int = DEFAULT_MIN_CPU_CORES,
    min_cpu_ram_gb: int = DEFAULT_MIN_CPU_RAM_GB,
    require_consumer_cpu: bool = False,
    required_geolocation: str | None = None,
    required_country: str | None = None,
    excluded_machine_ids: Iterable[int] = (),
    limit: int = 20,
) -> list[dict[str, Any]]:
    if min_cpu_cores_effective < 1 or min_cpu_ram_gb < 1 or num_gpus < 1:
        raise OpsError("GPU and CPU requirements must be positive")
    query = (
        f"gpu_name={gpu_name} num_gpus={num_gpus} rentable=true verified=true "
        f"dph<={max_dph} disk_space>={disk_gb} "
        f"cuda_max_good>={min_cuda} inet_down>={min_inet_down_mbps} "
        f"cpu_cores_effective>={min_cpu_cores_effective} "
        # Vast's search DSL accepts cpu_ram thresholds in GB, although raw
        # offer records report cpu_ram in MB.
        f"cpu_ram>={min_cpu_ram_gb} direct_port_count>=1"
    )
    # ``disk_space`` filters for host capacity, while ``--storage`` sets the
    # allocation priced by the resulting offer.  Keep both: omitting the
    # allocation makes the quote use Vast's default (currently 5 GiB), so a
    # later create with the manifest's requested disk can legitimately have a
    # different hourly price.
    offers = run_vast("search", "offers", query, "--storage", disk_gb, "-o", "dph")
    if not isinstance(offers, list):
        raise OpsError("Vast offer search did not return a list")
    excluded = set(excluded_machine_ids)
    eligible = [
        offer
        for offer in offers
        if isinstance(offer, dict)
        and offer.get("machine_id") not in excluded
        and (not require_consumer_cpu or not _is_server_cpu(offer))
        and (
            required_geolocation is None
            or offer.get("geolocation") == required_geolocation
        )
        and (required_country is None or _matches_country(offer, required_country))
    ]
    # ``limit`` is retained for CLI compatibility, but is deliberately not
    # applied here.  Launch selection must see every eligible offer returned by
    # Vast; truncating before cost-first selection can silently choose a more
    # expensive machine.  The read-only ``search`` command applies its display
    # limit after this function returns.
    del limit
    return eligible


def pick_offer(
    offers: list[dict[str, Any]],
    *,
    prefer_consumer_cpu: bool = True,
    price_band: float | None = None,
) -> dict[str, Any]:
    if price_band is None:
        price_band = OFFER_PRICE_BAND
    if not offers:
        raise OpsError("no offers satisfy the manifest limits")
    valid_offers = []
    for offer in offers:
        try:
            _finite_offer_number(offer, "dph_total")
        except OpsError:
            continue
        valid_offers.append(offer)
    if not valid_offers:
        raise OpsError("eligible offers did not report a finite hourly price")
    valid_offers = [
        offer for offer in valid_offers
        if _finite_offer_number(offer, "dph_total")
        <= OFFER_MAX_DPH_BY_GPU.get(
            _normalized_hardware_name(offer.get("gpu_name", "")), float("inf")
        )
    ]
    if not valid_offers:
        raise OpsError("no eligible offers under the per-GPU hourly price cap")

    # Avoid lemons: the very cheapest offers are disproportionately unreliable
    # hosts. Consider every offer within OFFER_PRICE_BAND of the cheapest eligible
    # price and take the most reliable one; price only breaks ties. max_dph and
    # max_total_usd still bound spend because offers were already filtered.
    floor = min(_finite_offer_number(offer, "dph_total") for offer in valid_offers)
    ceiling = floor * (1.0 + max(0.0, float(price_band)))
    banded = [
        offer for offer in valid_offers
        if _finite_offer_number(offer, "dph_total") <= ceiling
    ]

    def sort_key(offer: dict[str, Any]) -> tuple[float, int, float, float, float, float, int, int]:
        price = _finite_offer_number(offer, "dph_total")
        consumer_rank = (
            0 if not _is_server_cpu(offer) else 1
        ) if prefer_consumer_cpu else 0
        cores = _optional_finite_offer_number(offer, "cpu_cores_effective")
        ram = _optional_finite_offer_number(offer, "cpu_ram")
        reliability = _optional_finite_offer_number(offer, "reliability2")
        inet_down = _optional_finite_offer_number(offer, "inet_down")
        return (
            -reliability,
            consumer_rank,
            price,
            -inet_down,
            -cores,
            -ram,
            _offer_identity_number(offer, "machine_id"),
            _offer_identity_number(offer, "id"),
        )

    return min(banded, key=sort_key)


def validate_offer_for_purchase(
    offer: object,
    manifest: RunManifest,
    *,
    expected_offer_id: int | None = None,
) -> dict[str, Any]:
    """Validate the immutable identity and finite price immediately pre-create."""

    if not isinstance(offer, dict):
        raise OpsError("selected offer is not an object")
    try:
        offer_id = _strict_positive_identity(offer.get("id"), "offer ID")
        _strict_positive_identity(offer.get("machine_id"), "machine ID")
    except OpsError as error:
        raise OpsError(f"selected offer has invalid identity: {error}") from error
    if expected_offer_id is not None:
        expected_offer_id = _strict_positive_identity(
            expected_offer_id, "expected offer ID"
        )
        if offer_id != expected_offer_id:
            raise OpsError("selected offer identity changed before purchase")
    try:
        price = _finite_offer_number(offer, "dph_total")
    except OpsError as exc:
        raise OpsError(f"selected {exc}") from exc
    if price <= 0 or price > manifest.max_dph:
        raise OpsError("selected offer has a price outside the manifest limit")
    return offer


def create_source_bundle(manifest: RunManifest) -> tuple[Path, Path]:
    # Fail locally, before any paid instance is created, if the process that
    # will drive SSH cannot actually read its configured identity.  Checking
    # only the public half allowed sandboxed operators to rent candidates that
    # they could never authenticate to.
    require_local_ssh_identity()
    if not git_is_clean():
        raise OpsError("refusing remote launch from a dirty Git worktree")
    head = git_head()
    if manifest.source_commit != head:
        raise OpsError(
            f"manifest source_commit {manifest.source_commit} does not equal HEAD {head}"
        )
    stage = STAGING_DIR / manifest.run_id
    if stage.exists():
        resolved_stage = stage.resolve()
        resolved_root = STAGING_DIR.resolve()
        if resolved_root not in resolved_stage.parents:
            raise OpsError("unsafe staging path")
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    bundle = stage / "source.bundle"
    process = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={ROOT.as_posix()}",
            "bundle",
            "create",
            str(bundle),
            "HEAD",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        **hidden_subprocess_options(),
    )
    if process.returncode != 0:
        raise OpsError(
            "git bundle failed: " + _redacted_diagnostic_text(process.stderr.strip())
        )
    if manifest.input_artifacts:
        raise OpsError(
            "QuantExperiments remote input artifacts are disabled; "
            "jobs download their own data at runtime"
        )
    prepared_inputs: list[dict[str, str]] = []
    resolved_manifest = stage / "run_manifest.json"
    atomic_write_json(
        resolved_manifest,
        {
            **asdict(manifest),
            "prepared_at": utc_now(),
            "prepared_input_artifacts": prepared_inputs,
        },
    )
    return bundle, resolved_manifest


_REMOTE_WATCHDOG_SOURCE = """\
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

DEADLINE = __DEADLINE__
REQUEST_TIMEOUT = __REQUEST_TIMEOUT__
PROCESS_TAG = "qe-vast-watchdog-v1"
ARMED_PATH = "/workspace/quant_experiments_ops/watchdog_armed.json"


INSTANCE_ID = os.environ.get("CONTAINER_ID", "")
CONTAINER_KEY = os.environ.get("CONTAINER_API_KEY", "")
if (
    not INSTANCE_ID.isdigit()
    or int(INSTANCE_ID) <= 0
    or not CONTAINER_KEY
    or len(sys.argv) != 2
    or sys.argv[1] != PROCESS_TAG
):
    raise SystemExit("Vast container kill-switch credentials are unavailable")


def destroy_self():
    encoded_id = urllib.parse.quote(INSTANCE_ID, safe="")
    request = urllib.request.Request(
        "https://console.vast.ai/api/v0/instances/" + encoded_id + "/",
        method="DELETE",
        headers={"Authorization": "Bearer " + CONTAINER_KEY},
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT):
            return True
    except urllib.error.HTTPError as error:
        return error.code == 404 or 200 <= error.code < 300
    except (OSError, TimeoutError):
        return False


# Convert the one necessary cross-machine wall-clock comparison into a local
# monotonic deadline immediately. Later host clock adjustments cannot defer the
# kill switch; the early safety margin absorbs modest initial clock skew.
MONOTONIC_DEADLINE = time.monotonic() + max(0.0, DEADLINE - time.time())


def process_identity():
    stat_text = open("/proc/self/stat", encoding="ascii").read()
    closing = stat_text.rfind(")")
    fields = stat_text[closing + 2 :].split()
    if closing < 0 or len(fields) <= 19:
        raise OSError("could not read watchdog process identity")
    cmdline = open("/proc/self/cmdline", "rb").read()
    return fields[19], hashlib.sha256(cmdline).hexdigest()


def publish_ready():
    start_ticks, cmdline_sha256 = process_identity()
    payload = json.dumps(
        {
            "cmdline_sha256": cmdline_sha256,
            "deadline_epoch": DEADLINE,
            "instance_id": INSTANCE_ID,
            "pid": os.getpid(),
            "proc_start_ticks": start_ticks,
            "process_tag": PROCESS_TAG,
            "schema": "qe-vast-watchdog-ready-v1",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    temporary = ARMED_PATH + "." + str(os.getpid()) + "." + str(time.monotonic_ns())
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as marker:
            marker.write(payload)
            marker.flush()
            os.fsync(marker.fileno())
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    os.replace(temporary, ARMED_PATH)
    directory = os.open(os.path.dirname(ARMED_PATH), os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


armed = False
while True:
    remaining = MONOTONIC_DEADLINE - time.monotonic()
    if remaining > 0:
        if not armed:
            try:
                publish_ready()
            except OSError:
                time.sleep(min(__RETRY_SLEEP__, remaining))
                continue
            armed = True
        time.sleep(min(__SLEEP_CHUNK__, remaining))
        continue
    if destroy_self():
        break
    time.sleep(__RETRY_SLEEP__)
"""


_REMOTE_WATCHDOG_PROBE_SOURCE = """\
import hashlib
import json
import math
import os
import stat
import time

MARKER = "/workspace/quant_experiments_ops/watchdog_armed.json"
BOOTSTRAP = "/workspace/quant_experiments_ops/bootstrap_state"
EXPECTED_INSTANCE = __EXPECTED_INSTANCE__
EXPECTED_DEADLINE = __EXPECTED_DEADLINE__
PROCESS_TAG = b"qe-vast-watchdog-v1"
MAX_MARKER_BYTES = 4096


def read_bounded_regular(path):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_MARKER_BYTES:
            raise SystemExit(1)
        data = os.read(descriptor, MAX_MARKER_BYTES + 1)
        if len(data) > MAX_MARKER_BYTES:
            raise SystemExit(1)
        return data
    finally:
        os.close(descriptor)


def sample():
    payload = json.loads(read_bounded_regular(MARKER).decode("ascii"))
    if not isinstance(payload, dict) or payload.get("schema") != "qe-vast-watchdog-ready-v1":
        raise SystemExit(1)
    if payload.get("instance_id") != EXPECTED_INSTANCE:
        raise SystemExit(1)
    deadline = payload.get("deadline_epoch")
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)):
        raise SystemExit(1)
    if not math.isfinite(float(deadline)) or not math.isclose(
        float(deadline), EXPECTED_DEADLINE, rel_tol=0.0, abs_tol=1e-6
    ):
        raise SystemExit(1)
    pid = payload.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise SystemExit(1)
    proc_root = "/proc/" + str(pid)
    stat_text = open(proc_root + "/stat", encoding="ascii").read()
    closing = stat_text.rfind(")")
    fields = stat_text[closing + 2 :].split()
    if closing < 0 or len(fields) <= 19 or fields[0] in {"Z", "X", "x"}:
        raise SystemExit(1)
    if payload.get("proc_start_ticks") != fields[19]:
        raise SystemExit(1)
    cmdline = open(proc_root + "/cmdline", "rb").read()
    if PROCESS_TAG not in cmdline.split(b"\\0"):
        raise SystemExit(1)
    if payload.get("process_tag") != PROCESS_TAG.decode("ascii"):
        raise SystemExit(1)
    if payload.get("cmdline_sha256") != hashlib.sha256(cmdline).hexdigest():
        raise SystemExit(1)
    return payload


if open(BOOTSTRAP, encoding="ascii").read().strip() != "INSTANCE_READY":
    raise SystemExit(1)
first = sample()
time.sleep(1.0)
second = sample()
if first != second:
    raise SystemExit(1)
print("SSH_READY")
"""


def campaign_deadline_schedule(
    manifest: RunManifest,
    *,
    campaign_started_at: str,
    campaign_accrued_cost_usd: float = 0.0,
) -> dict[str, str]:
    """Build the immutable pre-create controller/watchdog deadline schedule."""

    started = parse_utc(campaign_started_at)
    if started.tzinfo is None:
        raise OpsError("campaign start must include a timezone")
    accrued = _manifest_number(
        campaign_accrued_cost_usd, "campaign_accrued_cost_usd"
    )
    if accrued < 0 or accrued >= manifest.max_total_usd:
        raise OpsError("campaign accrued cost leaves no positive spend budget")
    ttl_deadline = started + timedelta(hours=manifest.ttl_hours)
    spend_anchor = started if accrued == 0 else datetime.now(UTC)
    spend_deadline = spend_anchor + timedelta(
        seconds=(manifest.max_total_usd - accrued) / manifest.max_dph * 3600.0
    )
    provider = min(ttl_deadline, spend_deadline)
    watchdog = provider - timedelta(seconds=REMOTE_WATCHDOG_SAFETY_SECONDS)
    controller = watchdog - timedelta(
        seconds=LOCAL_WATCHDOG_ORDERING_MARGIN_SECONDS
    )
    cleanup = provider - timedelta(
        seconds=cleanup_reserve_hours(manifest.poll_seconds) * 3600.0
    )
    recovery = cleanup + timedelta(seconds=HARD_LIMIT_RECOVERY_RESERVE_SECONDS)
    if cleanup <= started or recovery >= controller:
        raise OpsError("campaign limits leave no positive pre-cleanup work window")

    def stamp(value: datetime) -> str:
        return value.isoformat().replace("+00:00", "Z")

    return {
        "provider_deadline_at": stamp(provider),
        "remote_watchdog_deadline_at": stamp(watchdog),
        "controller_deadline_at": stamp(controller),
        "cleanup_start_at": stamp(cleanup),
        "recovery_deadline_at": stamp(recovery),
    }


def _record_deadline_monotonic(
    record: dict[str, Any], field: str, *, fallback_seconds: float
) -> float:
    """Convert one persisted UTC boundary to a local monotonic deadline once."""

    raw = record.get(field)
    if raw is None:
        return time.monotonic() + max(0.0, fallback_seconds)
    deadline = parse_utc(str(raw))
    if deadline.tzinfo is None:
        raise OpsError(f"persisted {field} must include a timezone")
    return time.monotonic() + max(
        0.0, (deadline - datetime.now(UTC)).total_seconds()
    )


def remote_watchdog_deadline_epoch(
    offer: dict[str, Any],
    manifest: RunManifest,
    *,
    campaign_started_at: str | None = None,
) -> float:
    """Compute the provider-side kill-switch deadline from pre-create state.

    The local supervisor remains the authoritative early cleanup path.  This
    deadline is a deliberately earlier, in-instance fallback and uses only
    data already validated before the purchase request is sent.
    """

    # The quoted offer may be cheaper than the approved ceiling.  Using that
    # price here would arm the in-container fallback after the manifest's
    # actual spend boundary, so the deadline must use the conservative
    # manifest maximum regardless of the selected offer.
    del offer
    started = campaign_started_at or utc_now()
    schedule = campaign_deadline_schedule(
        manifest, campaign_started_at=started
    )
    return parse_utc(schedule["remote_watchdog_deadline_at"]).timestamp()


def remote_watchdog_onstart_command(deadline_epoch: float) -> str:
    """Return a fail-closed detached shell command for the in-instance kill switch."""

    if not math.isfinite(deadline_epoch):
        raise OpsError("remote watchdog deadline must be finite")
    source = (
        _REMOTE_WATCHDOG_SOURCE.replace("__DEADLINE__", repr(float(deadline_epoch)))
        .replace("__REQUEST_TIMEOUT__", repr(float(REMOTE_WATCHDOG_REQUEST_TIMEOUT_SECONDS)))
        .replace("__SLEEP_CHUNK__", repr(float(REMOTE_WATCHDOG_SLEEP_CHUNK_SECONDS)))
        .replace("__RETRY_SLEEP__", repr(float(REMOTE_WATCHDOG_SLEEP_CHUNK_SECONDS)))
    )
    encoded = base64.b64encode(source.encode("ascii")).decode("ascii")
    # The payload is base64-only, making the command stable across Vast's
    # shell/JSON argument layers.  The credential is read only inside the
    # container at runtime; no account-wide VAST_API_KEY is transferred.
    return (
        "command -v python3 >/dev/null 2>&1 || exit 125; "
        "rm -f /workspace/quant_experiments_ops/watchdog_armed.json; "
        "nohup python3 -c "
        f'"import base64;exec(base64.b64decode(\'{encoded}\'))" '
        "qe-vast-watchdog-v1 >/dev/null 2>&1 &"
    )


def remote_watchdog_probe_command(
    instance_id: int, expected_deadline_epoch: float
) -> str:
    """Build the two-sample remote identity/liveness readiness probe."""

    instance = _strict_positive_identity(instance_id, "watchdog instance ID")
    if not math.isfinite(expected_deadline_epoch):
        raise OpsError("expected watchdog deadline must be finite")
    source = (
        _REMOTE_WATCHDOG_PROBE_SOURCE.replace(
            "__EXPECTED_INSTANCE__", repr(str(instance))
        ).replace("__EXPECTED_DEADLINE__", repr(float(expected_deadline_epoch)))
    )
    encoded = base64.b64encode(source.encode("ascii")).decode("ascii")
    return (
        "python3 -c "
        f'"import base64;exec(base64.b64decode(\'{encoded}\'))"'
    )


def create_instance(
    offer: dict[str, Any],
    manifest: RunManifest,
    *,
    label: str | None = None,
    campaign_started_at: str | None = None,
    deadline_monotonic: float | None = None,
) -> int:
    validated_offer = validate_offer_for_purchase(offer, manifest)
    offer_id = _strict_positive_identity(validated_offer.get("id"), "offer ID")
    expected_label = label or f"quant-experiments-{manifest.run_id}"
    if not isinstance(expected_label, str) or not INSTANCE_LABEL_PATTERN.fullmatch(
        expected_label
    ):
        raise OpsError("instance label is malformed")
    watchdog_deadline = remote_watchdog_deadline_epoch(
        validated_offer,
        manifest,
        campaign_started_at=campaign_started_at,
    )
    onstart = (
        "set -eu; mkdir -p /workspace/quant_experiments_ops; "
        f"{remote_watchdog_onstart_command(watchdog_deadline)} "
        "echo INSTANCE_READY > "
        "/workspace/quant_experiments_ops/bootstrap_state"
    )
    result = run_vast(
        "create",
        "instance",
        offer_id,
        "--image",
        DEFAULT_IMAGE,
        "--disk",
        manifest.disk_gb,
        "--env",
        DEFAULT_TEMPLATE_ENV,
        "--onstart-cmd",
        onstart,
        "--label",
        expected_label,
        "--jupyter",
        "--ssh",
        "--direct",
        "--cancel-unavail",
        timeout=_deadline_timeout_seconds(deadline_monotonic, default=180),
    )
    if not isinstance(result, dict) or result.get("success") is not True:
        raise OpsError("instance creation failed with a malformed or negative response")
    return _strict_positive_identity(result.get("new_contract"), "instance ID")


def verify_created_instance(
    instance_id: int,
    manifest: RunManifest,
    *,
    expected_label: str,
    expected_offer: dict[str, Any] | None = None,
    deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    instance_id = _strict_positive_identity(instance_id, "instance ID")
    if not isinstance(expected_label, str) or not INSTANCE_LABEL_PATTERN.fullmatch(
        expected_label
    ):
        raise OpsError("expected instance label is malformed")
    deadline = min(
        time.monotonic() + 90,
        deadline_monotonic if deadline_monotonic is not None else float("inf"),
    )
    while time.monotonic() < deadline:
        live = _require_provider_instance_rows(
            run_vast(
                "show",
                "instances",
                timeout=_deadline_timeout_seconds(deadline, default=180),
            ),
            "created-instance verification query",
            require_label=True,
        )
        matches = [
            item
            for item in live
            if isinstance(item, dict)
            and _offer_identity_number(item, "id") == instance_id
        ]
        if matches:
            instance = matches[0]
            if instance.get("label") != expected_label:
                raise OpsError("created instance label does not match the requested campaign")
            # instance_id is the create-response ID for this exact offer purchase.
            _validate_live_contract_constraints(
                instance,
                manifest,
                expected_offer=expected_offer,
                create_response_instance_id=instance_id,
            )
            return instance
        _bounded_sleep(5, deadline)
    raise OpsError("created instance did not appear in the live Vast API")


def _normalized_hardware_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def instance_started_at(instance: dict[str, Any]) -> datetime:
    """Return Vast's original billable rental start, never adoption time."""

    value = instance.get("start_date")
    if value is None:
        raise OpsError("live instance did not report its original start_date")
    try:
        if isinstance(value, (int, float)) or re.fullmatch(
            r"[0-9]+(?:\.[0-9]+)?", str(value).strip()
        ):
            started = datetime.fromtimestamp(float(value), tz=UTC)
        else:
            started = parse_utc(str(value))
    except (ValueError, OverflowError) as error:
        raise OpsError("live instance start_date is invalid") from error
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    now = datetime.now(UTC)
    if (started - now).total_seconds() > 60:
        raise OpsError("live instance start_date is implausibly in the future")
    return started


def _require_live_number(
    instance: dict[str, Any], key: str, *, alternate: str | None = None
) -> float:
    value = instance.get(key)
    if value is None and alternate is not None:
        value = instance.get(alternate)
    if value is None:
        raise OpsError(f"live instance did not report {key}")
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise OpsError(f"live instance reported invalid {key}")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise OpsError(f"live instance reported invalid {key}") from error
    if not math.isfinite(numeric):
        raise OpsError(f"live instance reported non-finite {key}")
    return numeric


def _require_live_positive_dph(instance: dict[str, Any]) -> float:
    dph = _require_live_number(instance, "dph_total", alternate="dph_base")
    if dph <= 0:
        raise OpsError("live instance reported a non-positive hourly price")
    return dph


def _copy_reconciled_live_dph(
    target: dict[str, Any], instance: dict[str, Any]
) -> bool:
    """Copy only a finite positive live price into cleanup/reconcile state."""

    if instance.get("dph_total") is None and instance.get("dph_base") is None:
        return True
    try:
        target["dph"] = _require_live_positive_dph(instance)
    except OpsError:
        target["provider_dph_invalid"] = True
        return False
    target.pop("provider_dph_invalid", None)
    return True


def _require_live_exact_positive_int(
    instance: dict[str, Any], key: str
) -> int:
    """Read a live provider identity/count without lossy coercion."""

    if key not in instance:
        raise OpsError(f"live instance did not report {key}")
    return _strict_positive_identity(instance[key], f"live {key}")


def _validate_live_offer_identity(
    instance: dict[str, Any],
    expected_offer: dict[str, Any],
    *,
    create_response_instance_id: int | None = None,
) -> None:
    """Bind a live instance to the offer that was purchased.

    Vast's ``show instances`` rows do not carry the purchased offer/ask ID under
    any key (observed live 2026-11-02 on instance 58748479: none of the keys
    below were present, and no key mentioned 'offer' or 'contract'). The
    binding Vast does provide is the create response: PUT /asks/<offer_id>/
    returns the new instance ID. So when no offer identity is reported, the
    instance is accepted only if its own ``id`` equals the persisted
    create-response instance ID for that exact offer purchase. Any reported
    offer identity must still be unambiguous and match the selected offer.
    """

    expected_id = _strict_positive_identity(
        expected_offer.get("id"), "selected offer ID"
    )
    keys = ("ask_contract_id", "offer_id", "actual_offer_id", "contract_offer_id")
    present = [key for key in keys if instance.get(key) is not None]
    if not present:
        if create_response_instance_id is None:
            raise OpsError(
                "live purchased instance did not report an unambiguous ask/offer "
                "contract ID and no create-response instance ID binds it"
            )
        bound_id = _strict_positive_identity(
            create_response_instance_id, "create-response instance ID"
        )
        if _offer_identity_number(instance, "id") != bound_id:
            raise OpsError(
                "live instance ID does not match the create-response instance ID"
            )
        return
    values = {
        _strict_positive_identity(instance[key], f"live {key}") for key in present
    }
    if len(values) != 1:
        raise OpsError("live purchased instance reported ambiguous offer identities")
    actual_id = next(iter(values))
    if actual_id != expected_id:
        raise OpsError("live instance offer identity does not match selected offer")


def _validate_reconcilable_purchase_instance(
    record: dict[str, Any], instance: dict[str, Any]
) -> None:
    """Require one exact ordinary-purchase record and live provider identity.

    A label alone is not an authorization to destroy: a stale or maliciously
    reused label must not let reconciliation terminate another rental.
    """

    run_id = record.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise OpsError("purchase record run_id is absent or malformed")
    expected_label = f"quant-experiments-{run_id}"
    if record.get("expected_label") != expected_label:
        raise OpsError("persisted purchase label is not bound to its run_id")
    if instance.get("label") != expected_label:
        raise OpsError("live purchase label does not match its run_id")
    if "candidates" in record:
        raise OpsError("purchase reconciliation cannot use a race-owned record")
    lifecycle_origin = record.get("lifecycle_origin")
    if lifecycle_origin not in {None, "new_instance_purchase"}:
        raise OpsError("purchase reconciliation record has a non-purchase origin")
    expected_offer_id = _strict_positive_identity(
        record.get("offer_id"), "persisted offer ID"
    )
    expected_machine_id = _strict_positive_identity(
        record.get("machine_id"), "persisted machine ID"
    )
    live_machine_id = _strict_positive_identity(
        instance.get("machine_id"), "live machine ID"
    )
    _validate_live_offer_identity(
        instance,
        {"id": expected_offer_id},
        create_response_instance_id=record.get("create_response_instance_id"),
    )
    if live_machine_id != expected_machine_id:
        raise OpsError("live machine identity does not match persisted purchase")


def _is_reconcilable_purchase_instance(
    record: dict[str, Any], instance: dict[str, Any]
) -> bool:
    """Return whether strict purchase identity validation succeeds."""

    try:
        _validate_reconcilable_purchase_instance(record, instance)
    except OpsError:
        return False
    return True


def _validate_live_contract_constraints(
    instance: dict[str, Any],
    manifest: RunManifest,
    *,
    expected_offer: dict[str, Any] | None = None,
    create_response_instance_id: int | None = None,
) -> float:
    """Validate every declared/searchable host constraint after purchase.

    This helper is shared by launch verification and adoption so that a newly
    purchased contract cannot receive a weaker post-purchase check than an
    existing contract.  Identity fields are required where Vast exposes them;
    the expected offer's physical machine ID is always required because the
    search/race policy depends on it.
    """

    gpu_name = getattr(manifest, "gpu_name", None)
    if not isinstance(gpu_name, str) or not gpu_name.strip():
        raise OpsError("manifest does not declare a GPU name")
    if _normalized_hardware_name(instance.get("gpu_name")) != (
        _normalized_hardware_name(gpu_name)
    ):
        raise OpsError("live instance GPU does not match manifest gpu_name")
    num_gpus = _require_live_exact_positive_int(instance, "num_gpus")
    if num_gpus != _strict_positive_identity(manifest.num_gpus, "manifest num_gpus"):
        raise OpsError("live instance GPU count does not match manifest num_gpus")

    dph = _require_live_positive_dph(instance)
    if dph > float(manifest.max_dph) + 1e-9:
        raise OpsError(
            f"live instance price ${dph:.4f}/h exceeds cap "
            f"${float(manifest.max_dph):.4f}/h"
        )
    disk_gb = _require_live_number(instance, "disk_space")
    if disk_gb + 1e-9 < float(manifest.disk_gb):
        raise OpsError("live instance disk is below manifest disk_gb")
    cuda = _require_live_number(instance, "cuda_max_good")
    if bool(getattr(manifest, "require_cuda", True)) and cuda + 1e-9 < float(
        manifest.min_cuda
    ):
        raise OpsError("live instance CUDA capability is below manifest min_cuda")
    inet_down = _require_live_number(instance, "inet_down")
    if inet_down + 1e-9 < float(manifest.min_inet_down_mbps):
        raise OpsError("live instance download rate is below manifest minimum")
    cores = _require_live_number(instance, "cpu_cores_effective")
    if cores + 1e-9 < float(manifest.min_cpu_cores_effective):
        raise OpsError("live instance CPU cores are below manifest minimum")
    cpu_ram_mb = _require_live_number(instance, "cpu_ram")
    if cpu_ram_mb / 1024 + 1e-9 < float(manifest.min_cpu_ram_gb):
        raise OpsError("live instance CPU RAM is below manifest minimum")
    if _require_live_number(instance, "direct_port_count") < 1:
        raise OpsError("live instance has no direct port for guarded SSH")
    if bool(getattr(manifest, "require_consumer_cpu", False)) and _is_server_cpu(
        instance
    ):
        raise OpsError("live instance has a server CPU but a consumer CPU is required")
    required_geolocation = getattr(manifest, "required_geolocation", None)
    if required_geolocation is not None and instance.get("geolocation") != required_geolocation:
        raise OpsError(
            "live instance geolocation "
            f"{instance.get('geolocation')!r} does not match required "
            f"{required_geolocation!r}"
        )
    required_country = getattr(manifest, "required_country", None)
    if required_country is not None and not _matches_country(instance, required_country):
        raise OpsError(
            "live instance country does not match required country "
            f"{required_country!r}"
        )
    machine_id = _strict_positive_identity(
        instance.get("machine_id"), "live machine ID"
    )
    if expected_offer is not None:
        expected_machine_id = _strict_positive_identity(
            expected_offer.get("machine_id"), "selected machine ID"
        )
        if machine_id != expected_machine_id:
            raise OpsError("live instance physical machine does not match selected offer")
        expected_offer_id = _strict_positive_identity(
            expected_offer.get("id"), "selected offer ID"
        )
        if expected_offer_id:
            _validate_live_offer_identity(
                instance,
                expected_offer,
                create_response_instance_id=create_response_instance_id,
            )
    verified = instance.get("verified") is True or str(
        instance.get("verification") or ""
    ).lower() == "verified"
    if not verified:
        raise OpsError("live instance is not reported as a verified host")
    return dph


def validate_adoption_instance(
    instance: dict[str, Any],
    manifest: RunManifest,
    *,
    tracked_records: Iterable[dict[str, Any]],
    blacklisted_machine_ids: Iterable[int] = (),
) -> tuple[datetime, float, float]:
    """Strictly validate an existing live contract and its remaining budget.

    Returns the original start time, elapsed hours, and accrued cost. Missing
    contract fields fail closed because adoption cannot safely reconstruct the
    approved host predicate from an ambiguous live record.
    """

    try:
        instance_id = _strict_positive_identity(instance["id"], "live instance ID")
    except KeyError as error:
        raise OpsError("live instance has no valid instance ID") from error
    for record in tracked_records:
        raw_tracked_id = record.get("instance_id")
        if raw_tracked_id is None:
            continue
        tracked_id = _strict_positive_identity(
            raw_tracked_id, "persisted tracked instance ID"
        )
        if tracked_id == instance_id:
            raise OpsError("instance is already tracked by QuantExperiments")
    label = str(instance.get("label") or "").strip()
    if label:
        raise OpsError(
            f"instance has an existing label {label!r}; refusing conflicting adoption"
        )
    if str(instance.get("actual_status") or "").lower() not in {"running", "loaded"}:
        raise OpsError("existing instance is not in a running state")
    dph = _validate_live_contract_constraints(instance, manifest)
    machine_id = _strict_positive_identity(instance.get("machine_id"), "live machine ID")
    if machine_id in set(manifest.avoid_machine_ids) | set(blacklisted_machine_ids):
        raise OpsError("live instance is on an excluded physical machine")
    started = instance_started_at(instance)
    elapsed_hours = max(
        0.0,
        (datetime.now(UTC) - started).total_seconds() / 3600,
    )
    accrued_cost = elapsed_hours * dph
    trigger_hours = cleanup_trigger_hours(manifest, dph)
    if elapsed_hours >= trigger_hours:
        raise OpsError(
            "existing rental has exhausted the approved runtime after reserving "
            f"cleanup capacity (elapsed={elapsed_hours:.3f}h, "
            f"cleanup_trigger={trigger_hours:.3f}h, accrued=${accrued_cost:.3f})"
        )
    return started, elapsed_hours, accrued_cost


REMOTE_ADOPTION_PREFLIGHT_PROGRAM = r"""import json, os, pathlib, shutil, subprocess
candidate_paths = [
    "/workspace/quant-experiments",
    "/workspace/quant_experiments_ops",
    "/workspace/quant_experiments_stages",
    "/workspace/source.bundle",
    "/workspace/run_manifest.json",
]
occupied = [value for value in candidate_paths if pathlib.Path(value).exists()]
self_pids = {os.getpid(), os.getppid()}
project_processes = []
for entry in pathlib.Path("/proc").iterdir():
    if not entry.name.isdigit() or int(entry.name) in self_pids:
        continue
    try:
        command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        continue
    if "/workspace/quant-experiments" in command or "/vast/remote_runner.py" in command:
        project_processes.append({"pid": int(entry.name), "command": command[:300]})
gpu_processes = []
errors = []
if not shutil.which("nvidia-smi"):
    errors.append("nvidia-smi unavailable")
else:
    probe = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    if probe.returncode != 0:
        errors.append("nvidia-smi compute-process query failed")
    else:
        gpu_processes = [
            line.strip()
            for line in probe.stdout.splitlines()
            if line.strip() and "No running processes" not in line
        ]
result = {
    "idle": not occupied and not project_processes and not gpu_processes and not errors,
    "occupied_paths": occupied,
    "project_processes": project_processes,
    "gpu_processes": gpu_processes,
    "errors": errors,
}
print(json.dumps(result, sort_keys=True))"""


def verify_adoption_host_idle(
    connection: tuple[str, str, int],
) -> dict[str, Any]:
    """Read-only proof that adoption will not overwrite a host or workload."""

    command = " ".join(
        shlex.quote(argument)
        for argument in ["python3", "-c", REMOTE_ADOPTION_PREFLIGHT_PROGRAM]
    )
    process = remote_exec(connection, command, check=False, timeout=45)
    if type(getattr(process, "returncode", None)) is not int or process.returncode != 0:
        raise OpsError("existing instance blank-host preflight command failed")
    stdout = getattr(process, "stdout", None)
    if not isinstance(stdout, str):
        raise OpsError("existing instance blank-host preflight returned invalid JSON")
    try:
        stdout_bytes = len(stdout.encode("utf-8"))
    except UnicodeError as error:
        raise OpsError(
            "existing instance blank-host preflight returned invalid JSON"
        ) from error
    if (
        stdout_bytes > _MAX_REMOTE_PREFLIGHT_BYTES
        or not _json_structure_within_diagnostic_limits(stdout)
    ):
        raise OpsError(
            "existing instance blank-host preflight returned malformed or oversized JSON"
        )
    try:
        report = json.loads(stdout, object_pairs_hook=_unique_json_object)
    except (json.JSONDecodeError, _DiagnosticSanitizationError) as error:
        raise OpsError("existing instance blank-host preflight returned invalid JSON") from error
    if not isinstance(report, dict) or report.get("idle") is not True:
        raise OpsError("existing instance is not blank and idle")
    return {"idle": True}


def label_adopted_instance(instance_id: int, run_id: str) -> str:
    """Label a recorded adoption and verify the live API reflects it."""

    instance_id = _strict_positive_identity(instance_id, "instance ID")
    label = f"quant-experiments-{run_id}"
    run_vast("label", "instance", instance_id, label, raw=False)
    live = _require_provider_instance_rows(
        run_vast("show", "instances"),
        "adopted-instance label query",
        require_label=True,
    )
    matches = [
        item
        for item in live
        if isinstance(item, dict)
        and _offer_identity_number(item, "id") == instance_id
    ]
    if len(matches) != 1 or matches[0].get("label") != label:
        raise OpsError("adopted instance label could not be verified in the live API")
    return label


def _require_ssh_connection(
    connection: tuple[str, str, int],
) -> tuple[str, str, int]:
    """Validate provider-derived OpenSSH operands before local command use."""

    if not isinstance(connection, tuple) or len(connection) != 3:
        raise OpsError("Vast returned an invalid SSH connection")
    user, host, port = connection
    if not isinstance(user, str) or not SSH_USER_PATTERN.fullmatch(user):
        raise OpsError("Vast returned an invalid SSH connection")
    if not isinstance(host, str) or not SSH_HOST_PATTERN.fullmatch(host):
        raise OpsError("Vast returned an invalid SSH connection")
    host_labels = host.split(".")
    if any(
        not label
        or len(label) > 63
        or label.startswith("-")
        or label.endswith("-")
        for label in host_labels
    ):
        raise OpsError("Vast returned an invalid SSH connection")
    if type(port) is not int or not 1 <= port <= 65_535:
        raise OpsError("Vast returned an invalid SSH connection")
    return user, host, port


def parse_ssh_url(value: str) -> tuple[str, str, int]:
    if not isinstance(value, str):
        raise OpsError("Vast returned an unparseable SSH URL")
    match = SSH_URL_PATTERN.fullmatch(value.strip())
    if not match:
        raise OpsError("Vast returned an unparseable SSH URL")
    user, host, port = match.groups()
    return _require_ssh_connection((user, host, int(port)))


def ssh_url(instance_id: int, *, timeout: int = 180) -> tuple[str, str, int]:
    value = run_vast("ssh-url", instance_id, raw=False, timeout=timeout).strip()
    return parse_ssh_url(value)


def _deadline_timeout_seconds(
    deadline_monotonic: float | None,
    *,
    default: int,
) -> int:
    """Return a positive subprocess timeout bounded by an absolute deadline."""

    if deadline_monotonic is None:
        return default
    remaining = deadline_monotonic - time.monotonic()
    if remaining < 1:
        raise OpsError("the bounded recovery deadline has elapsed")
    # Floor rather than ceil: a subprocess timeout must fit *inside* its lease.
    # The one-second admission floor also avoids starting a control-plane call
    # that is guaranteed to outlive the remaining window.
    return max(1, min(default, math.floor(remaining)))


def _bounded_sleep(seconds: float, deadline_monotonic: float | None) -> None:
    """Sleep without crossing an optional absolute monotonic deadline."""

    delay = max(0.0, float(seconds))
    if deadline_monotonic is not None:
        delay = min(delay, max(0.0, deadline_monotonic - time.monotonic()))
    if delay > 0:
        time.sleep(delay)


def _forget_known_host(host: str, port: int) -> None:
    """Drop a stale key for a reused Vast host:port before a fresh instance.

    Vast recycles proxy/direct endpoints across unrelated instances.  With
    ``StrictHostKeyChecking=accept-new`` a recycled endpoint presenting a new
    key is refused ("host identification has changed"), wasting the rental.
    Each new instance is therefore trust-on-first-use for its own endpoint.
    """

    if not KNOWN_HOSTS.exists():
        return
    keygen = shutil.which("ssh-keygen")
    if keygen is None:
        return
    target = host if int(port) == 22 else f"[{host}]:{int(port)}"
    try:
        subprocess.run(
            [keygen, "-R", target, "-f", str(KNOWN_HOSTS)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            **hidden_subprocess_options(),
        )
    except (OSError, subprocess.SubprocessError):
        pass


def known_hosts_option() -> str:
    """Quote the path for OpenSSH's own option parser.

    ``subprocess`` preserves an argument containing spaces, but OpenSSH parses
    ``UserKnownHostsFile`` as a configuration token list a second time.  Inner
    quotes are therefore required for repositories such as ``Python Projects``.
    Forward slashes avoid an additional Windows backslash parsing edge case.
    """

    return f'UserKnownHostsFile="{KNOWN_HOSTS.as_posix()}"'


def ssh_base(connection: tuple[str, str, int]) -> list[str]:
    user, host, port = _require_ssh_connection(connection)
    KNOWN_HOSTS.parent.mkdir(parents=True, exist_ok=True)
    return [
        shutil.which("ssh") or "ssh",
        "-i",
        str(SSH_PRIVATE_KEY),
        "-p",
        str(port),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=30",
        "-o",
        "ConnectionAttempts=3",
        "-o",
        "ServerAliveInterval=5",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        known_hosts_option(),
        "--",
        f"{user}@{host}",
    ]


def local_ssh_public_key() -> str:
    """Return the public key matching the identity used by :func:`ssh_base`."""

    try:
        value = SSH_PUBLIC_KEY.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise OpsError(f"local SSH public key is unavailable: {SSH_PUBLIC_KEY}") from error
    parts = value.split()
    if len(parts) < 2 or parts[0] not in {
        "ssh-ed25519",
        "ssh-rsa",
        "ecdsa-sha2-nistp256",
        "ecdsa-sha2-nistp384",
        "ecdsa-sha2-nistp521",
    }:
        raise OpsError("local SSH public key is malformed")
    if "PRIVATE KEY" in value or "\n" in value or "\r" in value:
        raise OpsError("local SSH public key must be one public-key line")
    return value


def require_local_ssh_identity() -> None:
    """Verify that this process can use both halves of the SSH identity.

    Read only one byte from the private key.  This validates filesystem access
    without ever loading or exposing the key material unnecessarily.
    """

    local_ssh_public_key()
    try:
        with SSH_PRIVATE_KEY.open("rb") as stream:
            if not stream.read(1):
                raise OpsError(f"local SSH private key is empty: {SSH_PRIVATE_KEY}")
    except OpsError:
        raise
    except OSError as error:
        raise OpsError(
            f"local SSH private key is unreadable: {SSH_PRIVATE_KEY}"
        ) from error


def attach_local_ssh_key(
    instance_id: int, *, deadline_monotonic: float | None = None
) -> None:
    """Explicitly authorize the exact local identity on one Vast instance."""

    # Vast's --ssh creation flag has intermittently failed to inject the
    # account-default key.  Per-instance attachment is idempotent and ensures
    # the key accepted by the host matches the private key OpenSSH will offer.
    run_vast(
        "attach",
        "ssh",
        instance_id,
        local_ssh_public_key(),
        raw=False,
        timeout=_deadline_timeout_seconds(deadline_monotonic, default=180),
    )


def remote_exec(
    connection: tuple[str, str, int],
    command: str,
    *,
    check: bool = True,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(
        [*ssh_base(connection), command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        **hidden_subprocess_options(),
    )
    if check and process.returncode != 0:
        raise OpsError(
            "remote command failed: "
            + _redacted_diagnostic_text(process.stderr.strip())
        )
    return process


def wait_for_ssh(
    instance_id: int,
    timeout_seconds: int = DEFAULT_SSH_READY_TIMEOUT_SECONDS,
    *,
    cancel_event: threading.Event | None = None,
    require_watchdog_armed: bool = False,
    expected_watchdog_deadline_epoch: float | None = None,
    deadline_monotonic: float | None = None,
) -> tuple[str, str, int]:
    if require_watchdog_armed and expected_watchdog_deadline_epoch is None:
        raise OpsError("watchdog readiness requires its exact expected deadline")
    attach_local_ssh_key(
        instance_id, deadline_monotonic=deadline_monotonic
    )
    deadline = min(
        time.monotonic() + timeout_seconds,
        deadline_monotonic if deadline_monotonic is not None else float("inf"),
    )
    last_error = "SSH URL not available"
    forgotten_endpoints: set[tuple[str, int]] = set()
    while time.monotonic() < deadline and not (
        cancel_event is not None and cancel_event.is_set()
    ):
        try:
            connection = ssh_url(
                instance_id,
                timeout=_deadline_timeout_seconds(deadline, default=180),
            )
            endpoint = (str(connection[1]), int(connection[2]))
            if endpoint not in forgotten_endpoints:
                forgotten_endpoints.add(endpoint)
                _forget_known_host(*endpoint)
            probe_command = "echo SSH_READY"
            if require_watchdog_armed:
                probe_command = remote_watchdog_probe_command(
                    instance_id,
                    float(expected_watchdog_deadline_epoch),
                )
            probe = remote_exec(
                connection,
                probe_command,
                check=False,
                timeout=_deadline_timeout_seconds(deadline, default=30),
            )
            if probe.returncode == 0 and probe.stdout.strip() == "SSH_READY":
                return connection
            last_error = (
                _redacted_diagnostic_text(probe.stderr.strip())
                if probe.stderr.strip()
                else "SSH probe failed"
            )
        except Exception as error:  # transient boot/route failures are expected
            last_error = _redacted_exception_text(error)
        if cancel_event is None:
            _bounded_sleep(15, deadline)
        else:
            cancel_event.wait(min(15, max(0.0, deadline - time.monotonic())))
    if cancel_event is not None and cancel_event.is_set():
        raise OpsError("SSH readiness cancelled after another race candidate won")
    raise OpsError(f"SSH did not become ready: {last_error}")


def verify_cuda_readiness(
    connection: tuple[str, str, int], manifest: RunManifest
) -> dict[str, Any]:
    """Run a lightweight, non-training CUDA probe before a race winner starts."""

    if not manifest.require_cuda:
        return {"cuda_ready": True, "skipped": True}
    torch_probe = (
        "import json,sys,torch; "
        "count=torch.cuda.device_count() if torch.cuda.is_available() else 0; "
        "print(json.dumps({'torch_cuda': torch.version.cuda, 'device_count': count})); "
        f"raise SystemExit(0 if count >= {manifest.num_gpus} else 1)"
    )
    quoted_torch_probe = shlex.quote(torch_probe)
    command = (
        "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader,nounits "
        "&& if test -x /venv/main/bin/python; then "
        f"/venv/main/bin/python -c {quoted_torch_probe}; "
        f"else python3 -c {quoted_torch_probe}; fi"
    )
    probe = remote_exec(connection, command, check=False, timeout=45)
    if probe.returncode != 0 or not probe.stdout.strip():
        raise OpsError(
            "lightweight CUDA readiness check failed: "
            + _redacted_diagnostic_text(probe.stderr.strip())[:1000]
        )
    try:
        report = json.loads(probe.stdout.strip().splitlines()[-1])
    except (TypeError, json.JSONDecodeError) as error:
        raise OpsError("lightweight CUDA readiness check returned invalid JSON") from error
    if (
        not isinstance(report, dict)
        or int(report.get("device_count", 0)) < manifest.num_gpus
    ):
        raise OpsError(
            "lightweight CUDA readiness check found fewer devices than manifest num_gpus"
        )
    cuda_version = report.get("torch_cuda")
    try:
        if cuda_version is not None and float(str(cuda_version)) + 1e-9 < manifest.min_cuda:
            raise OpsError(
                f"torch CUDA {cuda_version} is below manifest min_cuda {manifest.min_cuda}"
            )
    except ValueError as error:
        raise OpsError("torch reported an invalid CUDA version") from error
    report["cuda_ready"] = True
    report["nvidia_smi"] = "\n".join(probe.stdout.strip().splitlines()[:-1])
    safe_report = _redacted_json_value(report)
    if not isinstance(safe_report, dict):
        return {"cuda_ready": True, "diagnostic": _DIAGNOSTIC_WITHHELD}
    return safe_report


def scp_upload(
    connection: tuple[str, str, int],
    local_paths: Iterable[Path],
    remote: str,
    *,
    timeout_seconds: int = 600,
) -> None:
    user, host, port = _require_ssh_connection(connection)
    command = [
        shutil.which("scp") or "scp",
        "-i",
        str(SSH_PRIVATE_KEY),
        "-P",
        str(port),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "ConnectTimeout=30",
        "-o",
        "ConnectionAttempts=3",
        "-o",
        "ServerAliveInterval=5",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        known_hosts_option(),
        "--",
        *[str(path) for path in local_paths],
        f"{user}@{host}:{remote}",
    ]
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        **hidden_subprocess_options(),
    )
    if process.returncode != 0:
        raise OpsError(
            "SCP upload failed: "
            + _redacted_diagnostic_text(process.stderr.strip())
        )


def scp_download(
    connection: tuple[str, str, int],
    remote: str,
    local: Path,
    *,
    timeout_seconds: int = 1200,
) -> bool:
    user, host, port = _require_ssh_connection(connection)
    local.parent.mkdir(parents=True, exist_ok=True)
    command = [
        shutil.which("scp") or "scp",
        "-i",
        str(SSH_PRIVATE_KEY),
        "-P",
        str(port),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        known_hosts_option(),
        "--",
        f"{user}@{host}:{remote}",
        str(local),
    ]
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        **hidden_subprocess_options(),
    )
    return process.returncode == 0


def start_remote_run(
    connection: tuple[str, str, int],
    bundle: Path,
    manifest_path: Path,
    *,
    deadline_monotonic: float | None = None,
    workload_stop_at: str | None = None,
) -> None:
    resolved_manifest = read_json(manifest_path, None)
    if not isinstance(resolved_manifest, dict):
        raise OpsError("resolved launch manifest is malformed")
    prepared_inputs = resolved_manifest.get("prepared_input_artifacts", [])
    if not isinstance(prepared_inputs, list):
        raise OpsError("resolved input artifact list is malformed")
    if prepared_inputs or resolved_manifest.get("input_artifacts"):
        raise OpsError("resolved manifest attempted a forbidden input artifact upload")
    if workload_stop_at is not None:
        # Anchor the remote workload deadline to the persisted campaign
        # schedule (purchase time), not to whenever the runner happens to
        # start after provisioning; the runner uses the earlier of the two.
        stop_at = parse_utc(str(workload_stop_at))
        if stop_at.tzinfo is None:
            raise OpsError("workload_stop_at must include a timezone")
        resolved_manifest["workload_stop_at_epoch"] = (
            stop_at.timestamp() - REMOTE_PACKAGING_MARGIN_SECONDS
        )
        atomic_write_json(manifest_path, resolved_manifest)
    remote_exec(
        connection,
        f"mkdir -p {REMOTE_OPS}",
        timeout=_deadline_timeout_seconds(deadline_monotonic, default=120),
    )
    upload_paths = [bundle, manifest_path]
    for entry in prepared_inputs:
        if not isinstance(entry, dict) or not isinstance(entry.get("staged_name"), str):
            raise OpsError("resolved input artifact entry is malformed")
        upload_paths.append(manifest_path.parent / entry["staged_name"])
    scp_upload(
        connection,
        tuple(upload_paths),
        f"{REMOTE_ROOT}/",
        timeout_seconds=_deadline_timeout_seconds(
            deadline_monotonic, default=600
        ),
    )
    command = (
        f"bash {REMOTE_REPO}/vast/remote_bootstrap.sh "
        f"{REMOTE_ROOT}/source.bundle {REMOTE_ROOT}/run_manifest.json"
    )
    # The bootstrap script lives inside the bundle, so clone before invoking it.
    clone = (
        f"test ! -e {REMOTE_REPO} && "
        f"git clone {REMOTE_ROOT}/source.bundle {REMOTE_REPO}"
    )
    remote_exec(
        connection,
        clone,
        timeout=_deadline_timeout_seconds(deadline_monotonic, default=1200),
    )
    for entry in prepared_inputs:
        relative = PurePosixPath(entry["path"])
        destination = f"{REMOTE_REPO}/" + relative.as_posix()
        staged = f"{REMOTE_ROOT}/{entry['staged_name']}"
        expected = entry["sha256"]
        install = (
            f"mkdir -p {shlex.quote(str(PurePosixPath(destination).parent))} && "
            f"test $(sha256sum {shlex.quote(staged)} | cut -d' ' -f1) = "
            f"{shlex.quote(expected)} && "
            f"cp {shlex.quote(staged)} {shlex.quote(destination)}"
        )
        remote_exec(
            connection,
            install,
            timeout=_deadline_timeout_seconds(deadline_monotonic, default=1200),
        )
    remote_exec(
        connection,
        command,
        timeout=_deadline_timeout_seconds(deadline_monotonic, default=1200),
    )


def start_remote_session_stage(
    connection: tuple[str, str, int],
    bundle: Path,
    manifest_path: Path,
    stage_id: str,
) -> tuple[str, str]:
    """Upload and start a new SHA-pinned checkout on an existing session host."""

    resolved_manifest = read_json(manifest_path, None)
    if not isinstance(resolved_manifest, dict):
        raise OpsError("resolved stage manifest is malformed")
    prepared_inputs = resolved_manifest.get("prepared_input_artifacts", [])
    if not isinstance(prepared_inputs, list):
        raise OpsError("resolved stage input artifact list is malformed")
    if prepared_inputs or resolved_manifest.get("input_artifacts"):
        raise OpsError("resolved stage manifest attempted a forbidden input artifact upload")
    stage_root = f"{REMOTE_STAGES}/{stage_id}"
    stage_repo = f"{stage_root}/repo"
    stage_ops = f"{stage_root}/ops"
    remote_exec(
        connection,
        f"test ! -e {shlex.quote(stage_root)} && mkdir -p {shlex.quote(stage_root)}",
    )
    upload_paths = [bundle, manifest_path]
    for entry in prepared_inputs:
        if not isinstance(entry, dict) or not isinstance(entry.get("staged_name"), str):
            raise OpsError("resolved stage input artifact entry is malformed")
        upload_paths.append(manifest_path.parent / entry["staged_name"])
    scp_upload(connection, tuple(upload_paths), f"{stage_root}/")
    remote_exec(
        connection,
        f"git clone {shlex.quote(stage_root + '/source.bundle')} "
        f"{shlex.quote(stage_repo)}",
        timeout=1200,
    )
    for entry in prepared_inputs:
        relative = PurePosixPath(entry["path"])
        destination = f"{stage_repo}/" + relative.as_posix()
        staged = f"{stage_root}/{entry['staged_name']}"
        expected = entry["sha256"]
        install = (
            f"mkdir -p {shlex.quote(str(PurePosixPath(destination).parent))} && "
            f"test $(sha256sum {shlex.quote(staged)} | cut -d' ' -f1) = "
            f"{shlex.quote(expected)} && cp {shlex.quote(staged)} "
            f"{shlex.quote(destination)}"
        )
        remote_exec(connection, install, timeout=1200)
    python_choice = (
        "$(if [ -x /venv/main/bin/python ]; then "
        "printf /venv/main/bin/python; else command -v python3; fi)"
    )
    launch = (
        f"mkdir -p {shlex.quote(stage_ops)} && PYTHON_BIN={python_choice} && "
        f"$PYTHON_BIN -m pip install --disable-pip-version-check -e "
        f"{shlex.quote(stage_repo)} > {shlex.quote(stage_ops + '/bootstrap.log')} "
        f"2>&1 && (nohup $PYTHON_BIN "
        f"{shlex.quote(stage_repo + '/vast/remote_runner.py')} --manifest "
        f"{shlex.quote(stage_root + '/run_manifest.json')} --ops-dir "
        f"{shlex.quote(stage_ops)} --repo {shlex.quote(stage_repo)} >> "
        f"{shlex.quote(stage_ops + '/bootstrap.log')} 2>&1 < /dev/null & "
        f"echo $! > {shlex.quote(stage_ops + '/runner.pid')})"
    )
    remote_exec(connection, launch, timeout=1200)
    return stage_repo, stage_ops


REMOTE_EVENT_READ_PROGRAM = r"""import json, os, stat, sys
path = sys.argv[1]
limit = int(sys.argv[2])
withheld = {"event": "remote_event_withheld", "diagnostic": "[DIAGNOSTIC WITHHELD]"}
flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
try:
    descriptor = os.open(path, flags)
except FileNotFoundError:
    raise SystemExit(0)
try:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > limit:
        print(json.dumps(withheld, sort_keys=True))
        raise SystemExit(0)
    chunks = []
    remaining = limit + 1
    while remaining > 0:
        chunk = os.read(descriptor, min(65536, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    data = b"".join(chunks)
    if len(data) > limit or os.read(descriptor, 1):
        print(json.dumps(withheld, sort_keys=True))
    else:
        sys.stdout.buffer.write(data)
finally:
    os.close(descriptor)"""


def _json_structure_within_limits(
    value: str,
    *,
    max_depth: int,
    max_nodes: int,
) -> bool:
    """Cheaply reject pathological JSON structure before ``json.loads``."""

    depth = 0
    nodes = 1
    in_string = False
    escaped = False
    for character in value:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            nodes += 1
            if depth > max_depth:
                return False
        elif character in "]}":
            depth -= 1
            if depth < 0:
                return False
        elif character == ",":
            nodes += 1
        if nodes > max_nodes:
            return False
    return not in_string and not escaped and depth == 0


def _json_structure_within_diagnostic_limits(value: str) -> bool:
    return _json_structure_within_limits(
        value,
        max_depth=_MAX_DIAGNOSTIC_JSON_DEPTH,
        max_nodes=_MAX_DIAGNOSTIC_JSON_NODES,
    )


def _json_structure_within_provider_limits(value: str) -> bool:
    return _json_structure_within_limits(
        value,
        max_depth=_MAX_PROVIDER_JSON_DEPTH,
        max_nodes=_MAX_PROVIDER_JSON_NODES,
    )


def read_remote_events(
    connection: tuple[str, str, int],
    *,
    remote_ops: str = REMOTE_OPS,
    deadline_monotonic: float | None = None,
) -> list[dict[str, Any]]:
    event_path = str(PurePosixPath(remote_ops) / "events.jsonl")
    arguments = [
        "python3",
        "-c",
        REMOTE_EVENT_READ_PROGRAM,
        event_path,
        str(_MAX_REMOTE_EVENT_BYTES),
    ]
    process = remote_exec(
        connection,
        " ".join(shlex.quote(argument) for argument in arguments),
        check=False,
        timeout=_deadline_timeout_seconds(deadline_monotonic, default=120),
    )
    if (
        type(getattr(process, "returncode", None)) is not int
        or process.returncode != 0
        or not isinstance(process.stdout, str)
        or len(process.stdout) > _MAX_REMOTE_EVENT_BYTES
    ):
        return [
            {
                "event": "remote_event_withheld",
                "diagnostic": _DIAGNOSTIC_WITHHELD,
            }
        ]
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(io.StringIO(process.stdout), start=1):
        if line_number > _MAX_DIAGNOSTIC_JSON_ITEMS:
            return [
                {
                    "event": "remote_event_withheld",
                    "diagnostic": _DIAGNOSTIC_WITHHELD,
                }
            ]
        line = line.rstrip("\r\n")
        if not _json_structure_within_diagnostic_limits(line):
            return [
                {
                    "event": "remote_event_withheld",
                    "diagnostic": _DIAGNOSTIC_WITHHELD,
                }
            ]
        try:
            value = json.loads(line, object_pairs_hook=_unique_json_object)
        except Exception:
            return [
                {
                    "event": "remote_event_withheld",
                    "diagnostic": _DIAGNOSTIC_WITHHELD,
                }
            ]
        if not isinstance(value, dict):
            return [
                {
                    "event": "remote_event_withheld",
                    "diagnostic": _DIAGNOSTIC_WITHHELD,
                }
            ]
        events.append(value)
    return _sanitized_remote_events(events)


def _strict_remote_success_evidence(
    events: list[dict[str, Any]], expected_run_id: str
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Return the sole ordered ready/success pair for one exact remote run."""

    if (
        not events
        or not isinstance(expected_run_id, str)
        or not RUN_ID_PATTERN.fullmatch(expected_run_id)
    ):
        return None
    latest = events[-1]
    if any(event.get("run_id") != expected_run_id for event in events):
        return None
    if (
        latest.get("event") != "run_succeeded"
        or latest.get("run_id") != expected_run_id
    ):
        return None
    terminal_names = {
        "artifact_packaging_failed",
        "bootstrap_failed",
        "remote_event_withheld",
        "run_failed",
        "run_succeeded",
        "runner_exception",
        "ttl_reached",
    }
    terminal_events = [
        event for event in events if event.get("event") in terminal_names
    ]
    if len(terminal_events) != 1 or terminal_events[0] is not latest:
        return None
    if any(
        event.get("event") == "command_finished"
        and (type(event.get("exit_code")) is not int or event.get("exit_code") != 0)
        for event in events
    ):
        return None
    ready_events = [
        event for event in events[:-1] if event.get("event") == "artifacts_ready"
    ]
    if len(ready_events) != 1:
        return None
    ready = ready_events[0]
    if (
        ready.get("run_id") != expected_run_id
        or type(ready.get("exit_code")) is not int
        or ready.get("exit_code") != 0
    ):
        return None
    return ready, latest


# Kept as a Python one-liner payload rather than a shell pipeline so diagnostics
# can be invoked safely on paths already validated by RunManifest.  It performs
# no writes and emits one JSON object even when a particular host tool is absent.
REMOTE_DIAGNOSTICS_PROGRAM = r"""import fnmatch, json, os, pathlib, shutil, stat, subprocess, sys, time
roots = [pathlib.Path(value) for value in sys.argv[1:-2]]
tail_lines = int(sys.argv[-2])
tail_bytes = int(sys.argv[-1])
result = {
    "event": "diagnostics_snapshot",
    "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
gpu_fields = (
    "index,name,utilization.gpu,memory.used,memory.total,power.draw,power.limit,"
    "temperature.gpu"
)
if shutil.which("nvidia-smi"):
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=" + gpu_fields, "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    if proc.returncode == 0:
        names = [
            "index",
            "name",
            "utilization_gpu_pct",
            "memory_used_mib",
            "memory_total_mib",
            "power_draw_w",
            "power_limit_w",
            "temperature_c",
        ]
        result["gpus"] = [
            dict(zip(names, [part.strip() for part in line.split(",")]))
            for line in proc.stdout.splitlines()
            if line.strip()
        ]
    else:
        result["gpu_error"] = proc.stderr.strip()[:500]
else:
    result["gpu_error"] = "nvidia-smi unavailable"
try:
    mem = {}
    for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        mem[key] = int(value.strip().split()[0]) * 1024
    result["host"] = {
        "cpu_logical_cores": os.cpu_count(),
        "loadavg_1_5_15": list(os.getloadavg()),
        "memory_total_bytes": mem.get("MemTotal"),
        "memory_available_bytes": mem.get("MemAvailable"),
    }
except (OSError, ValueError):
    result["host"] = {"cpu_logical_cores": os.cpu_count()}
metric_files = []
scanned = 0
for root in roots:
    try:
        root_mode = root.lstat().st_mode
    except OSError:
        continue
    if not stat.S_ISDIR(root_mode):
        continue
    pending = [root]
    while pending and scanned < 4096 and len(metric_files) < 4:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    scanned += 1
                    if scanned > 4096 or len(metric_files) >= 4:
                        break
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            pending.append(pathlib.Path(entry.path))
                        elif entry.is_file(follow_symlinks=False) and fnmatch.fnmatch(
                            entry.name, "*metrics*.jsonl"
                        ):
                            metric_files.append(pathlib.Path(entry.path))
                    except OSError:
                        pass
        except OSError:
            pass
    if scanned > 4096 or len(metric_files) >= 4:
        break
metric_tails = {}
for index, path in enumerate(metric_files[:4], start=1):
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise OSError("not a regular file")
            start = max(0, metadata.st_size - tail_bytes)
            os.lseek(descriptor, start, os.SEEK_SET)
            chunks = []
            remaining = tail_bytes
            while remaining > 0:
                chunk = os.read(descriptor, min(65536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
        finally:
            os.close(descriptor)
        if start:
            boundary = data.find(b"\n")
            data = b"" if boundary < 0 else data[boundary + 1:]
        lines = data.decode("utf-8", errors="replace").splitlines()[-tail_lines:]
        parsed = []
        for line in lines:
            try:
                parsed.append(json.loads(line) if line.lstrip().startswith("{") else line)
            except json.JSONDecodeError:
                parsed.append(line)
        metric_tails[f"metric_{index}"] = parsed
    except OSError:
        metric_tails[f"metric_{index}"] = {"error": "unavailable"}
result["declared_metrics_tail"] = metric_tails
print(json.dumps(result, sort_keys=True))"""


REMOTE_BOUNDED_TEXT_TAIL_PROGRAM = r"""import os, stat, sys
path = sys.argv[1]
line_limit = int(sys.argv[2])
byte_limit = int(sys.argv[3])
flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
try:
    descriptor = os.open(path, flags)
except FileNotFoundError:
    raise SystemExit(0)
except OSError:
    raise SystemExit(2)
try:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise SystemExit(2)
    start = max(0, metadata.st_size - byte_limit)
    os.lseek(descriptor, start, os.SEEK_SET)
    chunks = []
    remaining = byte_limit
    while remaining > 0:
        chunk = os.read(descriptor, min(65536, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    data = b"".join(chunks)
finally:
    os.close(descriptor)
if start:
    boundary = data.find(b"\n")
    data = b"" if boundary < 0 else data[boundary + 1:]
lines = data.splitlines()
sys.stdout.buffer.write(b"\n".join(lines[-line_limit:]))"""


def diagnostic_command(
    manifest: RunManifest, metrics_tail: int, *, repo_root: str = REMOTE_REPO
) -> str:
    """Build a read-only remote inspection command for declared artifact roots."""

    if metrics_tail < 1 or metrics_tail > 100:
        raise OpsError("metrics_tail must be in [1, 100]")
    roots = [
        str(PurePosixPath(repo_root) / PurePosixPath(path.replace("\\", "/")))
        for path in manifest.artifact_paths
    ]
    # Python itself validates the roots supplied from the already-validated
    # manifest.  Quote every argument because SSH executes a remote shell.
    arguments = [
        "python3",
        "-c",
        REMOTE_DIAGNOSTICS_PROGRAM,
        *roots,
        str(metrics_tail),
        str(_MAX_MANUAL_METRIC_TAIL_BYTES),
    ]
    return " ".join(shlex.quote(argument) for argument in arguments)


def _bounded_remote_diagnostic_payload(
    process: Any, *, withheld_event: str, failure_message: str
) -> dict[str, Any]:
    """Validate and sanitize a manual diagnostic response before local use."""

    if type(getattr(process, "returncode", None)) is not int or process.returncode != 0:
        raise OpsError(failure_message)
    stdout = getattr(process, "stdout", None)
    if not isinstance(stdout, str):
        return {"event": withheld_event, "diagnostic": _DIAGNOSTIC_WITHHELD}
    try:
        raw_size = len(stdout.encode("utf-8"))
    except UnicodeError:
        raw_size = _MAX_MANUAL_DIAGNOSTIC_BYTES + 1
    if (
        raw_size > _MAX_MANUAL_DIAGNOSTIC_BYTES
        or not _json_structure_within_diagnostic_limits(stdout)
    ):
        return {"event": withheld_event, "diagnostic": _DIAGNOSTIC_WITHHELD}
    try:
        snapshot = json.loads(stdout, object_pairs_hook=_unique_json_object)
    except Exception:
        return {"event": withheld_event, "diagnostic": _DIAGNOSTIC_WITHHELD}
    if not isinstance(snapshot, dict):
        return {"event": withheld_event, "diagnostic": _DIAGNOSTIC_WITHHELD}
    safe_snapshot = _redacted_json_value(snapshot)
    if not isinstance(safe_snapshot, dict):
        return {"event": withheld_event, "diagnostic": _DIAGNOSTIC_WITHHELD}
    return safe_snapshot


def read_remote_diagnostics(
    record: dict[str, Any], *, metrics_tail: int = 10
) -> dict[str, Any]:
    """Collect a non-mutating host/GPU snapshot for one tracked run."""

    manifest = RunManifest.from_dict(record["manifest"])
    connection = ssh_url(record_instance_id(record))
    process = remote_exec(
        connection, diagnostic_command(manifest, metrics_tail), check=False, timeout=45
    )
    safe_snapshot = _bounded_remote_diagnostic_payload(
        process,
        withheld_event="remote_diagnostics_withheld",
        failure_message="remote diagnostics command failed",
    )
    safe_snapshot.update(
        {
            "run_id": record["run_id"],
            "instance_id": record_instance_id(record),
            "local_status": record.get("status"),
        }
    )
    return safe_snapshot


def read_remote_stage_diagnostics(
    stage: dict[str, Any], *, metrics_tail: int = 10
) -> dict[str, Any]:
    manifest = RunManifest.from_dict(stage["manifest"])
    connection = ssh_url(
        _strict_positive_identity(stage.get("instance_id"), "stage instance ID")
    )
    process = remote_exec(
        connection,
        diagnostic_command(manifest, metrics_tail, repo_root=str(stage["remote_repo"])),
        check=False,
        timeout=45,
    )
    safe_snapshot = _bounded_remote_diagnostic_payload(
        process,
        withheld_event="remote_stage_diagnostics_withheld",
        failure_message="remote stage diagnostics command failed",
    )
    safe_snapshot.update(
        {
            "stage_id": stage["stage_id"],
            "parent_run_id": stage["parent_run_id"],
            "instance_id": stage["instance_id"],
            "local_status": stage.get("status"),
        }
    )
    return safe_snapshot


def sha256_file(
    path: Path, *, deadline_monotonic: float | None = None
) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            if (
                deadline_monotonic is not None
                and time.monotonic() >= deadline_monotonic
            ):
                raise OpsError("artifact hashing crossed its recovery deadline")
            digest.update(chunk)
    return digest.hexdigest()


_MAX_TAR_TRAILING_ZERO_BYTES = 1024 * 1024


def _tar_octal(header: bytes, field_name: str) -> int:
    """Parse one bounded POSIX tar numeric field; reject GNU base-256 forms."""

    if not header or header[0] & 0x80:
        raise OpsError(f"artifact archive uses unsupported {field_name} encoding")
    raw = header.rstrip(b"\0 ").lstrip(b" ")
    if not raw:
        return 0
    if any(byte < ord("0") or byte > ord("7") for byte in raw):
        raise OpsError(f"artifact archive has invalid {field_name}")
    return int(raw, 8)


def _read_exact_gzip(
    stream: gzip.GzipFile,
    size: int,
    *,
    deadline_monotonic: float | None,
) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        if (
            deadline_monotonic is not None
            and time.monotonic() >= deadline_monotonic
        ):
            raise OpsError("artifact inspection crossed its recovery deadline")
        chunk = stream.read(min(64 * 1024, remaining))
        if not chunk:
            raise OpsError("artifact archive ended unexpectedly")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _discard_exact_gzip(
    stream: gzip.GzipFile,
    size: int,
    *,
    deadline_monotonic: float | None,
) -> None:
    """Consume an exact gzip span without retaining an artifact body in memory."""

    remaining = size
    while remaining:
        if (
            deadline_monotonic is not None
            and time.monotonic() >= deadline_monotonic
        ):
            raise OpsError("artifact inspection crossed its recovery deadline")
        chunk = stream.read(min(64 * 1024, remaining))
        if not chunk:
            raise OpsError("artifact archive ended unexpectedly")
        remaining -= len(chunk)


def _prescan_regular_ustar(
    archive: Path,
    *,
    max_unpacked_bytes: int,
    deadline_monotonic: float | None,
) -> dict[str, int]:
    """Bound gzip expansion before ``tarfile`` can consume PAX/GNU metadata."""

    member_sizes: dict[str, int] = {}
    total_size = 0
    zero_blocks = 0
    trailing_zero_bytes = 0
    try:
        with gzip.open(archive, "rb") as stream:
            while True:
                header = _read_exact_gzip(
                    stream, 512, deadline_monotonic=deadline_monotonic
                )
                if header == b"\0" * 512:
                    zero_blocks += 1
                    trailing_zero_bytes += 512
                    if trailing_zero_bytes > _MAX_TAR_TRAILING_ZERO_BYTES:
                        raise OpsError("artifact archive has excessive trailing padding")
                    if zero_blocks < 2:
                        continue
                    while True:
                        if (
                            deadline_monotonic is not None
                            and time.monotonic() >= deadline_monotonic
                        ):
                            raise OpsError(
                                "artifact inspection crossed its recovery deadline"
                            )
                        chunk = stream.read(64 * 1024)
                        if not chunk:
                            return member_sizes
                        trailing_zero_bytes += len(chunk)
                        if (
                            trailing_zero_bytes > _MAX_TAR_TRAILING_ZERO_BYTES
                            or any(chunk)
                        ):
                            raise OpsError(
                                "artifact archive has invalid trailing content"
                            )
                if zero_blocks:
                    raise OpsError("artifact archive has a malformed end marker")
                if header[257:263] not in {b"ustar\0", b"ustar "}:
                    raise OpsError("artifact archive must use bounded POSIX ustar")
                stored_checksum = _tar_octal(header[148:156], "checksum")
                computed_checksum = (
                    sum(header[:148]) + (8 * ord(" ")) + sum(header[156:])
                )
                if stored_checksum != computed_checksum:
                    raise OpsError("artifact archive header checksum mismatch")
                if header[156:157] not in {b"0", b"\0"}:
                    raise OpsError(
                        "artifact archives may contain only regular ustar files"
                    )
                size = _tar_octal(header[124:136], "file size")
                total_size += size
                if total_size > max_unpacked_bytes:
                    raise OpsError("artifact archive exceeds the unpacked-byte limit")
                name = header[:100].split(b"\0", 1)[0]
                prefix = header[345:500].split(b"\0", 1)[0]
                try:
                    path = (prefix + (b"/" if prefix else b"") + name).decode(
                        "utf-8"
                    )
                except UnicodeDecodeError as error:
                    raise OpsError("artifact archive path is not UTF-8") from error
                relative = PurePosixPath(path)
                if (
                    not path
                    or "\\" in path
                    or relative.is_absolute()
                    or ".." in relative.parts
                    or relative.as_posix() != path
                ):
                    raise OpsError("artifact archive contains a non-canonical path")
                if path in member_sizes:
                    raise OpsError("artifact archive contains duplicate file paths")
                member_sizes[path] = size
                if len(member_sizes) > MAX_ARTIFACT_FILE_COUNT:
                    raise OpsError("artifact archive exceeds the file-count limit")
                padded_size = ((size + 511) // 512) * 512
                _discard_exact_gzip(
                    stream,
                    padded_size,
                    deadline_monotonic=deadline_monotonic,
                )
    except (gzip.BadGzipFile, EOFError, OSError) as error:
        raise OpsError("artifact archive is not a valid bounded gzip stream") from error


def safe_extract_tar(
    archive: Path,
    destination: Path,
    *,
    max_unpacked_bytes: int,
    expected_files: set[str] | None = None,
    deadline_monotonic: float | None = None,
) -> None:
    member_sizes = _prescan_regular_ustar(
        archive,
        max_unpacked_bytes=max_unpacked_bytes,
        deadline_monotonic=deadline_monotonic,
    )
    member_names = set(member_sizes)
    if expected_files is not None and member_names != expected_files:
        raise OpsError(
            "artifact archive members do not exactly match the artifact manifest"
        )

    destination.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination.resolve()
    extracted_names: set[str] = set()
    extracted_size = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member_count, member in enumerate(tar, start=1):
            if (
                deadline_monotonic is not None
                and time.monotonic() >= deadline_monotonic
            ):
                raise OpsError("artifact extraction crossed its recovery deadline")
            if member_count > MAX_ARTIFACT_FILE_COUNT or not member.isfile():
                raise OpsError("artifact archive changed after validation")
            relative = PurePosixPath(member.name)
            if (
                not member.name
                or "\\" in member.name
                or relative.is_absolute()
                or ".." in relative.parts
                or relative.as_posix() != member.name
                or member.name not in member_names
                or member.name in extracted_names
                or member.size != member_sizes[member.name]
            ):
                raise OpsError("artifact archive changed after validation")
            extracted_names.add(member.name)
            extracted_size += int(member.size)
            if extracted_size > max_unpacked_bytes:
                raise OpsError("artifact archive changed after validation")
            target = (destination / member.name).resolve()
            if (
                target != resolved_destination
                and resolved_destination not in target.parents
            ):
                raise OpsError("artifact archive contains an unsafe member path")
            tar.extract(member, destination, filter="data")
    if extracted_names != member_names:
        raise OpsError("artifact archive changed after validation")


_PULL_RECEIPT_VERSION = 1


def _strict_pull_receipt_is_valid(
    marker: Path,
    destination: Path,
    record: dict[str, Any],
    *,
    deadline_monotonic: float | None = None,
) -> bool:
    """Validate a cached strict-success receipt and its bound local inputs."""

    try:
        if marker.stat().st_size > MAX_ARTIFACT_MANIFEST_BYTES:
            return False
        receipt = json.loads(
            marker.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
        if not isinstance(receipt, dict):
            return False
        run_manifest = RunManifest.from_dict(record["manifest"])
        instance_id = _strict_positive_identity(
            record.get("instance_id"), "record instance ID"
        )
        if (
            type(receipt.get("receipt_version")) is not int
            or receipt.get("receipt_version") != _PULL_RECEIPT_VERSION
            or receipt.get("verification_mode") != "strict_success"
            or type(receipt.get("artifact_exit_code")) is not int
            or receipt.get("artifact_exit_code") != 0
            or receipt.get("run_id") != record.get("run_id")
            or type(receipt.get("instance_id")) is not int
            or receipt.get("instance_id") != instance_id
            or receipt.get("source_commit") != record.get("source_commit")
            or receipt.get("destination_scope") != destination.resolve().as_posix()
            or type(receipt.get("file_count")) is not int
            or receipt.get("file_count") < 0
        ):
            return False
        manifest_digest = receipt.get("artifact_manifest_sha256")
        archive_digest = receipt.get("archive_sha256")
        if not (
            isinstance(manifest_digest, str)
            and re.fullmatch(r"[0-9a-f]{64}", manifest_digest)
            and isinstance(archive_digest, str)
            and re.fullmatch(r"[0-9a-f]{64}", archive_digest)
        ):
            return False
        artifact_manifest_path = destination / "artifact_manifest.json"
        archive_path = destination / "artifacts.tar.gz"
        if not artifact_manifest_path.is_file() or not archive_path.is_file():
            return False
        if (
            artifact_manifest_path.stat().st_size > MAX_ARTIFACT_MANIFEST_BYTES
            or archive_path.stat().st_size > run_manifest.max_artifact_archive_bytes
        ):
            return False
        if sha256_file(
            artifact_manifest_path, deadline_monotonic=deadline_monotonic
        ) != manifest_digest:
            return False
        if sha256_file(
            archive_path, deadline_monotonic=deadline_monotonic
        ) != archive_digest:
            return False
        artifact_manifest = json.loads(
            artifact_manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
        if (
            not isinstance(artifact_manifest, dict)
            or artifact_manifest.get("run_id") != record.get("run_id")
            or artifact_manifest.get("source_commit") != record.get("source_commit")
            or type(artifact_manifest.get("exit_code")) is not int
            or artifact_manifest.get("exit_code") != 0
            or artifact_manifest.get("archive_sha256") != archive_digest
            or not isinstance(artifact_manifest.get("files"), list)
            or len(artifact_manifest["files"]) != receipt.get("file_count")
        ):
            return False
    except (
        OSError,
        OpsError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        _DiagnosticSanitizationError,
    ):
        return False
    return True


def pull_and_verify(
    record: dict[str, Any],
    *,
    best_effort: bool = False,
    snapshot_name: str | None = None,
    destination: Path | None = None,
    remote_ops: str = REMOTE_OPS,
    deadline_monotonic: float | None = None,
) -> bool:
    run_id = record["run_id"]
    run_manifest = RunManifest.from_dict(record["manifest"])
    transfer_deadline = deadline_monotonic
    if transfer_deadline is not None:
        transfer_deadline -= LOCAL_ARTIFACT_VERIFICATION_RESERVE_SECONDS
        _deadline_timeout_seconds(transfer_deadline, default=1)
    destination = destination or ROOT / "artifacts" / "remote" / run_id
    if snapshot_name is not None:
        if not RUN_ID_PATTERN.fullmatch(snapshot_name):
            raise OpsError("snapshot_name must be a safe filename component")
        destination = destination / "snapshots" / snapshot_name
    verified_marker = destination / "pull_verified.json"
    if verified_marker.exists() and _strict_pull_receipt_is_valid(
        verified_marker,
        destination,
        record,
        deadline_monotonic=deadline_monotonic,
    ):
        return True
    connection = ssh_url(
        int(record["instance_id"]),
        timeout=_deadline_timeout_seconds(transfer_deadline, default=180),
    )
    files = {
        "artifact_manifest.json": f"{remote_ops}/artifact_manifest.json",
        "artifacts.tar.gz": f"{remote_ops}/artifacts.tar.gz",
    }
    downloaded: dict[str, bool] = {}
    for name, remote in files.items():
        try:
            timeout_seconds = _deadline_timeout_seconds(
                transfer_deadline, default=1200
            )
        except OpsError:
            downloaded[name] = False
            continue
        downloaded[name] = scp_download(
            connection,
            remote,
            destination / name,
            timeout_seconds=timeout_seconds,
        )
    if not downloaded["artifact_manifest.json"] or not downloaded["artifacts.tar.gz"]:
        if best_effort:
            _best_effort_append_local_event(
                run_id, "pull_incomplete", downloaded=downloaded
            )
            return False
        raise OpsError("required artifact bundle or manifest is unavailable")
    artifact_manifest_path = destination / "artifact_manifest.json"
    archive = destination / "artifacts.tar.gz"
    if artifact_manifest_path.stat().st_size > MAX_ARTIFACT_MANIFEST_BYTES:
        raise OpsError("remote artifact manifest exceeds its byte limit")
    if archive.stat().st_size > run_manifest.max_artifact_archive_bytes:
        raise OpsError("artifact archive exceeds its compressed-byte limit")
    _deadline_timeout_seconds(deadline_monotonic, default=1)
    try:
        manifest = json.loads(
            artifact_manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, json.JSONDecodeError, _DiagnosticSanitizationError) as error:
        raise OpsError("remote artifact manifest is malformed") from error
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("format_version")) is not int
        or manifest.get("format_version") != 1
        or not isinstance(manifest.get("files"), list)
    ):
        raise OpsError("remote artifact manifest is malformed")
    if manifest.get("run_id") != run_id:
        raise OpsError("remote artifact manifest run_id mismatch")
    if manifest.get("source_commit") != record["source_commit"]:
        raise OpsError("remote artifact manifest source commit mismatch")
    artifact_exit_code = manifest.get("exit_code")
    if type(artifact_exit_code) is not int:
        raise OpsError("artifact verification requires an exact integer exit_code")
    if not best_effort and artifact_exit_code != 0:
        raise OpsError(
            "successful artifact verification requires exact integer exit_code zero"
        )
    missing_declared_paths = manifest.get("missing_declared_paths")
    if not isinstance(missing_declared_paths, list):
        raise OpsError("remote artifact manifest missing-path list is malformed")
    if missing_declared_paths != []:
        raise OpsError("remote artifact bundle is missing declared paths")
    entries = manifest["files"]
    if len(entries) > MAX_ARTIFACT_FILE_COUNT:
        raise OpsError("remote artifact manifest exceeds the file-count limit")
    declared_unpacked_bytes = 0
    expected_files: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise OpsError("remote artifact file entry is malformed")
        path_value = entry.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise OpsError("remote artifact file path is malformed")
        relative = PurePosixPath(path_value)
        if (
            "\\" in path_value
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != path_value
        ):
            raise OpsError("unsafe path in artifact manifest")
        if path_value in expected_files:
            raise OpsError("remote artifact manifest contains duplicate file paths")
        expected_files.add(path_value)
        size = entry.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise OpsError("remote artifact file size is invalid")
        digest = entry.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise OpsError("remote artifact file SHA-256 is malformed")
        declared_unpacked_bytes += size
    declared_paths = [
        PurePosixPath(value.replace("\\", "/")).as_posix().rstrip("/")
        for value in run_manifest.artifact_paths
    ]
    if any(
        not any(
            file_path == declared_path
            or file_path.startswith(declared_path + "/")
            for file_path in expected_files
        )
        for declared_path in declared_paths
    ):
        raise OpsError("artifact manifest does not cover every declared artifact path")
    if any(
        not any(
            file_path == declared_path
            or file_path.startswith(declared_path + "/")
            for declared_path in declared_paths
        )
        for file_path in expected_files
    ):
        raise OpsError("artifact manifest contains a path outside declared artifacts")
    if declared_unpacked_bytes > run_manifest.max_artifact_unpacked_bytes:
        raise OpsError("remote artifact manifest exceeds the unpacked-byte limit")
    archive_digest = manifest.get("archive_sha256")
    if not isinstance(archive_digest, str) or not re.fullmatch(
        r"[0-9a-f]{64}", archive_digest
    ):
        raise OpsError("remote artifact archive SHA-256 is malformed")
    if sha256_file(
        archive, deadline_monotonic=deadline_monotonic
    ) != archive_digest:
        raise OpsError("artifact archive SHA-256 mismatch")
    extracted = destination / "extracted"
    if extracted.exists():
        shutil.rmtree(extracted)
    safe_extract_tar(
        archive,
        extracted,
        max_unpacked_bytes=run_manifest.max_artifact_unpacked_bytes,
        expected_files=expected_files,
        deadline_monotonic=deadline_monotonic,
    )
    for entry in entries:
        relative = PurePosixPath(entry["path"])
        local = extracted.joinpath(*relative.parts)
        if (
            not local.is_file()
            or local.stat().st_size != entry["size"]
            or sha256_file(local, deadline_monotonic=deadline_monotonic)
            != entry["sha256"]
        ):
            raise OpsError("artifact verification failed for a declared path")
    verification_mode = "best_effort_recovery" if best_effort else "strict_success"
    summary = {
        "receipt_version": _PULL_RECEIPT_VERSION,
        "verification_mode": verification_mode,
        "verified_at": utc_now(),
        "run_id": run_id,
        "instance_id": _strict_positive_identity(
            record.get("instance_id"), "record instance ID"
        ),
        "source_commit": record["source_commit"],
        "destination_scope": destination.resolve().as_posix(),
        "artifact_exit_code": artifact_exit_code,
        "file_count": len(manifest["files"]),
        "artifact_manifest_sha256": sha256_file(
            artifact_manifest_path, deadline_monotonic=deadline_monotonic
        ),
        "archive_sha256": archive_digest,
    }
    receipt_marker = (
        destination / "pull_recovered.json" if best_effort else verified_marker
    )
    atomic_write_json(receipt_marker, summary)
    _best_effort_append_local_event(
        run_id,
        "pull_recovered" if best_effort else "pull_verified",
        **{key: value for key, value in summary.items() if key != "run_id"},
    )
    return True


def _ordinary_cleanup_record(run_id: str) -> bool:
    """Return whether *run_id* is a single-instance lifecycle record.

    Race records have their own candidate-level cleanup supervisor.  The
    ordinary destroy primitive must not overwrite that aggregate state while it
    still provides the same provider-side destroy call to the race path.
    """

    with state_file_lock():
        records = _load_state_unlocked()
        matches = [record for record in records if record.get("run_id") == run_id]
        if not matches:
            return False
        if len(matches) != 1:
            raise OpsError(f"expected one tracked record for run {run_id!r}")
        current = matches[0]
        if "candidates" not in current:
            return True
        if isinstance(current.get("candidates"), list):
            return False
        raise OpsError("tracked cleanup record has malformed candidate ownership")


def _cleanup_generation(run_id: str) -> int:
    return _record_cleanup_generation(get_record(run_id))


def _claim_cleanup_authority(
    instance_id: int, run_id: str, reason: str
) -> dict[str, Any]:
    """Fence every later provisioning write before provider-side teardown."""

    with state_file_lock():
        records = _load_state_unlocked()
        matches = [record for record in records if record.get("run_id") == run_id]
        if len(matches) != 1:
            raise OpsError(f"expected one tracked record for run {run_id!r}")
        current = matches[0]
        if isinstance(current.get("candidates"), list):
            return dict(current)
        if current.get("status") == "destroyed" and not current.get(
            "cleanup_required"
        ):
            return dict(current)
        _record_cleanup_evidence(
            current,
            instance_id=_strict_positive_identity(instance_id, "instance ID"),
            status="cleanup_in_progress",
            cleanup_reason=reason,
            cleanup_started_at=utc_now(),
        )
        _save_state_unlocked(records)
        return dict(current)


def _mark_cleanup_required(
    instance_id: int,
    run_id: str,
    error: str,
    *,
    status: str = "cleanup_required",
) -> None:
    """Persist an unresolved provider cleanup without claiming success."""

    safe_error = _redacted_exception_text(OpsError(error))
    untracked = False
    race_owned = False
    with state_file_lock():
        records = _load_state_unlocked()
        matches = [record for record in records if record.get("run_id") == run_id]
        if not matches:
            untracked = True
        elif len(matches) != 1:
            raise OpsError(f"expected one tracked record for run {run_id!r}")
        elif isinstance(matches[0].get("candidates"), list):
            race_owned = True
        else:
            current = matches[0]
            _record_cleanup_evidence(
                current,
                instance_id=instance_id,
                status=status,
                cleanup_error=safe_error,
                cleanup_required_at=utc_now(),
            )
            _save_state_unlocked(records)
    if untracked:
        _best_effort_append_local_event(
            run_id,
            "cleanup_required_untracked",
            instance_id=instance_id,
            error=safe_error,
        )
        return
    if race_owned:
        # The race coordinator owns candidate state and will record the failure
        # in its own ledger.  Do not mutate its top-level lifecycle here.
        _best_effort_append_local_event(
            run_id,
            "cleanup_required_race_owned",
            instance_id=instance_id,
            error=safe_error,
        )
        return
    _best_effort_append_local_event(
        run_id,
        "cleanup_required",
        instance_id=instance_id,
        error=safe_error,
    )


def _best_effort_mark_cleanup_required(
    instance_id: int,
    run_id: str,
    error: str,
    *,
    status: str = "cleanup_required",
) -> bool:
    """Add secondary cleanup evidence after authority is already fail-closed."""

    try:
        _mark_cleanup_required(
            instance_id,
            run_id,
            error,
            status=status,
        )
        return True
    except Exception as persistence_error:
        _best_effort_update_record_fields(
            run_id,
            instance_id=instance_id,
            cleanup_required=True,
            cleanup_state_integrity_failed=True,
            cleanup_state_integrity_failed_at=utc_now(),
            cleanup_state_integrity_error=_redacted_exception_text(
                persistence_error
            ),
        )
        _best_effort_append_local_event(
            run_id,
            "cleanup_state_persistence_failed",
            instance_id=instance_id,
            error=_redacted_exception_text(persistence_error),
        )
        return False


def _live_instance_is_present(
    instance_id: int, *, deadline_monotonic: float | None = None
) -> bool:
    """Check provider truth for one exact instance identity."""

    live = _require_provider_instance_rows(
        run_vast(
            "show",
            "instances",
            timeout=_deadline_timeout_seconds(
                deadline_monotonic, default=DESTROY_API_TIMEOUT_SECONDS
            ),
        ),
        "Vast instance listing",
    )
    return any(_offer_identity_number(item, "id") == instance_id for item in live)


def _require_unfenced_purchase_cleanup_authority(
    run_id: str,
    instance_id: int,
    match: dict[str, Any],
) -> dict[str, Any]:
    """Revalidate the complete narrow purchase-cleanup authority proof."""

    with state_file_lock():
        return _require_unfenced_purchase_cleanup_authority_unlocked(
            run_id, instance_id, match
        )


def _require_unfenced_purchase_cleanup_authority_unlocked(
    run_id: str,
    instance_id: int,
    match: dict[str, Any],
) -> dict[str, Any]:
    """Validate narrow purchase authority while the caller holds state lock."""

    records = _load_state_unlocked()
    matches = [record for record in records if record.get("run_id") == run_id]
    if len(matches) != 1:
        raise OpsError(f"expected one tracked record for run {run_id!r}")
    record = matches[0]
    _validate_reconcilable_purchase_instance(record, match)
    if (
        _strict_positive_identity(match.get("id"), "live instance ID")
        != instance_id
    ):
        raise OpsError("exact purchase cleanup instance identity changed")
    return record


def _live_reconcilable_purchase_match(
    record: dict[str, Any],
    instance_id: int,
    *,
    deadline_monotonic: float | None = None,
) -> tuple[dict[str, Any] | None, bool]:
    """Return one still-exact live match and whether numeric-ID identity drifted."""

    live = _require_provider_instance_rows(
        run_vast(
            "show",
            "instances",
            timeout=_deadline_timeout_seconds(
                deadline_monotonic, default=DESTROY_API_TIMEOUT_SECONDS
            ),
        ),
        "exact purchase cleanup query",
        require_label=True,
    )
    same_id = [
        item
        for item in live
        if isinstance(item, dict)
        and _offer_identity_number(item, "id") == instance_id
    ]
    if not same_id:
        return None, False
    if len(same_id) != 1 or not _is_reconcilable_purchase_instance(
        record, same_id[0]
    ):
        return None, True
    return same_id[0], False


def _mark_destroyed_verified(
    instance_id: int,
    run_id: str,
    reason: str,
    *,
    ordinary: bool,
    expected_cleanup_generation: int | None = None,
) -> bool:
    """Record provisional absence iff no newer cleanup evidence superseded it."""

    if ordinary:
        with state_file_lock():
            records = _load_state_unlocked()
            matches = [record for record in records if record.get("run_id") == run_id]
            if len(matches) != 1:
                raise OpsError(f"expected one tracked record for run {run_id!r}")
            current = matches[0]
            if (
                expected_cleanup_generation is not None
                and _record_cleanup_generation(current)
                != expected_cleanup_generation
            ):
                return False
            current.update(
                {
                    "status": "destroy_followup_pending",
                    "destroy_verified_absent_at": utc_now(),
                    "reason": reason,
                    "cleanup_required": True,
                    "cleanup_error": (
                        "provider absence observed; delayed-reappearance follow-up "
                        "has not completed"
                    ),
                }
            )
            _save_state_unlocked(records)
    _best_effort_append_local_event(
        run_id,
        (
            "instance_absence_verified_followup_pending"
            if ordinary
            else "instance_destroyed"
        ),
        instance_id=instance_id,
        reason=reason,
        live_absence_verified=True,
    )
    return True


def _complete_destroy_reconciliation_if_current(
    run_id: str, *, expected_cleanup_generation: int
) -> bool:
    """Clear cleanup only if the final absence observation is still newest."""

    with state_file_lock():
        records = _load_state_unlocked()
        matches = [record for record in records if record.get("run_id") == run_id]
        if len(matches) != 1:
            raise OpsError(f"expected one tracked record for run {run_id!r}")
        current = matches[0]
        if _record_cleanup_generation(current) != expected_cleanup_generation:
            return False
        if current.get("cleanup_state_integrity_failed"):
            return False
        current.update(
            {
                "status": "destroyed",
                "destroyed_at": utc_now(),
                "cleanup_required": False,
                "cleanup_error": None,
                "destroy_reconciliation_verified_at": utc_now(),
                "destroy_reconciliation_status": "complete",
            }
        )
        _save_state_unlocked(records)
        return True


def destroy_instance(
    instance_id: int,
    run_id: str,
    reason: str,
    *,
    launch_supervisor: bool = True,
    deadline_monotonic: float | None = None,
    mark_terminal: bool = True,
    unfenced_exact_purchase_match: dict[str, Any] | None = None,
    untracked_orphan_label: str | None = None,
) -> None:
    """Destroy one instance and hand verified absence to bounded follow-up.

    Vast's successful CLI response is only an acknowledgement.  Every command
    attempt is followed by a provider ``show instances`` query.  Any command or
    query error is persisted as cleanup-required; ordinary records also receive
    a detached bounded reconciliation supervisor.  Initial absence is
    provisional: only that supervisor's complete delayed-reappearance lease can
    write the clean terminal state.  The supervisor caller uses
    ``launch_supervisor=False`` to avoid recursively spawning itself.
    """

    _strict_positive_identity(instance_id, "instance ID")
    if untracked_orphan_label is not None:
        expected_orphan_label = f"quant-experiments-{run_id}"
        if (
            reason != "explicit_orphan_cleanup"
            or unfenced_exact_purchase_match is not None
            or untracked_orphan_label != expected_orphan_label
        ):
            raise OpsError(
                "untracked orphan guard is restricted to exact manual cleanup"
            )
    if unfenced_exact_purchase_match is not None:
        if (
            launch_supervisor
            or mark_terminal
            or deadline_monotonic is None
            or reason
            not in {
                "purchase_outcome_reconciled",
                "late_purchase_outcome_reconciled",
            }
        ):
            raise OpsError(
                "unfenced purchase cleanup is restricted to the bounded exact-match "
                "purchase reconciler"
            )
        if (
            _strict_positive_identity(
                unfenced_exact_purchase_match.get("id"), "live instance ID"
            )
            != instance_id
        ):
            raise OpsError("unfenced purchase cleanup instance identity mismatch")
    ordinary = _ordinary_cleanup_record(run_id)
    if untracked_orphan_label is not None and ordinary:
        raise OpsError(
            "orphan cleanup target acquired a local lifecycle owner"
        )
    if unfenced_exact_purchase_match is not None:
        if not ordinary:
            raise OpsError(
                "unfenced purchase cleanup requires one ordinary lifecycle record"
            )
        _require_unfenced_purchase_cleanup_authority(
            run_id, instance_id, unfenced_exact_purchase_match
        )
    cleanup_state_integrity = True
    unfenced_cleanup_error: str | None = None
    if ordinary:
        try:
            cleanup_record = _claim_cleanup_authority(instance_id, run_id, reason)
        except OSError as state_error:
            if unfenced_exact_purchase_match is None:
                raise
            # Re-read and revalidate after the failed write.  Only the bounded
            # cleanup-only purchase reconciler may cross this narrow escape
            # hatch, and only for the exact provider identity it just observed.
            # State/read failures and all OpsError invariant failures remain
            # hard barriers.
            try:
                _require_unfenced_purchase_cleanup_authority(
                    run_id, instance_id, unfenced_exact_purchase_match
                )
            except Exception as identity_error:
                raise OpsError(
                    "exact purchase identity changed after cleanup claim failed"
                ) from identity_error
            ordinary = False
            unfenced_cleanup_error = _redacted_exception_text(state_error)
        else:
            transition_deadline = deadline_monotonic
            if transition_deadline is None:
                try:
                    manifest = RunManifest.from_dict(cleanup_record["manifest"])
                    transition_deadline = _record_deadline_monotonic(
                        cleanup_record,
                        "controller_deadline_at",
                        fallback_seconds=_campaign_hard_remaining_seconds(
                            cleanup_record, manifest
                        ),
                    )
                except (KeyError, OpsError, TypeError):
                    transition_deadline = (
                        time.monotonic() + STATE_LOCK_TIMEOUT_SECONDS
                    )
            try:
                # A bootstrap that committed first may finish, but cleanup cannot
                # race past it. Cleanup authority was already persisted, so no later
                # bootstrap can enter this critical section.
                with _lifecycle_transition_lock(
                    run_id, deadline_monotonic=transition_deadline
                ):
                    pass
            except OpsError as error:
                safe_error = _redacted_exception_text(error)
                message = (
                    "cleanup could not cross the provisioning barrier: " + safe_error
                )
                _best_effort_mark_cleanup_required(instance_id, run_id, message)
                raise OpsError(message) from error
    last_error: Exception | None = None
    command_attempt = 0
    verification_attempt = 0
    unfenced_refresh_attempt = 0
    purchase_record_for_verification: dict[str, Any] | None = None
    while (
        command_attempt < DESTROY_MAX_ATTEMPTS
        and verification_attempt < DESTROY_VERIFY_MAX_ATTEMPTS
    ):
        if (
            deadline_monotonic is not None
            and deadline_monotonic - time.monotonic() < 1
        ):
            last_error = OpsError("destroy deadline elapsed before verified absence")
            break
        if unfenced_exact_purchase_match is not None:
            if unfenced_refresh_attempt >= DESTROY_VERIFY_MAX_ATTEMPTS:
                last_error = OpsError(
                    "exact purchase identity refresh attempts were exhausted"
                )
                break
            unfenced_refresh_attempt += 1
            try:
                purchase_record = _require_unfenced_purchase_cleanup_authority(
                    run_id, instance_id, unfenced_exact_purchase_match
                )
                fresh_match, identity_drifted = _live_reconcilable_purchase_match(
                    purchase_record,
                    instance_id,
                    deadline_monotonic=deadline_monotonic,
                )
                if identity_drifted:
                    _best_effort_append_local_event(
                        run_id,
                        "exact_purchase_identity_drift_before_destroy",
                        instance_id=instance_id,
                    )
                if fresh_match is None:
                    _mark_destroyed_verified(
                        instance_id,
                        run_id,
                        reason,
                        ordinary=False,
                    )
                    return
                unfenced_exact_purchase_match = fresh_match
                purchase_record_for_verification = purchase_record
            except Exception as error:
                safe_error = _redacted_exception_text(error)
                last_error = OpsError(safe_error)
                _best_effort_append_local_event(
                    run_id,
                    "exact_purchase_pre_destroy_refresh_error",
                    instance_id=instance_id,
                    attempt=unfenced_refresh_attempt,
                    error=safe_error,
                )
                if unfenced_refresh_attempt < DESTROY_VERIFY_MAX_ATTEMPTS:
                    _bounded_sleep(
                        DESTROY_VERIFY_RETRY_DELAY_SECONDS, deadline_monotonic
                    )
                continue
        command_attempt += 1
        orphan_mutation_authorized = untracked_orphan_label is None
        try:
            command_deadline = deadline_monotonic
            if command_deadline is not None:
                # A mutation without a subsequent provider-truth query is not
                # verified cleanup. Reserve one full query slot before issuing
                # each destroy command.
                command_deadline -= DESTROY_API_TIMEOUT_SECONDS
            if untracked_orphan_label is not None:
                # Serialize the last local-ownership proof and provider mutation.
                # A lifecycle record appearing after the command's read-only
                # provider query must still fence this manual escape path.
                with state_file_lock():
                    _require_unrecorded_orphan_target_from_records(
                        instance_id,
                        untracked_orphan_label,
                        records=_load_state_unlocked(),
                        stages=load_stage_state(),
                    )
                    orphan_mutation_authorized = True
                    run_vast(
                        "destroy",
                        "instance",
                        instance_id,
                        "-y",
                        raw=False,
                        timeout=_deadline_timeout_seconds(
                            command_deadline,
                            default=DESTROY_API_TIMEOUT_SECONDS,
                        ),
                    )
            elif unfenced_exact_purchase_match is None:
                run_vast(
                    "destroy",
                    "instance",
                    instance_id,
                    "-y",
                    raw=False,
                    timeout=_deadline_timeout_seconds(
                        command_deadline, default=DESTROY_API_TIMEOUT_SECONDS
                    ),
                )
            else:
                # This emergency escape path exists only when the cleanup claim
                # could not be persisted. Serialize the last state proof and the
                # destructive call so no cooperating writer can change lifecycle
                # ownership in the check-to-mutation gap.
                with state_file_lock():
                    _require_unfenced_purchase_cleanup_authority_unlocked(
                        run_id, instance_id, unfenced_exact_purchase_match
                    )
                    run_vast(
                        "destroy",
                        "instance",
                        instance_id,
                        "-y",
                        raw=False,
                        timeout=_deadline_timeout_seconds(
                            command_deadline, default=DESTROY_API_TIMEOUT_SECONDS
                        ),
                    )
        except Exception as error:
            if (
                untracked_orphan_label is not None
                and not orphan_mutation_authorized
            ):
                raise
            safe_error = _redacted_exception_text(error)
            last_error = OpsError(safe_error)
            if ordinary and mark_terminal:
                if not _best_effort_mark_cleanup_required(
                    instance_id,
                    run_id,
                    f"destroy command failed: {safe_error}",
                    status="cleanup_in_progress",
                ):
                    cleanup_state_integrity = False
            _best_effort_append_local_event(
                run_id,
                "destroy_retry",
                instance_id=instance_id,
                attempt=command_attempt,
                error=safe_error,
            )
            if unfenced_cleanup_error is not None:
                _best_effort_append_local_event(
                    run_id,
                    "exact_purchase_cleanup_claim_write_failed",
                    instance_id=instance_id,
                    error=unfenced_cleanup_error,
                )
                unfenced_cleanup_error = None
            # A timeout or transport failure leaves the mutation outcome
            # unknown. Always take the promised provider-truth snapshot before
            # deciding whether another destructive retry is permitted.

        if unfenced_cleanup_error is not None:
            _best_effort_append_local_event(
                run_id,
                "exact_purchase_cleanup_claim_write_failed",
                instance_id=instance_id,
                error=unfenced_cleanup_error,
            )
            unfenced_cleanup_error = None

        verification_attempt += 1
        expected_cleanup_generation: int | None = None
        cleanup_generation_available = not (ordinary and mark_terminal)
        if ordinary and mark_terminal:
            try:
                expected_cleanup_generation = _cleanup_generation(run_id)
                cleanup_generation_available = True
            except Exception as state_error:
                _best_effort_append_local_event(
                    run_id,
                    "destroy_generation_read_failed",
                    instance_id=instance_id,
                    error=_redacted_exception_text(state_error),
                )
        try:
            if unfenced_exact_purchase_match is None:
                present = _live_instance_is_present(
                    instance_id, deadline_monotonic=deadline_monotonic
                )
            else:
                if purchase_record_for_verification is None:
                    raise OpsError(
                        "exact purchase verification lacks a pre-destroy authority proof"
                    )
                live_match, identity_drifted = _live_reconcilable_purchase_match(
                    purchase_record_for_verification,
                    instance_id,
                    deadline_monotonic=deadline_monotonic,
                )
                if identity_drifted:
                    _best_effort_append_local_event(
                        run_id,
                        "exact_purchase_identity_drift_after_destroy",
                        instance_id=instance_id,
                    )
                present = live_match is not None
                if live_match is not None:
                    unfenced_exact_purchase_match = live_match
                # This check deliberately follows the read-only provider query:
                # state drift must block interpretation and all later mutation,
                # but it must not suppress the post-mutation truth snapshot.
                _require_unfenced_purchase_cleanup_authority(
                    run_id, instance_id, unfenced_exact_purchase_match
                )
        except Exception as error:
            safe_error = _redacted_exception_text(error)
            last_error = OpsError(safe_error)
            if ordinary and mark_terminal:
                if not _best_effort_mark_cleanup_required(
                    instance_id,
                    run_id,
                    f"destroy absence query failed: {safe_error}",
                    status="cleanup_in_progress",
                ):
                    cleanup_state_integrity = False
            _best_effort_append_local_event(
                run_id,
                "destroy_absence_query_error",
                instance_id=instance_id,
                attempt=verification_attempt,
                error=safe_error,
            )
            if command_attempt < DESTROY_MAX_ATTEMPTS:
                _bounded_sleep(
                    DESTROY_VERIFY_RETRY_DELAY_SECONDS, deadline_monotonic
                )
            continue
        if not present:
            if not cleanup_generation_available or not cleanup_state_integrity:
                terminal_written = False
            else:
                try:
                    terminal_written = _mark_destroyed_verified(
                        instance_id,
                        run_id,
                        reason,
                        ordinary=ordinary and mark_terminal,
                        expected_cleanup_generation=expected_cleanup_generation,
                    )
                except Exception as state_error:
                    terminal_written = False
                    _best_effort_append_local_event(
                        run_id,
                        "destroy_terminal_write_failed",
                        instance_id=instance_id,
                        error=_redacted_exception_text(state_error),
                    )
            if not terminal_written:
                last_error = OpsError(
                    "newer cleanup evidence superseded the absence observation"
                )
                _best_effort_append_local_event(
                    run_id,
                    "destroy_terminal_write_fenced",
                    instance_id=instance_id,
                    attempt=verification_attempt,
                )
                if command_attempt < DESTROY_MAX_ATTEMPTS:
                    _bounded_sleep(
                        DESTROY_VERIFY_RETRY_DELAY_SECONDS, deadline_monotonic
                    )
                continue
            if ordinary and launch_supervisor:
                try:
                    _launch_destroy_reconciliation_with_retries(
                        run_id,
                        instance_id=instance_id,
                        deadline_monotonic=deadline_monotonic,
                    )
                except Exception as error:
                    safe_error = _redacted_exception_text(error)
                    message = (
                        "follow-up cleanup supervisor unavailable: " + safe_error
                    )
                    _best_effort_mark_cleanup_required(
                        instance_id, run_id, message
                    )
                    raise OpsError(message) from error
            return
        last_error = OpsError(
            f"Vast still reports instance {instance_id} after destroy acknowledgement"
        )
        if ordinary and mark_terminal:
            if not _best_effort_mark_cleanup_required(
                instance_id,
                run_id,
                str(last_error),
                status="cleanup_in_progress",
            ):
                cleanup_state_integrity = False
        _best_effort_append_local_event(
            run_id,
            "destroy_absence_not_verified",
            instance_id=instance_id,
            attempt=verification_attempt,
        )
        if command_attempt < DESTROY_MAX_ATTEMPTS:
            _bounded_sleep(DESTROY_RETRY_DELAY_SECONDS, deadline_monotonic)

    message = f"could not verify destruction of instance {instance_id}: {last_error}"
    _best_effort_mark_cleanup_required(instance_id, run_id, message)
    if ordinary and launch_supervisor:
        try:
            _launch_destroy_reconciliation_with_retries(
                run_id,
                instance_id=instance_id,
                deadline_monotonic=deadline_monotonic,
            )
        except Exception as error:
            safe_error = _redacted_exception_text(error)
            _best_effort_append_local_event(
                run_id,
                "cleanup_supervisor_launch_failed",
                instance_id=instance_id,
                error=safe_error,
            )
            message += f"; supervisor unavailable: {safe_error}"
    raise OpsError(message)


def destroy_verified_success(
    instance_id: int,
    run_id: str,
    *,
    deadline_monotonic: float | None = None,
) -> bool:
    try:
        destroy_instance(
            instance_id,
            run_id,
            "success_verified",
            deadline_monotonic=deadline_monotonic,
        )
    except Exception as error:
        safe_error = _redacted_exception_text(error)
        _best_effort_append_local_event(
            run_id, "success_destroy_retry", error=safe_error
        )
        try:
            print(
                json.dumps(
                    {"event": "success_destroy_retry", "error": safe_error},
                    sort_keys=True,
                ),
                flush=True,
            )
        except Exception:
            pass
        return False
    return True


def handle_verified_success(
    record: dict[str, Any],
    manifest: RunManifest,
    instance_id: int,
    *,
    deadline_monotonic: float | None = None,
) -> bool:
    """Apply the manifest's verified-success policy.

    Return True only when monitoring should stop. Research sessions deliberately
    remain monitored so their hard TTL and spend cleanup reserve still applies.
    """

    if manifest.success_policy == "destroy_after_verify":
        return destroy_verified_success(
            instance_id,
            manifest.run_id,
            deadline_monotonic=deadline_monotonic,
        )
    if record.get("status") != "session_ready":
        ready_at = utc_now()
        wrote, _current = _monitor_write_if_active(
            manifest.run_id,
            status="session_ready",
            session_ready_at=ready_at,
            artifacts_verified_at=ready_at,
            event="session_ready",
            event_fields={
                "instance_id": instance_id,
                "success_policy": manifest.success_policy,
            },
        )
        if not wrote:
            # A cleanup supervisor won while artifacts were being checked.  Its
            # transition is authoritative.  Continue the outer monitor once so
            # it can prove a cleanup supervisor owns the lease or take the
            # bounded cleanup fallback itself.
            return False
        _best_effort_emit_json(
            {
                "event": "session_ready",
                "instance_id": instance_id,
                "message": (
                    "artifacts verified; instance remains live until explicit "
                    "finalization or the hard cleanup reserve"
                ),
            }
        )
    return False


def retain_failed_debug_session(
    record: dict[str, Any],
    manifest: RunManifest,
    instance_id: int,
    *,
    deadline_monotonic: float | None = None,
) -> bool:
    """Keep an explicitly retained session live after a diagnostic failure.

    A failed experiment is useful evidence during an interactive debugging
    session. Recover what it produced, mark the pinned checkout ready for a
    follow-up stage, and leave the parent monitor enforcing the original hard
    spend and TTL limits.
    """

    if manifest.success_policy != "keep_alive_after_verify":
        return False
    if record.get("status") == "session_ready":
        return True
    artifacts_verified = False
    try:
        artifacts_verified = bool(
            pull_and_verify(
                record,
                best_effort=True,
                deadline_monotonic=deadline_monotonic,
            )
        )
    except Exception as error:
        _best_effort_append_local_event(
            manifest.run_id,
            "retained_failure_recovery_failed",
            error=_redacted_exception_text(error),
        )
    ready_at = utc_now()
    wrote, _current = _monitor_write_if_active(
        manifest.run_id,
        status="session_ready",
        session_ready_at=ready_at,
        retained_failure_at=ready_at,
        failure_artifacts_verified=artifacts_verified,
        event="session_ready_after_failure",
        event_fields={
            "instance_id": instance_id,
            "artifacts_verified": artifacts_verified,
        },
    )
    if not wrote:
        # Cleanup became authoritative during best-effort recovery.  Report the
        # monitor path handled so its caller does not start a stale failure
        # timer; the next monitor iteration exits on cleanup_required.
        return True
    _best_effort_emit_json(
        {
            "event": "session_ready_after_failure",
            "instance_id": instance_id,
            "artifacts_verified": artifacts_verified,
            "message": (
                "diagnostic failure retained; instance remains live until "
                "explicit finalization or the hard cleanup reserve"
            ),
        }
    )
    return True


NOT_RUNNING_POLLS_BEFORE_CLEANUP = 3
_NOT_RUNNING_ACTUAL_STATUSES = frozenset({"exited", "stopped", "offline"})


def _live_row_not_running(row: dict[str, Any]) -> bool:
    """True when Vast reports the instance stopped/exited (storage still bills)."""

    actual = str(row.get("actual_status") or "").strip().lower()
    intended = str(row.get("intended_status") or "").strip().lower()
    return actual in _NOT_RUNNING_ACTUAL_STATUSES or intended == "stopped"


def _monitor_run(run_id: str) -> int:
    record = get_record(run_id)
    instance_id = record_instance_id(record)
    manifest = RunManifest.from_dict(record["manifest"])
    last_signature = record.get("last_event_signature")
    failed_at: datetime | None = None
    transport_error_reported = False
    live_query_error_reported = False
    # Stall / stopped-instance detection.  A hung workload or a Vast instance
    # in a non-running state otherwise keeps billing until the TTL boundary,
    # and a stopped container cannot run its own emergency watchdog.
    last_progress_at = time.monotonic()
    terminal_seen = False
    not_running_polls = 0
    initial_hard_remaining = _campaign_hard_remaining_seconds(record, manifest)
    cleanup_deadline = _record_deadline_monotonic(
        record,
        "cleanup_start_at",
        fallback_seconds=max(
            0.0,
            initial_hard_remaining
            - cleanup_reserve_hours(manifest.poll_seconds) * 3600.0,
        ),
    )
    recovery_deadline = _record_deadline_monotonic(
        record,
        "recovery_deadline_at",
        fallback_seconds=max(
            0.0,
            initial_hard_remaining
            - (
                cleanup_reserve_hours(manifest.poll_seconds) * 3600.0
                - HARD_LIMIT_RECOVERY_RESERVE_SECONDS
            ),
        ),
    )
    controller_deadline = _record_deadline_monotonic(
        record,
        "controller_deadline_at",
        fallback_seconds=max(
            0.0,
            initial_hard_remaining
            - REMOTE_WATCHDOG_SAFETY_SECONDS
            - LOCAL_WATCHDOG_ORDERING_MARGIN_SECONDS,
        ),
    )
    while True:
        # A terminal destroy transition is authoritative. A delayed monitor
        # must never reverse it from a stale provider snapshot.
        try:
            record = get_record(run_id)
        except OpsError as state_error:
            # A transient state-lock timeout (another supervisor holding the
            # lock across provider I/O) must not kill the only process that
            # enforces TTL and spend.  Retry; the deadlines below still apply.
            _best_effort_append_local_event(
                run_id,
                "monitor_state_read_retry",
                instance_id=instance_id,
                error=_redacted_exception_text(state_error),
            )
            time.sleep(min(15.0, max(1.0, float(manifest.poll_seconds) / 4)))
            continue
        # Re-anchor the monotonic deadlines to the persisted UTC schedule
        # every iteration so a suspended laptop (where the monotonic clock may
        # not advance) cannot push cleanup later than the wall-clock boundary.
        for field_name, current in (
            ("cleanup_start_at", cleanup_deadline),
            ("recovery_deadline_at", recovery_deadline),
            ("controller_deadline_at", controller_deadline),
        ):
            if record.get(field_name) is None:
                continue
            refreshed = min(
                current,
                _record_deadline_monotonic(
                    record, field_name, fallback_seconds=0.0
                ),
            )
            if field_name == "cleanup_start_at":
                cleanup_deadline = refreshed
            elif field_name == "recovery_deadline_at":
                recovery_deadline = refreshed
            else:
                controller_deadline = refreshed
        if record.get("status") == "destroyed":
            return 0
        if record.get("status") == "missing_from_live_api":
            return 2
        if record.get("status") == "emergency_termination_unverified":
            return 2
        hard_reason = None
        if record.get("cleanup_required"):
            cleanup_owner = _supervisor_lock_owner_pid(
                MONITOR_DIR / f"destroy_reconcile_{run_id}.lock"
            )
            if cleanup_owner is not None:
                return 2
            hard_reason = "cleanup_required_retry"
        elif record.get("status") == "provisioning_cleanup_pending":
            hard_reason = "provisioning_cleanup_retry"
        elif record.get("lifecycle_origin") == "existing_instance_adoption" and (
            record.get("status") in {"adoption_pending", "adoption_provisioning"}
        ):
            # A monitor only starts after setup reaches running. Seeing either
            # state therefore means the adopting process was interrupted; do
            # not leave an already-billing blank or half-provisioned host idle.
            provisioning_owner_pid = record.get("provisioning_owner_pid")
            if (
                isinstance(provisioning_owner_pid, int)
                and provisioning_owner_pid > 0
                and process_is_alive(provisioning_owner_pid)
            ):
                # A mistakenly started monitor must not destroy a host while
                # the adopting process still owns the setup transition.
                return 0
            hard_reason = "adoption_setup_interrupted"
        elif time.monotonic() >= cleanup_deadline:
            hard_reason = "controller_cleanup_boundary_reached"
        elif not_running_polls >= NOT_RUNNING_POLLS_BEFORE_CLEANUP:
            hard_reason = "instance_not_running"
        elif (
            manifest.stall_minutes
            and not terminal_seen
            and record.get("status") != "session_ready"
            and time.monotonic() - last_progress_at >= manifest.stall_minutes * 60
        ):
            hard_reason = "remote_progress_stalled"
        if hard_reason:
            destroy_deadline = controller_deadline
            if destroy_deadline - time.monotonic() < 1:
                # Cleanup remains mandatory after a missed controller boundary,
                # but it can no longer be described as within the approved
                # client-side cap. Persist that operational failure explicitly.
                destroy_deadline = (
                    time.monotonic() + destroy_attempt_budget_seconds()
                )
                _best_effort_update_record_fields(
                    run_id,
                    cap_control_failed=True,
                    cap_control_failure_at=utc_now(),
                    cap_control_failure_reason=(
                        "cleanup began after the persisted controller deadline"
                    ),
                )
                _best_effort_append_local_event(
                    run_id,
                    "cap_control_failed_emergency_cleanup",
                    reason=hard_reason,
                )
            if hard_reason not in {
                "provisioning_cleanup_retry",
                "cleanup_required_retry",
            }:
                if record.get("status") in {"running", "session_ready"}:
                    try:
                        # The normal runner packages artifacts only after its
                        # command sequence exits. At a hard TTL/spend boundary
                        # it may still be training, so create a point-in-time
                        # archive before pulling and destroying the contract.
                        # Checkpoints are atomically replaced by the trainer,
                        # making a concurrent snapshot safe and resumable.
                        request_remote_session_snapshot(
                            record, deadline_monotonic=recovery_deadline
                        )
                        _best_effort_monitor_event_if_active(
                            run_id, "hard_limit_snapshot_ready"
                        )
                    except Exception as error:
                        _best_effort_monitor_event_if_active(
                            run_id,
                            "hard_limit_snapshot_failed",
                            error=_redacted_exception_text(error),
                        )
                try:
                    pull_and_verify(
                        record,
                        best_effort=True,
                        deadline_monotonic=recovery_deadline,
                    )
                except Exception as error:
                    _best_effort_monitor_event_if_active(
                        run_id,
                        "hard_limit_recovery_failed",
                        error=_redacted_exception_text(error),
                    )
            try:
                destroy_instance(
                    instance_id,
                    run_id,
                    hard_reason,
                    deadline_monotonic=destroy_deadline,
                )
                return 2
            except Exception as error:
                safe_error = _redacted_exception_text(error)
                try:
                    print(
                        json.dumps(
                            {"event": "destroy_failed", "error": safe_error},
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                except Exception:
                    pass
                # ``destroy_instance`` already tried detached supervision and
                # a bounded attached fallback. The unresolved manual state is
                # authoritative; do not start a fresh post-boundary loop.
                return 2

        try:
            live = _require_provider_instance_rows(
                run_vast(
                    "show",
                    "instances",
                    timeout=_deadline_timeout_seconds(
                        cleanup_deadline, default=DESTROY_API_TIMEOUT_SECONDS
                    ),
                ),
                "monitor live instance query",
            )
            live_ids: set[int] = set()
            for item in live:
                if not isinstance(item, dict) or "id" not in item:
                    raise OpsError("monitor live instance entry is malformed")
                live_ids.add(
                    _strict_positive_identity(item.get("id"), "live instance ID")
                )
            live_query_error_reported = False
            live_row = next(
                (
                    item
                    for item in live
                    if isinstance(item, dict)
                    and _strict_positive_identity(item.get("id"), "live instance ID")
                    == instance_id
                ),
                None,
            )
            if live_row is not None and _live_row_not_running(live_row):
                not_running_polls += 1
                if not_running_polls == 1:
                    _best_effort_monitor_event_if_active(
                        run_id,
                        "instance_not_running",
                        actual_status=live_row.get("actual_status"),
                        intended_status=live_row.get("intended_status"),
                    )
            else:
                not_running_polls = 0
            if instance_id not in live_ids:
                try:
                    wrote, current = _monitor_write_if_active(
                        run_id,
                        status="emergency_termination_unverified",
                        cleanup_required=True,
                        cleanup_error=(
                            "instance disappeared before ordinary verified "
                            "destruction; provider watchdog or external "
                            "termination may have fired"
                        ),
                        reconciled_at=utc_now(),
                        event="provider_emergency_termination_unverified",
                    )
                except Exception as state_error:
                    # Provider absence still requires a reconciliation handoff.
                    # A failed state replica forbids a clean conclusion, but it
                    # must not suppress that handoff.
                    wrote = True
                    current = record
                    _best_effort_update_record_fields(
                        run_id,
                        status="emergency_termination_unverified",
                        cleanup_required=True,
                        cleanup_error=(
                            "provider disappearance state persistence failed: "
                            + _redacted_exception_text(state_error)
                        ),
                    )
                if not wrote:
                    # A cleanup supervisor won the race; its terminal state is
                    # authoritative and must not be replaced by this stale API
                    # snapshot.
                    return 0 if current.get("status") == "destroyed" else 2
                try:
                    _launch_destroy_reconciliation_with_retries(
                        run_id,
                        instance_id=instance_id,
                        deadline_monotonic=controller_deadline,
                    )
                except Exception as supervisor_error:
                    _best_effort_append_local_event(
                        run_id,
                        "emergency_termination_reconciliation_failed",
                        error=_redacted_exception_text(supervisor_error),
                    )
                _best_effort_emit_json(
                    {"event": "provider_emergency_termination_unverified"}
                )
                return 2
        except Exception as error:
            if not live_query_error_reported:
                safe_error = _redacted_exception_text(error)
                _best_effort_monitor_event_if_active(
                    run_id, "live_query_error", error=safe_error
                )
                try:
                    print(
                        json.dumps(
                            {"event": "live_query_error", "error": safe_error}
                        ),
                        flush=True,
                    )
                except Exception:
                    pass
                live_query_error_reported = True

        try:
            connection = ssh_url(
                instance_id,
                timeout=_deadline_timeout_seconds(cleanup_deadline, default=180),
            )
            events = _sanitized_remote_events(
                read_remote_events(
                    connection, deadline_monotonic=cleanup_deadline
                )
            )
            transport_error_reported = False
        except Exception as error:
            events = []
            if not transport_error_reported:
                safe_error = _redacted_exception_text(error)
                _best_effort_monitor_event_if_active(
                    run_id, "monitor_transport_error", error=safe_error
                )
                try:
                    print(
                        json.dumps(
                            {
                                "event": "monitor_transport_error",
                                "error": safe_error,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                except Exception:
                    pass
                transport_error_reported = True

        if events:
            latest = events[-1]
            signature = hashlib.sha256(
                json.dumps(latest, sort_keys=True).encode("utf-8")
            ).hexdigest()
            if signature != last_signature:
                last_progress_at = time.monotonic()
                # Heartbeats prove liveness but are intentionally not forwarded as
                # model-visible progress. The local state still records the newest
                # heartbeat for crash recovery and stall diagnosis.
                if latest.get("event") != "heartbeat":
                    _best_effort_emit_json(latest)
                wrote, current = _monitor_write_if_active(
                    run_id,
                    last_event_signature=signature,
                    last_remote_event=latest,
                    event=(
                        "remote_state_change"
                        if latest.get("event") != "heartbeat"
                        else None
                    ),
                    event_fields={"remote": latest},
                )
                if not wrote:
                    return 0 if current.get("status") == "destroyed" else 2
                last_signature = signature

            event_name = latest.get("event")
            success_evidence = _strict_remote_success_evidence(events, run_id)
            if success_evidence is not None or _is_bound_remote_failure_event(
                latest, run_id
            ):
                # Terminal outcomes have their own verify/grace handling.
                terminal_seen = True
            if success_evidence is not None:
                if record.get("status") == "session_ready":
                    verified = True
                else:
                    try:
                        verified = pull_and_verify(
                            record, deadline_monotonic=cleanup_deadline
                        )
                    except Exception as error:
                        verified = False
                        safe_error = _redacted_exception_text(error)
                        _best_effort_monitor_event_if_active(
                            run_id,
                            "success_verification_retry",
                            error=safe_error,
                        )
                        try:
                            print(
                                json.dumps(
                                    {
                                        "event": "success_verification_retry",
                                        "error": safe_error,
                                    },
                                    sort_keys=True,
                                ),
                                flush=True,
                            )
                        except Exception:
                            pass
                if verified and handle_verified_success(
                    record,
                    manifest,
                    instance_id,
                    deadline_monotonic=controller_deadline,
                ):
                    return 0
            bound_failure = _is_bound_remote_failure_event(latest, run_id)
            if (
                bound_failure
                and event_name == "run_failed"
                and retain_failed_debug_session(
                    record,
                    manifest,
                    instance_id,
                    deadline_monotonic=cleanup_deadline,
                )
            ):
                failed_at = None
            elif bound_failure:
                if failed_at is None:
                    # The remote timestamp remains diagnostic only. Its clock
                    # domain is untrusted, so failure grace always starts when
                    # this controller first receives the terminal event.
                    failed_at = datetime.now(UTC)
                try:
                    pull_and_verify(
                        record,
                        best_effort=True,
                        deadline_monotonic=cleanup_deadline,
                    )
                except Exception as error:
                    _best_effort_monitor_event_if_active(
                        run_id,
                        "failure_recovery_failed",
                        error=_redacted_exception_text(error),
                    )
                grace = manifest.failure_grace_minutes * 60
                if (datetime.now(UTC) - failed_at).total_seconds() >= grace:
                    try:
                        destroy_instance(
                            instance_id,
                            run_id,
                            "failure_grace_elapsed",
                            deadline_monotonic=controller_deadline,
                        )
                        return 1
                    except Exception as error:
                        safe_error = _redacted_exception_text(error)
                        try:
                            print(
                                json.dumps(
                                    {"event": "destroy_failed", "error": safe_error},
                                    sort_keys=True,
                                ),
                                flush=True,
                            )
                        except Exception:
                            pass

        _bounded_sleep(manifest.poll_seconds, cleanup_deadline)


def process_is_alive(pid: int) -> bool:
    if os.name == "nt":
        # os.kill(pid, 0) is not a portable liveness probe on Windows. Query the
        # process handle directly so a live monitor's lock is not mistaken for
        # stale state and replaced by a duplicate monitor.
        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(
            process_query_limited_information, False, int(pid)
        )
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


_SUPERVISOR_LOCK_HANDLES: dict[Path, Any] = {}


def _read_supervisor_lock_pid(lock_path: Path) -> int:
    """Read PID metadata stored after the byte-range lock sentinel."""

    try:
        with lock_path.open("rb") as stream:
            stream.seek(1)
            raw = stream.read().decode("ascii").strip()
        if not raw.isdigit():
            return -1
        owner = int(raw)
        return owner if 0 < owner < 2**63 - 1 else -1
    except (OSError, UnicodeDecodeError, ValueError):
        return -1


def _claim_supervisor_lock(lock_path: Path, description: str) -> Path:
    """Claim an OS-held supervisor lease, recovering stale metadata safely."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = _open_lock_stream(lock_path)
    try:
        if os.name == "nt":
            import msvcrt

            _prepare_windows_lock_byte(stream)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError) as error:
        stream.close()
        if not _lock_is_contended(error):
            raise OpsError(f"{description} lock probe failed") from error
        owner = _read_supervisor_lock_pid(lock_path)
        if owner > 0 and process_is_alive(owner):
            raise OpsError(f"{description} is already supervised by PID {owner}") from error
        raise OpsError(f"{description} could not acquire its supervisor lock") from error
    stream.seek(0)
    stream.truncate()
    # Byte zero is the stable OS lease; metadata starts at byte one so Windows
    # readers can inspect it without violating the locked byte range.
    stream.write(b"\0" + str(os.getpid()).encode("ascii"))
    stream.flush()
    _SUPERVISOR_LOCK_HANDLES[lock_path] = stream
    return lock_path


def _supervisor_lock_owner_pid(lock_path: Path) -> int | None:
    """Return the metadata PID only when the corresponding OS lease is held.

    A live unrelated process can reuse a stale PID, so PID liveness alone is
    never supervision evidence.  The stable lock inode is authoritative; its
    text is read only after a nonblocking acquisition proves another process
    currently holds the lease.
    """

    if not lock_path.is_file():
        return None
    stream = lock_path.open("r+b")
    acquired = False
    try:
        try:
            if os.name == "nt":
                import msvcrt

                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except (BlockingIOError, OSError) as error:
            if not _lock_is_contended(error):
                raise OpsError("supervisor lock probe failed") from error
            owner = _read_supervisor_lock_pid(lock_path)
            if owner <= 0:
                # A held lock with damaged metadata is still real supervision;
                # fail closed rather than launching a duplicate writer.
                raise OpsError(
                    "held supervisor lock has invalid owner metadata"
                ) from None
            return owner
        return None
    finally:
        if acquired:
            if os.name == "nt":
                import msvcrt

                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def _release_supervisor_lock(lock_path: Path) -> None:
    """Release the OS lease while preserving the stable lock-file inode.

    Unlinking after unlock has an unavoidable handoff race: a successor can
    acquire and rewrite the same path between the old owner's metadata read and
    unlink, causing the old process to remove the successor's live lock inode.
    Stale PID text is harmless because acquisition is OS-lock-first, so the
    lock file remains as durable metadata instead.
    """

    stream = _SUPERVISOR_LOCK_HANDLES.pop(lock_path, None)
    if stream is not None:
        try:
            if os.name == "nt":
                import msvcrt

                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


def _wait_for_supervisor_lease(
    lock_path: Path,
    process: Any,
    description: str,
    *,
    deadline_monotonic: float | None = None,
) -> int:
    """Require two stable child-owned lease samples before handoff succeeds."""

    pid = _strict_positive_identity(process.pid, f"{description} PID")
    deadline = min(
        time.monotonic() + SUPERVISOR_HANDSHAKE_TIMEOUT_SECONDS,
        deadline_monotonic if deadline_monotonic is not None else float("inf"),
    )
    stable_owner_samples = 0
    while time.monotonic() < deadline:
        poll = getattr(process, "poll", None)
        if callable(poll):
            return_code = poll()
            if return_code is not None:
                raise OpsError(
                    f"{description} exited before a stable OS-lease handoff "
                    f"(return code {return_code})"
                )
        try:
            owner = _supervisor_lock_owner_pid(lock_path)
        except OpsError as error:
            if "invalid owner metadata" not in str(error):
                raise
            # The child acquires the byte-range lease immediately before it
            # truncates and rewrites PID metadata.  A reader can observe that
            # tiny interval; retry it instead of converting a healthy handoff
            # into a launch failure.
            owner = None
        if owner == pid:
            stable_owner_samples += 1
            if stable_owner_samples >= 2:
                if callable(poll) and poll() is not None:
                    raise OpsError(
                        f"{description} exited during its stable OS-lease handoff"
                    )
                return pid
        else:
            stable_owner_samples = 0
        _bounded_sleep(SUPERVISOR_HANDSHAKE_POLL_SECONDS, deadline)
    raise OpsError(f"{description} did not hold a stable OS lease before timeout")


def monitor_run(run_id: str) -> int:
    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = MONITOR_DIR / f"{run_id}.lock"
    _claim_supervisor_lock(lock_path, f"run {run_id!r} monitor")
    try:
        # Register the process that actually owns the monitor lease.  Detached
        # launches already write the child PID from the parent, but attached
        # monitoring runs in the provisioning process itself.  Without this
        # update the purchase reconciler could mistake a healthy attached
        # monitor for an incomplete handoff at the campaign deadline.
        update_record_fields(
            run_id,
            monitor_pid=_strict_positive_identity(os.getpid(), "monitor PID"),
            monitor_registered_at=utc_now(),
            provisioning_owner_pid=None,
        )
        return _monitor_run(run_id)
    finally:
        _release_supervisor_lock(lock_path)


def launch_detached_monitor(run_id: str) -> int:
    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = MONITOR_DIR / f"{run_id}.lock"
    lock_owner = _supervisor_lock_owner_pid(lock_path)
    if lock_owner is not None:
        return lock_owner
    log_path = MONITOR_DIR / f"{run_id}.log"
    command = [
        windowless_python_executable(),
        str(VAST_DIR / "ops.py"),
        "monitor",
        run_id,
    ]
    popen_options: dict[str, Any] = {
        "cwd": ROOT,
        "stdin": subprocess.DEVNULL,
        "stdout": None,
        "stderr": subprocess.STDOUT,
        "close_fds": True,
    }
    if os.name == "nt":
        popen_options["creationflags"] = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        )
    else:
        popen_options["start_new_session"] = True
    log_stream = None
    try:
        log_stream = log_path.open("a", encoding="utf-8")
        popen_options["stdout"] = log_stream
        process = _popen_detached(command, popen_options)
    except Exception as error:
        safe_error = _redacted_exception_text(error)
        _best_effort_record_secondary_failure(
            run_id,
            error=OpsError(safe_error),
            error_field="monitor_launch_error",
            error_at_field="monitor_launch_error_at",
            event="detached_monitor_launch_failed",
            state_fields={"monitor_launch_status": "launch_failed"},
        )
        raise OpsError(f"could not launch detached monitor: {safe_error}") from error
    finally:
        if log_stream is not None:
            log_stream.close()
    try:
        child_pid = _wait_for_supervisor_lease(
            lock_path, process, "ordinary monitor"
        )
    except Exception as error:
        safe_error = _redacted_exception_text(error)
        _best_effort_record_secondary_failure(
            run_id,
            error=OpsError(safe_error),
            error_field="monitor_launch_error",
            error_at_field="monitor_launch_error_at",
            event="detached_monitor_handshake_failed",
            state_fields={"monitor_launch_status": "launch_failed"},
        )
        raise OpsError(f"detached monitor handshake failed: {safe_error}") from error
    _best_effort_update_record_fields(
        run_id,
        monitor_pid=child_pid,
        monitor_log=str(log_path),
        monitor_started_at=utc_now(),
        monitor_launch_status="running",
    )
    _best_effort_append_local_event(
        run_id, "detached_monitor_started", pid=child_pid
    )
    return child_pid


def _destroy_reconciliation(run_id: str) -> int:
    """Boundedly watch a destroyed ordinary run for delayed reappearance."""

    lock_path = MONITOR_DIR / f"destroy_reconcile_{run_id}.lock"
    _claim_supervisor_lock(lock_path, f"destroy reconciliation for {run_id!r}")
    try:
        record = get_record(run_id)
        if isinstance(record.get("candidates"), list):
            raise OpsError("destroy reconciliation only supports ordinary runs")
        manifest = RunManifest.from_dict(record["manifest"])
        instance_id = record_instance_id(record)
        poll = max(
            DESTROY_RECONCILIATION_POLL_SECONDS,
            int(manifest.poll_seconds),
        )
        lease_seconds = destroy_reconciliation_lease_seconds(poll)
        controller_deadline = _record_deadline_monotonic(
            record,
            "controller_deadline_at",
            fallback_seconds=max(
                0.0,
                _campaign_hard_remaining_seconds(record, manifest)
                - REMOTE_WATCHDOG_SAFETY_SECONDS
                - LOCAL_WATCHDOG_ORDERING_MARGIN_SECONDS,
            ),
        )
        reconciliation_boundary = controller_deadline
        manual_boundary_raw = record.get("manual_destroy_reconciliation_deadline_at")
        if manual_boundary_raw is not None:
            manual_boundary = _record_deadline_monotonic(
                record,
                "manual_destroy_reconciliation_deadline_at",
                fallback_seconds=0,
            )
            reconciliation_boundary = max(
                reconciliation_boundary, manual_boundary
            )
        deadline = min(
            time.monotonic() + lease_seconds,
            reconciliation_boundary,
        )
        absent_scans = 0
        last_error: str | None = None
        last_absence_generation: int | None = None
        while deadline - time.monotonic() >= DESTROY_FOLLOWUP_MIN_QUERY_SECONDS:
            try:
                observation_generation = _cleanup_generation(run_id)
            except Exception as state_error:
                # Provider truth is still worth observing, but without the
                # generation token an absence cannot close cleanup terminally.
                observation_generation = None
                _best_effort_append_local_event(
                    run_id,
                    "destroy_reconciliation_generation_read_failed",
                    instance_id=instance_id,
                    error=_redacted_exception_text(state_error),
                )
            try:
                present = _live_instance_is_present(
                    instance_id, deadline_monotonic=deadline
                )
                last_error = None
            except Exception as error:
                last_error = _redacted_exception_text(error)
                _best_effort_mark_cleanup_required(
                    instance_id, run_id, last_error
                )
                _best_effort_append_local_event(
                    run_id,
                    "destroy_reconciliation_query_error",
                    instance_id=instance_id,
                    error=last_error,
                )
                present = None
            if present is False:
                absent_scans += 1
                last_absence_generation = observation_generation
            elif present is True:
                absent_scans = 0
                last_absence_generation = None
                _best_effort_mark_cleanup_required(
                    instance_id,
                    run_id,
                    "provider still reports instance during destroy follow-up",
                )
                try:
                    destroy_instance(
                        instance_id,
                        run_id,
                        "destroy_reconciliation_retry",
                        launch_supervisor=False,
                        deadline_monotonic=deadline,
                    )
                except Exception as error:
                    last_error = _redacted_exception_text(error)
                    _best_effort_append_local_event(
                        run_id,
                        "destroy_reconciliation_destroy_error",
                        instance_id=instance_id,
                        error=last_error,
                    )
                else:
                    # destroy_instance returned only after an exact live-API
                    # absence query, so that verification is the first clean
                    # observation after the reappearance.
                    absent_scans = 1
                    last_error = None
                    try:
                        last_absence_generation = _cleanup_generation(run_id)
                    except Exception as state_error:
                        last_absence_generation = None
                        _best_effort_append_local_event(
                            run_id,
                            "destroy_reconciliation_generation_read_failed",
                            instance_id=instance_id,
                            error=_redacted_exception_text(state_error),
                        )
            _bounded_sleep(poll, deadline)
        # Do not terminate this supervisor after the first couple of clean
        # snapshots.  Holding the full follow-up lease is what lets it catch a
        # delayed/reappearing contract.  Completion therefore requires the
        # final observations in that lease to be clean.
        if (
            absent_scans >= DESTROY_FOLLOWUP_ABSENT_SCANS
            and last_error is None
            and last_absence_generation is not None
        ):
            terminal_write_failed = False
            try:
                completed = _complete_destroy_reconciliation_if_current(
                    run_id,
                    expected_cleanup_generation=last_absence_generation,
                )
            except Exception as state_error:
                completed = False
                terminal_write_failed = True
                last_error = _redacted_exception_text(state_error)
                _best_effort_append_local_event(
                    run_id,
                    "destroy_reconciliation_terminal_write_failed",
                    instance_id=instance_id,
                    error=last_error,
                )
            if completed:
                _best_effort_append_local_event(
                    run_id,
                    "destroy_reconciliation_complete",
                    instance_id=instance_id,
                    absent_scans=absent_scans,
                )
                return 0
            _best_effort_append_local_event(
                run_id,
                "destroy_reconciliation_terminal_write_fenced",
                instance_id=instance_id,
            )
            if not terminal_write_failed:
                _best_effort_append_local_event(
                    run_id,
                    "destroy_reconciliation_expiry_suppressed_after_terminal_refusal",
                    instance_id=instance_id,
                )
                return 2
        message = last_error or "destroy follow-up lease expired before absence proof"
        _best_effort_mark_cleanup_required(
            instance_id, run_id, message, status="cleanup_required"
        )
        _best_effort_update_record_fields(
            run_id,
            destroy_reconciliation_status="expired",
            destroy_reconciliation_expired_at=utc_now(),
        )
        _best_effort_append_local_event(
            run_id,
            "destroy_reconciliation_expired",
            instance_id=instance_id,
            error=message,
        )
        return 2
    finally:
        _release_supervisor_lock(lock_path)


def launch_detached_destroy_reconciliation(
    run_id: str, *, deadline_monotonic: float | None = None
) -> int:
    """Start the ordinary-run delayed-destroy supervisor exactly once."""

    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    record = get_record(run_id)
    if isinstance(record.get("candidates"), list):
        raise OpsError("destroy reconciliation only supports ordinary runs")
    lock_owner = _supervisor_lock_owner_pid(
        MONITOR_DIR / f"destroy_reconcile_{run_id}.lock"
    )
    if lock_owner is not None:
        return lock_owner
    update_record_fields(
        run_id,
        destroy_reconciliation_status="launching",
        destroy_reconciliation_started_at=utc_now(),
    )
    lock_path = MONITOR_DIR / f"destroy_reconcile_{run_id}.lock"
    log_path = MONITOR_DIR / f"destroy_reconcile_{run_id}.log"
    command = [
        windowless_python_executable(),
        str(VAST_DIR / "ops.py"),
        "destroy-reconcile",
        run_id,
    ]
    popen_options: dict[str, Any] = {
        "cwd": ROOT,
        "stdin": subprocess.DEVNULL,
        "stderr": subprocess.STDOUT,
        "close_fds": True,
    }
    if os.name == "nt":
        popen_options["creationflags"] = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        )
    else:
        popen_options["start_new_session"] = True
    log_stream = None
    try:
        log_stream = log_path.open("a", encoding="utf-8")
        popen_options["stdout"] = log_stream
        process = _popen_detached(command, popen_options)
    except Exception as error:
        safe_error = _redacted_exception_text(error)
        _best_effort_record_secondary_failure(
            run_id,
            error=OpsError(safe_error),
            error_field="destroy_reconciliation_launch_error",
            error_at_field="destroy_reconciliation_launch_error_at",
            event="destroy_reconciliation_launch_failed",
        )
        try:
            update_record_fields(
                run_id,
                destroy_reconciliation_status="launch_failed",
                cleanup_required=True,
                cleanup_error=(
                    "could not launch destroy reconciliation: " + safe_error
                ),
                manual_reconciliation_required=True,
            )
        except Exception:
            pass
        raise OpsError(
            f"could not launch destroy reconciliation: {safe_error}"
        ) from error
    finally:
        if log_stream is not None:
            log_stream.close()
    try:
        child_pid = _wait_for_supervisor_lease(
            lock_path,
            process,
            "destroy reconciliation",
            deadline_monotonic=deadline_monotonic,
        )
    except Exception as error:
        safe_error = _redacted_exception_text(error)
        _best_effort_record_secondary_failure(
            run_id,
            error=OpsError(safe_error),
            error_field="destroy_reconciliation_handshake_error",
            error_at_field="destroy_reconciliation_handshake_error_at",
            event="destroy_reconciliation_handshake_failed",
        )
        try:
            update_record_fields(
                run_id,
                destroy_reconciliation_status="launch_failed",
                cleanup_required=True,
                cleanup_error=(
                    "destroy reconciliation handshake failed: " + safe_error
                ),
                manual_reconciliation_required=True,
            )
        except Exception:
            pass
        raise OpsError(
            f"destroy reconciliation handshake failed: {safe_error}"
        ) from error
    _best_effort_update_record_fields(
        run_id,
        destroy_reconciliation_pid=child_pid,
        destroy_reconciliation_status="running",
        manual_reconciliation_required=False,
    )
    _best_effort_append_local_event(
        run_id, "detached_destroy_reconciliation_started", pid=child_pid
    )
    return child_pid


def _launch_destroy_reconciliation_with_retries(
    run_id: str,
    *,
    instance_id: int,
    deadline_monotonic: float | None = None,
) -> int:
    """Hand off verified cleanup, with bounded spawn retries and fallback."""

    last_error: Exception | None = None
    for attempt in range(1, SUPERVISOR_LAUNCH_MAX_ATTEMPTS + 1):
        if (
            deadline_monotonic is not None
            and deadline_monotonic - time.monotonic() < 1
        ):
            last_error = OpsError("cleanup supervisor handoff deadline elapsed")
            break
        try:
            if deadline_monotonic is None:
                return launch_detached_destroy_reconciliation(run_id)
            return launch_detached_destroy_reconciliation(
                run_id, deadline_monotonic=deadline_monotonic
            )
        except Exception as error:
            safe_error = _redacted_exception_text(error)
            last_error = OpsError(safe_error)
            _best_effort_append_local_event(
                run_id,
                "cleanup_supervisor_launch_retry",
                instance_id=instance_id,
                attempt=attempt,
                error=safe_error,
            )
            if attempt < SUPERVISOR_LAUNCH_MAX_ATTEMPTS:
                _bounded_sleep(
                    SUPERVISOR_LAUNCH_RETRY_DELAY_SECONDS,
                    deadline_monotonic,
                )
    safe_last_error = _redacted_exception_text(
        last_error or OpsError("cleanup supervisor handoff failed")
    )
    try:
        update_record_fields(
            run_id,
            cleanup_required=True,
            manual_reconciliation_required=True,
            destroy_reconciliation_status="attached_fallback",
            cleanup_error=(
                "detached cleanup supervision unavailable: " + safe_last_error
            ),
        )
    except Exception:
        pass
    _best_effort_append_local_event(
        run_id,
        "destroy_reconciliation_attached_fallback",
        instance_id=instance_id,
        error=safe_last_error,
    )
    try:
        result = _destroy_reconciliation(run_id)
    except Exception as error:
        safe_error = _redacted_exception_text(error)
        raise OpsError(
            "detached cleanup supervision unavailable and attached "
            f"reconciliation failed: {safe_error}"
        ) from error
    if result != 0:
        raise OpsError(
            "detached cleanup supervision unavailable and attached "
            "reconciliation expired unresolved"
        )
    return os.getpid()


def _campaign_hard_remaining_seconds(
    record: dict[str, Any], manifest: RunManifest
) -> float:
    """Return time remaining to the original spend/TTL boundary."""

    persisted = record.get("provider_deadline_at")
    if persisted is not None:
        deadline = parse_utc(str(persisted))
        if deadline.tzinfo is None:
            raise OpsError("persisted provider_deadline_at must include a timezone")
        return max(0.0, (deadline - datetime.now(UTC)).total_seconds())
    started = record.get("campaign_started_at") or record.get("created_at")
    if not started:
        started = utc_now()
    elapsed = max(0.0, (datetime.now(UTC) - parse_utc(str(started))).total_seconds())
    try:
        campaign_offset = float(
            record.get("campaign_accrued_cost_usd", 0.0) or 0.0
        )
    except (TypeError, ValueError):
        campaign_offset = manifest.max_total_usd
    if not math.isfinite(campaign_offset) or campaign_offset < 0:
        campaign_offset = manifest.max_total_usd
    remaining_cap = max(0.0, manifest.max_total_usd - campaign_offset)
    # The approved hourly ceiling is authoritative.  A cheaper quote must not
    # extend a cleanup lease, and a stale recorded DPH must not weaken it.
    spend_seconds = remaining_cap / manifest.max_dph * 3600 - elapsed
    ttl_seconds = manifest.ttl_hours * 3600 - elapsed
    return max(0.0, min(spend_seconds, ttl_seconds))


def _purchase_reconciliation_budget_seconds(
    record: dict[str, Any], manifest: RunManifest
) -> float:
    """Return the primary lease remaining before the cleanup reserve."""

    reserve_seconds = cleanup_reserve_hours(manifest.poll_seconds) * 3600
    return max(
        0.0,
        _campaign_hard_remaining_seconds(record, manifest) - reserve_seconds,
    )


def _purchase_reconciliation_deadline(
    record: dict[str, Any], manifest: RunManifest
) -> datetime:
    """Return the persisted bounded cleanup deadline, creating it once."""

    raw = record.get("purchase_reconciliation_deadline_at") or record.get(
        "cleanup_start_at"
    )
    if raw is not None:
        try:
            deadline = parse_utc(str(raw))
            if deadline.tzinfo is not None:
                return deadline
        except (TypeError, ValueError):
            pass
    # ``_purchase_reconciliation_budget_seconds`` already subtracts elapsed
    # campaign time.  Add that remainder to *now*, not the original start, or a
    # legacy/restarted record would be charged elapsed time twice.
    deadline = datetime.now(UTC) + timedelta(
        seconds=_purchase_reconciliation_budget_seconds(record, manifest)
    )
    update_record_fields(
        manifest.run_id,
        purchase_reconciliation_deadline_at=deadline.isoformat().replace("+00:00", "Z"),
    )
    return deadline


def _late_create_reconciliation_deadline(
    record: dict[str, Any], manifest: RunManifest
) -> datetime:
    """Return the fixed cleanup-only late-create deadline without renewal."""

    raw = record.get("late_create_reconciliation_deadline_at")
    if raw is not None:
        try:
            deadline = parse_utc(str(raw))
            if deadline.tzinfo is not None:
                return deadline
        except (TypeError, ValueError):
            pass
    primary = _purchase_reconciliation_deadline(record, manifest)
    # The last bounded query is taken from the tail of this lease. Restarting
    # reads this persisted timestamp instead of granting another 24 hours.
    deadline = primary + timedelta(
        seconds=LATE_CREATE_RECONCILIATION_SECONDS
    )
    update_record_fields(
        manifest.run_id,
        late_create_reconciliation_deadline_at=(
            deadline.isoformat().replace("+00:00", "Z")
        ),
    )
    return deadline


def _scan_reconcilable_purchase_instances(
    record: dict[str, Any], *, deadline_monotonic: float
) -> list[dict[str, Any]]:
    """Return only exact label+offer+machine matches within one API lease."""

    live = _require_provider_instance_rows(
        run_vast(
            "show",
            "instances",
            timeout=_deadline_timeout_seconds(
                deadline_monotonic, default=DESTROY_API_TIMEOUT_SECONDS
            ),
        ),
        "purchase reconciliation query",
        require_label=True,
    )
    expected_label = str(record.get("expected_label") or "")
    return [
        item
        for item in live
        if isinstance(item, dict)
        and expected_label
        and _is_reconcilable_purchase_instance(record, item)
    ]


def _drain_reconcilable_purchase_instances(
    run_id: str,
    matches: list[dict[str, Any]],
    *,
    reason: str,
    deadline_monotonic: float,
    destroyed_ids: set[int],
) -> tuple[bool, bool]:
    """Destroy exact uncertain-create matches without declaring label closure."""

    observed = False
    all_destroyed = True
    for match in matches:
        observed = True
        instance_id = _offer_identity_number(match, "id")
        if instance_id >= 2**63 - 1:
            all_destroyed = False
            _best_effort_update_record_fields(
                run_id,
                reconciliation_error=(
                    "matching live instance did not report a positive ID"
                ),
                cleanup_required=True,
            )
            continue
        # ``destroy_instance`` claims lifecycle authority and persists this
        # exact ID before provider mutation.  No diagnostic replica may sit in
        # front of that control transition.
        _best_effort_append_local_event(
            run_id,
            "purchase_reconciled_instance_found",
            instance_id=instance_id,
            phase=reason,
        )
        try:
            destroy_instance(
                instance_id,
                run_id,
                reason,
                launch_supervisor=False,
                deadline_monotonic=deadline_monotonic,
                mark_terminal=False,
                unfenced_exact_purchase_match=match,
            )
        except Exception as error:
            all_destroyed = False
            safe_error = _redacted_exception_text(error)
            _best_effort_update_record_fields(
                run_id,
                status="purchase_reconciled_cleanup_pending",
                cleanup_required=True,
                cleanup_error=safe_error,
            )
            _best_effort_append_local_event(
                run_id,
                "purchase_reconciled_destroy_failed",
                instance_id=instance_id,
                phase=reason,
                error=safe_error,
            )
        else:
            destroyed_ids.add(instance_id)
            _best_effort_update_record_fields(
                run_id,
                status="purchase_reconciled_cleanup_pending",
                cleanup_required=True,
                last_reconciled_destroyed_at=utc_now(),
            )
    return observed, all_destroyed


def _try_drain_reconcilable_purchase_instances(
    run_id: str,
    matches: list[dict[str, Any]],
    *,
    reason: str,
    deadline_monotonic: float,
    destroyed_ids: set[int],
) -> tuple[bool, bool]:
    """Contain one drain bug so the bounded sentinel can scan again."""

    try:
        return _drain_reconcilable_purchase_instances(
            run_id,
            matches,
            reason=reason,
            deadline_monotonic=deadline_monotonic,
            destroyed_ids=destroyed_ids,
        )
    except Exception as error:
        safe_error = _redacted_exception_text(error)
        _best_effort_update_record_fields(
            run_id,
            reconciliation_error=safe_error,
            cleanup_required=True,
        )
        _best_effort_append_local_event(
            run_id,
            "purchase_reconciliation_drain_failed",
            phase=reason,
            error=safe_error,
        )
        return bool(matches), False


def _purchase_reconciliation(run_id: str) -> int:
    """Resolve an uncertain create, then watch boundedly for late visibility.

    The primary phase ends before the spend/TTL cleanup reserve.  A second,
    fixed 24-hour cleanup-only sentinel performs sparse exact-match scans after
    that boundary.  It never creates, adopts, bootstraps, or runs workload, and
    its persisted deadline cannot be renewed by a process restart.
    """

    lock_path = MONITOR_DIR / f"purchase_reconcile_{run_id}.lock"
    _claim_supervisor_lock(lock_path, f"purchase reconciliation for {run_id!r}")
    try:
        record = get_record(run_id)
        manifest = RunManifest.from_dict(record["manifest"])
        deadline_utc = _purchase_reconciliation_deadline(record, manifest)
        deadline = time.monotonic() + max(
            0.0, (deadline_utc - datetime.now(UTC)).total_seconds()
        )
        # Tests may inject a tiny primary lease. Production uses the persisted
        # campaign-derived deadline, not this convenience constant.
        if PURCHASE_RECONCILIATION_TIMEOUT_SECONDS < 180:
            deadline = min(
                deadline,
                time.monotonic() + max(0, PURCHASE_RECONCILIATION_TIMEOUT_SECONDS),
            )
        hard_deadline = _record_deadline_monotonic(
            record,
            "controller_deadline_at",
            fallback_seconds=max(
                0.0,
                _campaign_hard_remaining_seconds(record, manifest)
                - REMOTE_WATCHDOG_SAFETY_SECONDS
                - LOCAL_WATCHDOG_ORDERING_MARGIN_SECONDS,
            ),
        )
        destroyed_ids: set[int] = set()
        observed_match = False
        destroy_error_observed = False
        query_error_observed = False
        first_pass = True
        owner_pid = record.get("provisioning_owner_pid")
        while first_pass or time.monotonic() < deadline:
            first_pass = False
            record = get_record(run_id)
            if record.get("status") in {"running", "session_ready"}:
                monitor_owner = _supervisor_lock_owner_pid(
                    MONITOR_DIR / f"{run_id}.lock"
                )
                if monitor_owner is not None:
                    return 0
                owner_alive = (
                    isinstance(owner_pid, int)
                    and owner_pid > 0
                    and process_is_alive(owner_pid)
                )
                if owner_alive and time.monotonic() < deadline:
                    _bounded_sleep(PURCHASE_RECONCILIATION_POLL_SECONDS, deadline)
                    continue
                try:
                    launch_detached_monitor(run_id)
                except Exception as error:
                    safe_error = _redacted_exception_text(error)
                    _best_effort_update_record_fields(
                        run_id,
                        reconciliation_error=(
                            "remote run monitor takeover failed: " + safe_error
                        ),
                    )
                    if time.monotonic() < deadline:
                        _bounded_sleep(
                            PURCHASE_RECONCILIATION_POLL_SECONDS, deadline
                        )
                        continue
                    try:
                        instance_id = record_instance_id(record)
                        destroy_instance(
                            instance_id,
                            run_id,
                            "monitor_takeover_failed_at_reconciliation_deadline",
                            deadline_monotonic=hard_deadline,
                        )
                    except Exception as destroy_error:
                        safe_destroy_error = _redacted_exception_text(destroy_error)
                        _best_effort_update_record_fields(
                            run_id,
                            status="purchase_reconciled_cleanup_pending",
                            cleanup_required=True,
                            cleanup_error=safe_destroy_error,
                        )
                        _best_effort_append_local_event(
                            run_id,
                            "purchase_reconciliation_monitor_cleanup_failed",
                            error=safe_destroy_error,
                        )
                        return 2
                return 0
            if record.get("status") in {"race_failed", "race_spend_cap"}:
                return 0
            if (
                isinstance(owner_pid, int)
                and owner_pid > 0
                and process_is_alive(owner_pid)
                and time.monotonic() < deadline
            ):
                _bounded_sleep(PURCHASE_RECONCILIATION_POLL_SECONDS, deadline)
                continue
            if deadline - time.monotonic() < 1:
                break
            try:
                matches = _scan_reconcilable_purchase_instances(
                    record, deadline_monotonic=deadline
                )
            except Exception as error:
                query_error_observed = True
                _best_effort_update_record_fields(
                    run_id,
                    reconciliation_error=_redacted_exception_text(error),
                )
            else:
                observed, destroyed = _try_drain_reconcilable_purchase_instances(
                    run_id,
                    matches,
                    reason="purchase_outcome_reconciled",
                    deadline_monotonic=hard_deadline,
                    destroyed_ids=destroyed_ids,
                )
                observed_match = observed_match or observed
                destroy_error_observed = destroy_error_observed or not destroyed
            _bounded_sleep(PURCHASE_RECONCILIATION_POLL_SECONDS, deadline)

        # The final primary scan has its own explicit allocation in the cleanup
        # reserve and is still clamped to the original hard boundary.
        record = get_record(run_id)
        try:
            final_scan_deadline = min(
                hard_deadline,
                time.monotonic() + PURCHASE_FINAL_SCAN_RESERVE_SECONDS,
            )
            matches = _scan_reconcilable_purchase_instances(
                record, deadline_monotonic=final_scan_deadline
            )
        except Exception as error:
            query_error_observed = True
            _best_effort_update_record_fields(
                run_id,
                reconciliation_error=_redacted_exception_text(error),
            )
        else:
            observed, destroyed = _try_drain_reconcilable_purchase_instances(
                run_id,
                matches,
                reason="purchase_outcome_reconciled",
                deadline_monotonic=hard_deadline,
                destroyed_ids=destroyed_ids,
            )
            observed_match = observed_match or observed
            destroy_error_observed = destroy_error_observed or not destroyed

        pre_late_record = get_record(run_id)
        late_deadline_utc = _late_create_reconciliation_deadline(
            pre_late_record, manifest
        )
        late_deadline = time.monotonic() + max(
            0.0, (late_deadline_utc - datetime.now(UTC)).total_seconds()
        )
        # The persisted timestamp, not a mutable status string, decides whether
        # this invocation is already outside the fixed late-create lease.
        # ``reconcile`` may legitimately enrich or reopen the status after it
        # discovers provider truth; that must never suppress the one bounded
        # cleanup-only emergency scan.
        expired_manual_restart = late_deadline - time.monotonic() < 1
        if expired_manual_restart:
            # An explicit reconcile/restart after the fixed sentinel has ended
            # gets one fresh cleanup-only scan. It cannot be described as cap-
            # compliant because the original controller boundary has passed.
            emergency_scan_deadline = (
                time.monotonic() + PURCHASE_FINAL_SCAN_RESERVE_SECONDS
            )
            _best_effort_update_record_fields(
                run_id,
                cap_control_failed=True,
                cap_control_failure_at=utc_now(),
                cap_control_failure_reason=(
                    "manual late-create reconciliation after fixed deadline"
                ),
            )
            _best_effort_append_local_event(
                run_id, "late_create_manual_reconciliation_scan"
            )
            try:
                matches = _scan_reconcilable_purchase_instances(
                    pre_late_record,
                    deadline_monotonic=emergency_scan_deadline,
                )
            except Exception as error:
                query_error_observed = True
                _best_effort_update_record_fields(
                    run_id,
                    reconciliation_error=_redacted_exception_text(error),
                )
            else:
                emergency_destroy_deadline = (
                    time.monotonic() + destroy_attempt_budget_seconds()
                )
                observed, destroyed = _try_drain_reconcilable_purchase_instances(
                    run_id,
                    matches,
                    reason="late_purchase_outcome_reconciled",
                    deadline_monotonic=emergency_destroy_deadline,
                    destroyed_ids=destroyed_ids,
                )
                observed_match = observed_match or observed
                destroy_error_observed = destroy_error_observed or not destroyed
        late_poll_deadline = max(
            time.monotonic(),
            late_deadline - PURCHASE_FINAL_SCAN_RESERVE_SECONDS,
        )
        _best_effort_update_record_fields(
            run_id,
            status="late_create_reconciliation_active",
            purchase_reconciliation_status="late_watch_running",
            late_create_reconciliation_started_at=utc_now(),
            cleanup_required=True,
            reconciliation_observed_match=observed_match,
        )
        _best_effort_append_local_event(
            run_id,
            "late_create_reconciliation_started",
            deadline_at=late_deadline_utc.isoformat().replace("+00:00", "Z"),
        )
        while late_poll_deadline - time.monotonic() >= 1:
            record = get_record(run_id)
            try:
                matches = _scan_reconcilable_purchase_instances(
                    record, deadline_monotonic=late_poll_deadline
                )
            except Exception as error:
                query_error_observed = True
                _best_effort_update_record_fields(
                    run_id,
                    reconciliation_error=_redacted_exception_text(error),
                )
            else:
                late_destroy_deadline = min(
                    late_deadline,
                    time.monotonic() + destroy_attempt_budget_seconds(),
                )
                observed, destroyed = _try_drain_reconcilable_purchase_instances(
                    run_id,
                    matches,
                    reason="late_purchase_outcome_reconciled",
                    deadline_monotonic=late_destroy_deadline,
                    destroyed_ids=destroyed_ids,
                )
                observed_match = observed_match or observed
                destroy_error_observed = destroy_error_observed or not destroyed
            _bounded_sleep(
                LATE_CREATE_RECONCILIATION_POLL_SECONDS, late_poll_deadline
            )

        # Reserve the tail of the fixed late lease for one final bounded scan.
        if late_deadline - time.monotonic() >= 1:
            record = get_record(run_id)
            try:
                matches = _scan_reconcilable_purchase_instances(
                    record, deadline_monotonic=late_deadline
                )
            except Exception as error:
                query_error_observed = True
                _best_effort_update_record_fields(
                    run_id,
                    reconciliation_error=_redacted_exception_text(error),
                )
            else:
                late_destroy_deadline = min(
                    late_deadline,
                    time.monotonic() + destroy_attempt_budget_seconds(),
                )
                observed, destroyed = _try_drain_reconcilable_purchase_instances(
                    run_id,
                    matches,
                    reason="late_purchase_outcome_reconciled",
                    deadline_monotonic=late_destroy_deadline,
                    destroyed_ids=destroyed_ids,
                )
                observed_match = observed_match or observed
                destroy_error_observed = destroy_error_observed or not destroyed

        _best_effort_update_record_fields(
            run_id,
            status="late_create_reconciliation_expired",
            purchase_reconciliation_status="late_watch_expired",
            cleanup_required=True,
            manual_reconciliation_required=True,
            cap_control_failed=True,
            cap_control_failure_at=utc_now(),
            cap_control_failure_reason=(
                "fixed late-create reconciliation expired without provider closure"
            ),
            reconciliation_observed_match=observed_match,
            reconciliation_query_error_observed=query_error_observed,
            reconciliation_destroy_error_observed=destroy_error_observed,
            reconciliation_expired_at=utc_now(),
        )
        _best_effort_append_local_event(
            run_id,
            "late_create_reconciliation_expired",
            observed_match=observed_match,
            query_error_observed=query_error_observed,
            destroy_error_observed=destroy_error_observed,
            destroyed_instance_ids=sorted(destroyed_ids),
        )
        # Even a clean 24-hour observation cannot prove the provider will never
        # materialize an older request.  Preserve a fail-closed manual state.
        return 2
    finally:
        _release_supervisor_lock(lock_path)


def launch_detached_purchase_reconciliation(run_id: str) -> int:
    """Start a durable bounded setup/cleanup supervisor before purchase work."""

    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    lock_owner = _supervisor_lock_owner_pid(
        MONITOR_DIR / f"purchase_reconcile_{run_id}.lock"
    )
    if lock_owner is not None:
        return lock_owner
    record = get_record(run_id)
    manifest = RunManifest.from_dict(record["manifest"])
    deadline = _purchase_reconciliation_deadline(record, manifest)
    late_deadline = _late_create_reconciliation_deadline(record, manifest)
    # Register the durable lease before spawning.  The child can start and
    # write state immediately; the post-spawn update below merges into the
    # current record and therefore cannot overwrite child progress.
    update_record_fields(
        run_id,
        purchase_reconciliation_deadline_at=deadline.isoformat().replace("+00:00", "Z"),
        late_create_reconciliation_deadline_at=(
            late_deadline.isoformat().replace("+00:00", "Z")
        ),
        purchase_reconciliation_status="launching",
    )
    log_path = MONITOR_DIR / f"purchase_reconcile_{run_id}.log"
    command = [
        windowless_python_executable(),
        str(VAST_DIR / "ops.py"),
        "purchase-reconcile",
        run_id,
    ]
    popen_options: dict[str, Any] = {
        "cwd": ROOT,
        "stdin": subprocess.DEVNULL,
        "stderr": subprocess.STDOUT,
        "close_fds": True,
    }
    if os.name == "nt":
        popen_options["creationflags"] = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        )
    else:
        popen_options["start_new_session"] = True
    log_stream = None
    try:
        log_stream = log_path.open("a", encoding="utf-8")
        popen_options["stdout"] = log_stream
        process = _popen_detached(command, popen_options)
    except Exception as error:
        safe_error = _redacted_exception_text(error)
        _best_effort_update_record_fields(
            run_id,
            purchase_reconciliation_status="launch_failed",
            reconciliation_error=(
                "could not launch purchase reconciliation: " + safe_error
            ),
        )
        raise OpsError(
            f"could not launch purchase reconciliation: {safe_error}"
        ) from error
    finally:
        if log_stream is not None:
            log_stream.close()
    lock_path = MONITOR_DIR / f"purchase_reconcile_{run_id}.lock"
    try:
        child_pid = _wait_for_supervisor_lease(
            lock_path, process, "purchase reconciliation"
        )
    except Exception as error:
        safe_error = _redacted_exception_text(error)
        _best_effort_update_record_fields(
            run_id,
            purchase_reconciliation_status="launch_failed",
            reconciliation_error=(
                "purchase reconciliation handshake failed: " + safe_error
            ),
        )
        raise OpsError(
            f"purchase reconciliation handshake failed: {safe_error}"
        ) from error
    _best_effort_update_record_fields(
        run_id,
        purchase_reconciliation_pid=child_pid,
        purchase_reconciliation_started_at=utc_now(),
        purchase_reconciliation_status="running",
    )
    _best_effort_append_local_event(
        run_id, "detached_purchase_reconciliation_started", pid=child_pid
    )
    return child_pid


def _race_cleanup_pending(run_id: str) -> int:
    """Retry destruction of every persisted race candidate until all are gone."""

    lock_path = MONITOR_DIR / f"race_cleanup_{run_id}.lock"
    _claim_supervisor_lock(lock_path, f"race cleanup for {run_id!r}")
    try:
        return _race_cleanup_pending_unlocked(run_id)
    finally:
        _release_supervisor_lock(lock_path)


def _race_reconcile_candidates(
    record: dict[str, Any], manifest: RunManifest
) -> list[dict[str, Any]]:
    """Discover persisted candidate labels and mark absent contracts terminal."""

    live = _require_provider_instance_rows(
        run_vast("show", "instances"),
        "race reconciliation query",
        require_label=True,
    )
    live_by_id: dict[int, dict[str, Any]] = {}
    for item in live:
        if not isinstance(item, dict):
            continue
        item_id = _offer_identity_number(item, "id")
        if item_id < 2**63 - 1:
            live_by_id[item_id] = item
    changed = False
    for candidate in record.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        candidate_id = candidate.get("instance_id")
        label = str(candidate.get("label") or "")
        matches = [item for item in live if item.get("label") == label] if label else []
        valid_matches = [
            item
            for item in matches
            if _offer_identity_number(item, "id") < 2**63 - 1
        ]
        discovered_ids = sorted(
            {
                _strict_positive_identity(
                    item.get("id"), f"race candidate {label!r} live instance ID"
                )
                for item in valid_matches
            }
        )
        if candidate.get("discovered_instance_ids") != discovered_ids:
            candidate["discovered_instance_ids"] = discovered_ids
            changed = True
        known_candidate_id = _offer_identity_number({"id": candidate_id}, "id")
        if valid_matches and (
            known_candidate_id >= 2**63 - 1
            or known_candidate_id
            != _strict_positive_identity(
                valid_matches[0].get("id"), "race live instance ID"
            )
            or candidate.get("status") in {"destroyed", "absent_from_live_api"}
        ):
            candidate_id = _strict_positive_identity(
                valid_matches[0].get("id"), "race live instance ID"
            )
            candidate["instance_id"] = candidate_id
            candidate["status"] = "reconciled_after_interruption"
            if not _copy_reconciled_live_dph(candidate, valid_matches[0]):
                candidate["status"] = "cleanup_pending"
                candidate["cleanup_required"] = True
                record["status"] = "race_cleanup_required"
                record["cleanup_required"] = True
            changed = True
        candidate_id = _offer_identity_number(
            {"id": candidate.get("instance_id")}, "id"
        )
        if candidate_id >= 2**63 - 1:
            candidate_id = None
        if candidate_id is not None and candidate_id not in live_by_id:
            if candidate.get("status") not in {"destroyed", "absent_from_live_api"}:
                candidate["status"] = "absent_from_live_api"
                candidate["absent_from_live_at"] = utc_now()
                changed = True
        elif (
            candidate_id is None
            and not matches
            and candidate.get("status")
            not in {"destroyed", "absent_from_live_api"}
        ):
            if candidate.get("status") in {
                "purchase_outcome_unknown",
                "purchasing",
                "verifying_live_contract",
            }:
                candidate["live_miss_count"] = int(
                    candidate.get("live_miss_count", 0)
                ) + 1
                unknown_since = (
                    candidate.get("unknown_since")
                    or candidate.get("started_at")
                    or utc_now()
                )
                candidate["unknown_since"] = unknown_since
                elapsed = max(
                    0.0,
                    (
                        datetime.now(UTC) - parse_utc(str(unknown_since))
                    ).total_seconds(),
                )
                # Allow Vast's create endpoint and show endpoint to converge
                # for at least the same 90-second window used by live checks.
                if (
                    elapsed >= DEFAULT_SSH_READY_TIMEOUT_SECONDS
                    and int(candidate["live_miss_count"]) >= 2
                ):
                    candidate["status"] = "absent_from_live_api"
                    candidate["absent_from_live_at"] = utc_now()
                changed = True
            else:
                candidate["status"] = "absent_from_live_api"
                candidate["absent_from_live_at"] = utc_now()
                changed = True
    if changed:
        record["last_candidate_reconciliation_at"] = utc_now()
        replace_record(record)
    return live


def _race_cleanup_pending_unlocked(run_id: str) -> int:
    """Retry destruction until every known or discoverable contract is gone."""
    record = get_record(run_id)
    manifest = RunManifest.from_dict(record["manifest"])
    lock = threading.Lock()
    cleanup_only_losers = bool(record.get("cleanup_only_losers"))
    winner_index = record.get("winner_index")
    if not isinstance(winner_index, int) or winner_index < 0:
        cleanup_only_losers = False
        winner_index = -1
    hard_deadline = parse_utc(
        str(record.get("campaign_started_at") or record["created_at"])
    ).timestamp() + (
        min(
            manifest.ttl_hours,
            max(
                0.0,
                (manifest.max_total_usd
                 - float(record.get("campaign_accrued_cost_usd", 0.0) or 0.0))
                / max(
                    0.000001,
                    sum(
                        float(candidate.get("dph", 0.0) or 0.0)
                        for index, candidate in enumerate(record.get("candidates", []))
                        if isinstance(candidate, dict)
                        and not (cleanup_only_losers and index == winner_index)
                        and candidate.get("status")
                        not in {"destroyed", "absent_from_live_api"}
                    ),
                ),
            ),
        )
        * 3600
        - cleanup_reserve_hours(manifest.poll_seconds) * 3600
    )
    ttl_reported = False
    while True:
        _race_reconcile_candidates(record, manifest)
        _race_cleanup_losers(
            manifest,
            record,
            lock,
            winner_index if cleanup_only_losers else -1,
        )
        pending = [
            candidate
            for index, candidate in enumerate(record.get("candidates", []))
            if not (cleanup_only_losers and index == winner_index)
            if candidate.get("instance_id") is not None
            and candidate.get("status")
            not in {"destroyed", "absent_from_live_api"}
        ]
        unresolved = [
            candidate
            for index, candidate in enumerate(record.get("candidates", []))
            if not (cleanup_only_losers and index == winner_index)
            if candidate.get("instance_id") is None
            and candidate.get("status") not in {"absent_from_live_api", "destroyed"}
        ]
        if not pending and not unresolved:
            if cleanup_only_losers:
                record["status"] = record.get("cleanup_resume_status", "running")
                record.pop("cleanup_only_losers", None)
                record.pop("cleanup_resume_status", None)
            else:
                record["status"] = "race_failed"
            record["cleanup_finished_at"] = utc_now()
            replace_record(record)
            return 0
        if time.time() >= hard_deadline and not ttl_reported:
            record["status"] = "provisioning_cleanup_pending"
            record["cleanup_required"] = True
            record["cleanup_last_error"] = (
                "race cleanup reserve reached with unresolved live candidates; "
                "run reconcile after the provider converges"
            )
            replace_record(record)
            append_local_event(
                run_id,
                "race_cleanup_ttl_reached",
                pending_instances=[
                    candidate.get("instance_id") for candidate in pending
                ],
            )
            ttl_reported = True
            return 2
        time.sleep(min(30, manifest.poll_seconds))


def launch_detached_race_cleanup(run_id: str) -> int:
    """Start a bounded detached cleanup supervisor for an interrupted race."""

    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    log_path = MONITOR_DIR / f"race_cleanup_{run_id}.log"
    command = [
        windowless_python_executable(),
        str(VAST_DIR / "ops.py"),
        "race-cleanup",
        run_id,
    ]
    popen_options: dict[str, Any] = {
        "cwd": ROOT,
        "stdin": subprocess.DEVNULL,
        "stdout": None,
        "stderr": subprocess.STDOUT,
        "close_fds": True,
    }
    if os.name == "nt":
        popen_options["creationflags"] = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        )
    else:
        popen_options["start_new_session"] = True
    with log_path.open("a", encoding="utf-8") as log:
        popen_options["stdout"] = log
        process = _popen_detached(command, popen_options)
    update_record_fields(
        run_id,
        cleanup_monitor_pid=_strict_positive_identity(
            process.pid, "cleanup monitor PID"
        ),
        cleanup_monitor_log=str(log_path),
        cleanup_monitor_started_at=utc_now(),
    )
    append_local_event(run_id, "detached_race_cleanup_started", pid=process.pid)
    return process.pid


def _race_watchdog(run_id: str) -> int:
    """Watch an active coordinator and take over only after it stops safely."""

    lock_path = MONITOR_DIR / f"race_watchdog_{run_id}.lock"
    try:
        _claim_supervisor_lock(lock_path, f"race watchdog for {run_id!r}")
    except OpsError:
        return 0
    try:
        while True:
            record = get_record(run_id)
            status = record.get("status")
            if status in {
                "running",
                "session_ready",
                "race_failed",
                "race_spend_cap",
                "destroyed",
            }:
                return 0
            coordinator_pid = record.get("coordinator_pid")
            coordinator_alive = (
                isinstance(coordinator_pid, int)
                and coordinator_pid > 0
                and process_is_alive(coordinator_pid)
            )
            campaign_elapsed = elapsed_since(record, "campaign_started_at")
            created_candidates = [
                candidate
                for candidate in record.get("candidates", [])
                if isinstance(candidate, dict)
            ]
            actual_cost = sum(
                _race_candidate_cost(candidate) for candidate in created_candidates
            )
            cleanup_reserve_cost = sum(
                float(candidate.get("dph", 0.0) or 0.0)
                for candidate in created_candidates
                if candidate.get("status") not in {"destroyed", "absent_from_live_api"}
            ) * cleanup_reserve_hours(
                RunManifest.from_dict(record["manifest"]).poll_seconds
            )
            campaign_over_cap = (
                actual_cost + cleanup_reserve_cost
                >= float(record["campaign_cap_usd"])
            )
            manifest = RunManifest.from_dict(record["manifest"])
            ttl_cleanup_reached = campaign_elapsed >= max(
                0.0,
                manifest.ttl_hours - cleanup_reserve_hours(manifest.poll_seconds),
            )
            cleanup_required = status in {
                "provisioning_cleanup_pending",
                "race_cleanup_failed",
                "race_cleanup_required",
            }
            if (
                not coordinator_alive
                or cleanup_required
                or campaign_over_cap
                or ttl_cleanup_reached
            ):
                # The coordinator is no longer allowed to mutate state before
                # this process takes ownership of the cleanup ledger.
                record["status"] = "provisioning_cleanup_pending"
                record["watchdog_takeover_at"] = utc_now()
                record["watchdog_reason"] = (
                    "coordinator_lost"
                    if not coordinator_alive
                    else "cleanup_required_or_campaign_limit"
                )
                replace_record(record)
                append_local_event(
                    run_id,
                    "race_watchdog_takeover",
                    reason=record["watchdog_reason"],
                )
                return _race_cleanup_pending(run_id)
            time.sleep(min(30, RunManifest.from_dict(record["manifest"]).poll_seconds))
    finally:
        _release_supervisor_lock(lock_path)


def launch_detached_race_watchdog(run_id: str) -> int:
    """Start the one per-campaign provisioning watchdog before candidate creates."""

    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    log_path = MONITOR_DIR / f"race_watchdog_{run_id}.log"
    command = [
        windowless_python_executable(),
        str(VAST_DIR / "ops.py"),
        "race-watchdog",
        run_id,
    ]
    popen_options: dict[str, Any] = {
        "cwd": ROOT,
        "stdin": subprocess.DEVNULL,
        "stdout": None,
        "stderr": subprocess.STDOUT,
        "close_fds": True,
    }
    if os.name == "nt":
        popen_options["creationflags"] = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        )
    else:
        popen_options["start_new_session"] = True
    with log_path.open("a", encoding="utf-8") as log:
        popen_options["stdout"] = log
        process = _popen_detached(command, popen_options)
    update_record_fields(
        run_id,
        watchdog_pid=_strict_positive_identity(process.pid, "watchdog PID"),
        watchdog_log=str(log_path),
        watchdog_started_at=utc_now(),
    )
    append_local_event(run_id, "detached_race_watchdog_started", pid=process.pid)
    return process.pid


def start_requested_monitor(run_id: str, detach: bool) -> int:
    if not detach:
        return monitor_run(run_id)
    try:
        launch_detached_monitor(run_id)
    except Exception as error:
        safe_error = _redacted_exception_text(error)
        _best_effort_append_local_event(
            run_id,
            "detached_monitor_failed",
            error=safe_error,
            fallback="attached_monitor",
        )
        _best_effort_emit_json(
            {
                "event": "detached_monitor_failed",
                "error": safe_error,
                "fallback": "attached_monitor",
            }
        )
        return monitor_run(run_id)
    return 0


def _stage_artifact_destination(stage: dict[str, Any]) -> Path:
    return (
        ROOT
        / "artifacts"
        / "remote"
        / stage["parent_run_id"]
        / "stages"
        / stage["stage_id"]
    )


def _monitor_session_stage(stage_id: str) -> int:
    stage = get_stage_record(stage_id)
    parent = get_record(stage["parent_run_id"])
    manifest = RunManifest.from_dict(stage["manifest"])
    instance_id = record_instance_id(parent)
    last_signature = stage.get("last_event_signature")
    live_error_reported = False
    transport_error_reported = False
    while True:
        try:
            live = _require_provider_instance_rows(
                run_vast("show", "instances"), "stage live instance query"
            )
            live_ids: set[int] = set()
            for item in live:
                if not isinstance(item, dict) or "id" not in item:
                    raise OpsError("stage live instance entry is malformed")
                live_ids.add(
                    _strict_positive_identity(item.get("id"), "live instance ID")
                )
            live_error_reported = False
        except Exception as error:
            if not live_error_reported:
                _best_effort_append_local_event(
                    stage_id,
                    "stage_live_query_error",
                    error=_redacted_exception_text(error),
                )
                live_error_reported = True
            time.sleep(manifest.poll_seconds)
            continue
        if instance_id not in live_ids:
            stage["status"] = "instance_missing"
            stage["finished_at"] = utc_now()
            replace_stage_record(stage)
            _best_effort_append_local_event(stage_id, "stage_instance_missing")
            return 2
        try:
            connection = ssh_url(instance_id)
            events = _sanitized_remote_events(
                read_remote_events(connection, remote_ops=stage["remote_ops"])
            )
            transport_error_reported = False
        except Exception as error:
            if not transport_error_reported:
                _best_effort_append_local_event(
                    stage_id,
                    "stage_transport_error",
                    error=_redacted_exception_text(error),
                )
                transport_error_reported = True
            time.sleep(manifest.poll_seconds)
            continue
        if events:
            latest = events[-1]
            signature = hashlib.sha256(
                json.dumps(latest, sort_keys=True).encode("utf-8")
            ).hexdigest()
            if signature != last_signature:
                if latest.get("event") != "heartbeat":
                    _best_effort_emit_json(latest)
                last_signature = signature
                stage["last_event_signature"] = signature
                stage["last_remote_event"] = latest
                replace_stage_record(stage)
                if latest.get("event") != "heartbeat":
                    _best_effort_append_local_event(
                        stage_id, "stage_remote_state_change", remote=latest
                    )
            success_evidence = _strict_remote_success_evidence(events, stage_id)
            if success_evidence is not None:
                try:
                    verified = pull_and_verify(
                        stage,
                        destination=_stage_artifact_destination(stage),
                        remote_ops=stage["remote_ops"],
                    )
                except Exception as error:
                    verified = False
                    _best_effort_append_local_event(
                        stage_id,
                        "stage_success_verification_retry",
                        error=_redacted_exception_text(error),
                    )
                if verified:
                    stage["status"] = "stage_ready"
                    stage["verified_at"] = utc_now()
                    replace_stage_record(stage)
                    _best_effort_append_local_event(
                        stage_id,
                        "stage_ready",
                        parent_run_id=stage["parent_run_id"],
                        instance_id=instance_id,
                    )
                    return 0
            if _is_bound_remote_failure_event(latest, stage_id):
                try:
                    failure_artifacts_verified = pull_and_verify(
                        stage,
                        best_effort=True,
                        destination=_stage_artifact_destination(stage),
                        remote_ops=stage["remote_ops"],
                    )
                except Exception as error:
                    failure_artifacts_verified = False
                    _best_effort_append_local_event(
                        stage_id,
                        "stage_failure_recovery_error",
                        error=_redacted_exception_text(error),
                    )
                stage["status"] = "stage_failed"
                stage["finished_at"] = utc_now()
                stage["failure_artifacts_verified"] = bool(
                    failure_artifacts_verified
                )
                replace_stage_record(stage)
                _best_effort_append_local_event(
                    stage_id,
                    "stage_failed",
                    parent_run_id=stage["parent_run_id"],
                    instance_id=instance_id,
                    failure_artifacts_verified=bool(failure_artifacts_verified),
                )
                return 1
        time.sleep(manifest.poll_seconds)


def monitor_session_stage(stage_id: str) -> int:
    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = MONITOR_DIR / f"stage_{stage_id}.lock"
    _claim_supervisor_lock(lock_path, f"stage {stage_id!r} monitor")
    try:
        return _monitor_session_stage(stage_id)
    finally:
        _release_supervisor_lock(lock_path)


def launch_detached_stage_monitor(stage_id: str) -> int:
    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    log_path = MONITOR_DIR / f"stage_{stage_id}.log"
    command = [
        windowless_python_executable(),
        str(VAST_DIR / "ops.py"),
        "monitor-stage",
        stage_id,
    ]
    options: dict[str, Any] = {
        "cwd": ROOT,
        "stdin": subprocess.DEVNULL,
        "stdout": None,
        "stderr": subprocess.STDOUT,
        "close_fds": True,
    }
    if os.name == "nt":
        options["creationflags"] = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        )
    else:
        options["start_new_session"] = True
    with log_path.open("a", encoding="utf-8") as log:
        options["stdout"] = log
        process = _popen_detached(command, options)
    stage = get_stage_record(stage_id)
    stage["monitor_pid"] = process.pid
    stage["monitor_log"] = str(log_path)
    stage["monitor_started_at"] = utc_now()
    replace_stage_record(stage)
    append_local_event(stage_id, "detached_stage_monitor_started", pid=process.pid)
    return process.pid


def run_doctor() -> dict[str, Any]:
    checks: dict[str, Any] = {
        "secrets_file": SECRETS_FILE.exists(),
        "vast_api_key_present": False,
        "vast_cli": None,
        "vast_cli_imports": False,
        "ssh": shutil.which("ssh"),
        "scp": shutil.which("scp"),
        "ssh_public_key_valid": False,
        "ssh_private_key_readable": False,
        "git_head": None,
        "git_clean": False,
    }
    try:
        checks["vast_api_key_present"] = bool(load_secrets().get("VAST_API_KEY"))
        checks["vast_cli"] = vast_cli_path()
        process = subprocess.run(
            [checks["vast_cli"], "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            env=isolated_cli_environment(),
            **hidden_subprocess_options(),
        )
        checks["vast_cli_imports"] = process.returncode == 0
    except Exception as error:
        checks["vast_cli_error"] = _redacted_exception_text(error)
    try:
        local_ssh_public_key()
        checks["ssh_public_key_valid"] = True
        require_local_ssh_identity()
        checks["ssh_private_key_readable"] = True
    except Exception as error:
        checks["ssh_identity_error"] = _redacted_exception_text(error)
    try:
        checks["git_head"] = git_head()
        checks["git_clean"] = git_is_clean()
    except Exception as error:
        checks["git_error"] = _redacted_exception_text(error)
    required = (
        checks["secrets_file"],
        checks["vast_api_key_present"],
        checks["vast_cli"],
        checks["vast_cli_imports"],
        checks["ssh"],
        checks["scp"],
        checks["ssh_public_key_valid"],
        checks["ssh_private_key_readable"],
        checks["git_head"],
        checks["git_clean"],
    )
    checks["ready_to_launch"] = all(required)
    return checks


def command_doctor(_: argparse.Namespace) -> int:
    report = run_doctor()
    _best_effort_emit_json(report, indent=2)
    return 0 if report["ready_to_launch"] else 1


def command_validate_manifest(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    head = git_head()
    report = {
        "valid": True,
        "run_id": manifest.run_id,
        "source_commit": manifest.source_commit,
        "head": head,
        "source_matches_head": manifest.source_commit == head,
        "num_gpus": manifest.num_gpus,
        "estimated_ttl_ceiling_usd": manifest.ttl_hours * manifest.max_dph,
        "hard_total_cap_usd": manifest.max_total_usd,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["source_matches_head"] else 1


def command_search(args: argparse.Namespace) -> int:
    excluded_machine_ids = (
        load_machine_blacklist() | active_machine_ids() | set(args.avoid_machine)
    )
    offers = search_offers(
        args.gpu,
        args.max_dph,
        num_gpus=args.num_gpus,
        disk_gb=args.disk_gb,
        min_cuda=args.min_cuda,
        min_inet_down_mbps=args.min_inet,
        min_cpu_cores_effective=args.min_cpu_cores,
        min_cpu_ram_gb=args.min_cpu_ram_gb,
        require_consumer_cpu=args.require_consumer_cpu,
        required_country=args.country,
        excluded_machine_ids=excluded_machine_ids,
        limit=args.limit,
    )
    if args.limit < 1:
        raise OpsError("search limit must be positive")
    offers = sorted(
        offers,
        key=lambda offer: (
            _optional_finite_offer_number(offer, "dph_total"),
            _offer_identity_number(offer, "machine_id"),
            _offer_identity_number(offer, "id"),
        ),
    )[: args.limit]
    compact = [
        {
            "id": offer.get("id"),
            "machine_id": offer.get("machine_id"),
            "gpu_name": offer.get("gpu_name"),
            "num_gpus": offer.get("num_gpus"),
            "dph": offer.get("dph_total"),
            "reliability": offer.get("reliability2"),
            "inet_down": offer.get("inet_down"),
            "cpu_name": offer.get("cpu_name"),
            "cpu_cores_effective": offer.get("cpu_cores_effective"),
            "cpu_ram_gb": round(
                _optional_finite_offer_number(offer, "cpu_ram") / 1024, 1
            ),
            "server_cpu": _is_server_cpu(offer),
            "geolocation": offer.get("geolocation"),
        }
        for offer in offers
    ]
    _best_effort_emit_json(compact, indent=2)
    return 0


def command_launch(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    if any(record.get("run_id") == manifest.run_id for record in load_state()):
        raise OpsError(f"run_id {manifest.run_id!r} is already tracked")
    bundle, resolved_manifest = create_source_bundle(manifest)
    # A static blacklist carries prior account-specific failures.  Avoiding a
    # currently rented physical machine additionally prevents correlated host
    # and network failures if another run is live.
    excluded_machine_ids = (
        load_machine_blacklist()
        | set(manifest.avoid_machine_ids)
        | active_machine_ids()
    )
    offers = search_offers(
        manifest.gpu_name,
        manifest.max_dph,
        num_gpus=manifest.num_gpus,
        disk_gb=manifest.disk_gb,
        min_cuda=manifest.min_cuda,
        min_inet_down_mbps=manifest.min_inet_down_mbps,
        min_cpu_cores_effective=manifest.min_cpu_cores_effective,
        min_cpu_ram_gb=manifest.min_cpu_ram_gb,
        require_consumer_cpu=manifest.require_consumer_cpu,
        required_geolocation=manifest.required_geolocation,
        required_country=manifest.required_country,
        excluded_machine_ids=excluded_machine_ids,
    )
    if args.offer is not None:
        requested_offer_id = _strict_positive_identity(args.offer, "requested offer ID")
        matches = [
            offer
            for offer in offers
            if _offer_identity_number(offer, "id") == requested_offer_id
        ]
        if not matches:
            raise OpsError("requested offer is stale or outside manifest limits")
        offer = validate_offer_for_purchase(
            matches[0], manifest, expected_offer_id=requested_offer_id
        )
    else:
        offer = pick_offer(offers, prefer_consumer_cpu=manifest.prefer_consumer_cpu)
        offer = validate_offer_for_purchase(offer, manifest)
    expected_label = f"quant-experiments-{manifest.run_id}"
    assert_live_label_unused(expected_label)
    campaign_started_at = utc_now()
    deadline_schedule = campaign_deadline_schedule(
        manifest, campaign_started_at=campaign_started_at
    )
    if parse_utc(deadline_schedule["cleanup_start_at"]) <= datetime.now(UTC):
        raise OpsError("campaign limits leave no positive pre-cleanup work window")
    record = {
        "run_id": manifest.run_id,
        "instance_id": None,
        "offer_id": int(offer["id"]),
        "machine_id": offer.get("machine_id"),
        "dph": float(offer["dph_total"]),
        # Persist before the purchase call. This slightly overestimates cost and
        # lets reconcile recover a labeled instance if the client dies mid-call.
        "created_at": campaign_started_at,
        "source_commit": manifest.source_commit,
        "manifest_path": str(args.manifest.resolve()),
        "manifest": asdict(manifest),
        "expected_label": expected_label,
        "campaign_started_at": campaign_started_at,
        **deadline_schedule,
        "provisioning_owner_pid": _strict_positive_identity(
            os.getpid(), "provisioning owner PID"
        ),
        "status": "purchase_pending",
    }
    append_record(record)
    append_local_event(
        manifest.run_id,
        "purchase_pending",
        offer_id=record["offer_id"],
        dph=record["dph"],
        approved_max_usd=manifest.max_total_usd,
    )
    try:
        # Install the detached owner before the provider create call.  It is
        # deliberately also present during live verification, SSH, and
        # bootstrap so a killed client cannot leave a newly billed host behind.
        launch_detached_purchase_reconciliation(manifest.run_id)
    except Exception as watchdog_error:
        safe_error = _redacted_exception_text(watchdog_error)
        update_record_fields(
            manifest.run_id,
            status="purchase_watchdog_unavailable",
            watchdog_error=safe_error,
        )
        raise OpsError(
            "purchase watchdog could not start; no instance purchase was attempted: "
            f"{safe_error}"
        ) from watchdog_error
    instance_id: int | None = None
    create_response_instance_id: int | None = None
    cleanup_deadline = _record_deadline_monotonic(
        record,
        "cleanup_start_at",
        fallback_seconds=_purchase_reconciliation_budget_seconds(record, manifest),
    )
    controller_deadline = _record_deadline_monotonic(
        record,
        "controller_deadline_at",
        fallback_seconds=_campaign_hard_remaining_seconds(record, manifest),
    )
    try:
        create_response_instance_id = create_instance(
            offer,
            manifest,
            label=expected_label,
            campaign_started_at=record["campaign_started_at"],
            deadline_monotonic=cleanup_deadline,
        )
        _require_provisioning_transition(
            manifest.run_id,
            allowed_statuses={"purchase_pending"},
            create_response_instance_id=_strict_positive_identity(
                create_response_instance_id, "create-response instance ID"
            ),
            status="verifying_live_contract",
        )
        live_instance = verify_created_instance(
            create_response_instance_id,
            manifest,
            expected_label=expected_label,
            expected_offer=offer,
            deadline_monotonic=cleanup_deadline,
        )
        live_dph = _require_live_positive_dph(live_instance)
        # Only exact live label + offer + machine verification promotes a
        # create-response ID into ordinary teardown/bootstrap authority.
        instance_id = create_response_instance_id
        _require_provisioning_transition(
            manifest.run_id,
            allowed_statuses={"verifying_live_contract"},
            instance_id=_strict_positive_identity(instance_id, "instance ID"),
            dph=live_dph,
            status="provisioning",
        )
        record = get_record(manifest.run_id)
        _best_effort_append_local_event(
            manifest.run_id,
            "instance_created",
            instance_id=instance_id,
            dph=record["dph"],
            approved_max_usd=manifest.max_total_usd,
        )
        connection = wait_for_ssh(
            instance_id,
            timeout_seconds=LAUNCH_SSH_READY_TIMEOUT_SECONDS,
            require_watchdog_armed=True,
            expected_watchdog_deadline_epoch=parse_utc(
                str(record["remote_watchdog_deadline_at"])
            ).timestamp(),
            deadline_monotonic=cleanup_deadline,
        )
        with _lifecycle_transition_lock(
            manifest.run_id, deadline_monotonic=cleanup_deadline
        ):
            _require_provisioning_transition(
                manifest.run_id,
                allowed_statuses={"provisioning"},
                status="bootstrap_starting",
            )
            start_remote_run(
                connection,
                bundle,
                resolved_manifest,
                deadline_monotonic=cleanup_deadline,
                workload_stop_at=record.get("cleanup_start_at"),
            )
            _require_provisioning_transition(
                manifest.run_id,
                allowed_statuses={"bootstrap_starting"},
                status="running",
                remote_started_at=utc_now(),
            )
        _best_effort_append_local_event(
            manifest.run_id, "remote_started", instance_id=instance_id
        )
    except Exception as setup_error:
        setup_persistence_error: Exception | None = None
        try:
            _record_provisioning_setup_failure(
                manifest.run_id,
                instance_id=instance_id,
                error=setup_error,
            )
        except Exception as persistence_error:
            # Diagnostics must never sit in front of paid-instance cleanup.
            # The detached purchase reconciler was armed before create and
            # remains the fallback if local state storage itself is unhealthy.
            setup_persistence_error = persistence_error
        if instance_id is not None:
            try:
                destroy_instance(
                    instance_id,
                    manifest.run_id,
                    "provisioning_failed",
                    deadline_monotonic=controller_deadline,
                )
            except Exception as cleanup_error:
                monitor_error: Exception | None = None
                _best_effort_update_record_fields(
                    manifest.run_id,
                    instance_id=instance_id,
                    status="provisioning_cleanup_pending",
                    cleanup_required=True,
                    cleanup_required_at=utc_now(),
                )
                try:
                    launch_detached_monitor(manifest.run_id)
                except Exception as observed_monitor_error:
                    monitor_error = observed_monitor_error
                # Fallback ownership is attempted before any secondary
                # bookkeeping.  Both evidence writes are then leaf-level,
                # no-throw operations that cannot replace the setup failure.
                _best_effort_record_secondary_failure(
                    manifest.run_id,
                    error=cleanup_error,
                    error_field="provisioning_cleanup_error",
                    error_at_field="provisioning_cleanup_error_at",
                    event="provisioning_cleanup_pending",
                    instance_id=instance_id,
                )
                if monitor_error is not None:
                    _best_effort_record_secondary_failure(
                        manifest.run_id,
                        error=monitor_error,
                        error_field="provisioning_cleanup_monitor_error",
                        error_at_field="provisioning_cleanup_monitor_error_at",
                        event="provisioning_cleanup_monitor_failed",
                        instance_id=instance_id,
                    )
        else:
            try:
                _provisioning_write_if_active(
                    manifest.run_id,
                    allowed_statuses={"purchase_pending", "verifying_live_contract"},
                    status="purchase_outcome_unknown",
                    cleanup_required=True,
                    purchase_error_at=utc_now(),
                )
            except Exception as outcome_persistence_error:
                # The purchase reconciler was started before create.  If state
                # I/O is now unhealthy, preserve the original create failure
                # and leave that already-running owner in charge.
                _best_effort_record_secondary_failure(
                    manifest.run_id,
                    error=outcome_persistence_error,
                    error_field="purchase_outcome_persistence_error",
                    error_at_field="purchase_outcome_persistence_error_at",
                    event="purchase_outcome_persistence_failed",
                )
        if setup_persistence_error is not None:
            _best_effort_record_secondary_failure(
                manifest.run_id,
                error=setup_persistence_error,
                error_field="setup_error_persistence_error",
                error_at_field="setup_error_persistence_error_at",
                event="provisioning_setup_persistence_failed",
            )
        raise
    _best_effort_emit_json(
        {
            "run_id": manifest.run_id,
            "instance_id": instance_id,
            "dph": float(offer["dph_total"]),
            "max_total_usd": manifest.max_total_usd,
            "ttl_hours": manifest.ttl_hours,
            "status": "running",
        },
        indent=2,
    )
    return start_requested_monitor(manifest.run_id, args.detach)


def _race_candidate_id(manifest: RunManifest, index: int) -> str:
    """Return an internal, unique label for one candidate contract."""

    return f"{manifest.run_id}.candidate{index + 1}"


def _race_candidate_label(manifest: RunManifest, index: int) -> str:
    return f"quant-experiments-{manifest.run_id}-candidate{index + 1}"


def _race_update_candidate(
    record: dict[str, Any], lock: threading.Lock, index: int, **updates: Any
) -> None:
    with lock:
        record["candidates"][index].update(updates)
        replace_record(record)


def _race_candidate_cost(candidate: dict[str, Any]) -> float:
    started = candidate.get("instance_created_at")
    dph = candidate.get("dph")
    if started is None or dph is None:
        return 0.0
    try:
        ended = candidate.get("destroyed_at")
        end_time = (
            parse_utc(str(ended))
            if ended is not None
            else datetime.now(UTC)
        )
        elapsed = max(
            0.0,
            (end_time - parse_utc(str(started))).total_seconds()
            / 3600,
        )
        return elapsed * float(dph)
    except (TypeError, ValueError):
        return 0.0


def _race_candidate_worker(
    index: int,
    offer: dict[str, Any],
    manifest: RunManifest,
    bundle: Path,
    resolved_manifest: Path,
    record: dict[str, Any],
    lock: threading.Lock,
    winner_event: threading.Event,
    race_deadline: float,
) -> None:
    candidate_id = _race_candidate_id(manifest, index)
    instance_id: int | None = None
    create_response_instance_id: int | None = None
    won = False

    def check_budget() -> None:
        if time.monotonic() >= race_deadline:
            raise OpsError("shared race campaign cap or cleanup reserve reached")

    try:
        _race_update_candidate(
            record, lock, index, status="purchasing", started_at=utc_now()
        )
        check_budget()
        try:
            candidate_label = _race_candidate_label(manifest, index)
            create_response_instance_id = create_instance(
                offer,
                manifest,
                label=candidate_label,
                campaign_started_at=record["campaign_started_at"],
            )
        except Exception as create_error:
            # Vast can return an error after accepting a create request.  Keep
            # this candidate explicitly unknown so reconciliation searches its
            # immutable label instead of treating it as a clean failure.
            _race_update_candidate(
                record,
                lock,
                index,
                status="purchase_outcome_unknown",
                purchase_error_at=utc_now(),
                unknown_since=utc_now(),
                live_miss_count=0,
                error=_redacted_exception_text(create_error),
            )
            raise
        _race_update_candidate(
            record,
            lock,
            index,
            create_response_instance_id=create_response_instance_id,
            instance_created_at=utc_now(),
            status="verifying_live_contract",
        )
        if winner_event.is_set():
            raise OpsError("candidate lost the readiness race")
        check_budget()
        live_instance = verify_created_instance(
            create_response_instance_id,
            manifest,
            expected_label=candidate_label,
            expected_offer=offer,
        )
        check_budget()
        live_dph = _require_live_positive_dph(live_instance)
        instance_id = create_response_instance_id
        try:
            billable_start = instance_started_at(live_instance).isoformat().replace(
                "+00:00", "Z"
            )
        except OpsError:
            # Some mocked/older Vast responses omit start_date; the persisted
            # post-create timestamp remains a conservative local fallback.
            billable_start = record["candidates"][index].get(
                "instance_created_at", utc_now()
            )
        _race_update_candidate(
            record,
            lock,
            index,
            instance_id=instance_id,
            instance_created_at=billable_start,
            dph=live_dph,
            status="waiting_for_ssh",
        )
        check_budget()
        remaining = max(1, int(race_deadline - time.monotonic()))
        connection = wait_for_ssh(
            instance_id,
            timeout_seconds=min(DEFAULT_SSH_READY_TIMEOUT_SECONDS, remaining),
            cancel_event=winner_event,
            require_watchdog_armed=True,
            expected_watchdog_deadline_epoch=parse_utc(
                str(record["remote_watchdog_deadline_at"])
            ).timestamp(),
            deadline_monotonic=race_deadline,
        )
        if winner_event.is_set():
            raise OpsError("candidate lost the readiness race")
        check_budget()
        _race_update_candidate(record, lock, index, status="checking_cuda")
        check_budget()
        cuda_report = verify_cuda_readiness(connection, manifest)
        check_budget()
        if winner_event.is_set():
            raise OpsError("candidate lost the readiness race")
        with lock:
            if record.get("winner_index") is None:
                record["winner_index"] = index
                record["winner_instance_id"] = instance_id
                record["winner_connection"] = list(connection)
                record["candidates"][index].update(
                    {"status": "winner", "cuda_report": cuda_report}
                )
                winner_event.set()
                won = True
                replace_record(record)
            else:
                record["candidates"][index].update(
                    {"status": "lost", "cuda_report": cuda_report}
                )
                replace_record(record)
        if not won:
            raise OpsError("candidate lost the readiness race")
    except Exception as error:
        existing_status = record["candidates"][index].get("status")
        _race_update_candidate(
            record,
            lock,
            index,
            status=(
                "purchase_outcome_unknown"
                if existing_status == "purchase_outcome_unknown"
                else ("failed" if not won else "winner_failed")
            ),
            error=_redacted_exception_text(error),
        )
    finally:
        if instance_id is not None and not won:
            try:
                destroy_instance(instance_id, candidate_id, "race_loser_or_failure")
                _race_update_candidate(
                    record, lock, index, status="destroyed", destroyed_at=utc_now()
                )
            except Exception as cleanup_error:
                _race_update_candidate(
                    record,
                    lock,
                    index,
                    status="cleanup_pending",
                    cleanup_error=_redacted_exception_text(cleanup_error),
                )


def _race_cleanup_losers(
    manifest: RunManifest,
    record: dict[str, Any],
    lock: threading.Lock,
    winner_index: int,
) -> list[str]:
    errors: list[str] = []
    with lock:
        candidates = [dict(candidate) for candidate in record["candidates"]]
    for index, candidate in enumerate(candidates):
        if index == winner_index:
            continue
        if candidate.get("status") in {"destroyed", "absent_from_live_api"} and not candidate.get(
            "discovered_instance_ids"
        ):
            continue
        instance_ids: set[int] = set()
        try:
            primary_id = _optional_strict_identity(
                candidate.get("instance_id"),
                f"race candidate {index + 1} persisted instance ID",
            )
        except OpsError as error:
            safe_error = _redacted_exception_text(error)
            errors.append(safe_error)
            _race_update_candidate(
                record, lock, index, status="cleanup_pending", cleanup_error=safe_error
            )
            primary_id = None
        if primary_id is not None:
            instance_ids.add(primary_id)
        for raw_id in candidate.get("discovered_instance_ids", []):
            try:
                discovered_id = _strict_positive_identity(
                    raw_id, f"race candidate {index + 1} discovered instance ID"
                )
            except OpsError as error:
                safe_error = _redacted_exception_text(error)
                errors.append(safe_error)
                _race_update_candidate(
                    record, lock, index, status="cleanup_pending", cleanup_error=safe_error
                )
                continue
            instance_ids.add(discovered_id)
        for instance_id in sorted(instance_ids):
            try:
                destroy_instance(
                    instance_id,
                    _race_candidate_id(manifest, index),
                    "race_loser",
                )
            except Exception as error:
                safe_error = _redacted_exception_text(error)
                message = f"candidate {index + 1} instance {instance_id}: {safe_error}"
                errors.append(message)
                _race_update_candidate(
                    record, lock, index, status="cleanup_pending", cleanup_error=safe_error
                )
            else:
                _race_update_candidate(
                    record,
                    lock,
                    index,
                    status="destroyed",
                    destroyed_at=utc_now(),
                    discovered_instance_ids=[
                        value
                        for value in candidate.get("discovered_instance_ids", [])
                        if value != instance_id
                    ],
                )
    return errors


def command_race(args: argparse.Namespace) -> int:
    """Provision up to three candidates and start exactly the first ready winner."""

    if not ENABLE_CANDIDATE_RACE:
        raise OpsError(
            "candidate race is disabled pending its own aggregate-deadline audit"
        )
    candidate_count = int(args.candidates)
    if candidate_count < 1 or candidate_count > 3:
        raise OpsError("race candidates must be in [1, 3]")
    manifest = load_manifest(args.manifest)
    if any(record.get("run_id") == manifest.run_id for record in load_state()):
        raise OpsError(f"run_id {manifest.run_id!r} is already tracked")
    bundle, resolved_manifest = create_source_bundle(manifest)
    excluded_machine_ids = (
        load_machine_blacklist()
        | set(manifest.avoid_machine_ids)
        | active_machine_ids()
    )
    offers = search_offers(
        manifest.gpu_name,
        manifest.max_dph,
        num_gpus=manifest.num_gpus,
        disk_gb=manifest.disk_gb,
        min_cuda=manifest.min_cuda,
        min_inet_down_mbps=manifest.min_inet_down_mbps,
        min_cpu_cores_effective=manifest.min_cpu_cores_effective,
        min_cpu_ram_gb=manifest.min_cpu_ram_gb,
        require_consumer_cpu=manifest.require_consumer_cpu,
        required_geolocation=manifest.required_geolocation,
        required_country=manifest.required_country,
        excluded_machine_ids=excluded_machine_ids,
        limit=max(3, candidate_count * 3),
    )
    selected: list[dict[str, Any]] = []
    seen_machines: set[object] = set()
    for offer in offers:
        try:
            validate_offer_for_purchase(offer, manifest)
        except OpsError:
            # Invalid API rows cannot become paid candidates.  They are not
            # silently coerced into a purchase or into the race denominator.
            continue
        machine = offer.get("machine_id")
        # A race is specifically a physical-host hedge.  An offer without a
        # machine ID cannot prove that property, so fail closed instead of
        # silently racing multiple contracts on one unknown host.
        try:
            _strict_positive_identity(machine, "race machine ID")
        except OpsError:
            continue
        if machine in seen_machines:
            continue
        seen_machines.add(machine)
        selected.append(offer)
        if len(selected) == candidate_count:
            break
    if len(selected) != candidate_count:
        raise OpsError(
            f"only {len(selected)} distinct eligible offers are available; "
            f"race requires {candidate_count}"
        )
    dphs = [float(offer["dph_total"]) for offer in selected]
    reserve = cleanup_reserve_hours(manifest.poll_seconds)
    race_window_hours = min(manifest.ttl_hours, manifest.max_total_usd / sum(dphs)) - reserve
    if race_window_hours <= 0:
        raise OpsError(
            "manifest max_total_usd and TTL leave no positive race window after cleanup reserve"
        )
    race_started = time.monotonic()
    race_deadline = race_started + (
        race_window_hours * 3600
    )
    campaign_started_at = utc_now()
    # Retained only for internal contract tests while the race entry point is
    # disabled. A future paid race must replace this ordinary schedule with an
    # independently audited shared aggregate-spend schedule.
    deadline_schedule = campaign_deadline_schedule(
        manifest, campaign_started_at=campaign_started_at
    )
    record: dict[str, Any] = {
        "run_id": manifest.run_id,
        "instance_id": None,
        "created_at": campaign_started_at,
        "campaign_started_at": campaign_started_at,
        **deadline_schedule,
        "source_commit": manifest.source_commit,
        "manifest_path": str(args.manifest.resolve()),
        "manifest": asdict(manifest),
        "expected_label": f"quant-experiments-{manifest.run_id}",
        "lifecycle_origin": "race",
        "status": "race_pending",
        "campaign_cap_usd": manifest.max_total_usd,
        "race_candidate_count": candidate_count,
        "winner_index": None,
        "winner_instance_id": None,
        "candidates": [
            {
                "candidate_id": _race_candidate_id(manifest, index),
                "label": _race_candidate_label(manifest, index),
                "offer_id": int(offer["id"]),
                "machine_id": offer.get("machine_id"),
                "dph": float(offer["dph_total"]),
                "instance_id": None,
                "status": "pending",
            }
            for index, offer in enumerate(selected)
        ],
    }
    # Persist the complete candidate set before any create call.  This lets
    # reconcile identify every possible contract after a client interruption.
    append_record(record)
    append_local_event(
        manifest.run_id,
        "race_pending",
        candidate_count=candidate_count,
        candidate_labels=[candidate["label"] for candidate in record["candidates"]],
        campaign_cap_usd=manifest.max_total_usd,
    )
    record["coordinator_pid"] = os.getpid()
    replace_record(record)
    try:
        launch_detached_race_watchdog(manifest.run_id)
    except Exception as watchdog_error:
        record["status"] = "race_unavailable"
        record["watchdog_error"] = _redacted_exception_text(watchdog_error)
        replace_record(record)
        raise OpsError(
            "race watchdog could not be started; no candidate purchases were attempted"
        ) from watchdog_error
    lock = threading.Lock()
    winner_event = threading.Event()
    threads = [
        threading.Thread(
            target=_race_candidate_worker,
            args=(
                index,
                offer,
                manifest,
                bundle,
                resolved_manifest,
                record,
                lock,
                winner_event,
                race_deadline,
            ),
            name=f"vast-race-{manifest.run_id}-{index + 1}",
            daemon=True,
        )
        for index, offer in enumerate(selected)
    ]
    for thread in threads:
        thread.start()
    while not winner_event.is_set() and any(thread.is_alive() for thread in threads):
        winner_event.wait(0.1)
    for thread in threads:
        thread.join()
    winner_index = record.get("winner_index")
    if winner_index is None:
        cleanup_errors = _race_cleanup_losers(manifest, record, lock, -1)
        unknown_outcomes = [
            candidate
            for candidate in record.get("candidates", [])
            if candidate.get("status") == "purchase_outcome_unknown"
            or (
                candidate.get("instance_id") is None
                and candidate.get("status") in {"purchasing", "verifying_live_contract"}
            )
        ]
        if cleanup_errors or unknown_outcomes:
            record["status"] = "provisioning_cleanup_pending"
            record["cleanup_errors"] = cleanup_errors
            record["cleanup_required"] = bool(unknown_outcomes)
            replace_record(record)
            try:
                launch_detached_race_cleanup(manifest.run_id)
                supervisor = "detached race cleanup started"
            except Exception as supervisor_error:
                safe_supervisor_error = _redacted_exception_text(supervisor_error)
                append_local_event(
                    manifest.run_id,
                    "detached_race_cleanup_failed",
                    error=safe_supervisor_error,
                )
                supervisor = f"detached cleanup unavailable: {safe_supervisor_error}"
            live_candidates = [
                {
                    "instance_id": candidate.get("instance_id"),
                    "label": candidate.get("label"),
                    "status": candidate.get("status"),
                }
                for candidate in record.get("candidates", [])
                if candidate.get("instance_id") is not None
                and candidate.get("status")
                not in {"destroyed", "absent_from_live_api"}
            ]
            raise OpsError(
                "no race candidate became ready and loser cleanup failed; "
                f"live candidates={live_candidates}; "
                f"unknown candidates={len(unknown_outcomes)}; {supervisor}"
            )
        record["status"] = "race_failed"
        replace_record(record)
        raise OpsError("no race candidate passed live, SSH, and CUDA readiness")
    if record.get("status") == "provisioning_cleanup_pending" and record.get(
        "watchdog_takeover_at"
    ):
        raise OpsError(
            "race watchdog took over cleanup before a winner could start training"
        )
    winner_index = int(winner_index)
    winner = record["candidates"][winner_index]
    winner_instance_id = int(winner["instance_id"])
    winner_cost = _race_candidate_cost(record["candidates"][winner_index])
    loser_campaign_cost = sum(
        _race_candidate_cost(candidate)
        for index, candidate in enumerate(record["candidates"])
        if index != winner_index
    )
    campaign_cost = loser_campaign_cost + winner_cost
    record.update(
        {
            "instance_id": winner_instance_id,
            "offer_id": winner["offer_id"],
            "machine_id": winner.get("machine_id"),
            "dph": winner["dph"],
            # Keep the winner's original billable start so the TTL clock does
            # not reset at election.  Only loser cost is carried as an offset;
            # winner cost continues from this timestamp in elapsed_and_cost().
            "created_at": winner.get("instance_created_at") or utc_now(),
            "campaign_accrued_cost_usd": loser_campaign_cost,
            "race_total_cost_at_start_usd": campaign_cost,
            "status": "starting_remote",
        }
    )
    replace_record(record)
    cleanup_errors = _race_cleanup_losers(manifest, record, lock, winner_index)
    if cleanup_errors:
        winner_cleanup_error: Exception | None = None
        try:
            destroy_instance(
                winner_instance_id, manifest.run_id, "race_cleanup_failed"
            )
        except Exception as error:
            winner_cleanup_error = error
        # The race supervisor owns the complete candidate ledger.  It must be
        # used even when the immediate winner destroy succeeds because loser
        # contracts may still be live.
        record["status"] = "provisioning_cleanup_pending"
        record["cleanup_errors"] = cleanup_errors
        if winner_cleanup_error is not None:
            record["winner_cleanup_error"] = _redacted_exception_text(
                winner_cleanup_error
            )
        replace_record(record)
        try:
            launch_detached_race_cleanup(manifest.run_id)
            supervisor = "detached race cleanup started"
        except Exception as supervisor_error:
            safe_supervisor_error = _redacted_exception_text(supervisor_error)
            append_local_event(
                manifest.run_id,
                "detached_race_cleanup_failed",
                error=safe_supervisor_error,
            )
            supervisor = f"detached race cleanup unavailable: {safe_supervisor_error}"
        detail = "race loser cleanup failed"
        if winner_cleanup_error is not None:
            detail += " and winner destruction failed"
        raise OpsError(
            f"{detail}; winner={winner_instance_id}; {supervisor}; "
            + "; ".join(cleanup_errors)
        ) from winner_cleanup_error
    # Destruction timestamps are now fixed for every loser, so recompute the
    # sunk campaign spend before checking whether the winner can reserve its
    # own recovery/destruction window.
    winner_cost = _race_candidate_cost(record["candidates"][winner_index])
    loser_campaign_cost = sum(
        _race_candidate_cost(candidate)
        for index, candidate in enumerate(record["candidates"])
        if index != winner_index
    )
    campaign_cost = loser_campaign_cost + winner_cost
    record["campaign_accrued_cost_usd"] = loser_campaign_cost
    record["race_total_cost_at_start_usd"] = campaign_cost
    replace_record(record)
    winner_cleanup_reserve = float(winner["dph"]) * reserve
    if campaign_cost + winner_cleanup_reserve >= manifest.max_total_usd:
        try:
            destroy_instance(winner_instance_id, manifest.run_id, "race_spend_cap")
        except Exception as winner_cleanup_error:
            record["status"] = "provisioning_cleanup_pending"
            record["winner_cleanup_error"] = _redacted_exception_text(
                winner_cleanup_error
            )
            replace_record(record)
            try:
                launch_detached_monitor(manifest.run_id)
            except Exception as supervisor_error:
                append_local_event(
                    manifest.run_id,
                    "detached_monitor_failed",
                    error=_redacted_exception_text(supervisor_error),
                )
            raise OpsError(
                "race campaign cap left no winner cleanup reserve; "
                f"winner destruction failed for {winner_instance_id}"
            ) from winner_cleanup_error
        record["status"] = "race_spend_cap"
        replace_record(record)
        raise OpsError(
            "shared race campaign cap plus winner cleanup reserve was reached "
            "before training started"
        )
    try:
        connection = tuple(record.get("winner_connection") or ())
        if len(connection) != 3:
            connection = wait_for_ssh(winner_instance_id)
        start_remote_run(connection, bundle, resolved_manifest)
    except Exception as setup_error:
        try:
            destroy_instance(winner_instance_id, manifest.run_id, "race_winner_setup_failed")
        except Exception as cleanup_error:
            record["status"] = "provisioning_cleanup_pending"
            record["setup_error"] = _redacted_exception_text(setup_error)
            record["cleanup_error"] = _redacted_exception_text(cleanup_error)
            replace_record(record)
            try:
                launch_detached_monitor(manifest.run_id)
                supervisor = "detached winner monitor started"
            except Exception as supervisor_error:
                safe_supervisor_error = _redacted_exception_text(supervisor_error)
                append_local_event(
                    manifest.run_id,
                    "detached_monitor_failed",
                    error=safe_supervisor_error,
                )
                supervisor = (
                    f"detached winner monitor unavailable: {safe_supervisor_error}"
                )
            raise OpsError(
                "winner remote setup failed and winner destruction failed; "
                f"winner={winner_instance_id}; {supervisor}"
            ) from cleanup_error
        record["status"] = "race_failed"
        record["setup_error"] = _redacted_exception_text(setup_error)
        replace_record(record)
        raise
    record["status"] = "running"
    record["remote_started_at"] = utc_now()
    replace_record(record)
    append_local_event(
        manifest.run_id,
        "race_winner_started",
        instance_id=winner_instance_id,
        candidate_index=winner_index + 1,
        campaign_accrued_cost_usd=loser_campaign_cost,
        race_total_cost_at_start_usd=campaign_cost,
    )
    print(
        json.dumps(
            {
                "run_id": manifest.run_id,
                "instance_id": winner_instance_id,
                "winner_candidate": winner_index + 1,
                "race_candidates": candidate_count,
                "campaign_cap_usd": manifest.max_total_usd,
                "campaign_accrued_cost_usd": round(loser_campaign_cost, 6),
                "race_total_cost_at_start_usd": round(campaign_cost, 6),
                "status": "running",
            },
            indent=2,
        )
    )
    return start_requested_monitor(manifest.run_id, args.detach)


def command_race_cleanup(args: argparse.Namespace) -> int:
    return _race_cleanup_pending(args.run_id)


def command_race_watchdog(args: argparse.Namespace) -> int:
    return _race_watchdog(args.run_id)


def command_adopt_existing(args: argparse.Namespace) -> int:
    """Guard and adopt one already-billing blank instance without renting."""

    if not ENABLE_EXISTING_INSTANCE_ADOPTION:
        raise OpsError(
            "adopt-existing is disabled pending its own setup-supervisor audit"
        )
    manifest = load_manifest(args.manifest)
    requested_instance_id = _strict_positive_identity(args.instance, "--instance")
    require_adoption_approval(manifest, instance_id=requested_instance_id)
    records = load_state()
    if any(record.get("run_id") == manifest.run_id for record in records):
        raise OpsError(f"run_id {manifest.run_id!r} is already tracked")

    # Source preparation is still the same exact clean-commit gate as launch.
    # It happens before the host is modified.
    bundle, resolved_manifest = create_source_bundle(manifest)
    live = _require_provider_instance_rows(
        run_vast("show", "instances"), "existing-instance adoption query"
    )
    matches = [
        item
        for item in live
        if isinstance(item, dict)
        and _offer_identity_number(item, "id") == requested_instance_id
    ]
    if len(matches) != 1:
        raise OpsError("the confirmed existing instance is not present exactly once")
    instance = matches[0]
    started, elapsed_hours, accrued_cost = validate_adoption_instance(
        instance,
        manifest,
        tracked_records=records,
        blacklisted_machine_ids=load_machine_blacklist(),
    )
    connection = wait_for_ssh(requested_instance_id)
    verify_adoption_host_idle(connection)

    # SSH readiness can consume a material fraction of a short remaining cap.
    # Re-check the original rental clock immediately before adopting it.
    elapsed_hours = max(
        0.0,
        (datetime.now(UTC) - started).total_seconds() / 3600,
    )
    dph = _require_live_number(instance, "dph_total", alternate="dph_base")
    accrued_cost = elapsed_hours * dph
    if elapsed_hours >= cleanup_trigger_hours(manifest, dph):
        raise OpsError(
            "existing rental exhausted the cleanup-reserved cap during preflight"
        )

    # Re-read local state after network preflight so two operators cannot adopt
    # the same run ID or immutable instance based on the same stale snapshot.
    records = load_state()
    if any(record.get("run_id") == manifest.run_id for record in records):
        raise OpsError(f"run_id {manifest.run_id!r} became tracked during preflight")
    if any(
        record.get("instance_id") is not None
        and _strict_positive_identity(
            record["instance_id"], "persisted tracked instance ID"
        )
        == requested_instance_id
        for record in records
    ):
        raise OpsError("instance became tracked during adoption preflight")

    rental_started_at = started.isoformat().replace("+00:00", "Z")
    deadline_schedule = campaign_deadline_schedule(
        manifest,
        campaign_started_at=rental_started_at,
        campaign_accrued_cost_usd=accrued_cost,
    )
    if parse_utc(deadline_schedule["cleanup_start_at"]) <= datetime.now(UTC):
        raise OpsError("adopted rental has no positive pre-cleanup work window")
    record = {
        "run_id": manifest.run_id,
        "instance_id": requested_instance_id,
        "offer_id": (
            _strict_positive_identity(
                instance.get("ask_contract_id"), "adopted ask contract ID"
            )
            if instance.get("ask_contract_id") is not None
            else None
        ),
        "machine_id": instance.get("machine_id"),
        "dph": dph,
        # The monitor's cost and TTL clock intentionally begins at Vast's
        # original rental start, not when this command adopted the contract.
        "created_at": rental_started_at,
        "campaign_started_at": rental_started_at,
        "rental_started_at": rental_started_at,
        "adopted_at": utc_now(),
        "accrued_cost_at_adoption_usd": accrued_cost,
        "elapsed_hours_at_adoption": elapsed_hours,
        "source_commit": manifest.source_commit,
        "manifest_path": str(args.manifest.resolve()),
        "manifest": asdict(manifest),
        "expected_label": f"quant-experiments-{manifest.run_id}",
        "observed_label": str(instance.get("label") or ""),
        "lifecycle_origin": "existing_instance_adoption",
        "provisioning_owner_pid": _strict_positive_identity(
            os.getpid(), "adoption provisioning owner PID"
        ),
        "status": "adoption_pending",
        **deadline_schedule,
    }
    # This is the point of adoption: persist the immutable instance identity and
    # original billable start before upload, checkout creation, or process start.
    append_record(record)
    append_local_event(
        manifest.run_id,
        "adoption_pending",
        instance_id=requested_instance_id,
        dph=dph,
        elapsed_hours=elapsed_hours,
        accrued_cost_usd=accrued_cost,
        approved_max_usd=manifest.max_total_usd,
    )
    try:
        verified_label = label_adopted_instance(requested_instance_id, manifest.run_id)
        update_record_fields(
            manifest.run_id,
            observed_label=verified_label,
            status="adoption_provisioning",
        )
        append_local_event(
            manifest.run_id,
            "instance_adopted",
            instance_id=requested_instance_id,
            label=verified_label,
        )
        start_remote_run(connection, bundle, resolved_manifest)
        update_record_fields(
            manifest.run_id,
            status="running",
            remote_started_at=utc_now(),
        )
        append_local_event(
            manifest.run_id, "remote_started", instance_id=requested_instance_id
        )
    except Exception as setup_error:
        try:
            destroy_instance(
                requested_instance_id, manifest.run_id, "adoption_setup_failed"
            )
        except OpsError as cleanup_error:
            update_record_fields(
                manifest.run_id,
                status="provisioning_cleanup_pending",
                setup_error=_redacted_exception_text(setup_error),
                cleanup_error=_redacted_exception_text(cleanup_error),
                provisioning_owner_pid=None,
            )
            append_local_event(
                manifest.run_id,
                "adoption_cleanup_pending",
                error=_redacted_exception_text(cleanup_error),
            )
            try:
                launch_detached_monitor(manifest.run_id)
            except Exception as monitor_error:
                append_local_event(
                    manifest.run_id,
                    "adoption_cleanup_monitor_failed",
                    error=_redacted_exception_text(monitor_error),
                )
        raise

    print(
        json.dumps(
            {
                "run_id": manifest.run_id,
                "instance_id": requested_instance_id,
                "dph": dph,
                "accrued_cost_at_adoption_usd": round(accrued_cost, 4),
                "max_total_usd": manifest.max_total_usd,
                "ttl_hours": manifest.ttl_hours,
                "status": "running",
                "lifecycle_origin": "existing_instance_adoption",
            },
            indent=2,
        )
    )
    return start_requested_monitor(manifest.run_id, args.detach)


def command_monitor(args: argparse.Namespace) -> int:
    return monitor_run(args.run_id)


def command_purchase_reconcile(args: argparse.Namespace) -> int:
    return _purchase_reconciliation(args.run_id)


def command_destroy_reconcile(args: argparse.Namespace) -> int:
    return _destroy_reconciliation(args.run_id)


def command_status(_: argparse.Namespace) -> int:
    try:
        live = _require_provider_instance_rows(
            run_vast("show", "instances"), "status live instance query"
        )
    except OpsError as error:
        live = []
        _best_effort_emit_text(
            f"live query failed: {_redacted_exception_text(error)}",
            stream=sys.stderr,
        )
    live_by_id: dict[int, dict[str, Any]] = {}
    for item in live:
        if not isinstance(item, dict):
            continue
        item_id = _offer_identity_number(item, "id")
        if item_id < 2**63 - 1:
            live_by_id[item_id] = item
    rows = []
    for record in load_state():
        elapsed, cost = elapsed_and_cost(record)
        instance_id = record.get("instance_id")
        remote = live_by_id.get(int(instance_id)) if instance_id is not None else None
        rows.append(
            {
                "run_id": record["run_id"],
                "instance_id": instance_id,
                "local_status": record.get("status"),
                "live_status": remote.get("actual_status") if remote else None,
                "elapsed_hours": round(elapsed, 3),
                "estimated_cost_usd": round(cost, 3),
                "last_remote_event": record.get("last_remote_event", {}).get("event"),
            }
        )
    _best_effort_emit_json(rows, indent=2)
    return 0


def command_logs(args: argparse.Namespace) -> int:
    record = get_record(args.run_id)
    connection = ssh_url(record_instance_id(record))
    if args.train_tail:
        if args.train_tail < 1 or args.train_tail > 10_000:
            raise OpsError("--train-tail must be in [1, 10000]")
        arguments = [
            "python3",
            "-c",
            REMOTE_BOUNDED_TEXT_TAIL_PROGRAM,
            f"{REMOTE_OPS}/train.log",
            str(args.train_tail),
            str(_MAX_MANUAL_DIAGNOSTIC_BYTES),
        ]
        process = remote_exec(
            connection,
            " ".join(shlex.quote(argument) for argument in arguments),
            check=False,
        )
        stdout = getattr(process, "stdout", None)
        if (
            type(getattr(process, "returncode", None)) is not int
            or process.returncode != 0
            or not isinstance(stdout, str)
            or len(stdout) > _MAX_MANUAL_DIAGNOSTIC_BYTES
        ):
            _best_effort_emit_text(_DIAGNOSTIC_WITHHELD)
        else:
            _best_effort_emit_text(stdout)
    else:
        _best_effort_emit_json(read_remote_events(connection), indent=2)
    return 0


_PEEK_MAX_BYTES = 256 * 1024


def _peek_remote_path(value: str) -> str:
    """Restrict project-agent inspection to experiment outputs."""

    raw = (value or "artifacts").replace("\\", "/")
    if raw == "train.log":
        return f"{REMOTE_OPS}/train.log"
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or path.parts[0] != "artifacts"
    ):
        raise OpsError("peek paths must be train.log or stay under artifacts/")
    return f"{REMOTE_REPO}/{path.as_posix()}"


def command_peek(args: argparse.Namespace) -> int:
    """Read-only look at a live run's experiment outputs (for the project agent).

    Lists a directory, prints the head of a file, or tails it.  Output is
    bounded; nothing on the instance is modified and no lifecycle state is
    touched, so this is safe to call while the monitor owns the run.
    """

    remote_path = shlex.quote(_peek_remote_path(args.path))
    limit = _PEEK_MAX_BYTES
    if args.tail:
        if not 1 <= args.tail <= 10_000:
            raise OpsError("--tail must be in [1, 10000]")
        reader = f"tail -n {int(args.tail)} -- {remote_path}"
    else:
        reader = f"head -c {limit} -- {remote_path}"
    command = (
        f"if [ -d {remote_path} ]; then ls -la -- {remote_path}; "
        f"elif [ -f {remote_path} ]; then {reader}; "
        f"else echo 'peek: no such path' >&2; exit 3; fi | head -c {limit}"
    )
    record = get_record(args.run_id)
    connection = ssh_url(record_instance_id(record))
    process = remote_exec(connection, command, check=False)
    stdout = getattr(process, "stdout", "")
    if process.returncode != 0:
        _best_effort_emit_text(
            _redacted_diagnostic_text(str(getattr(process, "stderr", "")).strip())
            or "peek failed"
        )
        return 1
    _best_effort_emit_text(stdout if isinstance(stdout, str) else "")
    return 0


def command_ssh_command(args: argparse.Namespace) -> int:
    """Print the exact SSH invocation (repo known-hosts, scoped key) for a run.

    For the project agent's read-only inspection and for mayfly ``--ssh``.
    The account API key is never part of it.
    """

    record = get_record(args.run_id)
    connection = ssh_url(record_instance_id(record))
    argv = ssh_base(connection)
    _best_effort_emit_json(
        {
            "run_id": args.run_id,
            "argv": argv,
            "posix": shlex.join(argv),
            "known_hosts": KNOWN_HOSTS.as_posix(),
            "remote_artifacts": f"{REMOTE_REPO}/artifacts",
            "remote_train_log": f"{REMOTE_OPS}/train.log",
            "remote_events": f"{REMOTE_OPS}/events.jsonl",
        },
        indent=2,
    )
    return 0


def command_diagnose(args: argparse.Namespace) -> int:
    """Print a read-only utilization/host/metrics snapshot for a tracked run."""

    _best_effort_emit_json(
        read_remote_diagnostics(
            get_record(args.run_id), metrics_tail=args.metrics_tail
        ),
        indent=2,
    )
    return 0


def command_ssh(args: argparse.Namespace) -> int:
    record = get_record(args.run_id)
    connection = ssh_url(record_instance_id(record))
    print(subprocess.list2cmdline(ssh_base(connection)))
    return 0


def command_pull(args: argparse.Namespace) -> int:
    return (
        0
        if pull_and_verify(get_record(args.run_id), best_effort=args.best_effort)
        else 1
    )


def command_session_stage(args: argparse.Namespace) -> int:
    """Start a separately pinned stage on an existing verified research session."""

    parent = get_record(args.parent_run)
    instance_id = record_instance_id(parent)
    if str(args.confirm_instance) != str(instance_id):
        raise OpsError("--confirm-instance must exactly match the tracked instance ID")
    if args.confirm_parent != args.parent_run:
        raise OpsError("--confirm-parent must exactly match the parent run ID")
    parent_manifest = RunManifest.from_dict(parent["manifest"])
    if parent_manifest.success_policy != "keep_alive_after_verify":
        raise OpsError("session-stage requires a keep_alive_after_verify parent")
    if parent.get("status") != "session_ready":
        # A retained debugging authorization may intentionally begin with a
        # failing probe. Permit a follow-up pinned stage while the exact
        # adopted host is still live, but only for that terminal failed event;
        # ordinary sessions still require verified artifacts.
        last_event = parent.get("last_remote_event")
        if not (
            parent.get("status") == "running"
            and isinstance(last_event, dict)
            and last_event.get("event") == "run_failed"
            and last_event.get("run_id") == args.parent_run
        ):
            raise OpsError("parent run must have verified session_ready status")
    stage_manifest = load_manifest(args.manifest)
    if stage_manifest.run_id != args.confirm_stage:
        raise OpsError("--confirm-stage must exactly match the stage manifest run_id")
    if args.confirm_source_commit != stage_manifest.source_commit:
        raise OpsError("--confirm-source-commit must exactly match the stage source")
    if any(
        stage.get("stage_id") == stage_manifest.run_id for stage in load_stage_state()
    ):
        raise OpsError(f"stage_id {stage_manifest.run_id!r} is already tracked")
    if stage_manifest.success_policy != "keep_alive_after_verify":
        raise OpsError("session stages must use keep_alive_after_verify")
    if stage_manifest.max_dph > parent_manifest.max_dph:
        raise OpsError("stage max_dph exceeds the approved parent manifest")
    if stage_manifest.max_total_usd > parent_manifest.max_total_usd:
        raise OpsError("stage total cap exceeds the approved parent manifest")
    elapsed_hours, _ = elapsed_and_cost(parent)
    remaining = (
        cleanup_trigger_hours(parent_manifest, float(parent["dph"])) - elapsed_hours
    )
    if remaining <= cleanup_reserve_hours(stage_manifest.poll_seconds):
        raise OpsError("insufficient parent session time remains for a stage")
    if stage_manifest.ttl_hours > remaining:
        raise OpsError(
            "stage TTL exceeds time remaining before the parent cleanup reserve"
        )
    live = _require_provider_instance_rows(
        run_vast("show", "instances"), "session-stage live instance query"
    )
    matches = [item for item in live if item["id"] == instance_id]
    if len(matches) != 1:
        raise OpsError("parent instance is not uniquely present in the live API")
    bundle, resolved_manifest = create_source_bundle(stage_manifest)
    connection = ssh_url(instance_id)
    stage = {
        "stage_id": stage_manifest.run_id,
        "run_id": stage_manifest.run_id,
        "parent_run_id": args.parent_run,
        "instance_id": instance_id,
        "source_commit": stage_manifest.source_commit,
        "manifest_path": str(args.manifest.resolve()),
        "manifest": asdict(stage_manifest),
        "created_at": utc_now(),
        "dph": float(parent["dph"]),
        "status": "stage_uploading",
    }
    records = load_stage_state()
    records.append(stage)
    save_stage_state(records)
    try:
        stage_repo, stage_ops = start_remote_session_stage(
            connection,
            bundle,
            resolved_manifest,
            stage_manifest.run_id,
        )
    except Exception as error:
        stage["status"] = "stage_start_failed"
        stage["start_error"] = _redacted_exception_text(error)
        replace_stage_record(stage)
        raise
    stage["remote_repo"] = stage_repo
    stage["remote_ops"] = stage_ops
    stage["status"] = "stage_running"
    stage["remote_started_at"] = utc_now()
    replace_stage_record(stage)
    pid = launch_detached_stage_monitor(stage_manifest.run_id)
    print(
        json.dumps(
            {
                "stage_id": stage_manifest.run_id,
                "parent_run_id": args.parent_run,
                "instance_id": instance_id,
                "source_commit": stage_manifest.source_commit,
                "monitor_pid": pid,
                "status": "stage_running",
            },
            indent=2,
        )
    )
    return 0


def command_monitor_stage(args: argparse.Namespace) -> int:
    return monitor_session_stage(args.stage_id)


def command_stage_status(_: argparse.Namespace) -> int:
    report = []
    for stage in load_stage_state():
        report.append(
            {
                "stage_id": stage["stage_id"],
                "parent_run_id": stage["parent_run_id"],
                "instance_id": stage["instance_id"],
                "status": stage.get("status"),
                "source_commit": stage["source_commit"],
                "last_remote_event": (stage.get("last_remote_event") or {}).get(
                    "event"
                ),
            }
        )
    _best_effort_emit_json(report, indent=2)
    return 0


def command_diagnose_stage(args: argparse.Namespace) -> int:
    _best_effort_emit_json(
        read_remote_stage_diagnostics(
            get_stage_record(args.stage_id), metrics_tail=args.metrics_tail
        ),
        indent=2,
    )
    return 0


def command_pull_stage(args: argparse.Namespace) -> int:
    stage = get_stage_record(args.stage_id)
    return (
        0
        if pull_and_verify(
            stage,
            best_effort=args.best_effort,
            destination=_stage_artifact_destination(stage),
            remote_ops=stage["remote_ops"],
        )
        else 1
    )


def command_destroy(args: argparse.Namespace) -> int:
    record = get_record(args.run_id)
    instance_id = record_instance_id(record)
    if str(args.confirm_instance) != str(instance_id):
        raise OpsError("--confirm-instance must exactly match the tracked instance ID")
    destroy_instance(instance_id, args.run_id, "explicit_request")
    return 0


def request_remote_session_snapshot(
    record: dict[str, Any], *, deadline_monotonic: float | None = None
) -> None:
    """Repackage every declared artifact on a live pinned research checkout."""

    connection = ssh_url(
        record_instance_id(record),
        timeout=_deadline_timeout_seconds(deadline_monotonic, default=180),
    )
    command = " ".join(
        shlex.quote(item)
        for item in (
            "python3",
            f"{REMOTE_REPO}/vast/remote_runner.py",
            "--manifest",
            f"{REMOTE_ROOT}/run_manifest.json",
            "--ops-dir",
            REMOTE_OPS,
            "--repo",
            REMOTE_REPO,
            "--snapshot-only",
        )
    )
    process = remote_exec(
        connection,
        command,
        check=False,
        timeout=_deadline_timeout_seconds(deadline_monotonic, default=1200),
    )
    if process.returncode != 0:
        detail = _redacted_diagnostic_text(
            (process.stderr or process.stdout).strip()
        )[:1000]
        raise OpsError(f"remote session snapshot failed: {detail}")


def command_finalize_session(args: argparse.Namespace) -> int:
    """Verify the declared artifact snapshot, then end a research session."""

    record = get_record(args.run_id)
    instance_id = record_instance_id(record)
    if str(args.confirm_instance) != str(instance_id):
        raise OpsError("--confirm-instance must exactly match the tracked instance ID")
    manifest = RunManifest.from_dict(record["manifest"])
    if manifest.success_policy != "keep_alive_after_verify":
        raise OpsError("finalize-session requires keep_alive_after_verify")
    unfinished = [
        stage["stage_id"]
        for stage in load_stage_state()
        if stage.get("parent_run_id") == args.run_id
        and stage.get("status") != "stage_ready"
    ]
    if unfinished:
        raise OpsError(
            "cannot finalize with unverified session stages: " + ", ".join(unfinished)
        )
    request_remote_session_snapshot(record)
    if not pull_and_verify(record, snapshot_name="final"):
        raise OpsError("artifact verification failed; session remains live")
    destroy_instance(instance_id, args.run_id, "session_finalized_verified")
    return 0


def command_recover_destroy(args: argparse.Namespace) -> int:
    """Snapshot a stalled run, verify it locally, and only then destroy it."""

    record = get_record(args.run_id)
    instance_id = record_instance_id(record)
    if str(args.confirm_instance) != str(instance_id):
        raise OpsError("--confirm-instance must exactly match the tracked instance ID")
    request_remote_session_snapshot(record)
    if not pull_and_verify(record, snapshot_name="stalled_recovery"):
        raise OpsError("stalled-run recovery verification failed; instance remains live")
    destroy_instance(instance_id, args.run_id, "stalled_recovery_verified")
    return 0


def _direct_lifecycle_instance_ids(
    record: object, *, context: str
) -> set[int]:
    """Collect every persisted ID that can own or clean up a Vast contract."""

    if not isinstance(record, dict):
        raise OpsError(f"{context} is malformed")
    instance_ids: set[int] = set()
    for field_name in (
        "instance_id",
        "create_response_instance_id",
        "winner_instance_id",
    ):
        value = record.get(field_name)
        if value is not None:
            instance_ids.add(
                _strict_positive_identity(value, f"{context} {field_name}")
            )
    reconciliation_ids = record.get("reconciliation_exact_match_ids")
    if reconciliation_ids is not None:
        if not isinstance(reconciliation_ids, list):
            raise OpsError(
                f"{context} reconciliation_exact_match_ids are malformed"
            )
        for index, value in enumerate(reconciliation_ids):
            instance_ids.add(
                _strict_positive_identity(
                    value,
                    f"{context} reconciliation exact-match instance ID "
                    f"{index + 1}",
                )
            )

    candidates = record.get("candidates")
    if candidates is None:
        return instance_ids
    if not isinstance(candidates, list):
        raise OpsError(f"{context} candidates are malformed")
    for index, candidate in enumerate(candidates):
        candidate_context = f"{context} candidate {index + 1}"
        if not isinstance(candidate, dict):
            raise OpsError(f"{candidate_context} is malformed")
        for field_name in ("instance_id", "create_response_instance_id"):
            value = candidate.get(field_name)
            if value is not None:
                instance_ids.add(
                    _strict_positive_identity(
                        value, f"{candidate_context} {field_name}"
                    )
                )
        discovered = candidate.get("discovered_instance_ids")
        if discovered is None:
            continue
        if not isinstance(discovered, list):
            raise OpsError(
                f"{candidate_context} discovered_instance_ids are malformed"
            )
        for discovered_index, value in enumerate(discovered):
            instance_ids.add(
                _strict_positive_identity(
                    value,
                    f"{candidate_context} discovered instance ID "
                    f"{discovered_index + 1}",
                )
            )
    return instance_ids


def _require_unrecorded_orphan_target(instance_id: int, label: str) -> None:
    """Fence the manual orphan path from every local lifecycle owner."""

    _require_unrecorded_orphan_target_from_records(
        instance_id,
        label,
        records=load_state(),
        stages=load_stage_state(),
    )


def _require_unrecorded_orphan_target_from_records(
    instance_id: int,
    label: str,
    *,
    records: list[Any],
    stages: list[Any],
) -> None:
    """Validate one already-read lifecycle snapshot before orphan cleanup."""

    run_id = label.removeprefix("quant-experiments-")
    for index, record in enumerate(records):
        context = f"local lifecycle record {index + 1}"
        if not isinstance(record, dict):
            raise OpsError(f"{context} is malformed")
        record_run_id = record.get("run_id")
        if not isinstance(record_run_id, str) or not RUN_ID_PATTERN.fullmatch(
            record_run_id
        ):
            raise OpsError(f"{context} run_id is malformed")
        expected_label = record.get("expected_label")
        if expected_label is not None and (
            not isinstance(expected_label, str)
            or not expected_label.startswith("quant-experiments-")
            or expected_label == "quant-experiments-"
        ):
            raise OpsError(f"{context} expected_label is malformed")
        direct_ids = _direct_lifecycle_instance_ids(record, context=context)
        if instance_id in direct_ids:
            raise OpsError(
                "instance is referenced by local lifecycle state; use its "
                "owning cleanup or reconciliation path"
            )
        if record_run_id == run_id or expected_label == label:
            raise OpsError(
                "project label belongs to local lifecycle state; use its "
                "owning cleanup or reconciliation path"
            )
        candidates = record.get("candidates")
        if isinstance(candidates, list):
            for candidate_index, candidate in enumerate(candidates):
                if not isinstance(candidate, dict):
                    raise OpsError(
                        f"{context} candidate {candidate_index + 1} is malformed"
                    )
                candidate_label = candidate.get("label")
                if candidate_label is not None and (
                    not isinstance(candidate_label, str)
                    or not candidate_label.startswith("quant-experiments-")
                    or candidate_label == "quant-experiments-"
                ):
                    raise OpsError(
                        f"{context} candidate {candidate_index + 1} label is malformed"
                    )
                if candidate_label == label:
                    raise OpsError(
                        "project label belongs to local lifecycle state; use its "
                        "owning cleanup or reconciliation path"
                    )

    for index, stage in enumerate(stages):
        context = f"local session-stage record {index + 1}"
        if not isinstance(stage, dict):
            raise OpsError(f"{context} is malformed")
        stage_instance_id = stage.get("instance_id")
        if stage_instance_id is not None and _strict_positive_identity(
            stage_instance_id, f"{context} instance_id"
        ) == instance_id:
            raise OpsError(
                "instance is referenced by local session-stage state; use its "
                "owning lifecycle path"
            )
        stage_id = stage.get("stage_id")
        parent_run_id = stage.get("parent_run_id")
        if stage_id is not None and (
            not isinstance(stage_id, str)
            or not RUN_ID_PATTERN.fullmatch(stage_id)
        ):
            raise OpsError(f"{context} stage_id is malformed")
        if parent_run_id is not None and (
            not isinstance(parent_run_id, str)
            or not RUN_ID_PATTERN.fullmatch(parent_run_id)
        ):
            raise OpsError(f"{context} parent_run_id is malformed")
        if stage_id == run_id or parent_run_id == run_id:
            raise OpsError(
                "project label belongs to local session-stage state; use its "
                "owning lifecycle path"
            )


def command_destroy_orphan(args: argparse.Namespace) -> int:
    instance_id = _strict_positive_identity(args.instance, "--instance")
    label = args.confirm_label
    run_id = (
        label.removeprefix("quant-experiments-")
        if isinstance(label, str)
        else ""
    )
    if (
        not isinstance(label, str)
        or not label.startswith("quant-experiments-")
        or not RUN_ID_PATTERN.fullmatch(run_id)
    ):
        raise OpsError(
            "--confirm-label must exactly match a live quant-experiments-* label"
        )
    _require_unrecorded_orphan_target(instance_id, label)
    live = _require_provider_instance_rows(
        run_vast("show", "instances"),
        "orphan-destroy live instance query",
        require_label=True,
    )
    matches = [
        item
        for item in live
        if isinstance(item, dict)
        and _offer_identity_number(item, "id") == instance_id
    ]
    if len(matches) != 1:
        raise OpsError("the confirmed orphan instance is not present in the live API")
    live_label = str(matches[0].get("label") or "")
    if live_label != label:
        raise OpsError(
            "--confirm-label must exactly match a live quant-experiments-* label"
        )
    # Recheck after provider I/O so lifecycle ownership that appeared while the
    # read-only query was in flight still fences the destructive call.
    _require_unrecorded_orphan_target(instance_id, label)
    destroy_instance(
        instance_id,
        run_id,
        "explicit_orphan_cleanup",
        untracked_orphan_label=label,
    )
    return 0


def _reconciliation_absence_is_supervised(record: dict[str, Any]) -> bool:
    """Return whether an active owner should arbitrate one absent snapshot."""

    run_id = str(record.get("run_id") or "")
    status = str(record.get("status") or "")
    if (
        run_id
        and status == "destroy_followup_pending"
        and _destroy_reconciliation_lease_is_active(run_id)
    ):
        return True
    if record.get("cleanup_required"):
        return False
    owner_pid = record.get("provisioning_owner_pid")
    if (
        isinstance(owner_pid, int)
        and owner_pid > 0
        and process_is_alive(owner_pid)
    ):
        return True
    if not run_id:
        return False
    lock_paths: list[Path] = []
    if status in {
        "purchase_pending",
        "verifying_live_contract",
        "provisioning",
        "bootstrap_starting",
        "purchase_outcome_unknown",
        "purchase_reconciliation_pending",
    }:
        lock_paths.append(MONITOR_DIR / f"purchase_reconcile_{run_id}.lock")
    if status in {
        "running",
        "session_ready",
        "adoption_pending",
        "adoption_provisioning",
    }:
        lock_paths.append(MONITOR_DIR / f"{run_id}.lock")
    for lock_path in lock_paths:
        try:
            if _supervisor_lock_owner_pid(lock_path) is not None:
                return True
        except OpsError:
            # A held lease with metadata in transition is still supervision.
            return True
    return False


def _destroy_reconciliation_lease_is_active(run_id: str) -> bool:
    """Treat an OS-held destroy-followup lease as the current authority."""

    try:
        return (
            _supervisor_lock_owner_pid(
                MONITOR_DIR / f"destroy_reconcile_{run_id}.lock"
            )
            is not None
        )
    except OpsError:
        return True


def command_reconcile(args: argparse.Namespace) -> int:
    """Reconcile provider truth without holding the state lock across I/O."""

    pre_query_records = {
        str(record.get("run_id")): record for record in load_state()
    }
    live = _require_provider_instance_rows(
        run_vast("show", "instances"),
        "reconciliation live instance query",
        require_label=True,
    )
    with state_file_lock():
        report, restart_requests = _command_reconcile_locked(
            args, live, pre_query_records
        )
    _restart_reconciled_supervisors(restart_requests, report)
    _best_effort_emit_json(report, indent=2)
    return 0


def _command_reconcile_locked(
    args: argparse.Namespace,
    live: list[dict[str, Any]],
    pre_query_records: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    """Merge one already-fetched provider snapshot under the short state lock."""

    live_by_id: dict[int, dict[str, Any]] = {}
    for item in live:
        if not isinstance(item, dict):
            continue
        item_id = _offer_identity_number(item, "id")
        if item_id < 2**63 - 1:
            live_by_id[item_id] = item
    records = _load_state_unlocked()
    report = {
        "tracked": [],
        "untracked_live": [],
        "purchase_cleanup_matches": [],
        "monitors_restarted": [],
    }
    tracked_ids = set()
    for record in records:
        pre_query_record = pre_query_records.get(str(record.get("run_id")))
        snapshot_generation_matches = (
            pre_query_record is not None and pre_query_record == record
        )
        # A race has one campaign record but several candidate contracts.  The
        # nested identities and labels are authoritative until the coordinator
        # promotes one candidate to the ordinary winner record.
        race_candidates = record.get("candidates")
        if isinstance(race_candidates, list):
            race_report: list[dict[str, Any]] = []
            for candidate_index, candidate in enumerate(race_candidates):
                if not isinstance(candidate, dict):
                    continue
                candidate_id = candidate.get("instance_id")
                label_matches: list[dict[str, Any]] = []
                if candidate_id is None:
                    label = str(candidate.get("label") or "")
                    label_matches = [
                        item
                        for item in live
                        if label and item.get("label") == label
                    ]
                    label_id = (
                        _offer_identity_number(label_matches[0], "id")
                        if len(label_matches) == 1
                        else 2**63 - 1
                    )
                    known_id = _offer_identity_number({"id": candidate_id}, "id")
                    if len(label_matches) == 1 and label_id < 2**63 - 1 and (
                        known_id >= 2**63 - 1 or known_id != label_id
                    ):
                        candidate_id = label_id
                        candidate["instance_id"] = candidate_id
                        candidate["status"] = "orphaned_after_race"
                        if not _copy_reconciled_live_dph(
                            candidate, label_matches[0]
                        ):
                            candidate["status"] = "cleanup_pending"
                            candidate["cleanup_required"] = True
                            record["status"] = "race_cleanup_required"
                            record["cleanup_required"] = True
                candidate_id = _offer_identity_number({"id": candidate_id}, "id")
                if candidate_id >= 2**63 - 1:
                    candidate_id = None
                if candidate_id is not None:
                    tracked_ids.add(candidate_id)
                live_candidate = (
                    candidate_id is not None and candidate_id in live_by_id
                )
                if live_candidate and candidate.get("status") in {
                    "destroyed",
                    "absent_from_live_api",
                }:
                    # Provider state is authoritative.  A destroy response can
                    # race the next snapshot, so a locally terminal candidate
                    # that is live must become cleanup work again; treating it
                    # as absent would orphan a billable contract.
                    candidate["status"] = "cleanup_pending"
                    candidate["cleanup_required"] = True
                    candidate["cleanup_error"] = (
                        "provider still reports this locally terminal race candidate"
                    )
                    record["cleanup_required"] = True
                    if (
                        isinstance(record.get("winner_index"), int)
                        and record["winner_index"] != candidate_index
                    ):
                        record["cleanup_only_losers"] = True
                        record.setdefault(
                            "cleanup_resume_status", record.get("status")
                        )
                    record["status"] = "race_cleanup_required"
                if (
                    candidate_id is not None
                    and not live_candidate
                    and snapshot_generation_matches
                    and candidate.get("status")
                    not in {"destroyed", "absent_from_live_api"}
                ):
                    candidate["status"] = "absent_from_live_api"
                    candidate["absent_from_live_at"] = utc_now()
                elif (
                    candidate_id is None
                    and not label_matches
                    and snapshot_generation_matches
                    and candidate.get("status")
                    not in {"destroyed", "absent_from_live_api"}
                ):
                    if candidate.get("status") in {
                        "purchase_outcome_unknown",
                        "purchasing",
                        "verifying_live_contract",
                    }:
                        candidate["live_miss_count"] = int(
                            candidate.get("live_miss_count", 0)
                        ) + 1
                        candidate["unknown_since"] = candidate.get(
                            "unknown_since", candidate.get("started_at", utc_now())
                        )
                    else:
                        candidate["status"] = "absent_from_live_api"
                        candidate["absent_from_live_at"] = utc_now()
                race_report.append(
                    {
                        "candidate_id": candidate.get("candidate_id"),
                        "instance_id": candidate_id,
                        "label": candidate.get("label"),
                        "status": candidate.get("status"),
                        "live": live_candidate,
                    }
                )
            record["race_candidates_reconciled"] = utc_now()
            report.setdefault("race_candidates", []).append(
                {"run_id": record["run_id"], "candidates": race_report}
            )
            incomplete = any(
                candidate.get("status")
                not in {"destroyed", "absent_from_live_api"}
                for candidate in race_candidates
                if isinstance(candidate, dict)
            )
            coordinator_pid = record.get("coordinator_pid")
            coordinator_alive = (
                isinstance(coordinator_pid, int)
                and coordinator_pid > 0
                and process_is_alive(coordinator_pid)
            )
            if incomplete and not coordinator_alive and record.get("status") not in {
                "running",
                "session_ready",
            }:
                record["status"] = "race_cleanup_required"
                record["cleanup_required"] = True
                record["reconciled_at"] = utc_now()
        instance_value = record.get("instance_id")
        expected_label = record.get("expected_label")
        raw_label_matches = [
            item
            for item in live
            if expected_label and item.get("label") == expected_label
        ]
        label_matches = [
            item
            for item in raw_label_matches
            if _is_reconcilable_purchase_instance(record, item)
        ]
        known_instance_id = _offer_identity_number(
            {"id": instance_value}, "id"
        )
        exact_match_ids = {
            match_id
            for item in label_matches
            if (match_id := _offer_identity_number(item, "id")) < 2**63 - 1
        }
        exact_match_has_invalid_id = len(exact_match_ids) != len(label_matches)
        additional_exact_purchase_match = (
            instance_value is not None
            and (
                exact_match_has_invalid_id
                or any(match_id != known_instance_id for match_id in exact_match_ids)
            )
        )
        # All exact label+offer+machine matches belong to this campaign.  If the
        # provider exposes a new identity after one was already tracked, retain
        # purchase-reconciliation authority so it can drain every duplicate.
        # Do not overwrite the old identity here: it may also reappear during
        # the bounded late-create watch.
        if additional_exact_purchase_match:
            tracked_ids.update(exact_match_ids)
            _record_cleanup_evidence(
                record,
                status="purchase_reconciled_cleanup_pending",
                manual_reconciliation_required=exact_match_has_invalid_id,
                reconciliation_exact_match_ids=sorted(exact_match_ids),
                reconciliation_exact_match_invalid_id=exact_match_has_invalid_id,
                reconciliation_error=(
                    "additional exact label+offer+machine identity observed; "
                    "purchase reconciliation must drain all matches"
                ),
            )
            report["purchase_cleanup_matches"].append(
                {
                    "run_id": record.get("run_id"),
                    "persisted_instance_id": instance_value,
                    "exact_live_instance_ids": sorted(exact_match_ids),
                    "invalid_live_instance_id": exact_match_has_invalid_id,
                }
            )
        if raw_label_matches and not label_matches:
            _record_cleanup_evidence(
                record,
                manual_reconciliation_required=True,
                reconciliation_error=(
                    "matching label lacked the exact persisted offer+machine identity; "
                    "no destructive action was authorized"
                ),
            )
        if label_matches and record.get("status") == "destroyed":
            # A destroy acknowledgement can race the next API snapshot.  Keep
            # the record restartable so a late duplicate label is destroyed.
            _record_cleanup_evidence(
                record,
                status="purchase_reconciled_cleanup_pending",
                reconciliation_error=(
                    "late matching label observed after destroy; cleanup resumed"
                ),
            )
        if instance_value is None:
            if len(label_matches) == 1:
                instance_value = _offer_identity_number(label_matches[0], "id")
                if instance_value >= 2**63 - 1:
                    instance_value = None
                if instance_value is None:
                    _record_cleanup_evidence(
                        record,
                        status="purchase_reconciled_cleanup_pending",
                        manual_reconciliation_required=True,
                        reconciliation_error=(
                            "matching label did not report a positive ID"
                        ),
                    )
                else:
                    record["instance_id"] = instance_value
                    _record_cleanup_evidence(
                        record,
                        status="orphaned_after_purchase",
                        reconciled_at=utc_now(),
                    )
                if not _copy_reconciled_live_dph(record, label_matches[0]):
                    _record_cleanup_evidence(
                        record,
                        status="purchase_reconciled_cleanup_pending",
                        manual_reconciliation_required=True,
                        reconciliation_error=(
                            "exact live purchase match reported an invalid hourly price; "
                            "cleanup remains required"
                        ),
                    )
            elif len(label_matches) > 1 and record.get("status") in {
                "purchase_outcome_unknown",
                "purchase_reconciliation_pending",
                "purchase_reconciliation_expired",
                "purchase_reconciled_cleanup_pending",
                "late_create_reconciliation_active",
                "late_create_reconciliation_expired",
                "destroyed",
            }:
                _record_cleanup_evidence(
                    record,
                    status="purchase_reconciled_cleanup_pending",
                    reconciliation_error=(
                        "multiple live instances share expected label; destroy all matches"
                    ),
                )
        if instance_value is None:
            report["tracked"].append(
                {
                    "run_id": record["run_id"],
                    "instance_id": None,
                    "status": record.get("status"),
                    "live": False,
                }
            )
            continue
        try:
            instance_id = _strict_positive_identity(
                instance_value, f"run {record.get('run_id')!r} persisted instance ID"
            )
        except OpsError as error:
            if additional_exact_purchase_match:
                # A malformed stale identity cannot revoke cleanup authority
                # over independently authorized exact live matches. Purchase
                # reconciliation scans by label+offer+machine and can still
                # drain every valid live ID.
                _record_cleanup_evidence(
                    record,
                    status="purchase_reconciled_cleanup_pending",
                    persisted_instance_id_error=_redacted_exception_text(error),
                    manual_reconciliation_required=exact_match_has_invalid_id,
                )
            else:
                _record_cleanup_evidence(
                    record,
                    status="invalid_instance_id",
                    manual_reconciliation_required=True,
                    reconciliation_error=_redacted_exception_text(error),
                )
            report["tracked"].append(
                {
                    "run_id": record["run_id"],
                    "instance_id": None,
                    "status": record.get("status"),
                    "live": False,
                }
            )
            continue
        tracked_ids.add(instance_id)
        instance_is_live = instance_id in live_by_id
        incomplete_destroy_followup = (
            not instance_is_live
            and record.get("status")
            in {"destroyed", "destroy_followup_pending"}
            and record.get("destroy_reconciliation_status") != "complete"
            and not _destroy_reconciliation_lease_is_active(
                str(record.get("run_id"))
            )
        )
        if incomplete_destroy_followup:
            _record_cleanup_evidence(
                record,
                status="emergency_termination_unverified",
                cleanup_error=(
                    "destroy absence was observed but the required follow-up lease "
                    "did not complete"
                ),
                reconciled_at=utc_now(),
            )
        absence_is_supervised = (
            not instance_is_live
            and _reconciliation_absence_is_supervised(record)
        )
        if instance_is_live and record.get("status") in {
            "destroyed",
            "destroy_followup_pending",
            "missing_from_live_api",
            "emergency_termination_unverified",
        }:
            _record_cleanup_evidence(
                record,
                status="cleanup_required",
                manual_reconciliation_required=False,
                reconciliation_error=(
                    "provider reports a live instance after a locally absent or "
                    "terminal state; cleanup resumed"
                ),
                reconciled_at=utc_now(),
            )
        elif (
            not instance_is_live
            and record.get("status") != "destroyed"
            and snapshot_generation_matches
            and not absence_is_supervised
            and not additional_exact_purchase_match
        ):
            # One provider omission is not terminal truth.  Keep a cleanup
            # verifier active so a transiently missing contract is destroyed
            # if it reappears, matching the ordinary monitor path.
            _record_cleanup_evidence(
                record,
                status="emergency_termination_unverified",
                cleanup_error=(
                    "instance absent from one reconciliation snapshot; provider "
                    "termination remains unverified"
                ),
                reconciled_at=utc_now(),
            )
        elif not instance_is_live and (
            not snapshot_generation_matches or absence_is_supervised
        ):
            report.setdefault("stale_snapshots_ignored", []).append(
                {
                    "run_id": record.get("run_id"),
                    "reason": (
                        "active lifecycle owner retained authority over one "
                        "absent provider snapshot"
                        if absence_is_supervised
                        else "local lifecycle changed during provider query"
                    ),
                }
            )
        report["tracked"].append(
            {
                "run_id": record["run_id"],
                "instance_id": instance_id,
                "status": record.get("status"),
                "live": instance_is_live,
            }
        )
    for instance_id, instance in live_by_id.items():
        if instance_id not in tracked_ids:
            report["untracked_live"].append(
                {
                    "instance_id": instance_id,
                    "gpu_name": instance.get("gpu_name"),
                    "dph": instance.get("dph_total"),
                    "label": instance.get("label"),
                    "cleanup": (
                        "use destroy-orphan with the exact instance ID and label"
                        if str(instance.get("label") or "").startswith(
                            "quant-experiments-"
                        )
                        else "not managed by QuantExperiments"
                    ),
                }
            )
    restart_requests: list[tuple[str, str]] = []
    if args.restart_monitors:
        for record in records:
            purchase_statuses = {
                "purchase_pending",
                "verifying_live_contract",
                "provisioning",
                "bootstrap_starting",
                "purchase_outcome_unknown",
                "purchase_reconciliation_pending",
                "purchase_reconciliation_expired",
                "purchase_reconciled_cleanup_pending",
                "late_create_reconciliation_active",
                "late_create_reconciliation_expired",
                "orphaned_after_purchase",
            }
            if (
                record.get("status") in purchase_statuses
                and record.get("cleanup_required", True)
            ):
                restart_requests.append(
                    ("purchase_reconciliation", str(record["run_id"]))
                )
                continue
            if (
                isinstance(record.get("candidates"), list)
                and record.get("status")
                in {
                    "provisioning_cleanup_pending",
                    "race_cleanup_failed",
                    "race_cleanup_required",
                }
            ):
                restart_requests.append(("race_cleanup", str(record["run_id"])))
                continue
            if (
                record.get("status")
                in {
                    "destroy_followup_pending",
                    "emergency_termination_unverified",
                }
                and record.get("instance_id") not in live_by_id
            ):
                manifest = RunManifest.from_dict(record["manifest"])
                controller_raw = record.get("controller_deadline_at")
                controller_expired = (
                    controller_raw is not None
                    and parse_utc(str(controller_raw)) <= datetime.now(UTC)
                )
                if controller_expired:
                    manual_deadline = datetime.now(UTC) + timedelta(
                        seconds=destroy_reconciliation_lease_seconds(
                            max(
                                DESTROY_RECONCILIATION_POLL_SECONDS,
                                int(manifest.poll_seconds),
                            )
                        )
                    )
                    record["manual_destroy_reconciliation_deadline_at"] = (
                        manual_deadline.isoformat().replace("+00:00", "Z")
                    )
                    record["cap_control_failed"] = True
                    record["cap_control_failure_at"] = utc_now()
                    record["cap_control_failure_reason"] = (
                        "manual absence verification began after the persisted "
                        "controller deadline"
                    )
                restart_requests.append(
                    ("destroy_reconciliation", str(record["run_id"]))
                )
                continue
            if (
                not isinstance(record.get("candidates"), list)
                and record.get("cleanup_required")
                and record.get("instance_id") in live_by_id
            ):
                # A live instance past its controller deadline still needs a
                # destroy attempt.  Without a fresh manual boundary the
                # reconciliation supervisor's deadline is already in the past
                # and it would exit without ever calling the provider.
                manifest = RunManifest.from_dict(record["manifest"])
                controller_raw = record.get("controller_deadline_at")
                if (
                    controller_raw is not None
                    and parse_utc(str(controller_raw)) <= datetime.now(UTC)
                ):
                    manual_deadline = datetime.now(UTC) + timedelta(
                        seconds=destroy_reconciliation_lease_seconds(
                            max(
                                DESTROY_RECONCILIATION_POLL_SECONDS,
                                int(manifest.poll_seconds),
                            )
                        )
                    )
                    record["manual_destroy_reconciliation_deadline_at"] = (
                        manual_deadline.isoformat().replace("+00:00", "Z")
                    )
                    record["cap_control_failed"] = True
                    record["cap_control_failure_at"] = utc_now()
                    record["cap_control_failure_reason"] = (
                        "live instance still present after the persisted "
                        "controller deadline; manual destroy reconciliation"
                    )
                restart_requests.append(
                    ("destroy_reconciliation", str(record["run_id"]))
                )
                continue
            if (
                record.get("lifecycle_origin") == "existing_instance_adoption"
                and record.get("status")
                in {"adoption_pending", "adoption_provisioning"}
            ):
                owner_pid = record.get("provisioning_owner_pid")
                if (
                    isinstance(owner_pid, int)
                    and owner_pid > 0
                    and process_is_alive(owner_pid)
                ):
                    continue
            if record.get("instance_id") in live_by_id and record.get("status") != "destroyed":
                restart_requests.append(("monitor", str(record["run_id"])))
    _save_state_unlocked(records)
    return report, restart_requests


def _restart_reconciled_supervisors(
    restart_requests: list[tuple[str, str]], report: dict[str, Any]
) -> None:
    """Launch requested supervisors only after releasing the state lock."""

    for kind, run_id in restart_requests:
        try:
            if kind == "purchase_reconciliation":
                lock_path = MONITOR_DIR / f"purchase_reconcile_{run_id}.lock"
                owner = _supervisor_lock_owner_pid(lock_path)
                if owner is not None:
                    continue
                pid = launch_detached_purchase_reconciliation(run_id)
            elif kind == "race_cleanup":
                pid = launch_detached_race_cleanup(run_id)
            elif kind == "destroy_reconciliation":
                lock_path = MONITOR_DIR / f"destroy_reconcile_{run_id}.lock"
                owner = _supervisor_lock_owner_pid(lock_path)
                if owner is not None:
                    continue
                pid = launch_detached_destroy_reconciliation(run_id)
            elif kind == "monitor":
                lock_path = MONITOR_DIR / f"{run_id}.lock"
                owner = _supervisor_lock_owner_pid(lock_path)
                if owner is not None:
                    continue
                pid = launch_detached_monitor(run_id)
            else:  # pragma: no cover - internal request construction is closed
                raise OpsError(f"unknown reconciliation supervisor kind {kind!r}")
        except OpsError as error:
            report["monitors_restarted"].append(
                {
                    "run_id": run_id,
                    "error": _redacted_exception_text(error),
                    "kind": kind,
                }
            )
        else:
            report["monitors_restarted"].append(
                {"run_id": run_id, "pid": pid, "kind": kind}
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command_name", required=True)

    sub = commands.add_parser("doctor")
    sub.set_defaults(function=command_doctor)

    sub = commands.add_parser("validate-manifest")
    sub.add_argument("manifest", type=Path)
    sub.set_defaults(function=command_validate_manifest)

    sub = commands.add_parser("search")
    sub.add_argument("--gpu", default="RTX_5090")
    sub.add_argument("--num-gpus", type=int, default=1)
    sub.add_argument("--max-dph", type=float, required=True)
    sub.add_argument("--disk-gb", type=int, default=30)
    sub.add_argument("--min-cuda", type=float, default=12.8)
    sub.add_argument("--min-inet", type=int, default=100)
    sub.add_argument("--min-cpu-cores", type=int, default=DEFAULT_MIN_CPU_CORES)
    sub.add_argument("--min-cpu-ram-gb", type=int, default=DEFAULT_MIN_CPU_RAM_GB)
    sub.add_argument("--require-consumer-cpu", action="store_true")
    sub.add_argument("--country")
    sub.add_argument(
        "--avoid-machine",
        type=int,
        action="append",
        default=[],
        help="Additional physical machine ID to exclude from this read-only search.",
    )
    sub.add_argument("--limit", type=int, default=20)
    sub.set_defaults(function=command_search)

    sub = commands.add_parser("launch")
    sub.add_argument("manifest", type=Path)
    sub.add_argument("--offer", type=int)
    sub.add_argument("--detach", action="store_true")
    sub.set_defaults(function=command_launch)

    sub = commands.add_parser("race")
    sub.add_argument("manifest", type=Path)
    sub.add_argument(
        "--candidates",
        type=int,
        default=3,
        help="Number of distinct physical-machine candidates to provision (1-3).",
    )
    sub.add_argument("--detach", action="store_true")
    sub.set_defaults(function=command_race)

    sub = commands.add_parser("race-cleanup")
    sub.add_argument("run_id")
    sub.set_defaults(function=command_race_cleanup)

    sub = commands.add_parser("race-watchdog")
    sub.add_argument("run_id")
    sub.set_defaults(function=command_race_watchdog)

    sub = commands.add_parser("adopt-existing")
    sub.add_argument("manifest", type=Path)
    sub.add_argument("--instance", type=int, required=True)
    sub.add_argument("--detach", action="store_true")
    sub.set_defaults(function=command_adopt_existing)

    sub = commands.add_parser("monitor")
    sub.add_argument("run_id")
    sub.set_defaults(function=command_monitor)

    sub = commands.add_parser("purchase-reconcile")
    sub.add_argument("run_id")
    sub.set_defaults(function=command_purchase_reconcile)

    sub = commands.add_parser("destroy-reconcile")
    sub.add_argument("run_id")
    sub.set_defaults(function=command_destroy_reconcile)

    sub = commands.add_parser("status")
    sub.set_defaults(function=command_status)

    sub = commands.add_parser("logs")
    sub.add_argument("run_id")
    sub.add_argument("--train-tail", type=int, default=0)
    sub.set_defaults(function=command_logs)

    sub = commands.add_parser(
        "peek", help="read-only look at a run's artifacts/ or train.log"
    )
    sub.add_argument("run_id")
    sub.add_argument("path", nargs="?", default="artifacts")
    sub.add_argument("--tail", type=int, default=0)
    sub.set_defaults(function=command_peek)

    sub = commands.add_parser(
        "ssh-command", help="print the exact SSH command for a tracked run"
    )
    sub.add_argument("run_id")
    sub.set_defaults(function=command_ssh_command)

    sub = commands.add_parser("diagnose")
    sub.add_argument("run_id")
    sub.add_argument(
        "--metrics-tail",
        type=int,
        default=10,
        help="Maximum JSONL records per declared metrics file (1-100).",
    )
    sub.set_defaults(function=command_diagnose)

    sub = commands.add_parser("ssh")
    sub.add_argument("run_id")
    sub.set_defaults(function=command_ssh)

    sub = commands.add_parser("pull")
    sub.add_argument("run_id")
    sub.add_argument("--best-effort", action="store_true")
    sub.set_defaults(function=command_pull)

    sub = commands.add_parser("session-stage")
    sub.add_argument("manifest", type=Path)
    sub.add_argument("--parent-run", required=True)
    sub.add_argument("--confirm-parent", required=True)
    sub.add_argument("--confirm-instance", required=True)
    sub.add_argument("--confirm-stage", required=True)
    sub.add_argument("--confirm-source-commit", required=True)
    sub.set_defaults(function=command_session_stage)

    sub = commands.add_parser("monitor-stage")
    sub.add_argument("stage_id")
    sub.set_defaults(function=command_monitor_stage)

    sub = commands.add_parser("stage-status")
    sub.set_defaults(function=command_stage_status)

    sub = commands.add_parser("diagnose-stage")
    sub.add_argument("stage_id")
    sub.add_argument("--metrics-tail", type=int, default=10)
    sub.set_defaults(function=command_diagnose_stage)

    sub = commands.add_parser("pull-stage")
    sub.add_argument("stage_id")
    sub.add_argument("--best-effort", action="store_true")
    sub.set_defaults(function=command_pull_stage)

    sub = commands.add_parser("destroy")
    sub.add_argument("run_id")
    sub.add_argument("--confirm-instance", required=True)
    sub.set_defaults(function=command_destroy)

    sub = commands.add_parser("finalize-session")
    sub.add_argument("run_id")
    sub.add_argument("--confirm-instance", required=True)
    sub.set_defaults(function=command_finalize_session)

    sub = commands.add_parser("recover-destroy")
    sub.add_argument("run_id")
    sub.add_argument("--confirm-instance", required=True)
    sub.set_defaults(function=command_recover_destroy)

    sub = commands.add_parser("destroy-orphan")
    sub.add_argument("--instance", type=int, required=True)
    sub.add_argument("--confirm-label", required=True)
    sub.set_defaults(function=command_destroy_orphan)

    sub = commands.add_parser("reconcile")
    sub.add_argument("--restart-monitors", action="store_true")
    sub.set_defaults(function=command_reconcile)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        status = int(args.function(args))
    except Exception as error:
        try:
            print(
                f"ERROR: {_redacted_exception_text(error)}",
                file=sys.stderr,
            )
        except Exception:
            pass
        status = 2
    raise SystemExit(status)


if __name__ == "__main__":
    main()
