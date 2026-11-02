#!/usr/bin/env bash
set -Eeuo pipefail

BUNDLE_PATH="${1:?source bundle path is required}"
MANIFEST_PATH="${2:?manifest path is required}"
REPO_PATH="/workspace/quant-experiments"
OPS_PATH="/workspace/quant_experiments_ops"
BOOTSTRAP_LOG="${OPS_PATH}/bootstrap.log"
EVENTS_PATH="${OPS_PATH}/events.jsonl"

mkdir -p "${OPS_PATH}"

record_bootstrap_failure() {
    local code="$?"
    local timestamp
    timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '{"timestamp":"%s","run_id":"unknown","event":"bootstrap_failed","exit_code":%s}\n' \
        "${timestamp}" "${code}" >> "${EVENTS_PATH}"
    exit "${code}"
}
trap record_bootstrap_failure ERR

test -f "${BUNDLE_PATH}"
test -f "${MANIFEST_PATH}"
test -d "${REPO_PATH}/.git"

if [[ -x /venv/main/bin/python ]]; then
    PYTHON_BIN=/venv/main/bin/python
else
    PYTHON_BIN="$(command -v python3)"
fi

# Dependencies: the image already ships torch/CUDA. Extra job dependencies go
# in requirements-remote.txt (installed with deps). A pyproject/setup.py, if
# present, is installed editable without deps so it cannot replace torch.
# Without either, the runner puts the repo on PYTHONPATH.
#
# Host checks first, so a slow host fails in about a minute rather than after a
# 20-minute bootstrap timeout: a real download-speed probe, then a hard time
# limit on dependency setup.
if [[ -f "${REPO_PATH}/vast/net_probe.py" ]]; then
    "${PYTHON_BIN}" "${REPO_PATH}/vast/net_probe.py" \
        --min-mb-per-s "${QE_NET_PROBE_MIN_MB_PER_S:-25}" \
        >> "${BOOTSTRAP_LOG}" 2>&1
fi
SETUP_TIMEOUT="${QE_SETUP_TIMEOUT_SECONDS:-360}"
if [[ -f "${REPO_PATH}/requirements-remote.txt" ]]; then
    timeout "${SETUP_TIMEOUT}" "${PYTHON_BIN}" -m pip install --disable-pip-version-check \
        -r "${REPO_PATH}/requirements-remote.txt" >> "${BOOTSTRAP_LOG}" 2>&1
fi
if [[ -f "${REPO_PATH}/pyproject.toml" || -f "${REPO_PATH}/setup.py" ]]; then
    timeout "${SETUP_TIMEOUT}" "${PYTHON_BIN}" -m pip install --disable-pip-version-check \
        --no-deps -e "${REPO_PATH}" >> "${BOOTSTRAP_LOG}" 2>&1
fi

nohup "${PYTHON_BIN}" "${REPO_PATH}/vast/remote_runner.py" \
    --manifest "${MANIFEST_PATH}" \
    --ops-dir "${OPS_PATH}" \
    --repo "${REPO_PATH}" \
    >> "${BOOTSTRAP_LOG}" 2>&1 < /dev/null &

printf '%s\n' "$!" > "${OPS_PATH}/runner.pid"
trap - ERR
