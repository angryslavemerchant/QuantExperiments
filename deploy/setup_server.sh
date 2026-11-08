#!/usr/bin/env bash
# One-time setup of the shapequant 27B server on a Linux (or WSL2) box with an NVIDIA GPU (24 GB+).
#   HF_TOKEN=hf_... bash deploy/setup_server.sh            (run from the repo root)
# Creates ~/sq-venv (vLLM + CuPy + this repo), downloads the private packed model and exports a
# vLLM checkpoint to ~/models/q27-sq. Re-running is safe (skips what exists).
# Env: VENV (~/sq-venv), MODEL_DIR (~/models/q27-sq), PACKED (HF repo:file of the packed model),
#   VISION (1: also export the base model's vision tower (+1 GB) so the server accepts images).
set -euo pipefail
VENV=${VENV:-$HOME/sq-venv}
MODEL_DIR=${MODEL_DIR:-$HOME/models/q27-sq}
PACKED=${PACKED:-luckymushy/Qwen3.8-27B-shapequant-2.5bpw:s16_25_ft.safetensors}
PY=${PYTHON:-python3}
VISION_ARG=""; [ "${VISION:-}" = 1 ] && VISION_ARG=--vision
log() { echo "[setup] $*"; }

[ -f pyproject.toml ] && grep -q shapequant pyproject.toml || { echo "run from the repo root"; exit 1; }
command -v nvidia-smi > /dev/null || { echo "nvidia-smi not found: install the NVIDIA driver (WSL2: the Windows driver)"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "${drv:-0}" -lt 580 ]; then
  echo "NVIDIA driver $drv is too old: vLLM 0.33 needs a CUDA 13 driver (580+). Update the driver first."; exit 1
fi
[ -n "${HF_TOKEN:-}" ] || { echo "set HF_TOKEN (a read token with access to the private model repo)"; exit 1; }

if [ ! -x "$VENV/bin/python" ]; then
  log "creating $VENV (vLLM, ~15 GB of disk)"
  $PY -m pip install -q --user uv
  $PY -m uv venv "$VENV" --python "$PY"
  $PY -m uv pip install --python "$VENV/bin/python" --torch-backend=auto vllm
fi
$PY -m uv pip install --python "$VENV/bin/python" -q --no-deps -e .
if ! "$VENV/bin/python" -c "import cupy" 2> /dev/null; then
  for pkg in cupy-cuda13x cupy-cuda12x; do
    $PY -m uv pip install --python "$VENV/bin/python" -q "$pkg"
    if "$VENV/bin/python" -c "import cupy as cp; k = cp.RawKernel(r'extern \"C\" __global__ void f(float* x){x[0]=1.f;}', 'f'); x = cp.zeros(1, dtype=cp.float32); k((1,), (1,), (x,)); assert float(x[0]) == 1"; then
      log "cupy: $pkg"; break
    fi
    $PY -m uv pip uninstall --python "$VENV/bin/python" -q "$pkg" || true
  done
fi
"$VENV/bin/python" -c "import vllm, torch, cupy; print('vllm', vllm.__version__, 'torch', torch.__version__, torch.cuda.get_device_name(0))"

if [ ! -f "$MODEL_DIR/config.json" ]; then
  log "downloading + exporting the model to $MODEL_DIR (~10 GB)"
  mkdir -p "$(dirname "$MODEL_DIR")"
  "$VENV/bin/python" scripts/export_vllm_checkpoint.py --packed-hf "$PACKED" --out "$MODEL_DIR" $VISION_ARG
elif [ "${VISION:-}" = 1 ] && [ ! -f "$MODEL_DIR/visual.safetensors" ]; then
  log "$MODEL_DIR is text-only: re-export with images by removing it first (rm -rf $MODEL_DIR)"
fi
log "done. Start the server with: bash deploy/serve.sh"
