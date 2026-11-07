#!/usr/bin/env bash
# The shapequant vLLM plugin on one rented GPU: correctness against the same weights dequantized to
# bf16 (stock vLLM) and throughput of both, per model.
#   bash scripts/vllm_box.sh <out_dir> name=REPO:FILE@BASE [name=REPO:FILE@BASE ...]
# REPO:FILE: a self-contained packed file on the Hub; BASE: its base model repo (for the bf16 ref).
# Results: <out_dir>/<name>/{ours,ref}.json, compare.json, logs.
set -uo pipefail
OUT=$1; shift
PY=${PYTHON:-$(command -v python3 || command -v python)}
VENV=/workspace/vllm_env
CK=/workspace/vllm_ckpt
mkdir -p "$OUT" "$CK"
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "building vLLM env"
(
  $PY -m pip install -q uv &&
  $PY -m uv venv "$VENV" --python "$PY" -q &&
  $PY -m uv pip install --python "$VENV/bin/python" -q --torch-backend=auto vllm datasets &&
  $PY -m uv pip install --python "$VENV/bin/python" -q --no-deps -e . &&
  "$VENV/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, torch.version.cuda)"
) > "$OUT/vllm_env.log" 2>&1 || { log "vLLM env build failed"; tail -20 "$OUT/vllm_env.log"; exit 1; }
tail -1 "$OUT/vllm_env.log"

PROBE='import cupy as cp, torch
k = cp.RawKernel(r"""extern "C" __global__ void f(float* x){ x[threadIdx.x] += 1.0f; }""", "f")
x = cp.zeros(32, dtype=cp.float32); k((1,), (32,), (x,)); assert float(x.sum()) == 32.0; print("cupy ok", cp.__version__)'
ok=0
for pkgs in "cupy-cuda13x" "cupy-cuda12x" "cupy-cuda12x cuda-toolkit[nvrtc,cudart]==12.*"; do
  $PY -m uv pip install --python "$VENV/bin/python" -q $pkgs >> "$OUT/vllm_env.log" 2>&1
  if "$VENV/bin/python" -c "$PROBE" >> "$OUT/vllm_env.log" 2>&1; then ok=1; log "cupy via: $pkgs"; break; fi
done
[ $ok = 1 ] || { log "cupy setup failed"; tail -20 "$OUT/vllm_env.log"; exit 1; }
"$VENV/bin/python" -c "from vllm.plugins import load_general_plugins as l; l()
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS as q; assert 'shapequant' in q; print('plugin registered')" \
  >> "$OUT/vllm_env.log" 2>&1 || { log "plugin not registered"; tail -20 "$OUT/vllm_env.log"; exit 1; }
log "plugin registered"

for spec in "$@"; do
  name=${spec%%=*}; rest=${spec#*=}; src=${rest%@*}; base=${rest##*@}
  D="$OUT/$name"; mkdir -p "$D"
  log "== $name: exporting ($src, base $base)"
  $PY scripts/export_vllm_checkpoint.py --packed-hf "$src" --base "$base" --out "$CK/$name.sq" > "$D/export_ours.log" 2>&1 \
    || { log "plugin export failed"; tail -20 "$D/export_ours.log"; continue; }
  $PY scripts/export_eval_checkpoint.py --base "$base" --packed-hf "$src" --out "$CK/$name.ref" \
      --record "$D/export_ref.json" > "$D/export_ref.log" 2>&1 \
    || { log "bf16 export failed"; tail -20 "$D/export_ref.log"; continue; }
  log "running ours (plugin)"
  if ! "$VENV/bin/python" scripts/vllm_smoke.py run --model "$CK/$name.sq" --name "$name-shapequant" --out "$D/ours.json" > "$D/ours.log" 2>&1; then
    log "ours failed (compiled); tail:"; tail -30 "$D/ours.log"
    log "retrying ours with --eager"
    "$VENV/bin/python" scripts/vllm_smoke.py run --model "$CK/$name.sq" --name "$name-shapequant-eager" --eager \
        --out "$D/ours.json" > "$D/ours_eager.log" 2>&1 || { log "ours eager failed"; tail -30 "$D/ours_eager.log"; }
  fi
  grep -E "^\[" "$D/ours.log" "$D/ours_eager.log" 2>/dev/null | tail -14
  log "running ref (bf16)"
  "$VENV/bin/python" scripts/vllm_smoke.py run --model "$CK/$name.ref" --name "$name-bf16" --out "$D/ref.json" > "$D/ref.log" 2>&1 \
    || { log "ref failed"; tail -30 "$D/ref.log"; }
  grep -E "^\[" "$D/ref.log" | tail -14
  if [ -e "$D/ours.json" ] && [ -e "$D/ref.json" ]; then
    "$VENV/bin/python" scripts/vllm_smoke.py compare "$D/ours.json" "$D/ref.json" --out "$D/compare.json"
  fi
  rm -rf "$CK/$name.sq" "$CK/$name.ref"
done
log "all done"
