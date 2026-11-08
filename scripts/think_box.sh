#!/usr/bin/env bash
# Thinking-mode GPQA-diamond + MATH-500 for one model on one rented GPU (scripts/think_eval.py).
#   bash scripts/think_box.sh <out_dir> <name> <SOURCE>
# SOURCE: packed-hf:REPO:FILE (served by the shapequant vLLM plugin) |
#         gguf-hf:REPO:FILE    (dequantized to a bf16 checkpoint of BASE, stock vLLM)
# Env: BASE (default Qwen/Qwen3.8-27B), TASKS (default gpqa,math500), EVAL_ARGS.
set -uo pipefail
OUT=$1; NAME=$2; SRC=$3
BASE=${BASE:-Qwen/Qwen3.8-27B}
PY=${PYTHON:-$(command -v python3 || command -v python)}
VENV=/workspace/vllm_env
CK=/workspace/ckpt/$NAME
mkdir -p "$OUT" /workspace/ckpt
log() { echo "[$(date +%H:%M:%S)] $*"; }
kind=${SRC%%:*}; path=${SRC#*:}

log "building vLLM env"
(
  $PY -m pip install -q uv &&
  $PY -m uv venv "$VENV" --python "$PY" -q &&
  $PY -m uv pip install --python "$VENV/bin/python" -q --torch-backend=auto vllm datasets math-verify &&
  $PY -m uv pip install --python "$VENV/bin/python" -q --no-deps -e .
) > "$OUT/env.log" 2>&1 || { log "env build failed"; tail -20 "$OUT/env.log"; exit 1; }
if [ "$kind" = packed-hf ]; then
  ok=0
  for pkgs in "cupy-cuda13x" "cupy-cuda12x"; do
    $PY -m uv pip install --python "$VENV/bin/python" -q $pkgs >> "$OUT/env.log" 2>&1
    if "$VENV/bin/python" -c "import cupy as cp; k = cp.RawKernel(r'extern \"C\" __global__ void f(float* x){x[0]=1.f;}', 'f'); x = cp.zeros(1, dtype=cp.float32); k((1,), (1,), (x,)); assert float(x[0]) == 1" >> "$OUT/env.log" 2>&1; then ok=1; break; fi
  done
  [ $ok = 1 ] || { log "cupy setup failed"; tail -20 "$OUT/env.log"; exit 1; }
fi
"$VENV/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, torch.cuda.get_device_name(0))"

log "exporting $NAME ($SRC)"
if [ "$kind" = packed-hf ]; then
  $PY scripts/export_vllm_checkpoint.py --packed-hf "$path" --base "$BASE" --out "$CK" > "$OUT/export.log" 2>&1
else
  $PY scripts/export_eval_checkpoint.py --base "$BASE" --gguf-hf "$path" --out "$CK" --record "$OUT/export.json" > "$OUT/export.log" 2>&1
fi || { log "export failed"; tail -20 "$OUT/export.log"; exit 1; }
tail -2 "$OUT/export.log"

log "evaluating"
"$VENV/bin/python" scripts/think_eval.py --model "$CK" --name "$NAME" --out "$OUT" --tasks "${TASKS:-gpqa,math500}" ${EVAL_ARGS:-} \
  > "$OUT/eval.log" 2>&1
log "eval exited $?"
grep -E "^\[" "$OUT/eval.log" | tail -12
