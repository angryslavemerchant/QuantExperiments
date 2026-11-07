#!/usr/bin/env bash
# Benchmark suite on one rented GPU for one or more quantized models.
#   bash scripts/eval_box.sh <out_dir> <base_repo> name=SOURCE [name=SOURCE ...]
# SOURCE: packed-hf:REPO:FILE | gguf-hf:REPO:FILE
# Exports each model to a bf16 HF checkpoint (scripts/export_eval_checkpoint.py, main env; the
# next export runs while the previous model is being evaluated), then runs scripts/eval_suite.py
# in a separate vLLM env. Results: <out_dir>/<name>/summary.json (+ samples, logs).
set -uo pipefail
OUT=$1; BASE=$2; shift 2
PY=${PYTHON:-python}
VENV=/workspace/vllm_env
CK=/workspace/eval_ckpt
mkdir -p "$OUT" "$CK"
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "prefetching $BASE"
$PY -c "from huggingface_hub import snapshot_download as s; s('$BASE')" > "$OUT/prefetch.log" 2>&1 &
PREFETCH=$!

log "building vLLM env"
(
  $PY -m pip install -q uv &&
  uv venv "$VENV" --python 3.12 -q &&
  uv pip install --python "$VENV/bin/python" -q --torch-backend=auto vllm "lm_eval[ifeval,math]==0.4.12" math-verify &&
  "$VENV/bin/python" -c "import vllm, lm_eval, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
) > "$OUT/vllm_env.log" 2>&1
RC=$?
tail -3 "$OUT/vllm_env.log"
if [ $RC -ne 0 ]; then log "vLLM env build failed"; exit 1; fi
wait $PREFETCH || { log "base prefetch failed"; tail -5 "$OUT/prefetch.log"; exit 1; }

# exports, one after another, in the background
(
  for spec in "$@"; do
    name=${spec%%=*}; src=${spec#*=}; kind=${src%%:*}; path=${src#*:}
    mkdir -p "$OUT/$name"
    if [ "$kind" = packed-hf ]; then arg=(--packed-hf "$path"); else arg=(--gguf-hf "$path"); fi
    $PY scripts/export_eval_checkpoint.py --base "$BASE" "${arg[@]}" --out "$CK/$name" \
        --record "$OUT/$name/export.json" > "$OUT/$name/export.log" 2>&1 \
      && touch "$CK/$name.ready" || touch "$CK/$name.failed"
  done
) &

for spec in "$@"; do
  name=${spec%%=*}
  log "waiting for export of $name"
  while [ ! -e "$CK/$name.ready" ] && [ ! -e "$CK/$name.failed" ]; do sleep 20; done
  tail -2 "$OUT/$name/export.log"
  if [ -e "$CK/$name.failed" ]; then log "export of $name failed"; tail -20 "$OUT/$name/export.log"; continue; fi
  log "evaluating $name"
  "$VENV/bin/python" scripts/eval_suite.py --ckpt "$CK/$name" --name "$name" --out "$OUT/$name" \
      > "$OUT/$name/eval.log" 2>&1
  log "eval of $name exited $?"
  grep -E "\] [a-z_0-9]+: " "$OUT/$name/eval.log" | tail -12
  rm -rf "$CK/$name"
done
wait
log "all done"
