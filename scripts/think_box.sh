#!/usr/bin/env bash
# Thinking-mode GPQA-diamond + MATH-500 for one model on one rented GPU, through lighteval 0.13.0
# (scripts/lighteval_run.py: lighteval's own tasks / prompts / metrics; only vLLM engine settings
# adjusted), with scripts/think_eval.py --style lighteval as the fallback if lighteval cannot run.
#   bash scripts/think_box.sh <out_dir> <name> <SOURCE>
# SOURCE: packed-hf:REPO:FILE (served by the shapequant vLLM plugin) |
#         gguf-hf:REPO:FILE    (dequantized to a bf16 checkpoint of BASE, stock vLLM)
# Env: BASE (default Qwen/Qwen3.8-27B); RUNS (lighteval runs "task:seed ...", default GPQA-diamond
#      2 x 4 samples, then MATH-500 1 sample); FALLBACK_ARGS.
set -uo pipefail
OUT=$1; NAME=$2; SRC=$3
BASE=${BASE:-Qwen/Qwen3.8-27B}
PY=${PYTHON:-$(command -v python3 || command -v python)}
VENV=/workspace/vllm_env
VPY=$VENV/bin/python
CK=/workspace/ckpt/$NAME
mkdir -p "$OUT" /workspace/ckpt
log() { echo "[$(date +%H:%M:%S)] $*"; }
kind=${SRC%%:*}; path=${SRC#*:}

log "building env (vllm, lighteval 0.13.0)"
(
  $PY -m pip install -q uv &&
  $PY -m uv venv "$VENV" --python "$PY" -q &&
  $PY -m uv pip install --python "$VPY" -q --torch-backend=auto vllm datasets math-verify &&
  $PY -m uv pip install --python "$VPY" -q --torch-backend=auto "lighteval==0.13.0" ray more_itertools &&
  $PY -m uv pip install --python "$VPY" -q --no-deps -e .
) > "$OUT/env.log" 2>&1 || { log "env build failed"; tail -20 "$OUT/env.log"; exit 1; }
if [ "$kind" = packed-hf ]; then
  ok=0
  for pkgs in "cupy-cuda13x" "cupy-cuda12x"; do
    $PY -m uv pip install --python "$VPY" -q $pkgs >> "$OUT/env.log" 2>&1
    if "$VPY" -c "import cupy as cp; k = cp.RawKernel(r'extern \"C\" __global__ void f(float* x){x[0]=1.f;}', 'f'); x = cp.zeros(1, dtype=cp.float32); k((1,), (1,), (x,)); assert float(x[0]) == 1" >> "$OUT/env.log" 2>&1; then ok=1; break; fi
  done
  [ $ok = 1 ] || { log "cupy setup failed"; tail -20 "$OUT/env.log"; exit 1; }
fi
"$VPY" -c "import vllm, torch, lighteval; print('vllm', vllm.__version__, 'torch', torch.__version__, 'lighteval', getattr(lighteval, '__version__', '?'), torch.cuda.get_device_name(0))" 2>&1 | tail -1

log "exporting $NAME ($SRC)"
if [ "$kind" = packed-hf ]; then
  $PY scripts/export_vllm_checkpoint.py --packed-hf "$path" --base "$BASE" --out "$CK" > "$OUT/export.log" 2>&1
else
  $PY scripts/export_eval_checkpoint.py --base "$BASE" --gguf-hf "$path" --out "$CK" --record "$OUT/export.json" > "$OUT/export.log" 2>&1
fi || { log "export failed"; tail -20 "$OUT/export.log"; exit 1; }
tail -2 "$OUT/export.log"

margs() {   # $1 seed, $2 max_new_tokens
  echo "model_name=$CK,dtype=bfloat16,max_model_length=36864,gpu_memory_utilization=0.9,max_num_seqs=256,max_num_batched_tokens=8192,seed=$1,generation_parameters={max_new_tokens:$2,temperature:0.6,top_p:0.95,top_k:20}"
}
summ() {    # print the metrics of the newest lighteval results json under $1
  "$VPY" - "$1" <<'PYEOF'
import glob, json, os, sys
fs = sorted(glob.glob(os.path.join(sys.argv[1], "results", "**", "results_*.json"), recursive=True))
if not fs:
    print("no results json"); sys.exit(1)
d = json.load(open(fs[-1]))
for t, m in d.get("results", {}).items():
    print(t, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
PYEOF
}

log "lighteval smoke (4 GPQA questions, 2 samples, 2k tokens)"
if "$VPY" scripts/lighteval_run.py vllm "$(margs 1234 2048)" "gpqa:diamond@k=1@n=2|0" --max-samples 4 \
     --output-dir "$OUT/le_smoke" > "$OUT/le_smoke.log" 2>&1 && summ "$OUT/le_smoke"; then
  grep -E "\[lighteval_run\]" "$OUT/le_smoke.log" | head -3
  i=0
  for run in ${RUNS:-"gpqa:diamond@k=1@n=4|0:1234" "gpqa:diamond@k=1@n=4|0:1235" "math_500@k=1@n=1|0:1234"}; do
    task=${run%:*}; seed=${run##*:}; i=$((i + 1))
    log "lighteval run $i: $task (seed $seed)"
    "$VPY" scripts/lighteval_run.py vllm "$(margs "$seed" 32768)" "$task" --save-details \
        --output-dir "$OUT/le_$i" > "$OUT/le_$i.log" 2>&1
    log "run $i exited $?"; summ "$OUT/le_$i"
  done
else
  log "lighteval smoke failed; tail:"; tail -25 "$OUT/le_smoke.log"
  log "fallback: think_eval.py --style lighteval (GPQA n=8, then MATH-500 n=1)"
  "$VPY" scripts/think_eval.py --model "$CK" --name "$NAME" --out "$OUT/fallback" --tasks gpqa --style lighteval \
      --n 8 --chunk 100 ${FALLBACK_ARGS:-} > "$OUT/fallback_gpqa.log" 2>&1
  grep -E "^\[" "$OUT/fallback_gpqa.log" | tail -4
  "$VPY" scripts/think_eval.py --model "$CK" --name "$NAME" --out "$OUT/fallback" --tasks math500 --style lighteval \
      --n 1 ${FALLBACK_ARGS:-} > "$OUT/fallback_math.log" 2>&1
  grep -E "^\[" "$OUT/fallback_math.log" | tail -4
fi
log "all done"
