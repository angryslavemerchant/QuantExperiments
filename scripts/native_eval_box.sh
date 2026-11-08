#!/usr/bin/env bash
# Thinking-mode GPQA-diamond for ours + GGUFs, each on its native engine, on one multi-GPU box:
#   ours    vLLM serve + shapequant plugin, one server per GPU in OURS_GPUS
#   GGUFs   llama.cpp llama-server (prebuilt CUDA, bench_ship.LLAMA_TAG), one per GPU
# One client (scripts/serve_eval.py) per model sends identical requests to that model's servers.
# Gates: env / downloads / export per model -> server health + one tiny chat -> 2-question smoke ->
# full run (round-major, every sample saved as it finishes, no new requests after DEADLINE_MIN).
#   bash scripts/native_eval_box.sh <out_dir>
# Env: OURS_GPUS ("0 1"), IQ2S_GPUS ("2 3 4"), Q4KM_GPUS ("5 6 7"), IQ2S_SLOTS / Q4KM_SLOTS
#      (llama-server parallel slots per GPU, each with SLOT_CTX tokens), GPQA_N (2), TASKS (gpqa),
#      DEADLINE_MIN / HARD_MIN (minutes from script start), LAYOUT (shapequant word layout).
set -uo pipefail
T0=$(date +%s)
OUT=$1
PY=${PYTHON:-$(command -v python3 || command -v python)}
VENV=/workspace/vllm_env
VPY=$VENV/bin/python
PACKED=${PACKED:-luckymushy/Qwen3.8-27B-shapequant-2.5bpw:s16_25_ft.safetensors}
GGUF_REPO=${GGUF_REPO:-unsloth/Qwen3.8-27B-GGUF}
OURS_GPUS=${OURS_GPUS:-"0 1"}; IQ2S_GPUS=${IQ2S_GPUS:-"2 3 4"}; Q4KM_GPUS=${Q4KM_GPUS:-"5 6 7"}
# f16 KV ~64 KB per token (16 attention layers): 7 x 36.9k-token slots = 17 GB next to IQ2_S (8.4 GB), 4 next to Q4_K_M (17 GB) on 32 GB
IQ2S_SLOTS=${IQ2S_SLOTS:-7}; Q4KM_SLOTS=${Q4KM_SLOTS:-4}; SLOT_CTX=${SLOT_CTX:-36864}
GPQA_N=${GPQA_N:-2}; TASKS=${TASKS:-gpqa}
DEADLINE_MIN=${DEADLINE_MIN:-72}; HARD_MIN=${HARD_MIN:-88}
LAYOUT=${LAYOUT:-lane}
mkdir -p "$OUT/servers" /workspace/gguf /workspace/ckpt
log() { echo "[$(date +%H:%M:%S)] $*"; }
mins() { echo $(( ($(date +%s) - T0) / 60 )); }
nvidia-smi --query-gpu=index,name,memory.total --format=csv | tee "$OUT/gpus.txt"

# ---------------------------------------------------------------- gate 1: env, downloads, export
log "env (vllm + cupy + this checkout)"
(
  $PY -m pip install -q uv &&
  $PY -m uv venv "$VENV" --python "$PY" -q &&
  $PY -m uv pip install --python "$VPY" -q --torch-backend=auto vllm datasets math-verify &&
  $PY -m uv pip install --python "$VPY" -q --no-deps -e .
) > "$OUT/env.log" 2>&1 || { log "env build failed"; tail -20 "$OUT/env.log"; exit 1; }
ok=0
for pkgs in cupy-cuda13x cupy-cuda12x; do
  $PY -m uv pip install --python "$VPY" -q $pkgs >> "$OUT/env.log" 2>&1
  if "$VPY" -c "import cupy as cp; k = cp.RawKernel(r'extern \"C\" __global__ void f(float* x){x[0]=1.f;}', 'f'); x = cp.zeros(1, dtype=cp.float32); k((1,), (1,), (x,)); assert float(x[0]) == 1" >> "$OUT/env.log" 2>&1; then ok=1; break; fi
done
OURS_OK=$ok; [ $ok = 1 ] || log "cupy setup failed: ours disabled"
"$VPY" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)" | tail -1

log "GGUF downloads (background)"
for f in UD-IQ2_S UD-Q4_K_M; do
  "$VPY" -c "from huggingface_hub import hf_hub_download as d; print(d('$GGUF_REPO', 'Qwen3.8-27B-$f.gguf', local_dir='/workspace/gguf'))" \
    > "$OUT/dl_$f.log" 2>&1 &
done
log "llama.cpp binaries"
"$VPY" - > "$OUT/llama_setup.log" 2>&1 <<'PYEOF'
import sys; sys.path.insert(0, "scripts")
from pathlib import Path
import bench_ship as b
env, d = b.llama_setup(Path("/workspace/llama_work"))
print("SERVER", next(d.rglob("llama-server")))
print("LIBS", env["LD_LIBRARY_PATH"])
PYEOF
LSERVER=$(grep '^SERVER ' "$OUT/llama_setup.log" | cut -d' ' -f2-)
LLIBS=$(grep '^LIBS ' "$OUT/llama_setup.log" | cut -d' ' -f2-)
[ -x "$LSERVER" ] || { log "llama.cpp setup failed: GGUFs disabled"; tail -5 "$OUT/llama_setup.log"; IQ2S_GPUS=""; Q4KM_GPUS=""; }

CK=/workspace/ckpt/ours
if [ "$OURS_OK" = 1 ]; then
  log "export ours"
  $PY scripts/export_vllm_checkpoint.py --packed-hf "$PACKED" --out "$CK" > "$OUT/export.log" 2>&1 \
    || { log "export failed: ours disabled"; tail -5 "$OUT/export.log"; OURS_OK=0; }
fi
[ "$OURS_OK" = 1 ] || OURS_GPUS=""
wait   # GGUF downloads (no servers started yet)
GG_IQ2S=$(tail -1 "$OUT/dl_UD-IQ2_S.log"); GG_Q4KM=$(tail -1 "$OUT/dl_UD-Q4_K_M.log")
[ -f "$GG_IQ2S" ] || { log "IQ2_S download failed"; IQ2S_GPUS=""; }
[ -f "$GG_Q4KM" ] || { log "Q4_K_M download failed"; Q4KM_GPUS=""; }
log "setup done at $(mins) min"

# ---------------------------------------------------------------- gate 2: servers
declare -A EPS CONC
start_vllm() {   # gpu port
  CUDA_VISIBLE_DEVICES=$1 SHAPEQUANT_LAYOUT=$LAYOUT PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup "$VENV/bin/vllm" serve "$CK" --served-model-name ours --host 127.0.0.1 --port $2 \
      --dtype bfloat16 --max-model-len $SLOT_CTX --gpu-memory-utilization 0.92 --max-num-seqs 64 \
      --max-num-batched-tokens 8192 --seed 0 > "$OUT/servers/vllm_$2.log" 2>&1 &
}
start_llama() {  # gpu port gguf slots ; flag variants for older / newer llama-server builds
  local gpu=$1 port=$2 gguf=$3 slots=$4
  for flags in "--flash-attn on --reasoning-format none" "--flash-attn on" "-fa"; do
    CUDA_VISIBLE_DEVICES=$gpu LD_LIBRARY_PATH=$LLIBS nohup "$LSERVER" -m "$gguf" -ngl 99 $flags --jinja \
      -np $slots -c $((slots * SLOT_CTX)) --host 127.0.0.1 --port $port -t 8 \
      > "$OUT/servers/llama_$port.log" 2>&1 &
    local pid=$!
    sleep 20
    if kill -0 $pid 2>/dev/null; then echo "$flags" > "$OUT/servers/llama_$port.flags"; return 0; fi
    log "llama-server :$port exited with flags '$flags'; tail:"; tail -3 "$OUT/servers/llama_$port.log"
  done
  return 1
}
healthy() {      # port timeout_s -> 0 when /health is 200 and a tiny chat answers
  local port=$1 deadline=$(( $(date +%s) + $2 ))
  while [ $(date +%s) -lt $deadline ]; do
    if curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
      curl -sf "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' \
        -d '{"model":"x","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8,"chat_template_kwargs":{"enable_thinking":false}}' \
        > "$OUT/servers/hello_$port.json" 2>&1 && return 0
    fi
    sleep 10
  done
  return 1
}
i=0; for g in $OURS_GPUS; do start_vllm $g $((8000 + i)); i=$((i + 1)); done
i=0; for g in $IQ2S_GPUS; do start_llama $g $((8100 + i)) "$GG_IQ2S" $IQ2S_SLOTS; i=$((i + 1)); done
i=0; for g in $Q4KM_GPUS; do start_llama $g $((8200 + i)) "$GG_Q4KM" $Q4KM_SLOTS; i=$((i + 1)); done
check() {        # model base_port gpus conc timeout
  local m=$1 base=$2 n=0 eps=""
  for g in $3; do
    local port=$((base + n)); n=$((n + 1))
    if healthy $port $5; then eps="$eps${eps:+,}http://127.0.0.1:$port"; log "$m :$port up"
    else log "$m :$port NOT healthy: dropped"; tail -5 "$OUT/servers/"*"_$port.log"; fi
  done
  EPS[$m]=$eps; CONC[$m]=$4
}
check ours 8000 "$OURS_GPUS" 64 1500
check iq2s 8100 "$IQ2S_GPUS" $IQ2S_SLOTS 600
check q4km 8200 "$Q4KM_GPUS" $Q4KM_SLOTS 600
nvidia-smi --query-gpu=index,memory.used --format=csv > "$OUT/servers/mem_after_load.txt"
up=""; for m in ours iq2s q4km; do [ -n "${EPS[$m]}" ] && up="$up $m"; done
[ -n "$up" ] || { log "no server came up: stopping"; exit 1; }
log "servers up at $(mins) min:$up"

# ---------------------------------------------------------------- gate 3: smoke (2 questions, 1024 tokens)
run_ok=""
for m in $up; do
  if "$VPY" scripts/serve_eval.py --endpoints "${EPS[$m]}" --conc 2 --model-name $m --name "$m-smoke" \
       --out "$OUT/smoke_$m" --tasks gpqa --n 1 --limit 2 --max-tokens 1024 --retries 0 > "$OUT/smoke_$m.log" 2>&1; then
    run_ok="$run_ok $m"; log "$m smoke ok"; tail -1 "$OUT/smoke_$m.log"
  else
    log "$m smoke FAILED: skipped"; tail -5 "$OUT/smoke_$m.log"
  fi
done
[ -n "$run_ok" ] || { log "every smoke failed: stopping"; exit 1; }

# ---------------------------------------------------------------- gate 4: full runs, all models at once
el=$(mins)
EVAL_PIDS=""
for m in $run_ok; do
  "$VPY" scripts/serve_eval.py --endpoints "${EPS[$m]}" --conc ${CONC[$m]} --model-name $m --name $m \
      --out "$OUT/$m" --tasks $TASKS --n $GPQA_N --deadline-min $((DEADLINE_MIN - el)) --hard-min $((HARD_MIN - el)) \
      > "$OUT/run_$m.log" 2>&1 &
  EVAL_PIDS="$EVAL_PIDS $!"
done
log "full runs started at $el min (new requests stop at $DEADLINE_MIN, hard stop $HARD_MIN):$run_ok"
for p in $EVAL_PIDS; do wait $p; done      # not the servers (also children of this shell)
for m in $run_ok; do tail -2 "$OUT/run_$m.log"; done
for f in "$OUT"/servers/*.log; do tail -c 20000 "$f" > "$f.tail"; done
log "all done at $(mins) min"
pkill -f "vllm serve" 2>/dev/null; pkill -f llama-server 2>/dev/null
exit 0
