#!/usr/bin/env bash
# Speculative decoding on the 24 GB dev box: ours (vLLM + plugin) with no drafter, the RedHat DSpark
# drafter, and Qwen3.8's own MTP head; decode tok/s at conc 1 / 4 and acceptance (spec_bench.py),
# plus a 7x8 sanity answer per config.
#   bash scripts/spec_stage.sh <out_dir>
# Env: CONFIGS ("base dspark mtp"), DSPARK_K (7), MTP_K (3), MAX_LEN (16384); for a local box: VENV, CK, CKM
# (checkpoint dirs), PACKED_LOCAL (export from a local packed file), ALLOC_CONF ("" on WSL2: expandable
# segments make the plugin's shape tuning crawl there).
set -uo pipefail
OUT=$1
PY=${PYTHON:-$(command -v python3 || command -v python)}
VENV=${VENV:-/workspace/vllm_env}; VPY=$VENV/bin/python
PACKED=${PACKED:-luckymushy/Qwen3.8-27B-shapequant-2.5bpw:s16_25_ft.safetensors}
CONFIGS=${CONFIGS:-"base dspark mtp"}
DSPARK_K=${DSPARK_K:-7}; MTP_K=${MTP_K:-3}; MAX_LEN=${MAX_LEN:-16384}
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*"; }
SRC=(--packed-hf "$PACKED"); [ -n "${PACKED_LOCAL:-}" ] && SRC=(--packed "$PACKED_LOCAL")
ALLOC_CONF=${ALLOC_CONF-expandable_segments:True}
CK=${CK:-/workspace/vllm_ckpt/q27.sq}
[ -e "$CK/config.json" ] || $PY scripts/export_vllm_checkpoint.py "${SRC[@]}" --out "$CK" > "$OUT/export.log" 2>&1
CKM=${CKM:-/workspace/vllm_ckpt/q27.sq.mtp}     # bf16 embedding / head (shareable with drafters) + the MTP head
if [[ " $CONFIGS " == *" mtp "* || " $CONFIGS " == *" dspark_bf16 "* ]] && [ ! -e "$CKM/mtp.safetensors" ]; then
  $PY scripts/export_vllm_checkpoint.py "${SRC[@]}" --out "$CKM" --embed-head bf16 --mtp > "$OUT/export_mtp.log" 2>&1 \
    || { log "MTP export failed"; tail -5 "$OUT/export_mtp.log"; }
fi
stop() {         # kill every vLLM process and wait until the GPU is actually free (s40c s3: a slow DSpark
  pkill -f "vllm serve" 2>/dev/null   # start outlived the health wait, kept the GPU, and the next config's bench hit it)
  for _ in $(seq 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    pgrep -f "vllm serve|EngineCore" > /dev/null || { [ "${used:-0}" -lt 2000 ] && break; }
    pkill -9 -f "vllm serve|EngineCore" 2>/dev/null; sleep 5
  done
}
for cfg in $CONFIGS; do
  stop
  case $cfg in
    base)        ck=$CK;  spec="" ;;
    dspark)      ck=$CK;  spec="{\"model\":\"RedHatAI/Qwen3.8-27B-speculator.dspark\",\"num_speculative_tokens\":$DSPARK_K,\"method\":\"dspark\"}" ;;
    dspark_bf16) ck=$CKM; spec="{\"model\":\"RedHatAI/Qwen3.8-27B-speculator.dspark\",\"num_speculative_tokens\":$DSPARK_K,\"method\":\"dspark\"}" ;;
    mtp)         ck=$CKM; spec="{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":$MTP_K}" ;;
    *) log "unknown config $cfg"; continue ;;
  esac
  log "server: $cfg"
  env ${ALLOC_CONF:+PYTORCH_CUDA_ALLOC_CONF=$ALLOC_CONF} nohup "$VENV/bin/vllm" serve "$ck" --served-model-name ours \
    --host 127.0.0.1 --port 8000 --dtype bfloat16 --max-model-len $MAX_LEN --max-num-seqs 8 --gpu-memory-utilization 0.93 \
    --max-num-batched-tokens 4096 --seed 0 ${spec:+--speculative-config "$spec"} > "$OUT/server_$cfg.log" 2>&1 &
  pid=$!; up=0
  for _ in $(seq 270); do        # 45 min: a drafter download + compile can take ~30
    curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1 && { up=1; break; }
    kill -0 $pid 2>/dev/null || break; sleep 10
  done
  if [ $up = 0 ]; then log "$cfg: server did not come up"; grep -E "Error|error" "$OUT/server_$cfg.log" | tail -6; continue; fi
  grep -E "KV cache size|took .* GiB" "$OUT/server_$cfg.log" | tail -2
  curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"ours","messages":[{"role":"user","content":"What is 7 times 8? Reply with only the number."}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
    > "$OUT/sanity_$cfg.json"; log "$cfg sanity: $(head -c 400 "$OUT/sanity_$cfg.json" | grep -o '"content":"[^"]*"' | head -1)"
  "$VPY" scripts/spec_bench.py --endpoint http://127.0.0.1:8000 --model ours --name $cfg --out "$OUT/spec_$cfg.json" \
    > "$OUT/spec_$cfg.log" 2>&1; tail -3 "$OUT/spec_$cfg.log"
done
stop
for f in "$OUT"/server_*.log; do tail -c 30000 "$f" > "$f.tail"; rm -f "$f"; done
log "spec stage done"
