#!/usr/bin/env bash
# vLLM-side bisection of a broken plugin run (NLL ~ log(vocab)): short NLL + a few decode steps
# per variant, one engine at a time.
#   bash scripts/vllm_bisect.sh <out_dir>
# Env: VARIANTS ("eager_debug bf16head"), PACKED.
#   eager_debug: packed embed/head checkpoint, --eager, SHAPEQUANT_DEBUG=1 (per-op trace)
#   bf16head:    checkpoint exported with --embed-head bf16 (stock vLLM embedding / lm_head)
#   nocompact:   packed checkpoint, SHAPEQUANT_COMPACT=0 (needs > 24 GB)
#   notune:      packed checkpoint, SHAPEQUANT_TUNE=0
set -uo pipefail
OUT=$1
PY=${PYTHON:-$(command -v python3 || command -v python)}
VENV=/workspace/vllm_env; VPY=$VENV/bin/python
PACKED=${PACKED:-luckymushy/Qwen3.8-27B-shapequant-2.5bpw:s16_25_ft.safetensors}
VARIANTS=${VARIANTS:-"eager_debug bf16head"}
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*"; }
CK=/workspace/vllm_ckpt/q27.sq; CKB=/workspace/vllm_ckpt/q27.sq.bf16
[ -e "$CK/config.json" ] || $PY scripts/export_vllm_checkpoint.py --packed-hf "$PACKED" --out "$CK" > "$OUT/export.log" 2>&1
ARGS="--windows 2 --max-model-len 8192 --gpu-mem 0.90 --max-num-seqs 16 --max-num-batched-tokens 4096 --conc 1,8 --gen 64"
for v in $VARIANTS; do
  ck=$CK; extra=""; envs=""
  case $v in
    eager_debug) extra="--eager"; envs="SHAPEQUANT_DEBUG=1" ;;
    bf16head)    ck=$CKB
                 [ -e "$CKB/config.json" ] || $PY scripts/export_vllm_checkpoint.py --packed-hf "$PACKED" --out "$CKB" \
                   --embed-head bf16 > "$OUT/export_bf16.log" 2>&1 || { log "bf16 export failed"; tail -3 "$OUT/export_bf16.log"; continue; } ;;
    nocompact)   envs="SHAPEQUANT_COMPACT=0" ;;
    notune)      envs="SHAPEQUANT_TUNE=0" ;;
    *) log "unknown variant $v"; continue ;;
  esac
  log "variant $v"
  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $envs "$VPY" scripts/vllm_smoke.py run --model "$ck" --name "q27-$v" \
    --out "$OUT/vllm_$v.json" $ARGS $extra > "$OUT/vllm_$v.log" 2>&1 || log "$v failed"
  grep -E "^\[[0-9]|Error" "$OUT/vllm_$v.log" | tail -6
  grep -E "shapequant (debug|trace)" "$OUT/vllm_$v.log" | head -12
  "$VPY" -c "import json; d = json.load(open('$OUT/vllm_$v.json')); print('chat:', repr(d['chat']['text'][:160]))" 2>/dev/null
done
for f in "$OUT"/vllm_*.log; do [ $(stat -c %s "$f") -gt 400000 ] && { tail -c 400000 "$f" > "$f.t"; mv "$f.t" "$f"; }; done
log "bisect done"
