#!/usr/bin/env bash
# Vision on the 24 GB box: export the multimodal checkpoint (packed text model + the base model's
# bf16 vision tower), serve it with vLLM + the plugin, check image answers and a text sanity prompt.
#   bash scripts/vision_stage.sh <out_dir>
set -uo pipefail
OUT=$1
PY=${PYTHON:-$(command -v python3 || command -v python)}
VENV=/workspace/vllm_env; VPY=$VENV/bin/python
PACKED=${PACKED:-luckymushy/Qwen3.8-27B-shapequant-2.5bpw:s16_25_ft.safetensors}
CKV=/workspace/vllm_ckpt/q27.sq.vl
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*"; }
if [ ! -e "$CKV/visual.safetensors" ]; then
  log "export (vision)"
  $PY scripts/export_vllm_checkpoint.py --packed-hf "$PACKED" --out "$CKV" --vision > "$OUT/export_vision.log" 2>&1 \
    || { log "vision export failed"; tail -8 "$OUT/export_vision.log"; exit 0; }
  tail -3 "$OUT/export_vision.log"
fi
pkill -f "vllm serve" 2>/dev/null; sleep 10
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup "$VENV/bin/vllm" serve "$CKV" --served-model-name ours \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 --max-model-len 16384 --max-num-seqs 8 --gpu-memory-utilization 0.92 \
  --max-num-batched-tokens 4096 --limit-mm-per-prompt '{"image": 2}' --seed 0 > "$OUT/server_vision.log" 2>&1 &
up=0; for _ in $(seq 150); do curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1 && { up=1; break; }; sleep 10; done
if [ $up = 1 ]; then
  grep -E "KV cache size|took .* GiB" "$OUT/server_vision.log" | tail -2
  curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"ours","messages":[{"role":"user","content":"What is 7 times 8? Reply with only the number."}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
    > "$OUT/sanity_text.json"; log "text sanity: $(grep -o '"content":"[^"]*"' "$OUT/sanity_text.json" | head -1)"
  "$VPY" scripts/vision_check.py --endpoint http://127.0.0.1:8000 --model ours --out "$OUT/vision.json" 2>&1 | tail -4
else
  log "vision server did not come up"; grep -E "Error|error" "$OUT/server_vision.log" | tail -8
fi
pkill -f "vllm serve" 2>/dev/null; sleep 10
tail -c 40000 "$OUT/server_vision.log" > "$OUT/server_vision.log.tail"; rm -f "$OUT/server_vision.log"
log "vision stage done"
