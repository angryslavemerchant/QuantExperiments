#!/usr/bin/env bash
# Wishlist stage on the 24 GB dev box: quality prompts + long-context retrieval (incl. >128k) and
# KV-cache quantization, ours (vLLM serve + plugin) vs UD-Q4_K_M (llama-server), one server at a time.
#   bash scripts/wishlist_stage.sh <out_dir>
# Env: STEPS ("quality longctx kvquant"), LENGTHS, DEPTHS, OURS_MAX_LEN (131072), GGUF, SKIP_LLAMA (0).
set -uo pipefail
OUT=$1
PY=${PYTHON:-$(command -v python3 || command -v python)}
VENV=/workspace/vllm_env; VPY=$VENV/bin/python
PACKED=${PACKED:-luckymushy/Qwen3.8-27B-shapequant-2.5bpw:s16_25_ft.safetensors}
GGUF=${GGUF:-unsloth/Qwen3.8-27B-GGUF:Qwen3.8-27B-UD-Q4_K_M.gguf}
STEPS=${STEPS:-"quality longctx kvquant"}
LENGTHS=${LENGTHS:-8000,32000,64000,128000}
DEPTHS=${DEPTHS:-0.1,0.5,0.9}
OURS_MAX_LEN=${OURS_MAX_LEN:-131072}
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }
CK=/workspace/vllm_ckpt/q27.sq
[ -e "$CK/config.json" ] || $PY scripts/export_vllm_checkpoint.py --packed-hf "$PACKED" --out "$CK" > "$OUT/export.log" 2>&1
"$VPY" -m pip --version > /dev/null 2>&1 || true

wait_up() {      # port timeout_s
  local d=$(( $(date +%s) + $2 ))
  while [ $(date +%s) -lt $d ]; do curl -sf "http://127.0.0.1:$1/health" > /dev/null 2>&1 && return 0; sleep 10; done
  return 1
}
stop_all() { pkill -f "vllm serve" 2>/dev/null; pkill -f llama-server 2>/dev/null; sleep 15; }
ours_up() {      # tag max_len kv_dtype [max_seqs]: on a KV-cache shortfall, retry once at vLLM's estimated max length
  local len=$2 try
  for try in 1 2; do
    stop_all
    log "ours server ($1: max_len $len, kv $3)"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup "$VENV/bin/vllm" serve "$CK" --served-model-name ours       --host 127.0.0.1 --port 8000 --dtype bfloat16 --max-model-len $len --max-num-seqs ${4:-8} --gpu-memory-utilization 0.94       --max-num-batched-tokens 2048 --kv-cache-dtype $3 --seed 0 > "$OUT/server_ours_$1.log" 2>&1 &
    if wait_up 8000 1500; then
      grep -E "KV cache size|Maximum concurrency|took .* GiB" "$OUT/server_ours_$1.log" | tail -3
      echo "$len" > "$OUT/ours_$1.max_len"
      return 0
    fi
    local est=$(grep -oE "estimated maximum model length is [0-9]+" "$OUT/server_ours_$1.log" | grep -oE "[0-9]+$" | tail -1)
    log "ours ($1) did not come up at $len${est:+ (vLLM estimate $est)}"; grep -E "Error" "$OUT/server_ours_$1.log" | tail -3
    [ -n "$est" ] && [ "$try" = 1 ] || return 1
    len=$(( est / 1024 * 1024 )); mv "$OUT/server_ours_$1.log" "$OUT/server_ours_$1.try1.log"
  done
  return 1
}
LSERVER=""; LLIBS=""
llama_setup() {
  [ "${SKIP_LLAMA:-0}" = 1 ] && return 1
  [ -n "$LSERVER" ] && return 0
  "$VPY" - > "$OUT/llama_setup.log" 2>&1 <<'PYEOF'
import sys; sys.path.insert(0, "scripts")
from pathlib import Path
import bench_ship as b
env, d = b.llama_setup(Path("/workspace/llama_work"))
print("SERVER", next(d.rglob("llama-server")))
print("LIBS", env["LD_LIBRARY_PATH"])
PYEOF
  LSERVER=$(grep '^SERVER ' "$OUT/llama_setup.log" | cut -d' ' -f2-); LLIBS=$(grep '^LIBS ' "$OUT/llama_setup.log" | cut -d' ' -f2-)
  GG=$("$VPY" -c "from huggingface_hub import hf_hub_download as d; r, f = '$GGUF'.split(':', 1); print(d(r, f, local_dir='/workspace/gguf'))" | tail -1)
  [ -x "$LSERVER" ] && [ -f "$GG" ]
}
llama_up() {     # tag ctx kv_type(f16|q8_0|q4_0) [slots]
  stop_all
  log "llama-server Q4_K_M ($1: ctx $2, kv $3)"
  LD_LIBRARY_PATH=$LLIBS nohup "$LSERVER" -m "$GG" -ngl 99 --flash-attn on --jinja --reasoning-format none -np ${4:-1} -c $2 \
    -ctk $3 -ctv $3 --host 127.0.0.1 --port 8100 -t 8 > "$OUT/server_q4km_$1.log" 2>&1 &
  wait_up 8100 600 || { log "llama ($1) did not come up"; tail -5 "$OUT/server_q4km_$1.log"; return 1; }
}

if has quality; then
  ours_up quality 32768 auto && "$VPY" scripts/quality_prompts.py --target ours=http://127.0.0.1:8000 --model ours --max-tokens 6144 \
      --out "$OUT/quality_ours.json" > "$OUT/quality_ours.log" 2>&1; tail -3 "$OUT/quality_ours.log"
  if llama_setup && llama_up quality 65536 f16 4; then
    "$VPY" scripts/quality_prompts.py --target q4km=http://127.0.0.1:8100 --model q4km --conc 4 --max-tokens 6144 \
      --out "$OUT/quality_q4km.json" > "$OUT/quality_q4km.log" 2>&1; tail -3 "$OUT/quality_q4km.log"
  fi
fi
if has longctx; then
  if ours_up longctx $OURS_MAX_LEN auto 2; then
    "$VPY" scripts/long_ctx.py --endpoint http://127.0.0.1:8000 --model ours --name ours-bf16kv --lengths "$LENGTHS" \
      --depths "$DEPTHS" --out "$OUT/longctx_ours_bf16kv.json" > "$OUT/longctx_ours_bf16kv.log" 2>&1; tail -8 "$OUT/longctx_ours_bf16kv.log"
  fi
  if llama_setup && llama_up longctx 131072 q8_0; then
    "$VPY" scripts/long_ctx.py --endpoint http://127.0.0.1:8100 --model q4km --name q4km-q8kv --lengths "$LENGTHS" \
      --depths "$DEPTHS" --out "$OUT/longctx_q4km_q8kv.json" > "$OUT/longctx_q4km_q8kv.log" 2>&1; tail -8 "$OUT/longctx_q4km_q8kv.log"
  fi
fi
if has kvquant; then
  # ours with fp8 KV out to the model's 262k; llama q4_0 KV at 128k; llama f16 KV at 128k (expected not to fit)
  if ours_up fp8kv 262144 fp8 2; then
    "$VPY" scripts/long_ctx.py --endpoint http://127.0.0.1:8000 --model ours --name ours-fp8kv --lengths "$LENGTHS,200000,250000" \
      --depths "$DEPTHS" --out "$OUT/longctx_ours_fp8kv.json" > "$OUT/longctx_ours_fp8kv.log" 2>&1; tail -8 "$OUT/longctx_ours_fp8kv.log"
  fi
  if llama_setup && llama_up q4kv 131072 q4_0; then
    "$VPY" scripts/long_ctx.py --endpoint http://127.0.0.1:8100 --model q4km --name q4km-q4kv --lengths "$LENGTHS" \
      --depths "$DEPTHS" --out "$OUT/longctx_q4km_q4kv.json" > "$OUT/longctx_q4km_q4kv.log" 2>&1; tail -8 "$OUT/longctx_q4km_q4kv.log"
  fi
  llama_setup && { llama_up f16kv 131072 f16 && log "Q4_K_M f16 KV at 128k: fits" || log "Q4_K_M f16 KV at 128k: does not fit"; }
fi
stop_all
for f in "$OUT"/server_*.log; do tail -c 30000 "$f" > "$f.tail"; rm -f "$f"; done
log "wishlist stage done"
