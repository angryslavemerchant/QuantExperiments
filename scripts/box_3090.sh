#!/usr/bin/env bash
# RTX 3090 (sm_86, 24 GB) bring-up and tuning for the 27B packed model, one box:
#   1. kernel tests + per-matrix bench + vLLM plugin (q4 embed / q8 head from the packed file):
#      wikitext NLL and throughput by concurrency (scripts/kernel_dev_box.sh, VLLM_RUN=1)
#   2. standalone runtime decode launch-config sweep (scripts/tune_decode.py)
#   3. standalone runtime tg/pp + llama.cpp on Q4_K_M and IQ2_S, same card (scripts/bench_ship.py)
#   bash scripts/box_3090.sh <out_dir>
set -uo pipefail
OUT=$1
PY=${PYTHON:-$(command -v python3 || command -v python)}
PACKED=${PACKED:-luckymushy/Qwen3.8-27B-shapequant-2.5bpw:s16_25_ft.safetensors}
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*"; }
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,power.limit --format=csv | tee "$OUT/gpu.txt"

log "step 1: kernel dev stage (tests, per-matrix bench, vLLM plugin)"
VLLM_RUN=1 CONFIGS="${CONFIGS:-fused:1:64 fused:0:32}" \
  SMOKE_ARGS="${SMOKE_ARGS:---windows 8 --max-model-len 4096 --gpu-mem 0.88 --max-num-seqs 128 --conc 1,8,16,32,64,128}" \
  MS="${MS:-1,8,16,32,48,64,96,128}" \
  bash scripts/kernel_dev_box.sh "$OUT/kdev"

PK=$($PY -c "from huggingface_hub import hf_hub_download as d; r, f = '$PACKED'.split(':', 1); print(d(r, f))") \
  || { log "packed download failed"; exit 1; }
VPY=/workspace/vllm_env/bin/python

log "step 2: standalone decode launch sweep"
"$VPY" scripts/tune_decode.py --packed "$PK" --out "$OUT/tune_decode.json" > "$OUT/tune_decode.log" 2>&1 \
  || log "tune_decode failed"
grep -E "^\[" "$OUT/tune_decode.log" | tail -12

log "step 3: standalone vs llama.cpp (Q4_K_M, IQ2_S)"
"$VPY" scripts/bench_ship.py --packed "$PK" --out "$OUT/ship" --max-len 2048 \
  --gguf unsloth/Qwen3.8-27B-GGUF:Qwen3.8-27B-UD-Q4_K_M.gguf \
  --gguf unsloth/Qwen3.8-27B-GGUF:Qwen3.8-27B-UD-IQ2_S.gguf > "$OUT/ship.log" 2>&1 || log "bench_ship failed"
grep -E "tok/s|rc " "$OUT/ship.log" | tail -20
log "all done"
