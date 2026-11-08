#!/usr/bin/env bash
# 3090 follow-up (s38b): vLLM plugin with the in-place ("lane") word layout (the "tile" copy
# doubled the weights: 19.8 GB at load, OOM on 24 GB), smaller prefill chunks; per-matrix lane vs
# tile; standalone tg with the Ampere launch default.
#   bash scripts/box_3090b.sh <out_dir>
set -uo pipefail
OUT=$1
PY=${PYTHON:-$(command -v python3 || command -v python)}
PACKED=${PACKED:-luckymushy/Qwen3.8-27B-shapequant-2.5bpw:s16_25_ft.safetensors}
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*"; }
nvidia-smi --query-gpu=name,memory.total --format=csv | tee "$OUT/gpu.txt"
# env + per-matrix bench (lane vs tile); vLLM below
TESTS="tests/test_vllm_ext_core.py" MS="1,8,16,32,48,64" bash scripts/kernel_dev_box.sh "$OUT/kdev" --lane
VPY=/workspace/vllm_env/bin/python
PK=$($PY -c "from huggingface_hub import hf_hub_download as d; r, f = '$PACKED'.split(':', 1); print(d(r, f))")
CK=/workspace/vllm_ckpt/q27.sq
[ -e "$CK/config.json" ] || $PY scripts/export_vllm_checkpoint.py --packed-hf "$PACKED" --out "$CK" > "$OUT/export.log" 2>&1
for cfg in ${CONFIGS:-lane:1:64:2048 lane:0:32:2048}; do
  IFS=: read -r layout xs bmax mbt <<< "$cfg"
  tag="${layout}_xs${xs}_b${bmax}_t${mbt}"
  log "vLLM plugin ($tag)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SHAPEQUANT_LAYOUT=$layout SHAPEQUANT_XS=$xs SHAPEQUANT_BMAX=$bmax \
  "$VPY" scripts/vllm_smoke.py run --model "$CK" --name "q27-$tag" --out "$OUT/vllm_$tag.json" \
      --windows 8 --max-model-len 4096 --gpu-mem 0.90 --max-num-seqs 128 --max-num-batched-tokens "$mbt" \
      --conc 1,8,16,32,64,128 > "$OUT/vllm_$tag.log" 2>&1 || log "smoke failed"
  grep -E "^\[|Error|took .* GiB" "$OUT/vllm_$tag.log" | tail -16
done
log "standalone tg (Ampere default launch)"
"$VPY" scripts/bench_ship.py --packed "$PK" --out "$OUT/ship" --max-len 2048 > "$OUT/ship.log" 2>&1 || log "bench_ship failed"
grep -E "tok/s" "$OUT/ship.log" | tail -5
log "all done"
