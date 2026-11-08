#!/usr/bin/env bash
# Overnight dev stage on a kept-alive box (3090 first): kernel tests, small-batch launch study,
# per-matrix bench, standalone batched decode sweep, vLLM plugin (compact auto, packed embed/head):
# wikitext NLL + throughput. Each step's failure is logged and the next step still runs.
#   bash scripts/dev3090_stage.sh <out_dir>
# Env: STEPS (default "tests small mid standalone vllm"), VLLM_CONFIGS ("large:xs:bmax ..."),
#      SMOKE_ARGS, NAMES.
set -uo pipefail
OUT=$1
PY=${PYTHON:-$(command -v python3 || command -v python)}
VENV=/workspace/vllm_env; VPY=$VENV/bin/python
PACKED=${PACKED:-luckymushy/Qwen3.8-27B-shapequant-2.5bpw:s16_25_ft.safetensors}
NAMES=${NAMES:-model.layers.0.mlp.gate_proj,model.layers.0.mlp.down_proj,model.layers.0.linear_attn.in_proj_qkv,model.layers.3.self_attn.o_proj}
STEPS=${STEPS:-"tests small mid standalone vllm"}
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }
nvidia-smi --query-gpu=name,memory.total --format=csv | tee "$OUT/gpu.txt"
# env (reused by later stages) + this checkout; tests "none" so kernel_dev_box only builds
TESTS=none MS=1 NAMES=model.layers.0.mlp.down_proj bash scripts/kernel_dev_box.sh "$OUT/setup" > "$OUT/setup.log" 2>&1
tail -3 "$OUT/setup.log"
PK=$($PY -c "from huggingface_hub import hf_hub_download as d; r, f = '$PACKED'.split(':', 1); print(d(r, f))") || { log "packed download failed"; exit 1; }
if has tests; then
  log "tests"
  "$VPY" -m pytest tests/test_batched_runs.py tests/test_vllm_ext_core.py tests/test_cuda_dequant.py tests/test_runs_gemm.py \
     -q -p no:cacheprovider > "$OUT/tests.log" 2>&1; log "tests exited $?"; tail -4 "$OUT/tests.log"
fi
if has small; then
  log "small-batch launch study"
  "$VPY" scripts/bench_small_b.py --packed "$PK" --names "$NAMES" --out "$OUT/small_b.json" > "$OUT/small_b.log" 2>&1 || log "small_b failed"
  grep "B=" "$OUT/small_b.log" | tail -30
fi
if has mid; then
  log "per-matrix mid-batch bench"
  "$VPY" scripts/bench_mid_batch.py --packed "$PK" --names "$NAMES" --Ms 1,8,32,64,128,256 --out "$OUT/mid.json" > "$OUT/mid.log" 2>&1 || log "mid failed"
  grep " M=" "$OUT/mid.log" | tail -30
fi
if has standalone; then
  log "standalone batched decode sweep"
  "$VPY" scripts/bench_batched.py --packed "$PK" --ctx 512 --tg 64 --tune --stat median --out "$OUT/standalone_batched.json" \
     > "$OUT/standalone_batched.log" 2>&1 || log "standalone failed"
  grep -iE "B=|tok/s" "$OUT/standalone_batched.log" | tail -12
fi
if has vllm; then
  CK=/workspace/vllm_ckpt/q27.sq
  [ -e "$CK/config.json" ] || $PY scripts/export_vllm_checkpoint.py --packed-hf "$PACKED" --out "$CK" > "$OUT/export.log" 2>&1
  for cfg in ${VLLM_CONFIGS:-fused:1:64}; do
    IFS=: read -r large xs bmax <<< "$cfg"
    tag="${large}_xs${xs}_b${bmax}"
    log "vLLM plugin ($tag, compact ${SHAPEQUANT_COMPACT:-auto})"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SHAPEQUANT_LARGE=$large SHAPEQUANT_XS=$xs SHAPEQUANT_BMAX=$bmax \
      "$VPY" scripts/vllm_smoke.py run --model "$CK" --name "q27-$tag" --out "$OUT/vllm_$tag.json" \
      ${SMOKE_ARGS:---windows 8 --max-model-len 8192 --gpu-mem 0.90 --max-num-seqs 64 --max-num-batched-tokens 4096 --conc 1,2,4,8,16,32,64} \
      > "$OUT/vllm_$tag.log" 2>&1 || log "vLLM failed"
    grep -E "^\[|took .* GiB|KV cache size|Error" "$OUT/vllm_$tag.log" | tail -16
  done
fi
log "stage done"
