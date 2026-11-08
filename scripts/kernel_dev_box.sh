#!/usr/bin/env bash
# Kernel development stage on a retained GPU box: (re)uses /workspace/vllm_env (vLLM + CuPy +
# pytest + this checkout installed editable), runs the kernel tests, then the per-matrix bench on
# the 27B packed file (HF cache persists across stages).
#   bash scripts/kernel_dev_box.sh <out_dir> [bench args...]
# Env: TESTS (pytest targets, default the kernel tests; "none" skips), PACKED (REPO:FILE),
#      NAMES (matrices), MS (batch sizes), VLLM_RUN=1 (also the vLLM plugin smoke on the 27B,
#      CONFIGS="large:xs:bmax ...", SMOKE_ARGS).
set -uo pipefail
OUT=$1; shift
PY=${PYTHON:-$(command -v python3 || command -v python)}
VENV=/workspace/vllm_env
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*"; }

if [ ! -e "$VENV/.ready" ]; then
  log "building vLLM env"
  (
    $PY -m pip install -q uv &&
    $PY -m uv venv "$VENV" --python "$PY" -q &&
    $PY -m uv pip install --python "$VENV/bin/python" -q --torch-backend=auto vllm datasets pytest
  ) > "$OUT/env.log" 2>&1 || { log "env build failed"; tail -20 "$OUT/env.log"; exit 1; }
  ok=0
  for pkgs in "cupy-cuda13x" "cupy-cuda12x"; do
    $PY -m uv pip uninstall --python "$VENV/bin/python" -q cupy-cuda13x cupy-cuda12x >> "$OUT/env.log" 2>&1
    rm -rf "$HOME/.cupy/kernel_cache"        # a failed build's cached cubins (s40: CUDA_ERROR_INVALID_IMAGE)
    $PY -m uv pip install --python "$VENV/bin/python" -q $pkgs >> "$OUT/env.log" 2>&1
    if "$VENV/bin/python" -c "import cupy as cp; k = cp.RawKernel(r'extern \"C\" __global__ void f(float* x){x[0]=1.f;}', 'f'); x = cp.zeros(1, dtype=cp.float32); k((1,), (1,), (x,)); assert float(x[0]) == 1" >> "$OUT/env.log" 2>&1; then
      ok=1; break
    fi
  done
  [ $ok = 1 ] || { log "cupy setup failed"; tail -20 "$OUT/env.log"; exit 1; }
  touch "$VENV/.ready"
fi
# this stage's checkout, editable (entry point for the vLLM plugin)
$PY -m uv pip install --python "$VENV/bin/python" -q --no-deps -e . > "$OUT/install.log" 2>&1 || { log "install failed"; tail "$OUT/install.log"; exit 1; }
"$VENV/bin/python" -c "import torch, vllm, cupy; print('torch', torch.__version__, 'vllm', vllm.__version__, 'cupy', cupy.__version__, torch.cuda.get_device_name(0))"

TESTS=${TESTS:-tests/test_batched_runs.py tests/test_runs_gemm.py tests/test_cuda_dequant.py tests/test_vllm_ext_core.py}
if [ "$TESTS" != none ]; then
  log "tests: $TESTS"
  "$VENV/bin/python" -m pytest $TESTS -q -x -p no:cacheprovider > "$OUT/tests.log" 2>&1
  log "tests exited $?"; tail -3 "$OUT/tests.log"
fi

PACKED=${PACKED:-luckymushy/Qwen3.8-27B-shapequant-2.5bpw:s16_25_ft.safetensors}
PK=$($PY -c "from huggingface_hub import hf_hub_download as d; r, f = '$PACKED'.split(':', 1); print(d(r, f))") \
  || { log "packed download failed"; exit 1; }
NAMES=${NAMES:-model.layers.0.mlp.gate_proj,model.layers.0.mlp.down_proj,model.layers.0.linear_attn.in_proj_qkv,model.layers.3.self_attn.o_proj}
MS=${MS:-1,8,16,32,48,64,96,128,192,256}
log "bench"
"$VENV/bin/python" scripts/bench_mid_batch.py --packed "$PK" --names "$NAMES" --Ms "$MS" --out "$OUT/bench.json" "$@" \
  > "$OUT/bench.log" 2>&1 || log "bench failed"
grep -E " M=|Error" "$OUT/bench.log" | tail -60

if [ -n "${VLLM_RUN:-}" ]; then
  CK=/workspace/vllm_ckpt/q27.sq
  [ -e "$CK/config.json" ] || $PY scripts/export_vllm_checkpoint.py --packed-hf "$PACKED" --out "$CK" > "$OUT/export.log" 2>&1
  # CONFIGS: large-M path : xs (0/1) : bmax of the batched kernel
  for cfg in ${CONFIGS:-fused:0:32 fused:1:128}; do
    IFS=: read -r large xs bmax <<< "$cfg"
    tag="${large}_xs${xs}_b${bmax}"
    log "vLLM plugin smoke ($tag)"
    SHAPEQUANT_LARGE=$large SHAPEQUANT_XS=$xs SHAPEQUANT_BMAX=$bmax "$VENV/bin/python" scripts/vllm_smoke.py run \
        --model "$CK" --name "q27-$tag" --out "$OUT/vllm_$tag.json" ${SMOKE_ARGS:-} > "$OUT/vllm_$tag.log" 2>&1 || log "smoke failed"
    grep -E "^\[|Error" "$OUT/vllm_$tag.log" | tail -14
  done
fi
log "stage done"
