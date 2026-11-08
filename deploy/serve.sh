#!/usr/bin/env bash
# Start the OpenAI-compatible server for the shapequant 27B (vLLM + the shapequant plugin).
#   SQ_API_KEY=some-long-secret bash deploy/serve.sh
# Defaults suit a 24 GB card (RTX 3090 / 4090): compact weights (~12 GB), 32k context per request,
# up to 16 requests at once. Env overrides:
#   HOST (127.0.0.1; use your Tailscale IP to reach it from home), PORT (8000), MAX_LEN (32768),
#   MAX_SEQS (16), GPU_UTIL (0.92), MODEL_DIR (~/models/q27-sq), VENV (~/sq-venv), SQ_NAME (served model
#   name), KV_DTYPE (auto | fp8: halves KV memory -> longer contexts), TOOL_PARSER (enables tool calls,
#   e.g. qwen3_coder; see RUNBOOK), SPEC (dspark: RedHat's DSpark drafter, ~2x single-request speed,
#   less KV room -> use MAX_LEN<=16384, MAX_SEQS<=4; mtp: Qwen3.8's own MTP head, needs a checkpoint exported
#   with --embed-head bf16 --mtp, see RUNBOOK), SPEC_K (draft tokens; default dspark 7, mtp 2),
#   ALLOC_CONF (PYTORCH_CUDA_ALLOC_CONF; default expandable_segments:True, set "" on WSL2), IMAGES (max images per request when the model
#   was exported with VISION=1; default 4), EXTRA (more vllm serve args).
set -euo pipefail
VENV=${VENV:-$HOME/sq-venv}
MODEL_DIR=${MODEL_DIR:-$HOME/models/q27-sq}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
MAX_LEN=${MAX_LEN:-32768}
MAX_SEQS=${MAX_SEQS:-16}
GPU_UTIL=${GPU_UTIL:-0.92}
NAME=${SQ_NAME:-qwen3.8-27b-sq}   # not $NAME: WSL sets it to the hostname
KV_DTYPE=${KV_DTYPE:-auto}
[ -n "${SQ_API_KEY:-}" ] || { echo "set SQ_API_KEY (clients send it as 'Authorization: Bearer <key>')"; exit 1; }
[ -f "$MODEL_DIR/config.json" ] || { echo "no model at $MODEL_DIR: run deploy/setup_server.sh first"; exit 1; }
ALLOC_CONF=${ALLOC_CONF-expandable_segments:True}
[ -n "$ALLOC_CONF" ] && export PYTORCH_CUDA_ALLOC_CONF=$ALLOC_CONF
SPEC_ARGS=()
[ -f "$MODEL_DIR/visual.safetensors" ] && SPEC_ARGS+=(--limit-mm-per-prompt "{\"image\": ${IMAGES:-4}}")
case "${SPEC:-}" in
  "") ;;
  dspark) SPEC_ARGS+=(--speculative-config "{\"model\":\"RedHatAI/Qwen3.8-27B-speculator.dspark\",\"num_speculative_tokens\":${SPEC_K:-7},\"method\":\"dspark\"}") ;;
  mtp)
    [ -f "$MODEL_DIR/mtp.safetensors" ] || { echo "SPEC=mtp needs a checkpoint with the MTP head (export --embed-head bf16 --mtp; see RUNBOOK)"; exit 1; }
    SPEC_ARGS+=(--speculative-config "{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":${SPEC_K:-2}}") ;;
  *) echo "SPEC=$SPEC: use 'dspark' or 'mtp'"; exit 1 ;;
esac
# first start compiles the kernels and tunes each matrix shape (a few minutes); later starts reuse
# vLLM's compile cache, the shape tuning reruns (~1-2 min)
exec "$VENV/bin/vllm" serve "$MODEL_DIR" \
  --served-model-name "$NAME" --host "$HOST" --port "$PORT" --api-key "$SQ_API_KEY" \
  --dtype bfloat16 --max-model-len "$MAX_LEN" --max-num-seqs "$MAX_SEQS" \
  --gpu-memory-utilization "$GPU_UTIL" --max-num-batched-tokens 4096 --kv-cache-dtype "$KV_DTYPE" \
  --reasoning-parser qwen3 ${TOOL_PARSER:+--enable-auto-tool-choice --tool-call-parser $TOOL_PARSER} \
  ${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"} ${EXTRA:-}
