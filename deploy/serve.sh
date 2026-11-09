#!/usr/bin/env bash
# Start the OpenAI-compatible server for the shapequant 27B (vLLM + the shapequant plugin).
#   SQ_API_KEY=some-long-secret bash deploy/serve.sh
# Defaults suit a 24 GB card (RTX 3090 / 4090): compact weights (~12 GB), 32k context per request,
# up to 16 requests at once. Env overrides:
#   HOST (127.0.0.1; use your Tailscale IP to reach it from home), PORT (8000), MAX_LEN (32768),
#   MAX_SEQS (16), GPU_UTIL (default: computed at start from the VRAM free right now minus HEADROOM_MB
#   (1536), so memory other programs hold (the Windows desktop, ~1.3 GB on the work 3090) is left alone;
#   vLLM's own --gpu-memory-utilization is a share of the *total* VRAM. Too high a value spills VRAM to
#   system RAM on Windows/WSL and runs many times slower (OOM instead if the driver's CUDA Sysmem
#   Fallback Policy is off)), CHUNK (8192: prefill
#   chunk, --max-num-batched-tokens), MODEL_DIR (~/models/q27-sq), VENV (~/sq-venv), SQ_NAME (served model
#   name), KV_DTYPE (auto | fp8: halves KV memory -> longer contexts), TOOL_PARSER (enables tool calls,
#   e.g. qwen3_coder; see RUNBOOK), SPEC (dspark: DSpark drafter (default: the FP8 build of RedHat's), ~2.4x
#   single-request speed, less KV room; mtp: Qwen3.8's own MTP head, needs a checkpoint exported
#   with --embed-head bf16 --mtp, see RUNBOOK), SPEC_K (draft tokens; default dspark 7, mtp 2), SPEC_MODEL (DSpark drafter repo),
#   RSSM (1: RecoverSSM for the DeltaNet layers under SPEC, much more KV room; needs the patched venv from
#   deploy/patches/make_recoverssm_venv.sh, used automatically), ALLOC_CONF (PYTORCH_CUDA_ALLOC_CONF;
#   default expandable_segments:True, off on WSL2 where it stalls the plugin's shape tuning), IMAGES (max images per request when the model
#   was exported with VISION=1; default 4), EXTRA (more vllm serve args).
set -euo pipefail
VENV=${VENV:-$HOME/sq-venv}
MODEL_DIR=${MODEL_DIR:-$HOME/models/q27-sq}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
MAX_LEN=${MAX_LEN:-32768}
MAX_SEQS=${MAX_SEQS:-16}
if [ -z "${GPU_UTIL:-}" ]; then
  read -r free total < <(nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits | head -1 | tr -d ',')
  GPU_UTIL=$(awk -v f="$free" -v t="$total" -v h="${HEADROOM_MB:-1536}"     'BEGIN { u = (f - h) / t; if (u > 0.95) u = 0.95; if (u < 0.5) u = 0.5; printf "%.3f", u }')
  echo "GPU_UTIL=$GPU_UTIL (${free} MiB free of ${total}, ${HEADROOM_MB:-1536} MiB headroom)"
fi
CHUNK=${CHUNK:-8192}
NAME=${SQ_NAME:-qwen3.8-27b-sq}   # not $NAME: WSL sets it to the hostname
KV_DTYPE=${KV_DTYPE:-auto}
[ -n "${SQ_API_KEY:-}" ] || { echo "set SQ_API_KEY (clients send it as 'Authorization: Bearer <key>')"; exit 1; }
[ -f "$MODEL_DIR/config.json" ] || { echo "no model at $MODEL_DIR: run deploy/setup_server.sh first"; exit 1; }
if grep -qi microsoft /proc/version 2>/dev/null; then ALLOC_CONF=${ALLOC_CONF-}; else ALLOC_CONF=${ALLOC_CONF-expandable_segments:True}; fi
[ -n "$ALLOC_CONF" ] && export PYTORCH_CUDA_ALLOC_CONF=$ALLOC_CONF
SPEC_ARGS=()
[ -f "$MODEL_DIR/visual.safetensors" ] && SPEC_ARGS+=(--limit-mm-per-prompt "{\"image\": ${IMAGES:-4}}")
case "${SPEC:-}" in
  "") ;;
  dspark) SPEC_K=${SPEC_K:-7}
    SPEC_ARGS+=(--speculative-config "{\"model\":\"${SPEC_MODEL:-inference-optimization/Qwen3.8-27B-DSpark-FP8-DYNAMIC}\",\"num_speculative_tokens\":$SPEC_K,\"method\":\"dspark\"}") ;;
  mtp)
    [ -f "$MODEL_DIR/mtp.safetensors" ] || { echo "SPEC=mtp needs a checkpoint with the MTP head (export --embed-head bf16 --mtp; see RUNBOOK)"; exit 1; }
    SPEC_K=${SPEC_K:-2}
    SPEC_ARGS+=(--speculative-config "{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":$SPEC_K}") ;;
  *) echo "SPEC=$SPEC: use 'dspark' or 'mtp'"; exit 1 ;;
esac
if [ -n "${SPEC:-}" ]; then
  # CUDA graphs only up to what can run at once (MAX_SEQS x (K + 1) tokens), not vLLM's default 64+
  cap=$(( (MAX_SEQS * (SPEC_K + 1) + 7) / 8 * 8 ))
  SPEC_ARGS+=(--compilation-config "{\"max_cudagraph_capture_size\":$cap}")
fi
if [ "${RSSM:-}" = 1 ]; then
  [ -n "${SPEC:-}" ] || { echo "RSSM=1 needs SPEC (RecoverSSM replaces the speculative state copies)"; exit 1; }
  [ "$VENV" = "$HOME/sq-venv" ] && VENV=$HOME/sq-venv-rssm
  [ -x "$VENV/bin/vllm" ] || { echo "no RecoverSSM venv at $VENV: run deploy/patches/make_recoverssm_venv.sh"; exit 1; }
  SPEC_ARGS+=(--use-replayssm --mamba-backend triton)
fi
# first start compiles the kernels and tunes each matrix shape (a few minutes); later starts reuse
# vLLM's compile cache, the shape tuning reruns (~1-2 min)
exec "$VENV/bin/vllm" serve "$MODEL_DIR" \
  --served-model-name "$NAME" --host "$HOST" --port "$PORT" --api-key "$SQ_API_KEY" \
  --dtype bfloat16 --max-model-len "$MAX_LEN" --max-num-seqs "$MAX_SEQS" \
  --gpu-memory-utilization "$GPU_UTIL" --max-num-batched-tokens "$CHUNK" --kv-cache-dtype "$KV_DTYPE" \
  --reasoning-parser qwen3 ${TOOL_PARSER:+--enable-auto-tool-choice --tool-call-parser $TOOL_PARSER} \
  ${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"} ${EXTRA:-}
