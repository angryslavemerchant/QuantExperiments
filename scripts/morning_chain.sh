#!/usr/bin/env bash
# Last overnight stage on the 24 GB box: vision, MTP speculative decoding, long context / fp8 KV (ours only).
#   bash scripts/morning_chain.sh <out_dir>
set -uo pipefail
OUT=$1; mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*"; }
log "== vision";  bash scripts/vision_stage.sh "$OUT/vision" 2>&1 | tail -12
log "== mtp";     CONFIGS=mtp bash scripts/spec_stage.sh "$OUT/spec" 2>&1 | tail -10
log "== longctx"; SKIP_LLAMA=1 STEPS="longctx kvquant" bash scripts/wishlist_stage.sh "$OUT/wish" 2>&1 | tail -30
log "chain done"
