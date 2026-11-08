#!/usr/bin/env bash
# One-shot box: plugin tests + vLLM NLL / throughput (dev3090_stage.sh), then the vision check.
#   bash scripts/box_check.sh <out_dir>
set -uo pipefail
OUT=$1
mkdir -p "$OUT"
STEPS="tests vllm" bash scripts/dev3090_stage.sh "$OUT/check" 2>&1 | tail -30
bash scripts/vision_stage.sh "$OUT/vision" 2>&1 | tail -12
echo "box check done"
