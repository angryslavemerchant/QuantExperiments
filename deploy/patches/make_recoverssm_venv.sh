#!/usr/bin/env bash
# Build a second vLLM venv with RecoverSSM for Qwen3.5 GDN layers (vLLM PR #61309, draft, head 3a559cc)
# on top of vLLM 0.33.0, leaving the stock venv untouched.
#   bash deploy/patches/make_recoverssm_venv.sh            (run from the repo root, after setup_server.sh)
# Env: SRC (~/sq-venv), VENV (~/sq-venv-rssm).
#
# Why: under speculative decoding vLLM 0.33.0 keeps 1 + K copies of every Gated DeltaNet layer's state per
# request (rollback by snapshot); RecoverSSM keeps the checkpoint plus small per-token correction records and
# folds in only the accepted tokens. Work 3090, DSpark K=7: KV room 51k -> 81k tokens (MAX_LEN 16k,
# MAX_SEQS 4), single-request speed unchanged or better. Serve with RSSM=1 (deploy/serve.sh).
# The PR touches Python files only (no rebuild). Skipped: tests, the vllm/models/qwen3_5/nvidia/* files
# (that layout is newer than 0.33.0; 0.33.0 runs Qwen3.5 from model_executor/models/qwen3_5.py, which
# the PR also patches) and the qwen3_next_mtp.py hunk (not our model). Two hunks are re-applied by hand.
set -euo pipefail
SRC=${SRC:-$HOME/sq-venv}
VENV=${VENV:-$HOME/sq-venv-rssm}
HERE=$(cd "$(dirname "$0")" && pwd)
DIFF=$HERE/vllm-pr61309-recoverssm-3a559cc.diff
[ -x "$SRC/bin/python" ] || { echo "no stock venv at $SRC: run deploy/setup_server.sh first"; exit 1; }
"$SRC/bin/python" -c "import vllm, sys; sys.exit(0 if vllm.__version__.startswith('0.33.') else 1)" \
  || { echo "this patch is for vLLM 0.33.x"; exit 1; }
if [ ! -d "$VENV" ]; then
  echo "[rssm] copying $SRC -> $VENV (~15 GB)"
  cp -a "$SRC" "$VENV"
  grep -rlI "$SRC" "$VENV/bin" | xargs sed -i "s#$SRC#$VENV#g"     # console scripts / activate
fi
SITE=$("$VENV/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
if [ ! -f "$SITE/.recoverssm_3a559cc" ]; then
  echo "[rssm] applying $(basename "$DIFF")"
  (cd "$SITE" && patch -p1 -f --no-backup-if-mismatch -r - < "$DIFF" > /tmp/recoverssm_patch.log 2>&1) || true
  # expected misses only: tests/*, vllm/models/qwen3_5/nvidia/*, other models' files, 3 known hunks
  VENV=$VENV "$VENV/bin/python" "$HERE/fix_recoverssm_vllm033.py"
  touch "$SITE/.recoverssm_3a559cc"
fi
"$VENV/bin/python" - <<'EOF'
import vllm.config.vllm as v
from vllm.model_executor.models import qwen3_5
assert "Qwen3_5ForConditionalGeneration" in v._QWEN_GDN_RECOVERSSM_ARCHS
assert hasattr(qwen3_5, "SupportsReplaySSM")
print("[rssm] ok: vLLM", __import__("vllm").__version__, "+ RecoverSSM for Qwen3.5")
EOF
