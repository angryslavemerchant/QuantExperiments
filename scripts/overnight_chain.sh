#!/usr/bin/env bash
# Overnight chain on the kept-alive 24 GB box, gated on the vLLM correctness check:
#   1. verify: vLLM plugin wikitext NLL + throughput (dev3090_stage.sh vllm); stop unless NLL < 3
#   2. serve:  deploy/serve.sh as the 3090 server would run it (API key enforced, reasoning parser,
#              tool-call parser), checked with real requests
#   3. spec:   speculative decoding (spec_stage.sh: base / DSpark / MTP)
#   4. wish:   quality prompts, long context, KV-cache quantization (wishlist_stage.sh)
#   bash scripts/overnight_chain.sh <out_dir>
# Env: CHAIN ("verify serve spec wish"), plus each script's own env.
set -uo pipefail
OUT=$1
PY=${PYTHON:-$(command -v python3 || command -v python)}
VENV=/workspace/vllm_env; VPY=$VENV/bin/python
CHAIN=${CHAIN:-"verify serve spec wish"}
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*"; }
has() { [[ " $CHAIN " == *" $1 "* ]]; }
CK=/workspace/vllm_ckpt/q27.sq

if has verify; then
  log "== verify"
  STEPS=vllm SMOKE_ARGS="--windows 8 --max-model-len 8192 --gpu-mem 0.90 --max-num-seqs 64 --max-num-batched-tokens 4096 --conc 1,2,4,8,16,32,64" \
    bash scripts/dev3090_stage.sh "$OUT/verify" 2>&1 | tail -20
  nll=$("$VPY" -c "import json, glob; d = json.load(open(glob.glob('$OUT/verify/vllm_*.json')[0])); print(d['wikitext']['nll'])" 2>/dev/null)
  log "verify NLL: ${nll:-none}"
  "$VPY" -c "import sys; sys.exit(0 if float('${nll:-99}') < 3 else 1)" || { log "NLL check failed: stopping the chain"; exit 0; }
fi

if has serve; then
  log "== serve (deploy/serve.sh)"
  KEY=$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')          # throwaway, never written out
  pkill -f "vllm serve" 2>/dev/null; sleep 10
  SQ_API_KEY=$KEY VENV=$VENV MODEL_DIR=$CK TOOL_PARSER=qwen3_coder nohup bash deploy/serve.sh > "$OUT/serve.log" 2>&1 &
  up=0; for _ in $(seq 150); do curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1 && { up=1; break; }; sleep 10; done
  if [ $up = 1 ]; then
    grep -E "KV cache size|Maximum concurrency|took .* GiB" "$OUT/serve.log" | tail -3
    code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/v1/models)
    log "no key -> HTTP $code (want 401)"
    KEY=$KEY "$VPY" - > "$OUT/serve_check.json" 2> "$OUT/serve_check.err" <<'PYEOF'
import json, os, time, urllib.request
key = os.environ["KEY"]
def call(body):
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=900).read())
    return r, time.time() - t
out = {}
r, s = call({"model": "qwen3.8-27b-sq", "messages": [{"role": "user", "content": "What is 17 * 23? Answer briefly."}],
             "max_tokens": 2048, "temperature": 0.6, "top_p": 0.95, "top_k": 20})
m = r["choices"][0]["message"]
out["thinking"] = {"content": m.get("content"), "reasoning_chars": len(m.get("reasoning_content") or m.get("reasoning") or ""),
                   "tokens": r["usage"]["completion_tokens"], "seconds": round(s, 1)}
r, s = call({"model": "qwen3.8-27b-sq", "messages": [{"role": "user", "content": "Name three primary colors."}],
             "max_tokens": 200, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "chat_template_kwargs": {"enable_thinking": False}})
m = r["choices"][0]["message"]
out["no_thinking"] = {"content": m.get("content"), "reasoning_chars": len(m.get("reasoning_content") or m.get("reasoning") or "")}
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
r, s = call({"model": "qwen3.8-27b-sq", "messages": [{"role": "user", "content": "What's the weather in Toronto right now?"}],
             "tools": tools, "max_tokens": 2048, "temperature": 0.6, "top_p": 0.95, "top_k": 20})
m = r["choices"][0]["message"]
out["tool_call"] = {"tool_calls": m.get("tool_calls"), "content": (m.get("content") or "")[:300],
                    "finish": r["choices"][0]["finish_reason"]}
print(json.dumps(out, indent=1))
PYEOF
    head -c 2500 "$OUT/serve_check.json"; tail -3 "$OUT/serve_check.err"
  else
    log "serve.sh did not come up"; grep -E "Error|error" "$OUT/serve.log" | tail -5
  fi
  pkill -f "vllm serve" 2>/dev/null; sleep 15
  tail -c 30000 "$OUT/serve.log" > "$OUT/serve.log.tail"; rm -f "$OUT/serve.log"
fi

if has spec; then
  log "== spec"
  bash scripts/spec_stage.sh "$OUT/spec" 2>&1 | tail -30
fi
if has wish; then
  log "== wish"
  bash scripts/wishlist_stage.sh "$OUT/wish" 2>&1 | tail -60
fi
log "chain done"
