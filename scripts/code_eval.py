"""HumanEval+ / MBPP+ generation through any OpenAI-compatible chat server (vLLM, llama-server), so different
runtimes and quantizations are compared on identical prompts. Writes evalplus-format samples (score them with
`evalplus.evaluate <dataset> --samples <file>`) plus the raw responses (for loop / failure analysis).

Prompt: evalplus's instruct prompt ("Please provide a self-contained Python script ... in a markdown code block"),
chat template, thinking on or off (chat_template_kwargs.enable_thinking, honoured by vLLM and llama-server).
Code = the first ```python block in the answer (after any </think>), else the whole answer.

  python scripts/code_eval.py --endpoint http://127.0.0.1:8000 --api-key "$SQ_API_KEY" --dataset humaneval \
      --name sq --out artifacts/overnight/code/sq_humaneval      # -> .jsonl (samples) + .raw.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPT = ("Please provide a self-contained Python script that solves the following problem in a markdown code "
          "block:\n```\n{p}\n```")
BLOCK = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", re.S)


def extract(text: str) -> str:
    t = text.split("</think>")[-1]
    m = BLOCK.findall(t)
    if m:
        # prefer the block that defines functions
        for b in m:
            if "def " in b:
                return b
        return m[0]
    open_ = re.search(r"```(?:python|py)?[ \t]*\n(.*)", t, re.S)  # unterminated block (hit max tokens)
    return open_.group(1) if open_ else t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--api-key", default="none")
    ap.add_argument("--model")
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=0, help="default 2048, or 16384 with --think")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--extra", default="{}", help="extra JSON merged into the request body")
    ap.add_argument("--conc", type=int, default=2)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", required=True, help="path prefix: writes <out>.jsonl and <out>.raw.json")
    a = ap.parse_args()
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    probs = get_human_eval_plus() if a.dataset == "humaneval" else get_mbpp_plus()
    items = list(probs.items())[: a.limit]
    base = a.endpoint.rstrip("/")
    hdr = {"Content-Type": "application/json", "Authorization": f"Bearer {a.api_key}"}
    model = a.model or json.load(urllib.request.urlopen(urllib.request.Request(
        f"{base}/v1/models", headers=hdr)))["data"][0]["id"]
    max_tokens = a.max_tokens or (16384 if a.think else 2048)

    def run(item):
        tid, p = item
        body = {"model": model, "messages": [{"role": "user", "content": PROMPT.format(p=p["prompt"].strip())}],
                "max_tokens": max_tokens, "temperature": a.temperature, "seed": 0,
                "chat_template_kwargs": {"enable_thinking": a.think}}
        body.update(json.loads(a.extra))
        for attempt in range(3):
            try:
                t0 = time.time()
                req = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(), headers=hdr)
                r = json.load(urllib.request.urlopen(req, timeout=3600))
                msg = r["choices"][0]["message"]
                reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                content = msg.get("content") or ""
                return {"task_id": tid, "content": content, "reasoning": reasoning,
                        "finish": r["choices"][0].get("finish_reason"),
                        "tokens": r.get("usage", {}).get("completion_tokens"), "seconds": round(time.time() - t0, 1)}
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    return {"task_id": tid, "content": "", "reasoning": "", "finish": f"error: {e}", "tokens": 0,
                            "seconds": 0}
                time.sleep(10)

    t0 = time.time()
    out = []
    with ThreadPoolExecutor(a.conc) as ex:
        for i, r in enumerate(ex.map(run, items)):
            out.append(r)
            if (i + 1) % 25 == 0:
                print(f"[{a.name}] {i + 1}/{len(items)} {time.time() - t0:.0f}s", flush=True)
    with open(a.out + ".jsonl", "w") as f:
        for r in out:
            f.write(json.dumps({"task_id": r["task_id"], "solution": extract(r["content"])}) + "\n")
    meta = {"name": a.name, "model": model, "dataset": a.dataset, "think": a.think, "max_tokens": max_tokens,
            "temperature": a.temperature, "extra": json.loads(a.extra), "seconds": round(time.time() - t0, 1),
            "responses": out}
    with open(a.out + ".raw.json", "w") as f:
        json.dump(meta, f, indent=1)
    n_len = sum(r["finish"] == "length" for r in out)
    print(f"[{a.name}] done {len(out)} in {meta['seconds']:.0f}s, hit max_tokens: {n_len}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
