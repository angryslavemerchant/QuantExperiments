"""Next-token top-k at fixed token-id prefixes, through vLLM (/v1/completions, logprobs) or llama-server
(/completion, n_probs), so two runtimes are compared on identical inputs.

  python scripts/next_token_probe.py --prefixes p.json --endpoint http://127.0.0.1:8000 --api-key K --kind vllm --out a.json
  python scripts/next_token_probe.py --prefixes p.json --endpoint http://127.0.0.1:8100 --kind llama --out b.json
p.json: [{"name": ..., "ids": [...], "target": id}, ...]
"""
from __future__ import annotations

import argparse
import json
import math
import urllib.request


def post(url, body, key):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefixes", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--api-key", default="none")
    ap.add_argument("--kind", choices=["vllm", "llama"], required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base = a.endpoint.rstrip("/")
    out = []
    for p in json.load(open(a.prefixes)):
        if a.kind == "vllm":
            model = a.model or json.load(urllib.request.urlopen(urllib.request.Request(
                f"{base}/v1/models", headers={"Authorization": f"Bearer {a.api_key}"})))["data"][0]["id"]
            r = post(f"{base}/v1/completions", {"model": model, "prompt": p["ids"], "max_tokens": 1, "temperature": 0,
                                                "logprobs": a.k, "return_tokens_as_token_ids": True}, a.api_key)
            tl = r["choices"][0]["logprobs"]["top_logprobs"][0]
            top = sorted(((int(t.split(":")[1]), v) for t, v in tl.items()), key=lambda x: -x[1])
        else:
            r = post(f"{base}/completion", {"prompt": p["ids"], "n_predict": 1, "temperature": 0, "n_probs": a.k,
                                            "post_sampling_probs": False}, a.api_key)
            pr = r["completion_probabilities"][0]["top_logprobs"]
            top = sorted(((x["id"], x["logprob"]) for x in pr), key=lambda x: -x[1])
        out.append({"name": p["name"], "target": p.get("target"), "top": top})
        print(p["name"], [(t, round(v, 3)) for t, v in top[:3]], flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
