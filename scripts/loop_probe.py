"""Loop rate on realistic chat prompts: each prompt N times at the server's sampling defaults (or overrides),
thinking on or off, through any OpenAI-compatible chat server. A response counts as a loop when
scripts/loop_detect.py flags it or it hits max_tokens.

  python scripts/loop_probe.py --endpoint http://127.0.0.1:8000 --api-key "$SQ_API_KEY" --n 20 \
      --configs '{"default": {}, "pp15": {"presence_penalty": 1.5}}' --out artifacts/overnight/loops/probe.json
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from loop_detect import is_loop, loop_score

PROMPTS = {
    "hex16to32": "write a script to turn 16 byte hex rows into 32 byte ones",
    "hexdump_parse": "Write a Python function that parses the output of `xxd` back into bytes, handling the "
                     "final short line. Include a few tests.",
    "csv_merge": "Write a Python script that merges two CSV files on a shared 'id' column, keeping all rows "
                 "from both and reporting ids that appear in only one file.",
    "bit_reverse": "Write a C function that reverses the bit order of every byte in a buffer, and a table-"
                   "driven version. Explain the speed difference.",
    "regex_log": "Write a Python script that reads an nginx access log and prints the top 10 IPs, top 10 "
                 "paths and the count of each status code.",
    "list_pattern": "List the first 40 rows of a table converting decimal 0-39 to hex, binary and octal.",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--api-key", default="none")
    ap.add_argument("--model")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--think", choices=["on", "off", "both"], default="both")
    ap.add_argument("--configs", default='{"default": {}}', help="JSON {name: extra request fields}")
    ap.add_argument("--prompts", default=",".join(PROMPTS))
    ap.add_argument("--max-tokens", type=int, default=12288)
    ap.add_argument("--conc", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base = a.endpoint.rstrip("/")
    hdr = {"Content-Type": "application/json", "Authorization": f"Bearer {a.api_key}"}
    model = a.model or json.load(urllib.request.urlopen(urllib.request.Request(
        f"{base}/v1/models", headers=hdr)))["data"][0]["id"]
    configs = json.loads(a.configs)
    thinks = {"on": [True], "off": [False], "both": [True, False]}[a.think]
    jobs = [(c, th, p, i) for c in configs for th in thinks for p in a.prompts.split(",") for i in range(a.n)]

    def run(job):
        c, th, p, i = job
        body = {"model": model, "messages": [{"role": "user", "content": PROMPTS[p]}], "max_tokens": a.max_tokens,
                "seed": 1000 + i, "chat_template_kwargs": {"enable_thinking": th}, **configs[c]}
        t0 = time.time()
        try:
            req = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(), headers=hdr)
            r = json.load(urllib.request.urlopen(req, timeout=3600))
            m = r["choices"][0]["message"]
            text = (m.get("reasoning_content") or m.get("reasoning") or "") + "\n</think>\n" + (m.get("content") or "")
            fin = r["choices"][0].get("finish_reason")
            ntok = r.get("usage", {}).get("completion_tokens")
        except Exception as e:  # noqa: BLE001
            text, fin, ntok = "", f"error: {str(e)[:200]}", 0
        return {"config": c, "think": th, "prompt": p, "i": i, "finish": fin, "tokens": ntok,
                "seconds": round(time.time() - t0, 1), "loop": fin == "length" or is_loop(text, code=True),
                "score": loop_score(text), "text": text}

    t0 = time.time()
    res = []
    with ThreadPoolExecutor(a.conc) as ex:
        for k, r in enumerate(ex.map(run, jobs)):
            res.append(r)
            if (k + 1) % 20 == 0:
                print(f"{k + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
                json.dump({"model": model, "results": res}, open(a.out, "w"), indent=1)
    json.dump({"model": model, "results": res}, open(a.out, "w"), indent=1)
    print("config  think  prompt          loops/n  maxtok  errors  mean_tokens")
    for c in configs:
        for th in thinks:
            for p in a.prompts.split(","):
                g = [r for r in res if r["config"] == c and r["think"] == th and r["prompt"] == p]
                print(f"{c:8s} {str(th):5s}  {p:15s} {sum(r['loop'] for r in g):3d}/{len(g):<3d}  "
                      f"{sum(r['finish'] == 'length' for r in g):6d}  {sum(str(r['finish']).startswith('error') for r in g):6d}"
                      f"  {sum(r['tokens'] or 0 for r in g) / max(1, len(g)):8.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
