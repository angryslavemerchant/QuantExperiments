"""Thinking-mode GPQA-diamond / MATH-500 against OpenAI-compatible chat servers (vLLM serve with
the shapequant plugin, llama.cpp llama-server on GGUFs, ...): one client, the same requests for
every model, each model on its native engine.

Same prompts (think_eval --style lighteval: lighteval 0.13.0's wording), sampling (temperature 0.6,
top_p 0.95, top_k 20, thinking on via chat_template_kwargs) and scoring (think_eval.score) as
think_eval.py. Requests go out round-major (sample 0 of every question, then sample 1, ...), are
spread over all --endpoints by a shared queue (--conc in flight per endpoint), and every finished
sample is appended to <out>/samples.jsonl at once, so a cutoff leaves complete earlier rounds.
<out>/summary.json: per task, per-question mean over its finished samples, stderr over questions,
per-round accuracy and coverage. A rerun skips samples already in the file.

  python scripts/serve_eval.py --endpoints http://127.0.0.1:8000,http://127.0.0.1:8001 --conc 64 \
      --model-name ours --name ours --out artifacts/x/ours --tasks gpqa --n 2 --deadline-min 80
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("think_eval", ROOT / "scripts" / "think_eval.py")
te = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(te)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def seed_for(qid: int, k: int) -> int:
    return 1000 + 16 * qid + k


def response_text(choice: dict) -> str:
    """The generation as raw text (thinking + '</think>' + answer). Servers that split reasoning
    out (reasoning_content / reasoning) get it put back; a reasoning-only answer that hit the
    length limit stays unclosed (cut off while thinking)."""
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    if reasoning and "</think>" not in content:
        if not content and choice.get("finish_reason") == "length":
            return reasoning
        return reasoning + "</think>" + content
    return content


def post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def summarize(rows: list[dict], n_items: int, metric: str) -> dict:
    by_q: dict = {}
    for r in rows:
        by_q.setdefault(r["id"], []).append(r)
    qs = sorted(by_q)
    out: dict = {"questions": n_items, "questions_with_samples": len(qs), "samples": len(rows), "headline": metric}
    if not qs:
        return out
    per_q = {q: {m: sum(r["scores"][m] for r in by_q[q]) / len(by_q[q]) for m in by_q[q][0]["scores"]} for q in qs}
    for m in per_q[qs[0]]:
        vals = [per_q[q][m] for q in qs]
        p = sum(vals) / len(vals)
        out[m] = p
        out[m + "_stderr"] = math.sqrt(sum((v - p) ** 2 for v in vals) / (len(vals) - 1) / len(vals)) if len(vals) > 1 else None
    rounds: dict = {}
    for r in rows:
        rounds.setdefault(r["k"], []).append(r["scores"][metric])
    out["rounds"] = {str(k): {"n": len(v), metric: sum(v) / len(v)} for k, v in sorted(rounds.items())}
    out["coverage"] = {str(c): sum(1 for q in qs if len(by_q[q]) == c) for c in sorted({len(v) for v in by_q.values()})}
    out["mean_gen_tokens"] = sum(r["gen_tokens"] or 0 for r in rows) / len(rows)
    out["length_cutoff_frac"] = sum(r["finish"] == "length" for r in rows) / len(rows)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints", required=True, help="comma-separated base URLs")
    ap.add_argument("--conc", default="32", help="requests in flight per endpoint (one value or one per endpoint)")
    ap.add_argument("--model-name", default="default")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="gpqa")
    ap.add_argument("--n", type=int, default=1, help="samples per question (round-major)")
    ap.add_argument("--style", choices=["boxed", "lighteval"], default="lighteval")
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--limit", type=int, help="first N questions per task (smoke)")
    ap.add_argument("--deadline-min", type=float, help="stop issuing new requests after this many minutes")
    ap.add_argument("--hard-min", type=float, help="write the summary and exit after this many minutes")
    ap.add_argument("--timeout-s", type=float, default=3 * 3600)
    ap.add_argument("--retries", type=int, default=2)
    a = ap.parse_args(argv)
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    eps = [e.strip() for e in a.endpoints.split(",") if e.strip()]
    conc = [int(c) for c in a.conc.split(",")]
    conc = conc * len(eps) if len(conc) == 1 else conc
    assert len(conc) == len(eps), "--conc: one value or one per endpoint"

    tasks = a.tasks.split(",")
    items = {t: te.load_task(t, a.style) for t in tasks}
    if a.limit:
        items = {t: v[:a.limit] for t, v in items.items()}
    sp = out / "samples.jsonl"
    rows: list[dict] = []
    if sp.exists():
        rows = [json.loads(line) for line in sp.read_text(encoding="utf-8").splitlines() if line.strip()]
    have = {(r["task"], r["id"], r["k"]) for r in rows}
    jobs = [(t, it, k) for k in range(a.n) for t in tasks for it in items[t] if (t, it["id"], k) not in have]
    log(f"{a.name}: {sum(len(v) for v in items.values())} questions x {a.n}, {len(have)} samples done, "
        f"{len(jobs)} to go over {len(eps)} endpoints (conc {conc})")
    lock = threading.Lock()
    state = {"next": 0, "fresh": 0, "errors": 0}
    fh = open(sp, "a", encoding="utf-8")

    def write_summary():
        summ = {"name": a.name, "model_name": a.model_name, "endpoints": eps, "n": a.n, "style": a.style,
                "max_tokens": a.max_tokens, "elapsed_s": round(time.time() - t0), "errors": state["errors"], "tasks": {}}
        for t in tasks:
            summ["tasks"][t] = summarize([r for r in rows if r["task"] == t], len(items[t]),
                                         "exact_match" if t == "gpqa" else "math_verify")
        (out / "summary.json").write_text(json.dumps(summ, indent=2))
        return summ

    def worker(ep: str):
        while True:
            el = (time.time() - t0) / 60
            if (a.deadline_min and el > a.deadline_min) or (a.hard_min and el > a.hard_min):
                return
            with lock:
                if state["next"] >= len(jobs):
                    return
                t, it, k = jobs[state["next"]]
                state["next"] += 1
            body = {"model": a.model_name, "messages": [{"role": "user", "content": it["prompt"]}],
                    "temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": a.max_tokens,
                    "seed": seed_for(it["id"], k), "chat_template_kwargs": {"enable_thinking": True}}
            res, err = None, None
            for attempt in range(a.retries + 1):
                try:
                    t1 = time.time()
                    res = post(ep, body, a.timeout_s)
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError) as e:
                    err = repr(e)[:300]
                    if attempt < a.retries:
                        time.sleep(20)
            if res is None:
                with lock:
                    state["errors"] += 1
                log(f"{a.name}: {t} q{it['id']} k{k} failed on {ep}: {err}")
                continue
            ch = res["choices"][0]
            text = response_text(ch)
            row = {"task": t, "id": it["id"], "k": k, "seed": body["seed"], "scores": te.score(t, it["doc"], text),
                   "finish": ch.get("finish_reason"), "gen_tokens": (res.get("usage") or {}).get("completion_tokens"),
                   "seconds": round(time.time() - t1, 1), "endpoint": ep, "target": it["doc"], "text": text}
            with lock:
                if state.get("closed"):
                    return
                rows.append(row)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                state["fresh"] += 1
                if state["fresh"] % 10 == 0:
                    s = write_summary()
                    msg = ", ".join(f"{tk} {100 * v.get(v['headline'], 0):.1f} ({v['samples']} samples)"
                                    for tk, v in s["tasks"].items())
                    log(f"{a.name}: {state['fresh']} new samples; {msg}; {state['errors']} errors")

    threads = [threading.Thread(target=worker, args=(ep,), daemon=True) for ep, c in zip(eps, conc) for _ in range(c)]
    for th in threads:
        th.start()
    while any(th.is_alive() for th in threads):
        if a.hard_min and (time.time() - t0) / 60 > a.hard_min:
            log(f"{a.name}: hard stop at {a.hard_min} min ({sum(th.is_alive() for th in threads)} requests abandoned)")
            break
        time.sleep(2)
    with lock:
        state["closed"] = True
        s = write_summary()
        fh.close()
    for t, v in s["tasks"].items():
        m = v["headline"]
        if m in v:
            log(f"{a.name} {t}: {m} {100 * v[m]:.1f} +- {100 * (v[m + '_stderr'] or 0):.1f} over "
                f"{v['questions_with_samples']}/{v['questions']} questions, {v['samples']} samples, coverage {v['coverage']}")
        else:
            log(f"{a.name} {t}: no samples")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
