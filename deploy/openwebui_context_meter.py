"""
title: Context Meter
description: After each reply, shows how many tokens the conversation holds against the server's max context (the server's own usage numbers incl. tool definitions when the model has the "usage" capability, else vLLM's /tokenize of the messages), plus generation speed (tok/s) and time to first token.
version: 0.4
"""

# Install: Open WebUI -> Admin Panel -> Functions -> "+" -> paste this file -> Save -> enable it,
# then either toggle "Global" in its menu or enable it per model (Workspace -> Models -> Filters).
# Set the valves (gear icon): vLLM base URL as Open WebUI's machine sees it, and the API key.
#
# Speed = generated tokens / time from first to last streamed token (decode speed, prefill excluded;
# prefill shows as "first token after"). Generated tokens come from the server's usage numbers when
# the stream carries them, otherwise from tokenizing the reply incl. its thinking ("~" = estimate).

import re
import time

import aiohttp
from pydantic import BaseModel, Field

_REASONING = re.compile(r"<details[^>]*type=\"reasoning\"[^>]*>(.*?)</details>", re.S)
_SUMMARY = re.compile(r"<summary>.*?</summary>", re.S)


def _text(content) -> str:
    if isinstance(content, str):
        return _REASONING.sub("", content)
    if isinstance(content, list):  # multimodal: count text parts only
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _generated_text(content) -> str:
    """The reply as generated: thinking (Open WebUI stores it quoted inside <details>) + answer."""
    if not isinstance(content, str):
        return _text(content)
    parts = []
    for inner in _REASONING.findall(content):
        inner = _SUMMARY.sub("", inner)
        parts.append("\n".join(l[2:] if l.startswith("> ") else l.lstrip(">") for l in inner.splitlines()))
    parts.append(_REASONING.sub("", content))
    return "\n".join(parts)


class Filter:
    class Valves(BaseModel):
        base_url: str = Field(default="http://127.0.0.1:8000", description="vLLM server (no /v1)")
        api_key: str = Field(default="", description="same key the connection uses")
        model: str = Field(default="", description="served model name; empty = take it from the chat")
        warn_pct: int = Field(default=80, description="add a warning above this fill level")
        show_speed: bool = Field(default=True, description="show tok/s and time to first token")

    def __init__(self):
        self.valves = self.Valves()
        self._runs = {}  # chat id -> {"t0", "first", "last", "usage"}

    @staticmethod
    def _key(meta) -> str:
        meta = meta or {}
        return str(meta.get("chat_id") or meta.get("session_id") or "_")

    async def _post(self, path: str, body: dict) -> dict:
        headers = {"Authorization": f"Bearer {self.valves.api_key}"} if self.valves.api_key else {}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
            async with s.post(f"{self.valves.base_url.rstrip('/')}{path}", json=body, headers=headers) as r:
                r.raise_for_status()
                return await r.json()

    async def inlet(self, body: dict, __metadata__=None) -> dict:
        self._runs[self._key(__metadata__)] = {"t0": time.monotonic(), "first": None, "last": None, "usage": None,
                                                 "text": ""}
        if len(self._runs) > 200:  # forget stale chats
            for k in list(self._runs)[:100]:
                self._runs.pop(k, None)
        return body

    def stream(self, event: dict, __metadata__=None) -> dict:
        run = self._runs.get(self._key(__metadata__))
        if run is None and len(self._runs) == 1:  # metadata not passed to stream(): single active chat
            run = next(iter(self._runs.values()))
        if run is not None:
            for ch in event.get("choices") or []:
                d = ch.get("delta") or {}
                piece = "".join(x for x in (d.get("reasoning_content"), d.get("reasoning"), d.get("content"))
                                if isinstance(x, str))
                if piece:
                    run["text"] += piece
                    now = time.monotonic()
                    run["first"] = run["first"] or now
                    run["last"] = now
            if event.get("usage"):
                run["usage"] = event["usage"]
        return event

    async def outlet(self, body: dict, __event_emitter__=None, __model__=None, __metadata__=None) -> dict:
        if not __event_emitter__:
            return body
        messages = body.get("messages", [])
        model = self.valves.model or (__model__ or {}).get("id") or body.get("model", "")
        msgs = [{"role": m.get("role", "user"), "content": _text(m.get("content"))} for m in messages]
        run = self._runs.get(self._key(__metadata__))
        usage = (run or {}).get("usage") or {}
        try:
            j = await self._post("/tokenize", {"model": model, "messages": msgs, "add_generation_prompt": False})
            n, cap, exact = j["count"], j.get("max_model_len"), True
        except Exception:
            n, cap, exact = sum(len(m["content"]) for m in msgs) // 4, None, False  # rough fallback
        if usage.get("prompt_tokens"):
            # what the server actually held: system prompt + tool definitions + history + this reply
            n, exact = usage["prompt_tokens"] + usage.get("completion_tokens", 0), True
        desc = f"Context: {'' if exact else '~'}{n:,} tokens"
        if cap:
            pct = 100 * n / cap
            desc += f" / {cap:,} ({pct:.0f}%)"
            if pct >= self.valves.warn_pct:
                desc += " - nearly full, consider a new chat"

        run = self._runs.pop(self._key(__metadata__), None)
        if self.valves.show_speed and run and run["first"] and messages and messages[-1].get("role") == "assistant":
            gen, approx = None, False
            if run["usage"] and run["usage"].get("completion_tokens"):
                gen = run["usage"]["completion_tokens"]
            else:
                # exactly what streamed (thinking + answer); the stored message is a rewritten copy
                text = run["text"] or _generated_text(messages[-1].get("content"))
                try:
                    gen = (await self._post("/tokenize", {"model": model, "prompt": text,
                                                          "add_special_tokens": False}))["count"]
                except Exception:
                    gen = len(text) // 4
                approx = not run["text"]
            span = run["last"] - run["first"]
            if gen and gen > 1 and span > 0:
                desc += f" | {'~' if approx else ''}{(gen - 1) / span:.1f} tok/s ({gen:,} tokens)"
            desc += f" | first token after {run['first'] - run['t0']:.1f} s"

        await __event_emitter__({"type": "status", "data": {"description": desc, "done": True}})
        return body
