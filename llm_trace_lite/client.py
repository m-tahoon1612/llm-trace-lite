"""Thin client for OpenAI-compatible /chat/completions that logs every call.

Uses urllib so there's no dependency on the openai package. Should work with anything
that speaks the same API (OpenAI, vLLM, Ollama's /v1, LiteLLM proxy, ...).
"""

import json
import os
import time
import urllib.error
import urllib.request

from .pricing import estimate_cost
from .store import Call, TraceStore


class TracedClient:
    def __init__(self, store: TraceStore | None = None, base_url: str | None = None,
                 api_key: str | None = None, timeout: float = 60):
        self.store = store or TraceStore()
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.timeout = timeout
        if not self.api_key:
            raise ValueError("no API key: set OPENAI_API_KEY")

    def chat(self, model: str, messages: list[dict], tag: str | None = None, **params) -> dict:
        body = {"model": model, "messages": messages, **params}
        start = time.perf_counter()
        try:
            resp = self._post(body)
        except Exception as e:  # record failures too, then re-raise
            self.store.record(Call(model=model, latency_ms=self._elapsed(start),
                                   prompt_tokens=None, completion_tokens=None, cost_usd=None,
                                   status="error", tag=tag, request=body, error=str(e)[:500]))
            raise
        latency = self._elapsed(start)

        usage = resp.get("usage") or {}
        pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
        text = (resp.get("choices") or [{}])[0].get("message", {}).get("content")
        self.store.record(Call(
            model=resp.get("model", model), latency_ms=latency, prompt_tokens=pt,
            completion_tokens=ct,
            cost_usd=estimate_cost(model, pt or 0, ct or 0) if pt is not None else None,
            tag=tag, request=body, response_text=text,
        ))
        return resp

    @staticmethod
    def _elapsed(start: float) -> int:
        return int((time.perf_counter() - start) * 1000)

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}") from e
