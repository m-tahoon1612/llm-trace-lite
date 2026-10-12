"""Thin client for OpenAI-compatible /chat/completions that logs every call.

Uses urllib so there's no dependency on the openai package. Works with anything
that speaks the same API (OpenAI, vLLM, Ollama's /v1, LiteLLM proxy, ...).
"""

import json
import os
import random
import time
import urllib.error
import urllib.request

from .pricing import estimate_cost
from .store import Call, TraceStore


def _approx_tokens(text: str) -> int:
    # ~4 chars per token for English. Only used in mock mode.
    return max(1, len(text) // 4)


class TracedClient:
    def __init__(self, store: TraceStore | None = None, base_url: str | None = None,
                 api_key: str | None = None, mock: bool = False, timeout: float = 60,
                 seed: int | None = None):
        self.store = store or TraceStore()
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.mock = mock
        self.timeout = timeout
        self._rng = random.Random(seed)
        if not mock and not self.api_key:
            raise ValueError("no API key: set OPENAI_API_KEY or use mock=True")

    def chat(self, model: str, messages: list[dict], tag: str | None = None, **params) -> dict:
        body = {"model": model, "messages": messages, **params}
        start = time.perf_counter()
        try:
            resp = self._mock(body) if self.mock else self._post(body)
        except Exception as e:  # record failures too, then re-raise
            self.store.record(Call(model=model, latency_ms=self._elapsed(start),
                                   prompt_tokens=None, completion_tokens=None, cost_usd=None,
                                   status="error", tag=tag, request=body, error=str(e)[:500]))
            raise
        latency = resp.pop("_mock_latency_ms", None) or self._elapsed(start)

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

    def _mock(self, body: dict) -> dict:
        prompt = " ".join(str(m.get("content", "")) for m in body["messages"])
        if "FAIL" in prompt:
            raise RuntimeError("mock: simulated upstream error")
        reply = f"(mock reply to {len(body['messages'])} message(s))"
        pt = _approx_tokens(prompt) + 4 * len(body["messages"])
        ct = self._rng.randint(20, body.get("max_tokens") or 400)
        # bigger models are slower; rough numbers, only for demo data
        base = 900 if "mini" not in body["model"] and "haiku" not in body["model"] else 350
        return {
            "id": f"mock-{self._rng.randint(0, 10**9)}",
            "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": reply},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct,
                      "total_tokens": pt + ct},
            "_mock_latency_ms": int(base + ct * self._rng.uniform(4, 12)),
        }
