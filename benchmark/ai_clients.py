"""
AI clients with a common interface.

Each provider (OpenAI, Gemini, DeepSeek) is wrapped in a class that
inherits from AIClient and implements _call(). The public ask() method
is the same for all: it measures latency, counts tokens and computes
the cost.

Each provider's library is imported lazily (inside __init__) so the
project works even if you only installed one provider's SDK.

Note: DeepSeek is served through OpenRouter, which is OpenAI-compatible,
so it reuses the OpenAI client with a different base_url.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import config
from benchmark.logging_setup import log

# Common system prompt: defines the role and the answering rules.
SYSTEM_PROMPT = (
    "You are a financial analysis assistant for a personal finance "
    "management platform. Answer ONLY based on the data provided in the "
    "context. Be direct and objective. "
    "If the requested information is NOT in the data, answer exactly: "
    "'The information is not available in the provided data.' "
    "Never make up numbers, names or projections that are not in the context."
)


@dataclass
class AIResponse:
    """Result of a single call to an AI."""
    text: str
    latency_s: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    error: str | None = None


class AIClient:
    """Base class. Each provider implements _call()."""

    def __init__(self, name: str):
        self.name = name
        cfg = config.PROVIDERS[name]
        self.model = cfg["model"]
        self.input_price = cfg["input_price"]
        self.output_price = cfg["output_price"]
        self.base_url = cfg.get("base_url")
        self.api_key = config.api_key(name)

    def _compute_cost(self, tin: int, tout: int) -> float:
        """Cost in USD = tokens/1M * price per million, for input and output."""
        return (tin / 1_000_000) * self.input_price + (tout / 1_000_000) * self.output_price

    def _call(self, context: str, prompt: str) -> tuple[str, int, int]:
        """Must return (answer_text, input_tokens, output_tokens)."""
        raise NotImplementedError

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        """
        True if the error is temporary and worth retrying.

        Covers provider overload (503 / "overloaded" / "unavailable"),
        rate limits (429 / "rate limit"), timeouts and connection drops.
        An auth error or a bad request is NOT transient and fails fast.
        """
        msg = str(exc).lower()
        signals = (
            "503", "500", "502", "504", "overloaded", "unavailable",
            "429", "rate limit", "timeout", "timed out",
            "connection", "temporarily",
        )
        return any(s in msg for s in signals)

    def ask(self, context: str, question: str) -> AIResponse:
        """
        Common interface: builds the prompt, times it, retries transient
        errors with exponential backoff, and handles errors.
        """
        prompt = f"{context}\n\n=== QUESTION ===\n{question}"
        start = time.perf_counter()
        last_exc: Exception | None = None

        for attempt in range(config.MAX_RETRIES + 1):
            try:
                text, tin, tout = self._call(context, prompt)
                latency = time.perf_counter() - start
                return AIResponse(
                    text=text.strip(),
                    latency_s=latency,
                    input_tokens=tin,
                    output_tokens=tout,
                    cost_usd=self._compute_cost(tin, tout),
                )
            except Exception as exc:
                last_exc = exc
                transient = self._is_transient(exc)
                if transient and attempt < config.MAX_RETRIES:
                    wait = config.RETRY_BACKOFF_S * (2 ** attempt)
                    log.warning(
                        "[%s] transient error (attempt %d/%d), retrying in %.1fs: %s",
                        self.name, attempt + 1, config.MAX_RETRIES, wait, exc,
                    )
                    time.sleep(wait)
                    continue
                break  # non-transient, or retries exhausted

        latency = time.perf_counter() - start
        log.error("[%s] call error (gave up): %s", self.name, last_exc)
        return AIResponse("", latency, 0, 0, 0.0, error=str(last_exc))


# ---------------------------------------------------------------------
# OpenAI (and OpenAI-compatible endpoints such as OpenRouter)
# ---------------------------------------------------------------------
class OpenAIClient(AIClient):
    def __init__(self, name: str = "OpenAI"):
        super().__init__(name)
        from openai import OpenAI
        # base_url=None makes the SDK use the official OpenAI endpoint.
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _call(self, context: str, prompt: str) -> tuple[str, int, int]:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_OUTPUT_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return text, usage.prompt_tokens, usage.completion_tokens


# ---------------------------------------------------------------------
# DeepSeek via OpenRouter (OpenAI-compatible)
# ---------------------------------------------------------------------
class DeepSeekClient(OpenAIClient):
    """Reuses the OpenAI client; config supplies the OpenRouter base_url."""

    def __init__(self):
        super().__init__("DeepSeek")


# ---------------------------------------------------------------------
# Google Gemini via OpenRouter (OpenAI-compatible)
# ---------------------------------------------------------------------
class GeminiClient(OpenAIClient):
    """
    Gemini served through OpenRouter, which is OpenAI-compatible, so it
    reuses the OpenAI client. We use this instead of the direct Google
    API (google-genai) because the Google AI Studio billing could not be
    set up; OpenRouter bills Gemini usage through the OpenRouter credit.
    """

    def __init__(self):
        super().__init__("Gemini")


# Name -> class map, used by the factory.
_FACTORY = {
    "OpenAI": OpenAIClient,
    "Gemini": GeminiClient,
    "DeepSeek": DeepSeekClient,
}


def create_client(name: str) -> AIClient:
    """Instantiate the provider's client by name."""
    return _FACTORY[name]()
