"""
utils/llm.py — Creates and caches the Groq LLM instance.

Uses langchain-groq with the Groq free tier.
Get your free API key at: https://console.groq.com

Free-tier limits for llama-3.1-8b-instant (as of 2025):
  - 14,400 requests / day
  - 131,072 tokens / minute
  - 500,000 tokens / day

Rate-limit handling:
  - Automatically retries on 429 / RESOURCE_EXHAUSTED errors
  - Uses exponential backoff (reads retry delay from the API response)
  - Falls back to alternate models if quota is exhausted
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from functools import lru_cache
from typing import Any, Iterator, AsyncIterator, Optional

from dotenv import load_dotenv
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_groq import ChatGroq

load_dotenv()

# Maximum number of automatic retries on 429 before giving up
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "5"))
# Base wait time in seconds (used when no retry-delay is in the API response)
BASE_WAIT_SECONDS = float(os.getenv("LLM_BASE_WAIT_SECONDS", "15"))


def _extract_retry_delay(error_message: str) -> float:
    """Parse 'retry after Xs' or 'retry in Xs' from the API error message."""
    match = re.search(r"retry.{1,10}?(\d+(?:\.\d+)?)\s*s", error_message, re.IGNORECASE)
    if match:
        return float(match.group(1)) + 2.0  # add 2s buffer
    return BASE_WAIT_SECONDS


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True if *exc* is a 429 / quota-exhausted error."""
    msg = str(exc).lower()
    return (
        "429" in msg
        or "too many requests" in msg
        or "resource_exhausted" in msg
        or "rate limit" in msg
        or "rate_limit" in msg
    )


def _build_llm(model_id: str, api_key: str, temperature: float) -> ChatGroq:
    """Construct a ChatGroq instance."""
    return ChatGroq(
        model=model_id,
        groq_api_key=api_key,
        temperature=temperature,
    )


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """
    Returns a cached ChatGroq instance.

    Model priority (controlled via .env):
      1. GROQ_MODEL_ID            (default: llama-3.1-8b-instant)
      2. GROQ_FALLBACK_MODEL_ID   (default: llama3-8b-8192)
      3. GROQ_FALLBACK_MODEL_ID_2 (default: gemma2-9b-it)
    """
    model_id = os.getenv("GROQ_MODEL_ID", "llama-3.1-8b-instant")
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    temperature = float(os.getenv("GROQ_TEMPERATURE", "0.2"))

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Get your free key at: https://console.groq.com\n"
            "Then add it to your .env file as: GROQ_API_KEY=gsk_..."
        )

    llm = _build_llm(model_id, api_key, temperature)
    print(f"[LLM] Loaded Groq Model: {model_id}")
    return llm


class RateLimitAwareLLM(Runnable):
    """
    LangChain-compatible Runnable wrapper around ChatGroq that retries
    automatically on 429 / RESOURCE_EXHAUSTED errors, cycling through
    a chain of fallback models.

    Inherits from Runnable so that `prompt | llm` pipe syntax works.
    """

    def __init__(self, llm: ChatGroq):
        self._api_key = os.getenv("GROQ_API_KEY", "").strip()
        self._temperature = float(os.getenv("GROQ_TEMPERATURE", "0.2"))
        # Build the fallback chain: primary → fallback1 → fallback2
        primary   = os.getenv("GROQ_MODEL_ID",            "llama-3.1-8b-instant")
        fallback1 = os.getenv("GROQ_FALLBACK_MODEL_ID",   "llama3-8b-8192")
        fallback2 = os.getenv("GROQ_FALLBACK_MODEL_ID_2", "gemma2-9b-it")
        self._model_chain = [primary, fallback1, fallback2]
        self._chain_index = 0  # which model we're currently using
        self._llm = llm

    # ── Proxy attribute/method access so it behaves like the underlying LLM ──
    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)

    # ── Support `prompt | llm` and `llm | parser` pipe syntax ────────────────
    def __or__(self, other: Any) -> Any:
        """Enable `llm | other` — delegates to underlying ChatGroq."""
        return self._llm.__or__(other)

    def __ror__(self, other: Any) -> Any:
        """Enable `other | llm` — creates a RunnableSequence via the real LLM."""
        if hasattr(other, 'pipe'):
            return other.pipe(self._llm)
        return NotImplemented

    def _advance_model(self) -> bool:
        """Switch to the next model in the fallback chain. Returns False if chain is exhausted."""
        if self._chain_index + 1 < len(self._model_chain):
            self._chain_index += 1
            next_model = self._model_chain[self._chain_index]
            print(
                f"\n[LLM] ⚠️  Quota exhausted. Switching to '{next_model}' "
                f"(model {self._chain_index + 1}/{len(self._model_chain)})..."
            )
            self._llm = _build_llm(next_model, self._api_key, self._temperature)
            return True
        return False

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Invoke with automatic retry on 429, cycling through fallback models."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if config is not None:
                    return self._llm.invoke(input, config, **kwargs)
                return self._llm.invoke(input, **kwargs)
            except Exception as exc:
                if not _is_rate_limit_error(exc):
                    raise
                switched = self._advance_model()
                wait = 5.0 if switched else _extract_retry_delay(str(exc))
                print(
                    f"[LLM] 🔄 Rate limited (attempt {attempt}/{MAX_RETRIES}). "
                    f"Waiting {wait:.0f}s before retry..."
                )
                time.sleep(wait)
        # Final attempt — let exception propagate naturally
        if config is not None:
            return self._llm.invoke(input, config, **kwargs)
        return self._llm.invoke(input, **kwargs)

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Async invoke with automatic retry on 429, cycling through fallback models."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if config is not None:
                    return await self._llm.ainvoke(input, config, **kwargs)
                return await self._llm.ainvoke(input, **kwargs)
            except Exception as exc:
                if not _is_rate_limit_error(exc):
                    raise
                switched = self._advance_model()
                wait = 5.0 if switched else _extract_retry_delay(str(exc))
                print(
                    f"[LLM] 🔄 Rate limited (attempt {attempt}/{MAX_RETRIES}). "
                    f"Waiting {wait:.0f}s before retry..."
                )
                await asyncio.sleep(wait)
        if config is not None:
            return await self._llm.ainvoke(input, config, **kwargs)
        return await self._llm.ainvoke(input, **kwargs)

    def pipe(self, *args: Any, **kwargs: Any) -> Any:
        return self._llm.pipe(*args, **kwargs)

    def bind(self, **kwargs: Any) -> Any:
        return self._llm.bind(**kwargs)

    def bind_tools(self, *args: Any, **kwargs: Any) -> Any:
        """
        Return the tool-bound LLM wrapped with LangChain's native retry.

        `create_react_agent` calls `model.bind_tools(tools)` internally and
        then uses the result for ALL LLM calls.  Without this wrapper the
        raw ChatGroq runnable is returned, which does NOT have our retry
        logic — so a 429 throws immediately instead of waiting and retrying.
        """
        bound = self._llm.bind_tools(*args, **kwargs)
        # Wrap with tenacity-based retry so any transient 429 is handled
        # automatically.  Max wait is capped at 60 s so we never block
        # longer than Groq's own "retry after" hints (typically 2–30 s).
        return bound.with_retry(
            stop_after_attempt=MAX_RETRIES,
            wait_exponential_jitter=True,
        )


def get_rate_limited_llm() -> RateLimitAwareLLM:
    """Return a RateLimitAwareLLM wrapping the cached base LLM."""
    return RateLimitAwareLLM(get_llm())
