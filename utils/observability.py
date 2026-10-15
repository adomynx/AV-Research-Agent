"""
Observability — lightweight run tracking for the agent pipeline.

Two things an AI engineer always wants to see for an LLM system:
  * what it cost (tokens + dollars), and
  * how long it took.

``track_run`` is a context manager that captures both for every OpenAI call
made inside it, with zero changes to agent code. If the user has LangSmith
configured (env vars), full traces are also sent there automatically by
LangChain — ``enable_langsmith`` just reports whether that is on.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from langchain_community.callbacks.manager import get_openai_callback


@dataclass
class RunStats:
    """Aggregate cost/latency/token stats for one tracked block of work."""

    label: str
    seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    successful_requests: int = 0
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "seconds": round(self.seconds, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "llm_calls": self.successful_requests,
            **self.extra,
        }

    def summary(self) -> str:
        return (
            f"{self.label}: {self.seconds:.2f}s | "
            f"{self.total_tokens:,} tokens | "
            f"${self.total_cost_usd:.4f} | "
            f"{self.successful_requests} call(s)"
        )


@contextmanager
def track_run(label: str) -> Iterator[RunStats]:
    """
    Track cost, latency and token usage for all OpenAI calls in the block.

    Usage::

        with track_run("triage") as stats:
            plan = triage.decompose(topic)
        print(stats.summary())
    """
    stats = RunStats(label=label)
    start = time.perf_counter()
    with get_openai_callback() as cb:
        try:
            yield stats
        finally:
            stats.seconds = time.perf_counter() - start
            stats.prompt_tokens = cb.prompt_tokens
            stats.completion_tokens = cb.completion_tokens
            stats.total_tokens = cb.total_tokens
            stats.total_cost_usd = cb.total_cost
            stats.successful_requests = cb.successful_requests


def enable_langsmith() -> bool:
    """
    Return True if LangSmith tracing is configured.

    LangChain picks up tracing automatically from ``LANGCHAIN_TRACING_V2`` and
    ``LANGCHAIN_API_KEY``; this just tells the UI whether it's active so we can
    show a badge instead of silently doing nothing.
    """
    return os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true" and bool(
        os.getenv("LANGCHAIN_API_KEY")
    )
