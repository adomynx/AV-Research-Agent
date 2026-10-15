"""
Web search with a keyed primary provider and a free fallback.

* If ``TAVILY_API_KEY`` is set, Tavily is used first — a keyed search API built
  for LLMs that doesn't rate-limit like DuckDuckGo does under bursty agent load.
* Otherwise (or if Tavily fails), it falls back to ``ddgs`` (DuckDuckGo).

Both paths retry with backoff and degrade to ``[]`` so a transient failure never
crashes a research run. Results are normalized to {title, url, snippet} dicts.
"""

from __future__ import annotations

import os
import time

import requests
from ddgs import DDGS

_TAVILY_URL = "https://api.tavily.com/search"


def _tavily(query: str, max_results: int) -> list[dict]:
    resp = requests.post(
        _TAVILY_URL,
        json={
            "api_key": os.environ["TAVILY_API_KEY"],
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
        for r in data.get("results", [])
    ]


def _ddgs(query: str, max_results: int) -> list[dict]:
    raw = DDGS().text(query, max_results=max_results)
    return [
        {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
        for r in raw
    ]


def _try(fn, query: str, max_results: int, retries: int) -> list[dict]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            results = fn(query, max_results)
            if results:
                return results
        except Exception as e:  # noqa: BLE001
            last_error = e
            time.sleep(0.6 * (attempt + 1))
    if last_error:
        print(f"[web_search] {fn.__name__} failed: {last_error}")
    return []


def web_search(query: str, max_results: int = 5, retries: int = 2) -> list[dict]:
    """
    Search the web, preferring Tavily (if configured) and falling back to ddgs.

    Returns a list of {title, url, snippet} dicts, or [] if everything failed.
    """
    if os.getenv("TAVILY_API_KEY"):
        results = _try(_tavily, query, max_results, retries)
        if results:
            return results  # fall through to ddgs only if Tavily yields nothing
    return _try(_ddgs, query, max_results, retries)


def search_backend() -> str:
    """Name of the active primary search backend (for UI/logging)."""
    return "Tavily" if os.getenv("TAVILY_API_KEY") else "DuckDuckGo"


def format_results(results: list[dict]) -> str:
    """Render search results as text for an LLM prompt, with sources."""
    if not results:
        return "No web results available."
    return "\n\n".join(
        f"[{r['title']}]({r['url']})\n{r['snippet']}" for r in results
    )
