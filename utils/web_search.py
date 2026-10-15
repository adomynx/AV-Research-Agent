"""
Web search via the `ddgs` library directly.

LangChain's DuckDuckGoSearchResults wrapper was issuing malformed requests
(DNS failures against bogus hosts), so the research agent silently got zero web
results and the LLM filled the gap from memory. Calling `ddgs` directly is
reliable, and we add light retry/backoff plus graceful degradation so a
transient failure never crashes a research run.
"""

from __future__ import annotations

import time

from ddgs import DDGS


def web_search(query: str, max_results: int = 5, retries: int = 2) -> list[dict]:
    """
    Run a web search and return a list of {title, url, snippet} dicts.

    Retries with backoff on transient failures; returns [] if all attempts fail
    (the caller degrades gracefully rather than crashing).
    """
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            raw = DDGS().text(query, max_results=max_results)
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
                for r in raw
            ]
        except Exception as e:  # noqa: BLE001
            last_error = e
            time.sleep(0.6 * (attempt + 1))
    print(f"[web_search] failed after {retries + 1} attempts: {last_error}")
    return []


def format_results(results: list[dict]) -> str:
    """Render search results as text for an LLM prompt, with sources."""
    if not results:
        return "No web results available."
    return "\n\n".join(
        f"[{r['title']}]({r['url']})\n{r['snippet']}" for r in results
    )
