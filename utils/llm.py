"""
Provider-agnostic LLM and embedding factories.

The whole pipeline builds its models through here, so switching between OpenAI
and Anthropic (Claude) is a config change (`.env`), not a code change. This is
the "model-agnostic" property that lets you benchmark GPT vs Claude on the same
eval harness.

Selection (all via environment, e.g. `.env`):
    LLM_PROVIDER         "openai" (default) | "anthropic"
    OPENAI_MODEL         default "gpt-4o-mini"
    ANTHROPIC_MODEL      default "claude-opus-5-5"
    EMBEDDINGS_PROVIDER  "local" (default, free) | "openai"
    LOCAL_EMBED_MODEL    default "BAAI/bge-small-en-v1.5"

Notes:
* The newest Claude models (claude-opus-5-5, claude-sonnet-5-5, ...) reject a
  non-default `temperature` (HTTP 400) and rely on adaptive thinking/effort, so
  we only pass `temperature` for OpenAI.
* Anthropic has no embeddings API, so embeddings default to a local, free
  FastEmbed (ONNX) model — RAG needs no API key regardless of LLM provider.
"""

from __future__ import annotations

import os

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
DEFAULT_LOCAL_EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def get_provider() -> str:
    return os.getenv("LLM_PROVIDER", "openai").strip().lower()


def get_chat_model(temperature: float = 0.2, max_tokens: int | None = None) -> BaseChatModel:
    """
    Build the configured chat model.

    `temperature` applies to OpenAI only; it is omitted for Anthropic because
    the current Claude models reject a non-default temperature.
    """
    provider = get_provider()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs: dict = {
            "model": os.getenv("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL),
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            # Claude requires an explicit output cap; give a sane default.
            "max_tokens": max_tokens or 4096,
        }
        return ChatAnthropic(**kwargs)

    from langchain_openai import ChatOpenAI

    kwargs = {
        "model": os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        "api_key": os.getenv("OPENAI_API_KEY"),
        "temperature": temperature,
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    return ChatOpenAI(**kwargs)


def get_embeddings() -> Embeddings:
    """
    Build the configured embedding model. Defaults to a free local model so RAG
    works with no API key and no per-vector cost.
    """
    provider = os.getenv("EMBEDDINGS_PROVIDER", "local").strip().lower()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        )

    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

    return FastEmbedEmbeddings(
        model_name=os.getenv("LOCAL_EMBED_MODEL", DEFAULT_LOCAL_EMBED_MODEL)
    )


def message_text(message) -> str:
    """
    Normalize an LLM message's content to a plain string.

    OpenAI returns ``content`` as a string; Anthropic returns a list of content
    blocks (e.g. ``[{"type": "text", "text": ...}]``). This flattens either
    shape so downstream code (report rendering, eval metrics) always gets text.
    """
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text") or block.get("content") or "")
            else:
                parts.append(getattr(block, "text", "") or "")
        return "".join(parts)
    return str(content)


def has_llm_credentials() -> bool:
    """True if the selected provider has an API key available."""
    if get_provider() == "anthropic":
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    return bool(os.getenv("OPENAI_API_KEY"))


def provider_label() -> str:
    """Human-readable 'provider / model' for UI badges."""
    if get_provider() == "anthropic":
        return f"Anthropic / {os.getenv('ANTHROPIC_MODEL', DEFAULT_ANTHROPIC_MODEL)}"
    return f"OpenAI / {os.getenv('OPENAI_MODEL', DEFAULT_OPENAI_MODEL)}"
