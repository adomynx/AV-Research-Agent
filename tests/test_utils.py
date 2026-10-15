"""Unit tests for provider-agnostic helpers and web search formatting."""

from utils.llm import message_text
from utils.web_search import format_results


class _Msg:
    def __init__(self, content):
        self.content = content


def test_message_text_openai_string():
    assert message_text(_Msg("hello world")) == "hello world"


def test_message_text_anthropic_blocks():
    msg = _Msg([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
    assert message_text(msg) == "ab"


def test_message_text_mixed_blocks():
    msg = _Msg(["raw", {"text": "x"}])
    assert message_text(msg) == "rawx"


def test_message_text_plain_string_input():
    assert message_text("already text") == "already text"


def test_format_results_empty():
    assert format_results([]) == "No web results available."


def test_format_results_renders_fields():
    out = format_results([{"title": "T", "url": "http://x", "snippet": "S"}])
    assert "T" in out and "http://x" in out and "S" in out
