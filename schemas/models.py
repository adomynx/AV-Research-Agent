"""
Typed schemas for the AV Research Agent pipeline.

These Pydantic models are the single source of truth for the data that flows
between agents. They are used two ways:

1. As the target of LangChain's ``with_structured_output`` so the LLM is
   constrained to emit valid, typed data (no more fragile ``JsonOutputParser``
   that silently returns the wrong shape).
2. As the contract the evaluation harness scores against — every metric knows
   exactly what fields exist and what they mean.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Relevance(str, Enum):
    """How useful a fact is to the overall research topic."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ResearchPlan(BaseModel):
    """Output of the Triage agent: the topic broken into focused sub-queries."""

    sub_queries: list[str] = Field(
        ...,
        description=(
            "4-6 focused, searchable sub-queries covering different angles of "
            "the topic (architecture, benchmarks, deployment, SOTA, safety)."
        ),
        min_length=1,
        max_length=6,
    )

    @field_validator("sub_queries")
    @classmethod
    def _strip_and_drop_empty(cls, queries: list[str]) -> list[str]:
        cleaned = [q.strip() for q in queries if q and q.strip()]
        if not cleaned:
            raise ValueError("research plan must contain at least one sub-query")
        return cleaned


class ReflectionVerdict(BaseModel):
    """Output of the Reflection step: is the evidence good enough, or research more?"""

    sufficient: bool = Field(
        ...,
        description="True if the collected facts cover the topic well enough to write a thorough report.",
    )
    missing_aspects: list[str] = Field(
        default_factory=list,
        description="Important angles of the topic not yet covered by the facts.",
    )
    follow_up_queries: list[str] = Field(
        default_factory=list,
        description="Up to 3 targeted search queries that would close the biggest gaps.",
        max_length=3,
    )

    @field_validator("follow_up_queries", "missing_aspects")
    @classmethod
    def _strip(cls, items: list[str]) -> list[str]:
        return [i.strip() for i in items if i and i.strip()]


class Fact(BaseModel):
    """A single structured finding extracted by the Research agent."""

    title: str = Field(..., description="Short descriptive title (max ~10 words).")
    content: str = Field(..., description="The key finding in 2-3 sentences.")
    source: str = Field(..., description="Where this information came from.")
    relevance: Relevance = Field(
        default=Relevance.MEDIUM,
        description="How relevant this fact is to the topic.",
    )

    @field_validator("title", "content", "source")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("field must not be empty")
        return value


class FactSet(BaseModel):
    """Output of the Research agent: the facts extracted for one sub-query."""

    facts: list[Fact] = Field(default_factory=list)

    def high_and_medium(self) -> list[Fact]:
        """Return only the facts worth putting in a report."""
        return [f for f in self.facts if f.relevance in (Relevance.HIGH, Relevance.MEDIUM)]
