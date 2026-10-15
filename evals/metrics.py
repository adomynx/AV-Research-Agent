"""
Evaluation metrics for generated research reports.

Two families:

* Deterministic metrics — pure functions over text, no API key, no network.
  They catch the failures that actually embarrass a research tool: citing a
  source that was never collected, skipping required sections, or inventing a
  source in the bibliography.

* LLM-as-judge metrics — faithfulness and topic relevance, scored by a model.
  These need an OpenAI key and are skipped (reported as ``None``) without one.

Every metric returns a ``MetricScore`` in [0, 1] plus a human-readable detail
string, so the scorecard is self-explanatory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from schemas.models import Fact

# Sections the Editor agent is prompted to produce. Scored as coverage.
REQUIRED_SECTIONS = [
    "Executive Summary",
    "Key Findings",
    "Conclusion",
    "Sources",
]

# [text] that is NOT immediately followed by "(" (i.e. not a markdown link).
_CITATION_RE = re.compile(r"\[([^\]\[]+)\](?!\()")


@dataclass
class MetricScore:
    name: str
    value: float  # 0.0 - 1.0
    detail: str

    def as_dict(self) -> dict:
        return {"name": self.name, "value": round(self.value, 4), "detail": self.detail}


def _known_sources(facts: list[Fact]) -> list[str]:
    return [f.source.strip() for f in facts]


def _matches_a_source(citation: str, sources: list[str]) -> bool:
    """A citation matches if it shares substring overlap with any real source."""
    c = citation.strip().lower()
    if not c:
        return False
    return any(c in s.lower() or s.lower() in c for s in sources)


def citation_coverage(report: str, facts: list[Fact]) -> MetricScore:
    """Fraction of inline citations that point at a source we actually collected."""
    citations = [m.group(1) for m in _CITATION_RE.finditer(report)]
    if not citations:
        return MetricScore("citation_coverage", 0.0, "no inline [citations] found")

    sources = _known_sources(facts)
    grounded = [c for c in citations if _matches_a_source(c, sources)]
    value = len(grounded) / len(citations)
    hallucinated = sorted({c for c in citations if not _matches_a_source(c, sources)})
    detail = f"{len(grounded)}/{len(citations)} citations grounded"
    if hallucinated:
        detail += f"; ungrounded: {hallucinated}"
    return MetricScore("citation_coverage", value, detail)


def structure_coverage(report: str) -> MetricScore:
    """Fraction of required report sections that are present."""
    lower = report.lower()
    present = [s for s in REQUIRED_SECTIONS if s.lower() in lower]
    value = len(present) / len(REQUIRED_SECTIONS)
    missing = [s for s in REQUIRED_SECTIONS if s.lower() not in lower]
    detail = f"{len(present)}/{len(REQUIRED_SECTIONS)} sections"
    if missing:
        detail += f"; missing: {missing}"
    return MetricScore("structure_coverage", value, detail)


def source_grounding(report: str, facts: list[Fact]) -> MetricScore:
    """Fraction of entries in the Sources section that map to a collected fact."""
    match = re.search(r"##+\s*Sources\s*(.+)$", report, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return MetricScore("source_grounding", 0.0, "no Sources section")

    listed = [
        re.sub(r"^\s*\d+\.\s*", "", line).strip()
        for line in match.group(1).splitlines()
        if line.strip()
    ]
    if not listed:
        return MetricScore("source_grounding", 0.0, "empty Sources section")

    sources = _known_sources(facts)
    grounded = [s for s in listed if _matches_a_source(s, sources)]
    value = len(grounded) / len(listed)
    invented = sorted({s for s in listed if not _matches_a_source(s, sources)})
    detail = f"{len(grounded)}/{len(listed)} listed sources grounded"
    if invented:
        detail += f"; invented: {invented}"
    return MetricScore("source_grounding", value, detail)


def deterministic_metrics(report: str, facts: list[Fact]) -> list[MetricScore]:
    """Run every metric that needs no API key."""
    return [
        citation_coverage(report, facts),
        structure_coverage(report),
        source_grounding(report, facts),
    ]


# --------------------------------------------------------------------------- #
# LLM-as-judge metrics (optional; require an OpenAI API key)
# --------------------------------------------------------------------------- #

def _judge_llm():
    from pydantic import BaseModel, Field

    from utils.llm import get_chat_model

    class Verdict(BaseModel):
        score: float = Field(..., ge=0.0, le=1.0, description="0=fails, 1=perfect")
        reasoning: str = Field(..., description="one sentence justification")

    return get_chat_model(temperature=0.0).with_structured_output(Verdict)


def faithfulness(report: str, facts: list[Fact]) -> MetricScore:
    """Judge whether the report's claims are supported by the collected facts."""
    judge = _judge_llm()
    facts_text = "\n".join(f"- {f.title}: {f.content} (source: {f.source})" for f in facts)
    prompt = (
        "You are a strict fact-checker. Given a set of SOURCE FACTS and a REPORT, "
        "rate how faithful the report is to the facts. Penalise any claim that is "
        "not supported by the facts (hallucination). Score 0.0 to 1.0.\n\n"
        f"SOURCE FACTS:\n{facts_text}\n\nREPORT:\n{report}"
    )
    verdict = judge.invoke(prompt)
    return MetricScore("faithfulness", float(verdict.score), verdict.reasoning)


def topic_relevance(topic: str, report: str) -> MetricScore:
    """Judge whether the report actually answers the requested topic."""
    judge = _judge_llm()
    prompt = (
        "Rate from 0.0 to 1.0 how well this REPORT addresses the requested TOPIC "
        "— coverage, focus, and usefulness.\n\n"
        f"TOPIC: {topic}\n\nREPORT:\n{report}"
    )
    verdict = judge.invoke(prompt)
    return MetricScore("topic_relevance", float(verdict.score), verdict.reasoning)
