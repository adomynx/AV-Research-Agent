"""Unit tests for the Pydantic data contracts."""

import pytest
from pydantic import ValidationError

from schemas.models import Fact, FactSet, ReflectionVerdict, Relevance, ResearchPlan


def test_fact_rejects_empty_fields():
    with pytest.raises(ValidationError):
        Fact(title="   ", content="x", source="y")


def test_fact_defaults_relevance_medium():
    f = Fact(title="t", content="c", source="s")
    assert f.relevance == Relevance.MEDIUM


def test_factset_high_and_medium_filter():
    fs = FactSet(facts=[
        Fact(title="a", content="c", source="s", relevance=Relevance.HIGH),
        Fact(title="b", content="c", source="s", relevance=Relevance.LOW),
        Fact(title="c", content="c", source="s", relevance=Relevance.MEDIUM),
    ])
    assert len(fs.high_and_medium()) == 2


def test_research_plan_strips_and_drops_empty():
    plan = ResearchPlan(sub_queries=["  q1 ", "", "q2"])
    assert plan.sub_queries == ["q1", "q2"]


def test_research_plan_requires_at_least_one():
    with pytest.raises(ValidationError):
        ResearchPlan(sub_queries=["   ", ""])


def test_research_plan_caps_at_six():
    with pytest.raises(ValidationError):
        ResearchPlan(sub_queries=[f"q{i}" for i in range(7)])


def test_reflection_verdict_strips_lists():
    v = ReflectionVerdict(
        sufficient=False, missing_aspects=[" gap ", ""], follow_up_queries=["q", "  "]
    )
    assert v.missing_aspects == ["gap"]
    assert v.follow_up_queries == ["q"]
