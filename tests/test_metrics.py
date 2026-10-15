"""Unit tests for the deterministic eval metrics."""

from evals.metrics import citation_coverage, source_grounding, structure_coverage
from schemas.models import Fact, Relevance

FACTS = [
    Fact(title="A", content="c", source="BEVFusion paper", relevance=Relevance.HIGH),
    Fact(title="B", content="c", source="nuScenes", relevance=Relevance.MEDIUM),
]


def test_citation_coverage_all_grounded():
    report = "Claim one [BEVFusion paper]. Claim two [nuScenes]."
    assert citation_coverage(report, FACTS).value == 1.0


def test_citation_coverage_detects_hallucination():
    s = citation_coverage("Claim [BEVFusion paper] and [MadeUpBlog].", FACTS)
    assert s.value == 0.5
    assert "MadeUpBlog" in s.detail


def test_citation_coverage_no_citations():
    assert citation_coverage("no citations here", FACTS).value == 0.0


def test_citation_ignores_markdown_links():
    # [text](url) is a link, not a citation — should not be counted.
    s = citation_coverage("See [the repo](http://x) and [nuScenes].", FACTS)
    assert s.value == 1.0  # only [nuScenes] counts, and it is grounded


def test_structure_coverage_full():
    report = "## Executive Summary\nx\n## Key Findings\ny\n## Conclusion\nz\n## Sources\n1. a"
    assert structure_coverage(report).value == 1.0


def test_structure_coverage_partial():
    s = structure_coverage("## Executive Summary\nx\n## Key Findings\ny")
    assert 0.0 < s.value < 1.0


def test_source_grounding_all_match():
    report = "## Sources\n1. BEVFusion paper\n2. nuScenes"
    assert source_grounding(report, FACTS).value == 1.0


def test_source_grounding_detects_invented():
    s = source_grounding("## Sources\n1. BEVFusion paper\n2. FakeSource", FACTS)
    assert s.value == 0.5
    assert "FakeSource" in s.detail


def test_source_grounding_missing_section():
    assert source_grounding("no sources section", FACTS).value == 0.0
