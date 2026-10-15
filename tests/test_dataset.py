"""Integrity tests for the golden eval dataset."""

from evals.dataset import DATASET
from evals.metrics import deterministic_metrics


def test_dataset_has_enough_cases():
    assert len(DATASET) >= 6, "eval set should be large enough to be meaningful"


def test_case_ids_are_unique():
    ids = [c.case_id for c in DATASET]
    assert len(ids) == len(set(ids))


def test_every_case_is_well_formed():
    for c in DATASET:
        assert c.topic and c.sub_queries and c.facts
        assert c.reference_report, f"{c.case_id} missing reference report"


def test_deterministic_metrics_run_on_every_fixture():
    for c in DATASET:
        scores = deterministic_metrics(c.reference_report, c.facts)
        assert len(scores) == 3
        assert all(0.0 <= s.value <= 1.0 for s in scores)


def test_fixtures_are_mostly_well_structured():
    # Every fixture report should cover all required sections.
    for c in DATASET:
        structure = next(s for s in deterministic_metrics(c.reference_report, c.facts)
                         if s.name == "structure_coverage")
        assert structure.value == 1.0, f"{c.case_id} is missing required sections"
