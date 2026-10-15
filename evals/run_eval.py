"""
Evaluation runner — scores the Editor stage against the golden dataset and
writes a timestamped scorecard (JSON + Markdown) to ``evals/results/``.

Run from the repo root:

    # Offline: deterministic metrics only, no API key, no network, free.
    python -m evals.run_eval

    # Live: regenerate each report with the real Editor agent, then also run
    # the LLM-as-judge metrics (faithfulness, relevance). Needs an API key for
    # the selected provider (OpenAI or Anthropic).
    python -m evals.run_eval --live

This is the before/after instrument: change a prompt, model, or retrieval
strategy, re-run, and compare scorecards.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from evals.dataset import DATASET, EvalCase
from evals.metrics import MetricScore, deterministic_metrics, faithfulness, topic_relevance
from utils.llm import has_llm_credentials, provider_label

RESULTS_DIR = Path(__file__).parent / "results"


def _score_case(case: EvalCase, has_creds: bool, live: bool) -> dict:
    # Choose the report to score: freshly generated (live) or the fixture.
    if live and has_creds:
        from agents.editor import EditorAgent
        report = EditorAgent().compile_report(case.topic, case.facts)
    else:
        report = case.reference_report

    scores: list[MetricScore] = deterministic_metrics(report, case.facts)

    judge_error: str | None = None
    if has_creds:
        # LLM-judge calls can fail (no credits, rate limit, outage). Never let
        # that crash the run — deterministic metrics must still be reported.
        try:
            scores.append(faithfulness(report, case.facts))
            scores.append(topic_relevance(case.topic, report))
        except Exception as e:  # noqa: BLE001
            judge_error = f"{type(e).__name__}: {e}"
            print(f"[eval] LLM-judge skipped for {case.case_id}: {judge_error}")

    result = {
        "case_id": case.case_id,
        "topic": case.topic,
        "mode": "live" if (live and has_creds) else "fixture",
        "scores": [s.as_dict() for s in scores],
    }
    if judge_error:
        result["judge_error"] = judge_error
    return result


def _aggregate(case_results: list[dict]) -> dict:
    """Average each metric across all cases."""
    buckets: dict[str, list[float]] = {}
    for case in case_results:
        for score in case["scores"]:
            buckets.setdefault(score["name"], []).append(score["value"])
    return {name: round(statistics.mean(vals), 4) for name, vals in buckets.items()}


def _render_markdown(report_obj: dict) -> str:
    lines = [
        "# AV Research Agent — Evaluation Scorecard",
        "",
        f"- **Run:** {report_obj['timestamp']}",
        f"- **Cases:** {len(report_obj['cases'])}",
        f"- **LLM-judge metrics:** {'on' if report_obj['llm_judge'] else 'off (no API key)'}",
        "",
        "## Aggregate scores",
        "",
        "| Metric | Mean (0-1) |",
        "|--------|-----------|",
    ]
    for name, val in report_obj["aggregate"].items():
        lines.append(f"| {name} | {val:.3f} |")

    lines += ["", "## Per-case detail", ""]
    for case in report_obj["cases"]:
        lines.append(f"### {case['case_id']}  _({case['mode']})_")
        lines.append(f"> {case['topic']}")
        lines.append("")
        lines.append("| Metric | Score | Detail |")
        lines.append("|--------|-------|--------|")
        for s in case["scores"]:
            lines.append(f"| {s['name']} | {s['value']:.3f} | {s['detail']} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the AV Research Agent.")
    parser.add_argument("--live", action="store_true",
                        help="Regenerate reports with the Editor agent (needs API key).")
    args = parser.parse_args()

    load_dotenv()
    has_creds = has_llm_credentials()

    if args.live and not has_creds:
        print("[eval] --live requested but no API key for the selected provider; "
              "falling back to fixture reports (deterministic metrics only).")

    case_results = [_score_case(c, has_creds, args.live) for c in DATASET]
    aggregate = _aggregate(case_results)

    report_obj = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "llm_judge": has_creds,
        "live": bool(args.live and has_creds),
        "provider": provider_label() if has_creds else None,
        "aggregate": aggregate,
        "cases": case_results,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f"scorecard_{stamp}.json"
    md_path = RESULTS_DIR / f"scorecard_{stamp}.md"
    json_path.write_text(json.dumps(report_obj, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(report_obj), encoding="utf-8")

    # Console summary.
    print("\n=== AV Research Agent — Eval Summary ===")
    print(f"cases={len(case_results)}  llm_judge={'on' if has_creds else 'off'}  "
          f"mode={'live' if report_obj['live'] else 'fixture'}")
    for name, val in aggregate.items():
        print(f"  {name:<20} {val:.3f}")
    print(f"\nScorecard written to:\n  {json_path}\n  {md_path}")


if __name__ == "__main__":
    main()
