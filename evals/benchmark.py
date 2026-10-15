"""
Provider/model benchmark — run the eval across several LLM backends and
tabulate the scores side by side.

This answers the interview question "how does Claude compare to GPT on your
task?" with numbers from your own eval harness, not vibes.

WARNING: each combo runs the full live eval and spends real API credits. Keep
the combo list short. A combo whose provider has no API key is skipped.

Usage (from repo root):
    python -m evals.benchmark --combos anthropic:claude-haiku-5-5 openai:gpt-4o-mini
"""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from evals.dataset import DATASET
from evals.run_eval import _aggregate, _score_case


def _apply_combo(provider: str, model: str) -> None:
    os.environ["LLM_PROVIDER"] = provider
    if provider == "anthropic":
        os.environ["ANTHROPIC_MODEL"] = model
    else:
        os.environ["OPENAI_MODEL"] = model


def _has_key(provider: str) -> bool:
    key = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    return bool(os.getenv(key))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the eval across providers.")
    parser.add_argument(
        "--combos", nargs="+", default=["anthropic:claude-haiku-5-5"],
        help="provider:model pairs, e.g. anthropic:claude-haiku-5-5 openai:gpt-4o-mini",
    )
    args = parser.parse_args()
    load_dotenv()

    results: dict[str, dict] = {}
    metric_order: list[str] = []

    for combo in args.combos:
        provider, _, model = combo.partition(":")
        if not _has_key(provider):
            print(f"[benchmark] skipping {combo}: no API key for '{provider}'")
            continue

        print(f"[benchmark] running {combo} ...")
        _apply_combo(provider, model)
        case_results = [_score_case(c, has_creds=True, live=True) for c in DATASET]
        agg = _aggregate(case_results)
        results[combo] = agg
        for name in agg:
            if name not in metric_order:
                metric_order.append(name)

    if not results:
        print("[benchmark] nothing ran — no funded providers among the combos.")
        return

    # Print a comparison table.
    col_w = max((len(c) for c in results), default=10)
    header = "metric".ljust(22) + "".join(c.ljust(col_w + 3) for c in results)
    print("\n" + header)
    print("-" * len(header))
    for metric in metric_order:
        row = metric.ljust(22)
        for combo in results:
            val = results[combo].get(metric)
            row += (f"{val:.3f}" if val is not None else "  -  ").ljust(col_w + 3)
        print(row)


if __name__ == "__main__":
    main()
