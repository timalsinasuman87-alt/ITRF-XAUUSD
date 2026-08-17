"""Run the pre-registered v0.10.1 purged event-research comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from itrf_context import create_context_features
from itrf_event_research import (
    block_bootstrap_mean_ci,
    build_event_dataset,
    probability_metrics,
    walk_forward_predictions,
)
from itrf_research import create_features, load_market_data
from run_v010_clean_baseline import build_candidate_ledger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_FILE = PROJECT_ROOT / "data" / "XAUUSD.csv"
DEFAULT_EVENTS_FILE = PROJECT_ROOT / "data" / "processed" / "v0101_events.csv"
DEFAULT_PREDICTIONS_FILE = PROJECT_ROOT / "data" / "processed" / "v0101_predictions.csv"


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fold, model), group in predictions.groupby(["fold", "model"], sort=False):
        rows.append({"fold": fold, "model": model, **probability_metrics(group["target_success"], group["probability"])})
    return pd.DataFrame(rows)


def paired_brier_improvement(predictions: pd.DataFrame, model: str) -> pd.Series:
    pivot = predictions.pivot(index="event_id", columns="model", values=["target_success", "probability"])
    target = pivot[("target_success", "constant_null")]
    null_error = (pivot[("probability", "constant_null")] - target) ** 2
    model_error = (pivot[("probability", model)] - target) ** 2
    return null_error - model_error


def print_report(events: pd.DataFrame, predictions: pd.DataFrame, folds: pd.DataFrame) -> None:
    print("\nITRF v0.10.1 PURGED EVENT RESEARCH — FRAMEWORK DIAGNOSTIC")
    print("Already-inspected development data; no probability threshold may be selected.")
    print(f"\nSequential events: {len(events)}")
    print(f"3R target successes: {int(events['target_success'].sum())} ({100 * events['target_success'].mean():.2f}%)")
    print("\nFold integrity:")
    print(folds.to_string(index=False))
    metrics = summarize_predictions(predictions)
    print("\nFold probability metrics:")
    print(metrics.round(4).to_string(index=False))
    pooled_rows = []
    for model, group in predictions.groupby("model", sort=False):
        pooled_rows.append({"model": model, **probability_metrics(group["target_success"], group["probability"])})
    pooled = pd.DataFrame(pooled_rows)
    print("\nPooled walk-forward metrics:")
    print(pooled.round(4).to_string(index=False))
    print("\nPaired Brier improvement versus constant null:")
    for model in ("price_activity", "causal_context"):
        improvement = paired_brier_improvement(predictions, model)
        low, high = block_bootstrap_mean_ci(improvement)
        print(f"{model}: mean={improvement.mean():.5f}, block-bootstrap 95%=[{low:.5f}, {high:.5f}]")
    print("\nPositive improvement means lower Brier error than the null.")
    print("This inspected sample can reject or refine a hypothesis, but cannot prove an edge.")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ITRF v0.10.1 event research.")
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--events-file", type=Path, default=DEFAULT_EVENTS_FILE)
    parser.add_argument("--predictions-file", type=Path, default=DEFAULT_PREDICTIONS_FILE)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    features = create_features(load_market_data(arguments.data_file))
    context = create_context_features(features)
    ledger = build_candidate_ledger(context)
    events = build_event_dataset(context, ledger)
    predictions, folds = walk_forward_predictions(events)
    arguments.events_file.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(arguments.events_file, index=False)
    predictions.to_csv(arguments.predictions_file, index=False)
    print_report(events, predictions, folds)
    print(f"\nEvents: {arguments.events_file}")
    print(f"Predictions: {arguments.predictions_file}")


if __name__ == "__main__":
    main()
