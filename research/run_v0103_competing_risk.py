"""Run the pre-registered v0.10.3 competing-risk calibration study."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from itrf_competing_risk import (
    build_risk_set_rows,
    competing_risk_metrics,
    improvement_interval,
    walk_forward_hazard_predictions,
)
from itrf_context import create_context_features
from itrf_event_research import build_event_dataset
from itrf_failure_anatomy import build_lifecycle_panel
from itrf_research import create_features, load_market_data
from run_v010_clean_baseline import build_candidate_ledger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_FILE = PROJECT_ROOT / "data" / "XAUUSD.csv"
DEFAULT_ROWS_FILE = PROJECT_ROOT / "data" / "processed" / "v0103_risk_rows.csv"
DEFAULT_PREDICTIONS_FILE = PROJECT_ROOT / "data" / "processed" / "v0103_hazard_predictions.csv"


def print_report(rows: pd.DataFrame, predictions: pd.DataFrame, audits: pd.DataFrame) -> None:
    print("\nITRF v0.10.3 COMPETING-RISK CALIBRATION")
    print("Already-inspected development data; probabilities are not management actions.")
    print(f"\nRisk-set rows: {len(rows)} across {rows['event_id'].nunique()} events")
    print("\nFold integrity:")
    print(audits.to_string(index=False))
    metrics = []
    for model, group in predictions.groupby("model", sort=False):
        metrics.append({"model": model, **competing_risk_metrics(group)})
    print("\nPooled walk-forward calibration:")
    print(pd.DataFrame(metrics).round(5).to_string(index=False))
    print("\nEvent-balanced Brier improvement versus empirical holding-bar hazard:")
    for model in ("preentry_time", "dynamic_path"):
        mean, low, high = improvement_interval(predictions, model)
        print(f"{model}: mean={mean:.5f}, block-bootstrap 95%=[{low:.5f}, {high:.5f}]")
    print("\nPositive improvement means lower event-balanced Brier error.")
    print("No probability threshold, exit rule, or profitability claim follows from this report.")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ITRF v0.10.3 competing-risk study.")
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--rows-file", type=Path, default=DEFAULT_ROWS_FILE)
    parser.add_argument("--predictions-file", type=Path, default=DEFAULT_PREDICTIONS_FILE)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    frame = create_features(load_market_data(arguments.data_file))
    context = create_context_features(frame)
    ledger = build_candidate_ledger(context)
    events = build_event_dataset(context, ledger)
    panel = build_lifecycle_panel(context, ledger)
    rows = build_risk_set_rows(context, panel, events)
    predictions, audits = walk_forward_hazard_predictions(rows, events)
    arguments.rows_file.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(arguments.rows_file, index=False)
    predictions.to_csv(arguments.predictions_file, index=False)
    print_report(rows, predictions, audits)
    print(f"\nRisk rows: {arguments.rows_file}")
    print(f"Predictions: {arguments.predictions_file}")


if __name__ == "__main__":
    main()
