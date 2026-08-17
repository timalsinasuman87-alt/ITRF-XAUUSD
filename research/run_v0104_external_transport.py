"""Run the pre-registered v0.10.4 backward external-transport test."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from itrf_competing_risk import (
    PROBABILITY_COLUMNS,
    build_risk_set_rows,
    competing_risk_metrics,
    external_hazard_predictions,
    improvement_interval,
)
from itrf_context import create_context_features
from itrf_event_research import build_event_dataset
from itrf_failure_anatomy import build_lifecycle_panel
from itrf_research import create_features, load_market_data
from run_v010_clean_baseline import build_candidate_ledger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEVELOPMENT_FILE = PROJECT_ROOT / "data" / "XAUUSD.csv"
DEFAULT_FULL_FILE = PROJECT_ROOT / "data" / "XAUUSD_2025_2026.csv"
DEFAULT_EXTERNAL_ROWS_FILE = PROJECT_ROOT / "data" / "processed" / "v0104_external_risk_rows.csv"
DEFAULT_PREDICTIONS_FILE = PROJECT_ROOT / "data" / "processed" / "v0104_external_predictions.csv"
LOCKED_FULL_SHA256 = "014b6e90ca0e5731b2fd3b958adb7c772d2adf974cf96e9d6cd424964779d058"
EXTERNAL_START = pd.Timestamp("2025-01-01 18:00:00")
EXTERNAL_END = pd.Timestamp("2026-04-30 23:45:00")
DEVELOPMENT_START = pd.Timestamp("2026-05-01 00:00:00")
DEVELOPMENT_END = pd.Timestamp("2026-08-07 16:45:00")
LOCKED_EXTERNAL_ROWS = 31_330
LOCKED_DEVELOPMENT_ROWS = 6_466


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_locked_segments(development_file: Path, full_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate the locked file and return independent raw external/development segments."""
    if sha256(full_file) != LOCKED_FULL_SHA256:
        raise ValueError("full XAU/USD file hash differs from the pre-registered value")
    development = load_market_data(development_file)
    full = load_market_data(full_file)
    external = full.loc[
        (full["time"] >= EXTERNAL_START) & (full["time"] <= EXTERNAL_END)
    ].reset_index(drop=True)
    locked_development = full.loc[
        (full["time"] >= DEVELOPMENT_START) & (full["time"] <= DEVELOPMENT_END)
    ].reset_index(drop=True)
    if len(external) != LOCKED_EXTERNAL_ROWS:
        raise ValueError("external XAU/USD row count differs from the pre-registered value")
    if len(development) != LOCKED_DEVELOPMENT_ROWS or len(locked_development) != LOCKED_DEVELOPMENT_ROWS:
        raise ValueError("development XAU/USD row count differs from the pre-registered value")
    pd.testing.assert_frame_equal(
        development.reset_index(drop=True),
        locked_development.reset_index(drop=True),
        check_exact=True,
    )
    if external["time"].max() >= development["time"].min():
        raise ValueError("external and development intervals overlap")
    return development, external


def build_segment_risk_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    context = create_context_features(create_features(frame))
    ledger = build_candidate_ledger(context)
    events = build_event_dataset(context, ledger)
    panel = build_lifecycle_panel(context, ledger)
    rows = build_risk_set_rows(context, panel, events)
    return rows, events


def print_report(
    development_rows: pd.DataFrame,
    external_rows: pd.DataFrame,
    predictions: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    print("\nITRF v0.10.4 BACKWARD EXTERNAL-TRANSPORT CALIBRATION")
    print("Historical XAU/USD only; this is not a future chronological holdout.")
    print("\nTransport integrity:")
    print(audit.to_string(index=False))
    print(f"Eligible development events: {development_rows['event_id'].nunique()}")
    print(f"Eligible external events: {external_rows['event_id'].nunique()}")
    metrics = []
    for model, group in predictions.groupby("model", sort=False):
        metrics.append({"model": model, **competing_risk_metrics(group)})
    metrics_frame = pd.DataFrame(metrics)
    print("\nLocked external calibration:")
    print(metrics_frame.round(5).to_string(index=False))
    print("\nEvent-balanced Brier improvement versus empirical holding-bar hazard:")
    decisions: dict[str, tuple[float, float, float]] = {}
    for model in ("preentry_time", "dynamic_path"):
        decisions[model] = improvement_interval(predictions, model)
        mean, low, high = decisions[model]
        print(f"{model}: mean={mean:.5f}, block-bootstrap 95%=[{low:.5f}, {high:.5f}]")
    empirical = metrics_frame.set_index("model").loc["empirical_hazard"]
    dynamic = metrics_frame.set_index("model").loc["dynamic_path"]
    dynamic_mean, dynamic_low, _ = decisions["dynamic_path"]
    passed = bool(
        dynamic_mean > 0
        and dynamic_low > 0
        and dynamic["log_loss"] < empirical["log_loss"]
        and np.isfinite(predictions.loc[:, PROBABILITY_COLUMNS]).all().all()
    )
    print(f"\nPre-registered transport decision: {'PASS' if passed else 'FAIL'}")
    print("A pass would support continued research only; it would not authorize a management rule.")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ITRF v0.10.4 external transport.")
    parser.add_argument("--development-file", type=Path, default=DEFAULT_DEVELOPMENT_FILE)
    parser.add_argument("--full-file", type=Path, default=DEFAULT_FULL_FILE)
    parser.add_argument("--external-rows-file", type=Path, default=DEFAULT_EXTERNAL_ROWS_FILE)
    parser.add_argument("--predictions-file", type=Path, default=DEFAULT_PREDICTIONS_FILE)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    development, external = load_locked_segments(arguments.development_file, arguments.full_file)
    development_rows, _ = build_segment_risk_rows(development)
    external_rows, _ = build_segment_risk_rows(external)
    predictions, audit = external_hazard_predictions(development_rows, external_rows)
    arguments.external_rows_file.parent.mkdir(parents=True, exist_ok=True)
    external_rows.to_csv(arguments.external_rows_file, index=False)
    predictions.to_csv(arguments.predictions_file, index=False)
    print_report(development_rows, external_rows, predictions, audit)
    print(f"\nExternal risk rows: {arguments.external_rows_file}")
    print(f"External predictions: {arguments.predictions_file}")


if __name__ == "__main__":
    main()
