"""Run the pre-registered v0.10.5 hazard-management policy screen."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from itrf_competing_risk import build_risk_set_rows, external_hazard_predictions
from itrf_context import create_context_features
from itrf_event_research import build_event_dataset
from itrf_failure_anatomy import build_lifecycle_panel
from itrf_hazard_management import (
    build_paired_management_outcomes,
    paired_improvement_interval,
    paired_policy_metrics,
)
from itrf_research import create_features
from run_v010_clean_baseline import FROZEN_POLICY, build_candidate_ledger
from run_v0104_external_transport import (
    DEFAULT_DEVELOPMENT_FILE,
    DEFAULT_FULL_FILE,
    load_locked_segments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTCOMES_FILE = PROJECT_ROOT / "data" / "processed" / "v0105_paired_management.csv"


def build_components(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    context = create_context_features(create_features(frame))
    ledger = build_candidate_ledger(context)
    events = build_event_dataset(context, ledger)
    panel = build_lifecycle_panel(context, ledger)
    rows = build_risk_set_rows(context, panel, events)
    return ledger, panel, rows


def print_report(outcomes: pd.DataFrame) -> None:
    print("\nITRF v0.10.5 HAZARD-MANAGEMENT POLICY SCREEN")
    print("Previously inspected historical sample; not a profitability or deployment claim.")
    actions = outcomes.loc[outcomes["managed_action"] == 1, "action_holding_bar"]
    print(f"\nPaired eligible events: {len(outcomes)}")
    print(f"Managed defensive exits: {len(actions)} ({100.0 * len(actions) / len(outcomes):.2f}%)")
    if not actions.empty:
        print(f"Action holding bar: mean={actions.mean():.2f}, median={actions.median():.2f}")

    metrics = pd.DataFrame([paired_policy_metrics(outcomes, cost) for cost in (0.0, 0.05, 0.10)])
    print("\nPaired aggregate policy comparison:")
    print(metrics.round(5).to_string(index=False))
    mean, low, high = paired_improvement_interval(outcomes)
    print(
        "\nMean managed-minus-baseline R: "
        f"{mean:.5f}, circular block-bootstrap 95%=[{low:.5f}, {high:.5f}]"
    )
    cost_metrics = metrics.set_index("cost_r")
    passed = bool(
        mean > 0
        and low > 0
        and all(
            cost_metrics.loc[cost, "managed_profit_factor_r"]
            > cost_metrics.loc[cost, "baseline_profit_factor_r"]
            for cost in (0.05, 0.10)
        )
        and all(
            cost_metrics.loc[cost, "managed_maximum_drawdown_r"]
            >= cost_metrics.loc[cost, "baseline_maximum_drawdown_r"]
            for cost in (0.05, 0.10)
        )
    )
    print(f"\nPre-registered exploratory decision: {'PASS' if passed else 'FAIL'}")
    print("A pass can freeze the rule for future validation only; it cannot authorize trading.")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ITRF v0.10.5 hazard management.")
    parser.add_argument("--development-file", type=Path, default=DEFAULT_DEVELOPMENT_FILE)
    parser.add_argument("--full-file", type=Path, default=DEFAULT_FULL_FILE)
    parser.add_argument("--outcomes-file", type=Path, default=DEFAULT_OUTCOMES_FILE)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    development, external = load_locked_segments(arguments.development_file, arguments.full_file)
    _, _, development_rows = build_components(development)
    external_ledger, external_panel, external_rows = build_components(external)
    predictions, _ = external_hazard_predictions(development_rows, external_rows)
    outcomes = build_paired_management_outcomes(
        external,
        external_ledger,
        external_panel,
        predictions,
        target_r=FROZEN_POLICY.target_r,
    )
    arguments.outcomes_file.parent.mkdir(parents=True, exist_ok=True)
    outcomes.to_csv(arguments.outcomes_file, index=False)
    print_report(outcomes)
    print(f"\nPaired management outcomes: {arguments.outcomes_file}")


if __name__ == "__main__":
    main()
