"""Run the pre-registered v0.10.6 continuation-value policy screen."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from itrf_competing_risk import (
    external_hazard_predictions,
    walk_forward_hazard_predictions,
)
from itrf_context import create_context_features
from itrf_continuation_value import (
    DECISION_FEATURES,
    FixedWeightedRidge,
    attach_continuation_labels,
    build_causal_decision_rows,
    build_continuation_policy_outcomes,
    event_balanced_row_weights,
)
from itrf_event_research import build_event_dataset
from itrf_hazard_management import paired_improvement_interval, paired_policy_metrics
from itrf_research import create_features
from run_v0104_external_transport import (
    DEFAULT_DEVELOPMENT_FILE,
    DEFAULT_FULL_FILE,
    load_locked_segments,
)
from run_v0105_hazard_management import build_components


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_FILE = PROJECT_ROOT / "data" / "processed" / "v0106_value_training_rows.csv"
DEFAULT_DECISIONS_FILE = PROJECT_ROOT / "data" / "processed" / "v0106_external_decisions.csv"
DEFAULT_OUTCOMES_FILE = PROJECT_ROOT / "data" / "processed" / "v0106_paired_management.csv"


def print_report(training: pd.DataFrame, decisions: pd.DataFrame, outcomes: pd.DataFrame) -> None:
    print("\nITRF v0.10.6 CONTINUATION-VALUE POLICY SCREEN")
    print("Previously inspected historical sample; not a profitability or deployment claim.")
    print(
        f"\nOut-of-fold development value training: {len(training)} rows "
        f"across {training['event_id'].nunique()} events"
    )
    print(
        f"External causal decisions: {len(decisions)} rows "
        f"across {decisions['event_id'].nunique()} events"
    )
    actions = outcomes.loc[outcomes["managed_action"] == 1, "action_holding_bar"]
    print(f"Paired eligible events: {len(outcomes)}")
    print(f"Managed continuation exits: {len(actions)} ({100.0 * len(actions) / len(outcomes):.2f}%)")
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
    print("A pass can freeze the complete framework for future validation only.")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ITRF v0.10.6 continuation value.")
    parser.add_argument("--development-file", type=Path, default=DEFAULT_DEVELOPMENT_FILE)
    parser.add_argument("--full-file", type=Path, default=DEFAULT_FULL_FILE)
    parser.add_argument("--training-file", type=Path, default=DEFAULT_TRAINING_FILE)
    parser.add_argument("--decisions-file", type=Path, default=DEFAULT_DECISIONS_FILE)
    parser.add_argument("--outcomes-file", type=Path, default=DEFAULT_OUTCOMES_FILE)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    development, external = load_locked_segments(arguments.development_file, arguments.full_file)
    development_ledger, development_panel, development_rows = build_components(development)
    external_ledger, external_panel, external_rows = build_components(external)

    # build_components intentionally returns the ledger, panel and risk rows;
    # the accepted event dataset is reconstructed from the same causal context
    # to keep the frozen fold definitions unchanged.
    development_context = create_context_features(create_features(development))
    development_events = build_event_dataset(development_context, development_ledger)
    development_hazards, _ = walk_forward_hazard_predictions(
        development_rows,
        development_events,
    )
    training = build_causal_decision_rows(
        development,
        development_panel,
        development_rows,
        development_hazards,
    )
    training = attach_continuation_labels(training, development_ledger)
    weights = event_balanced_row_weights(training)
    value_model = FixedWeightedRidge(l2_strength=1.0).fit(
        training.loc[:, DECISION_FEATURES],
        training["continuation_advantage_r"],
        weights,
    )

    external_hazards, _ = external_hazard_predictions(development_rows, external_rows)
    decisions = build_causal_decision_rows(
        external,
        external_panel,
        external_rows,
        external_hazards,
    )
    predicted_advantage = value_model.predict(decisions.loc[:, DECISION_FEATURES])
    decisions["predicted_continuation_advantage_r"] = predicted_advantage
    outcomes = build_continuation_policy_outcomes(
        external_ledger,
        decisions,
        predicted_advantage,
    )

    for output in (arguments.training_file, arguments.decisions_file, arguments.outcomes_file):
        output.parent.mkdir(parents=True, exist_ok=True)
    training.to_csv(arguments.training_file, index=False)
    decisions.to_csv(arguments.decisions_file, index=False)
    outcomes.to_csv(arguments.outcomes_file, index=False)
    print_report(training, decisions, outcomes)
    print(f"\nValue training rows: {arguments.training_file}")
    print(f"External decisions: {arguments.decisions_file}")
    print(f"Paired outcomes: {arguments.outcomes_file}")


if __name__ == "__main__":
    main()
