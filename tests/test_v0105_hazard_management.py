from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from itrf_hazard_management import (
    build_paired_management_outcomes,
    paired_improvement_interval,
    paired_policy_metrics,
)


def fixtures(direction: str = "LONG", ambiguous: int = 0):
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=4, freq="15min"),
            "open": [100.0, 101.0, 102.0, 103.0],
        }
    )
    baseline = -1.0 if not ambiguous else -1.0
    ledger = pd.DataFrame(
        {
            "decision": ["ACCEPTED"],
            "direction": [direction],
            "entry": [100.0],
            "risk": [2.0],
            "exit_index": [3],
            "exit_time": [frame.loc[3, "time"]],
            "exit_reason": ["STOP" if not ambiguous else "AMBIGUOUS_STOP_TARGET"],
            "bars_held": [3],
            "gross_r_lower": [baseline],
            "gross_r_upper": [3.0 if ambiguous else baseline],
            "ambiguous": [ambiguous],
        }
    )
    panel = pd.DataFrame(
        {"event_id": [0, 0, 0], "holding_bar": [1, 2, 3], "bar_index": [1, 2, 3]}
    )
    predictions = pd.DataFrame(
        {
            "model": ["dynamic_path"] * 3,
            "event_id": [0, 0, 0],
            "holding_bar": [1, 2, 3],
            "outcome": ["NONE", "NONE", "STOP"],
            "prob_none": [0.60, 0.75, 0.60],
            "prob_stop": [0.30, 0.20, 0.30],
            "prob_target": [0.10, 0.05, 0.10],
        }
    )
    return frame, ledger, panel, predictions


def test_policy_exits_at_earliest_post_entry_qualifying_open() -> None:
    frame, ledger, panel, predictions = fixtures()
    outcomes = build_paired_management_outcomes(frame, ledger, panel, predictions)
    assert outcomes.loc[0, "managed_action"] == 1
    assert outcomes.loc[0, "action_holding_bar"] == 2
    assert outcomes.loc[0, "managed_exit_index"] == 2
    assert outcomes.loc[0, "managed_gross_r"] == 1.0
    assert outcomes.loc[0, "paired_improvement_r"] == 2.0


def test_bar_one_trigger_is_not_an_entry_filter() -> None:
    frame, ledger, panel, predictions = fixtures()
    predictions.loc[:, ["prob_stop", "prob_target"]] = [[0.50, 0.10], [0.10, 0.10], [0.10, 0.10]]
    predictions["prob_none"] = 1.0 - predictions["prob_stop"] - predictions["prob_target"]
    outcomes = build_paired_management_outcomes(frame, ledger, panel, predictions)
    assert outcomes.loc[0, "managed_action"] == 0
    assert outcomes.loc[0, "managed_gross_r"] == -1.0


def test_short_managed_r_is_directional() -> None:
    frame, ledger, panel, predictions = fixtures(direction="SHORT")
    outcomes = build_paired_management_outcomes(frame, ledger, panel, predictions)
    assert outcomes.loc[0, "managed_gross_r"] == -1.0


def test_ambiguous_events_are_excluded() -> None:
    frame, ledger, panel, predictions = fixtures(ambiguous=1)
    with pytest.raises(ValueError, match="non-empty"):
        build_paired_management_outcomes(frame, ledger, panel, predictions)


def test_paired_metrics_apply_equal_cost_and_bootstrap_delta() -> None:
    outcomes = pd.DataFrame(
        {
            "baseline_gross_r": [-1.0, 3.0, 0.0],
            "managed_gross_r": [0.0, 2.0, 0.5],
            "paired_improvement_r": [1.0, -1.0, 0.5],
        }
    )
    metrics = paired_policy_metrics(outcomes, 0.10)
    assert metrics["baseline_average_r"] == pytest.approx(2.0 / 3.0 - 0.10)
    assert metrics["managed_average_r"] == pytest.approx(2.5 / 3.0 - 0.10)
    assert metrics["paired_improvement_r"] == pytest.approx(1.0 / 6.0)
    mean, low, high = paired_improvement_interval(outcomes)
    assert mean == pytest.approx(1.0 / 6.0)
    assert low <= mean <= high
