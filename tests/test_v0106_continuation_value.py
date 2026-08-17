from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from itrf_competing_risk import DYNAMIC_FEATURES
from itrf_continuation_value import (
    DECISION_FEATURES,
    FixedWeightedRidge,
    attach_continuation_labels,
    build_causal_decision_rows,
    build_continuation_policy_outcomes,
    event_balanced_row_weights,
)


def test_event_balanced_weights_give_each_event_equal_total_weight() -> None:
    rows = pd.DataFrame({"event_id": [0, 0, 0, 1]})
    weights = event_balanced_row_weights(rows)
    totals = weights.groupby(rows["event_id"]).sum()
    assert totals.loc[0] == pytest.approx(totals.loc[1])
    assert weights.mean() == pytest.approx(1.0)


def test_fixed_weighted_ridge_recovers_linear_relation() -> None:
    x = pd.DataFrame({"x": np.linspace(-2, 2, 50)})
    y = pd.Series(1.5 + 2.0 * x["x"])
    weights = pd.Series(np.ones(len(x)))
    model = FixedWeightedRidge(l2_strength=0.0).fit(x, y, weights)
    assert model.predict(x) == pytest.approx(y.to_numpy(), abs=1e-9)


def test_fixed_weighted_ridge_handles_collinear_feature_with_locked_l2() -> None:
    x = pd.DataFrame({"x": np.linspace(-2, 2, 50), "constant": 1.0})
    y = pd.Series(1.5 + 2.0 * x["x"])
    weights = pd.Series(np.ones(len(x)))
    model = FixedWeightedRidge(l2_strength=1.0).fit(x, y, weights)
    prediction = model.predict(x)
    assert np.isfinite(model.coefficients).all()
    assert np.isfinite(prediction).all()


def causal_fixtures():
    frame = pd.DataFrame(
        {
            "open": [100.0, 100.0, 102.0],
            "time": pd.date_range("2025-01-01", periods=3, freq="15min"),
        }
    )
    panel = pd.DataFrame(
        {
            "event_id": [0, 0],
            "holding_bar": [1, 2],
            "bar_index": [1, 2],
            "direction": ["LONG", "LONG"],
            "entry": [100.0, 100.0],
            "risk": [2.0, 2.0],
        }
    )
    risk = pd.DataFrame(
        {
            "event_id": [0, 0],
            "holding_bar": [1, 2],
            **{feature: [0.0, 0.1] for feature in DYNAMIC_FEATURES},
        }
    )
    hazards = pd.DataFrame(
        {
            "model": ["dynamic_path", "dynamic_path"],
            "event_id": [0, 0],
            "holding_bar": [1, 2],
            "prob_none": [0.8, 0.7],
            "prob_stop": [0.1, 0.2],
            "prob_target": [0.1, 0.1],
        }
    )
    ledger = pd.DataFrame(
        {
            "decision": ["ACCEPTED"],
            "direction": ["LONG"],
            "exit_index": [2],
            "exit_time": [frame.loc[2, "time"]],
            "exit_reason": ["TARGET"],
            "bars_held": [2],
            "gross_r_lower": [3.0],
            "gross_r_upper": [3.0],
            "ambiguous": [0],
        }
    )
    return frame, panel, risk, hazards, ledger


def test_decision_rows_start_at_bar_two_and_use_current_open_r() -> None:
    frame, panel, risk, hazards, _ = causal_fixtures()
    rows = build_causal_decision_rows(frame, panel, risk, hazards)
    assert rows["holding_bar"].tolist() == [2]
    assert rows.loc[0, "exit_now_r"] == 1.0
    assert set(DECISION_FEATURES).issubset(rows.columns)


def test_continuation_label_is_terminal_minus_exit_now() -> None:
    frame, panel, risk, hazards, ledger = causal_fixtures()
    rows = build_causal_decision_rows(frame, panel, risk, hazards)
    labeled = attach_continuation_labels(rows, ledger)
    assert labeled.loc[0, "continuation_advantage_r"] == 2.0


def test_policy_exits_on_earliest_negative_predicted_advantage() -> None:
    frame, panel, risk, hazards, ledger = causal_fixtures()
    rows = build_causal_decision_rows(frame, panel, risk, hazards)
    outcomes = build_continuation_policy_outcomes(ledger, rows, np.array([-0.25]))
    assert outcomes.loc[0, "managed_action"] == 1
    assert outcomes.loc[0, "action_holding_bar"] == 2
    assert outcomes.loc[0, "managed_gross_r"] == 1.0
    assert outcomes.loc[0, "paired_improvement_r"] == -2.0


def test_policy_retains_baseline_when_value_is_nonnegative() -> None:
    frame, panel, risk, hazards, ledger = causal_fixtures()
    rows = build_causal_decision_rows(frame, panel, risk, hazards)
    outcomes = build_continuation_policy_outcomes(ledger, rows, np.array([0.0]))
    assert outcomes.loc[0, "managed_action"] == 0
    assert outcomes.loc[0, "managed_gross_r"] == 3.0


def test_one_bar_terminal_without_management_row_retains_baseline() -> None:
    _, _, _, _, ledger = causal_fixtures()
    ledger.loc[0, "bars_held"] = 1
    empty_rows = pd.DataFrame(
        columns=["event_id", "holding_bar", "predicted_continuation_advantage_r"]
    )
    outcomes = build_continuation_policy_outcomes(ledger, empty_rows, np.array([]))
    assert outcomes.loc[0, "managed_action"] == 0
    assert outcomes.loc[0, "managed_gross_r"] == 3.0
