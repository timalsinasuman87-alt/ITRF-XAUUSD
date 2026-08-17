from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from itrf_competing_risk import (
    BASELINE_FEATURES,
    CLASSES,
    DYNAMIC_FEATURES,
    EmpiricalHoldingHazard,
    FixedMultinomialModel,
    build_risk_set_rows,
    competing_risk_metrics,
    walk_forward_hazard_predictions,
)


def test_start_of_bar_path_features_use_only_prior_completed_bar() -> None:
    frame = pd.DataFrame({"close": [100.0, 101.0, 99.0]})
    panel = pd.DataFrame(
        {
            "event_id": [0, 0, 0],
            "bar_index": [0, 1, 2],
            "holding_bar": [1, 2, 3],
            "direction": ["LONG"] * 3,
            "entry": [100.0] * 3,
            "risk": [2.0] * 3,
            "bar_favorable_r": [0.5, 1.0, 0.2],
            "bar_adverse_r": [-0.2, -0.4, -1.0],
            "running_mfe_r": [0.5, 1.0, 1.0],
            "running_mae_r": [-0.2, -0.4, -1.0],
            "terminal_event": ["NONE", "NONE", "STOP"],
        }
    )
    events = pd.DataFrame({
        "event_id": [0], "signal_index": [0], "exit_index": [2], "ambiguous": [0],
        **{feature: [0.0] for feature in BASELINE_FEATURES},
    })
    rows = build_risk_set_rows(frame, panel, events)
    assert rows.loc[0, ["prior_mfe_r", "prior_mae_r", "prior_close_r"]].tolist() == [0.0, 0.0, 0.0]
    assert rows.loc[1, "prior_mfe_r"] == 0.5
    assert rows.loc[1, "prior_close_r"] == 0.0
    assert rows.loc[2, "prior_mfe_r"] == 1.0
    assert rows.loc[2, "prior_close_r"] == 0.5
    assert rows["outcome"].tolist() == ["NONE", "NONE", "STOP"]


def test_short_prior_close_is_directional() -> None:
    frame = pd.DataFrame({"close": [98.0, 101.0]})
    panel = pd.DataFrame({
        "event_id": [0, 0], "bar_index": [0, 1], "holding_bar": [1, 2],
        "direction": ["SHORT", "SHORT"], "entry": [100.0, 100.0], "risk": [2.0, 2.0],
        "bar_favorable_r": [1.0, 0.0], "bar_adverse_r": [0.0, -0.5],
        "running_mfe_r": [1.0, 1.0], "running_mae_r": [0.0, -0.5],
        "terminal_event": ["NONE", "STOP"],
    })
    events = pd.DataFrame({"event_id": [0], "signal_index": [0], "exit_index": [1], "ambiguous": [0], **{f: [0.0] for f in BASELINE_FEATURES}})
    rows = build_risk_set_rows(frame, panel, events)
    assert rows.loc[1, "prior_close_r"] == 1.0


def test_empirical_hazard_uses_fixed_jeffreys_smoothing() -> None:
    train = pd.DataFrame({"holding_bar": [1, 1, 1], "outcome": ["NONE", "STOP", "STOP"]})
    model = EmpiricalHoldingHazard().fit(train)
    probability = model.predict_probability(pd.DataFrame({"holding_bar": [1]}))[0]
    expected = np.array([1.5, 2.5, 0.5]) / 4.5
    assert probability == pytest.approx(expected)
    assert probability.sum() == pytest.approx(1.0)


def test_multinomial_solver_is_finite_and_normalized() -> None:
    rows = 60
    features = pd.DataFrame({feature: np.sin(np.arange(rows) * (index + 1)) for index, feature in enumerate(DYNAMIC_FEATURES)})
    target = pd.Series(np.resize(np.array(CLASSES), rows))
    model = FixedMultinomialModel().fit(features, target)
    probability = model.predict_probability(features)
    assert model.converged
    assert np.isfinite(model.coefficients).all()
    assert np.isfinite(probability).all()
    assert probability.sum(axis=1) == pytest.approx(np.ones(rows))
    assert ((probability > 0) & (probability < 1)).all()


def test_competing_metrics_are_zero_for_perfect_probabilities() -> None:
    predictions = pd.DataFrame(
        {
            "event_id": [0, 1, 2],
            "outcome": list(CLASSES),
            "prob_none": [1.0, 0.0, 0.0],
            "prob_stop": [0.0, 1.0, 0.0],
            "prob_target": [0.0, 0.0, 1.0],
        }
    )
    metrics = competing_risk_metrics(predictions)
    assert metrics["multiclass_brier"] == 0.0
    assert metrics["event_balanced_brier"] == 0.0
    assert metrics["log_loss"] == 0.0


def test_walk_forward_keeps_events_intact_and_predictions_aligned() -> None:
    event_count = 30
    events = pd.DataFrame(
        {
            "event_id": np.arange(event_count),
            "signal_index": np.arange(event_count) * 10,
            "exit_index": np.arange(event_count) * 10 + 3,
            "ambiguous": 0,
        }
    )
    risk_records = []
    for event_id in range(event_count):
        terminal = "TARGET" if event_id % 10 == 0 else "STOP" if event_id % 3 == 0 else "NONE"
        for holding_bar in (1, 2):
            record = {
                "event_id": event_id,
                "holding_bar": holding_bar,
                "outcome": terminal if holding_bar == 2 else "NONE",
            }
            for feature_number, feature in enumerate(DYNAMIC_FEATURES):
                record[feature] = np.sin(event_id + holding_bar * (feature_number + 1))
            risk_records.append(record)
    predictions, audits = walk_forward_hazard_predictions(pd.DataFrame(risk_records), events)
    assert (audits["realized_embargo_bars"] >= 32).all()
    assert predictions.duplicated(["model", "event_id", "holding_bar"]).sum() == 0
    model_counts = predictions.groupby("model").size()
    assert model_counts.nunique() == 1
    probabilities = predictions[["prob_none", "prob_stop", "prob_target"]]
    assert np.isfinite(probabilities).all().all()
    assert probabilities.sum(axis=1).to_numpy() == pytest.approx(np.ones(len(probabilities)))
