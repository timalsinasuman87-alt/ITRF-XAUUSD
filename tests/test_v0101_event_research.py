from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from itrf_event_research import (
    BASELINE_FEATURES,
    CONTEXT_FEATURES,
    FixedLogisticModel,
    WalkForwardConfig,
    block_bootstrap_mean_ci,
    probability_metrics,
    purged_walk_forward_folds,
    walk_forward_predictions,
)


def event_intervals(count: int = 20) -> pd.DataFrame:
    signal = np.arange(count) * 10
    return pd.DataFrame({"signal_index": signal, "exit_index": signal + 5})


def test_walk_forward_is_past_only_and_enforces_embargo() -> None:
    events = event_intervals()
    config = WalkForwardConfig(folds=3, initial_train_fraction=0.4, embargo_bars=12)
    folds = purged_walk_forward_folds(events, config)
    assert len(folds) == 3
    for fold in folds:
        assert fold.train_positions.max() < fold.test_positions.min()
        last_train_exit = events.iloc[fold.train_positions]["exit_index"].max()
        first_test_signal = events.iloc[fold.test_positions]["signal_index"].min()
        assert last_train_exit < first_test_signal - config.embargo_bars


def test_overlapping_training_label_is_purged() -> None:
    events = event_intervals()
    events.loc[7, "exit_index"] = 100
    fold = purged_walk_forward_folds(
        events, WalkForwardConfig(folds=3, initial_train_fraction=0.4, embargo_bars=0)
    )[0]
    assert 7 not in fold.train_positions


def test_non_chronological_events_are_rejected() -> None:
    events = event_intervals()
    events.loc[2, "signal_index"] = -1
    with pytest.raises(ValueError, match="chronological"):
        purged_walk_forward_folds(events)


def test_logistic_preprocessing_is_fitted_from_training_only() -> None:
    train = pd.DataFrame({"x": [0.0, 1.0, np.nan, 2.0]})
    target = pd.Series([0, 0, 1, 1])
    model = FixedLogisticModel().fit(train, target)
    training_mean = model.means.copy()
    probabilities = model.predict_probability(pd.DataFrame({"x": [1_000_000.0, np.nan]}))
    assert np.array_equal(model.means, training_mean)
    assert probabilities.shape == (2,)
    assert np.isfinite(probabilities).all()


def test_logistic_solver_remains_finite_under_near_separation() -> None:
    features = pd.DataFrame(
        {
            "rare": [0.0] * 18 + [1.0, 1.0],
            "large": [-1e6, 1e6] * 10,
        }
    )
    target = pd.Series([0] * 18 + [1, 1])
    model = FixedLogisticModel().fit(features, target)
    probabilities = model.predict_probability(features)
    assert model.converged
    assert np.isfinite(model.coefficients).all()
    assert np.isfinite(probabilities).all()
    assert ((probabilities > 0) & (probabilities < 1)).all()


def test_walk_forward_predictions_remain_finite() -> None:
    count = 40
    events = event_intervals(count)
    events["event_id"] = np.arange(count)
    events["target_success"] = (np.arange(count) % 7 == 0).astype(int)
    events["gross_r_lower"] = np.where(events["target_success"] == 1, 3.0, -1.0)
    events["ambiguous"] = 0
    for feature_number, feature in enumerate(CONTEXT_FEATURES):
        events[feature] = (
            np.sin(np.arange(count) * (feature_number + 1))
            if feature not in BASELINE_FEATURES or feature.startswith("signed_")
            else np.arange(count) % 2
        )
    predictions, folds = walk_forward_predictions(
        events,
        WalkForwardConfig(folds=3, initial_train_fraction=0.4, embargo_bars=2),
    )
    assert len(folds) == 3
    assert predictions["probability"].between(0.0, 1.0, inclusive="neither").all()
    assert np.isfinite(predictions["probability"]).all()


def test_probability_metrics_match_perfect_predictions() -> None:
    metrics = probability_metrics(pd.Series([0, 1]), pd.Series([0.0, 1.0]))
    assert metrics["brier_score"] < 1e-12
    assert metrics["roc_auc"] == 1.0


def test_bootstrap_interval_is_deterministic() -> None:
    values = pd.Series([0.1, -0.1, 0.2, 0.0, 0.3])
    assert block_bootstrap_mean_ci(values, samples=100) == block_bootstrap_mean_ci(
        values, samples=100
    )
