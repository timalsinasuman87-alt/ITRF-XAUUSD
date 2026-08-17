from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from itrf_competing_risk import CLASSES, DYNAMIC_FEATURES, external_hazard_predictions


def make_rows(events: int, phase: float = 0.0) -> pd.DataFrame:
    records = []
    for event_id in range(events):
        terminal = "TARGET" if event_id % 7 == 0 else "STOP" if event_id % 3 == 0 else "NONE"
        for holding_bar in (1, 2, 3):
            record = {
                "event_id": event_id,
                "holding_bar": holding_bar,
                "outcome": terminal if holding_bar == 3 else "NONE",
            }
            for number, feature in enumerate(DYNAMIC_FEATURES):
                record[feature] = np.sin(event_id + phase + holding_bar * (number + 1))
            records.append(record)
    return pd.DataFrame(records)


def test_external_transport_is_aligned_and_normalized() -> None:
    development = make_rows(60)
    external = make_rows(25, phase=0.5)
    predictions, audit = external_hazard_predictions(development, external)
    assert audit.loc[0, "development_events"] == 60
    assert audit.loc[0, "external_events"] == 25
    assert predictions.duplicated(["model", "event_id", "holding_bar"]).sum() == 0
    assert predictions.groupby("model").size().nunique() == 1
    probabilities = predictions[["prob_none", "prob_stop", "prob_target"]]
    assert np.isfinite(probabilities).all().all()
    assert probabilities.sum(axis=1).to_numpy() == pytest.approx(np.ones(len(probabilities)))


def test_external_outcomes_never_fit_or_change_transport_predictions() -> None:
    development = make_rows(60)
    external = make_rows(25, phase=0.5)
    changed = external.copy()
    changed["outcome"] = np.resize(np.array(CLASSES), len(changed))
    original_predictions, _ = external_hazard_predictions(development, external)
    changed_predictions, _ = external_hazard_predictions(development, changed)
    probability_columns = ["prob_none", "prob_stop", "prob_target"]
    assert changed_predictions[probability_columns].to_numpy() == pytest.approx(
        original_predictions[probability_columns].to_numpy()
    )


def test_external_transport_rejects_missing_or_empty_rows() -> None:
    development = make_rows(60)
    external = make_rows(25)
    with pytest.raises(ValueError, match="non-empty"):
        external_hazard_predictions(development, external.iloc[0:0])
    with pytest.raises(ValueError, match="missing external"):
        external_hazard_predictions(development, external.drop(columns=[DYNAMIC_FEATURES[-1]]))
