from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

import itrf_futures_confirmation
from itrf_futures_confirmation import (
    build_futures_confirmed_ledger,
    chronological_half_means,
    classify_futures_gate,
    gated_cost_interval,
    merge_exact_futures_bars,
    sequential_metrics,
)


def test_exact_join_does_not_fill_nearby_futures_bar() -> None:
    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:15"]),
            "open": [100.0, 101.0],
        }
    )
    futures = pd.DataFrame(
        {
            "time": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:14"]),
            "open": [200.0, 201.0],
            "close": [201.0, 202.0],
        }
    )
    joined = merge_exact_futures_bars(frame, futures)
    assert joined["futures_bar_available"].tolist() == [1, 0]
    assert pd.isna(joined.loc[1, "gc_open"])


@pytest.mark.parametrize(
    ("direction", "gc_open", "gc_close", "available", "passed", "reason"),
    [
        ("LONG", 100.0, 101.0, 1, True, "PASS"),
        ("SHORT", 100.0, 99.0, 1, True, "PASS"),
        ("LONG", 100.0, 99.0, 1, False, "FUTURES_DISAGREEMENT"),
        ("SHORT", 100.0, 101.0, 1, False, "FUTURES_DISAGREEMENT"),
        ("LONG", 100.0, 100.0, 1, False, "ZERO_BODY_FUTURES"),
        ("LONG", np.nan, np.nan, 0, False, "MISSING_FUTURES_BAR"),
    ],
)
def test_gate_is_the_locked_directional_sign(
    direction, gc_open, gc_close, available, passed, reason
) -> None:
    row = pd.Series(
        {
            "gc_open": gc_open,
            "gc_close": gc_close,
            "futures_bar_available": available,
        }
    )
    assert classify_futures_gate(row, direction) == (passed, reason)


def _execution_frame(rows: int = 300) -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=rows, freq="15min"),
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "atr": 1.0,
            "gc_open": 200.0,
            "gc_close": 201.0,
            "futures_bar_available": 1,
        }
    )
    return data


def test_rejected_gate_does_not_block_next_passing_candidate(monkeypatch) -> None:
    data = _execution_frame()
    data.loc[250, "gc_close"] = 199.0
    data.loc[252, "high"] = 105.0
    monkeypatch.setattr(
        itrf_futures_confirmation,
        "detect_setup",
        lambda row: "LONG" if row.name in {250, 251} else "NONE",
    )
    ledger = build_futures_confirmed_ledger(data)
    assert ledger["decision"].tolist() == [
        "REJECTED_FUTURES_DISAGREEMENT",
        "ACCEPTED",
    ]
    assert ledger.loc[1, "entry_index"] == 252


def test_sequential_metrics_and_halves_use_only_accepted_lower_bound() -> None:
    ledger = pd.DataFrame(
        {
            "decision": ["ACCEPTED"] * 4,
            "signal_index": [1, 2, 3, 4],
            "gross_r_lower": [1.0, -1.0, 3.0, -1.0],
            "gross_r_upper": [1.0, -1.0, 3.0, -1.0],
            "ambiguous": [0, 0, 0, 0],
        }
    )
    metrics = sequential_metrics(ledger, 0.05)
    assert metrics["trades"] == 4
    assert metrics["average_r"] == pytest.approx(0.45)
    assert metrics["total_r"] == pytest.approx(1.8)
    assert chronological_half_means(ledger, 0.05) == pytest.approx((-0.05, 0.95))
    mean, low, high = gated_cost_interval(ledger, 0.05)
    assert mean == pytest.approx(0.45)
    assert low <= mean <= high


def test_ambiguity_bounds_are_reported_separately() -> None:
    ledger = pd.DataFrame(
        {
            "decision": ["ACCEPTED"],
            "signal_index": [1],
            "gross_r_lower": [-1.0],
            "gross_r_upper": [3.0],
            "ambiguous": [1],
        }
    )
    assert sequential_metrics(ledger, 0.0, "gross_r_lower")["total_r"] == -1.0
    assert sequential_metrics(ledger, 0.0, "gross_r_upper")["total_r"] == 3.0
