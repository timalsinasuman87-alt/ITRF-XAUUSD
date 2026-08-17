from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from audit_v08_failure import (
    block_bootstrap_mean_ci,
    build_audit_frame,
    exact_v08_components,
    maximum_drawdown_r,
    trace_forward_path,
)


def test_exact_components_reproduce_weighted_sweep_score() -> None:
    row = pd.Series(
        {
            "trend": 1,
            "momentum_atr": 0.5,
            "delta_zscore": 1.2,
            "relative_volume": 1.6,
            "delta_proxy": 10.0,
            "bullish_sweep": 1,
            "bearish_sweep": 0,
        }
    )
    result = exact_v08_components(row, "LONG")
    assert result["v08_score"] == 6
    assert result["component_sweep"] == 1


def test_trace_identifies_target_then_later_stop_label_override() -> None:
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=4, freq="15min"),
            "high": [100.0, 102.0, 100.5, 100.0],
            "low": [100.0, 99.5, 98.4, 99.0],
        }
    )
    result = trace_forward_path(frame, 0, "LONG", entry=100.0, atr=1.0)
    assert result["bars_to_1r"] == 1
    assert result["bars_to_stop"] == 2
    assert result["target_then_later_stop"] == 1
    assert result["outcome_r"] == -1.0


def test_trace_marks_same_bar_path_as_ambiguous_and_conservative_stop() -> None:
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=2, freq="15min"),
            "high": [100.0, 102.0],
            "low": [100.0, 98.0],
        }
    )
    result = trace_forward_path(frame, 0, "LONG", entry=100.0, atr=1.0)
    assert result["same_bar_stop_target"] == 1
    assert result["first_event"] == "AMBIGUOUS_SAME_BAR"
    assert result["outcome_r"] == -1.0


def test_maximum_drawdown_uses_running_peak_with_zero_start() -> None:
    assert maximum_drawdown_r(pd.Series([1.0, -1.0, -1.0, 2.0])) == 2.0


def test_block_bootstrap_is_deterministic() -> None:
    values = pd.Series([-1.0, 1.0, 0.0, 2.0, -1.0])
    assert block_bootstrap_mean_ci(values, samples=100) == block_bootstrap_mean_ci(
        values, samples=100
    )


def test_audit_adds_first_touch_outcome_without_changing_frozen_label() -> None:
    rows = 284
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=rows, freq="15min"),
            "open": [100.0] * rows,
            "high": [100.0] * rows,
            "low": [100.0] * rows,
            "close": [100.0] * rows,
            "atr": [1.0] * rows,
            "trend": [0] * rows,
            "momentum_atr": [0.0] * rows,
            "delta_zscore": [0.0] * rows,
            "relative_volume": [0.0] * rows,
            "delta_proxy": [0.0] * rows,
            "bullish_sweep": [0] * rows,
            "bearish_sweep": [0] * rows,
            "high_volatility": [0] * rows,
        }
    )
    frame.loc[250, ["trend", "momentum_atr", "delta_zscore", "relative_volume", "delta_proxy"]] = [
        1, 1.0, 2.0, 2.0, 10.0
    ]
    frame.loc[251, "high"] = 102.0
    frame.loc[252, "low"] = 98.0
    result = build_audit_frame(frame)
    assert result.loc[0, "outcome_r"] == -1.0
    assert result.loc[0, "first_touch_1r_outcome"] == 1.0
