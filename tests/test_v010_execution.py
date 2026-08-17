from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from itrf_execution import BracketPolicy, deduct_cost_r, simulate_bracket_trade


def frame(rows: list[dict[str, float]]) -> pd.DataFrame:
    result = pd.DataFrame(rows)
    result["time"] = pd.date_range("2026-01-01", periods=len(result), freq="15min")
    return result


def test_entry_is_next_bar_open_not_signal_close() -> None:
    data = frame([
        {"open": 100, "high": 101, "low": 99, "close": 100},
        {"open": 105, "high": 106, "low": 104, "close": 105},
    ])
    result = simulate_bracket_trade(data, 0, "LONG", 1.0, BracketPolicy(maximum_holding_bars=1))
    assert result["entry"] == 105
    assert result["exit_reason"] == "TIMEOUT"
    assert result["gross_r_lower"] == 0.0


def test_unambiguous_target_terminates_trade() -> None:
    data = frame([
        {"open": 100, "high": 100, "low": 100, "close": 100},
        {"open": 100, "high": 104.6, "low": 99.5, "close": 104},
        {"open": 104, "high": 105, "low": 90, "close": 91},
    ])
    result = simulate_bracket_trade(data, 0, "LONG", 1.0)
    assert result["exit_reason"] == "TARGET"
    assert result["exit_index"] == 1
    assert result["gross_r_lower"] == 3.0


def test_same_bar_stop_target_returns_bounds() -> None:
    data = frame([
        {"open": 100, "high": 100, "low": 100, "close": 100},
        {"open": 100, "high": 105, "low": 98, "close": 101},
    ])
    result = simulate_bracket_trade(data, 0, "LONG", 1.0)
    assert result["exit_reason"] == "AMBIGUOUS_STOP_TARGET"
    assert result["gross_r_lower"] == -1.0
    assert result["gross_r_upper"] == 3.0


def test_short_timeout_marks_final_close_in_r() -> None:
    data = frame([
        {"open": 100, "high": 100, "low": 100, "close": 100},
        {"open": 100, "high": 100.5, "low": 99, "close": 99},
        {"open": 99, "high": 99.5, "low": 98, "close": 98.5},
    ])
    result = simulate_bracket_trade(
        data, 0, "SHORT", 1.0, BracketPolicy(maximum_holding_bars=2)
    )
    assert result["exit_reason"] == "TIMEOUT"
    assert result["gross_r_lower"] == 1.0


def test_invalid_policy_and_cost_are_rejected() -> None:
    data = frame([
        {"open": 100, "high": 100, "low": 100, "close": 100},
        {"open": 100, "high": 100, "low": 100, "close": 100},
    ])
    with pytest.raises(ValueError, match="target_r"):
        simulate_bracket_trade(data, 0, "LONG", 1.0, BracketPolicy(target_r=0))
    with pytest.raises(ValueError, match="direction"):
        simulate_bracket_trade(data, 0, "FLAT", 1.0)
    with pytest.raises(ValueError, match="round_trip_cost_r"):
        deduct_cost_r(pd.Series([1.0]), -0.1)
