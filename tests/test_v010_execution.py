from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from itrf_execution import BracketPolicy, deduct_cost_r, simulate_bracket_trade
import run_v010_clean_baseline
from run_v010_clean_baseline import build_candidate_ledger


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


def test_candidate_without_full_horizon_is_right_censored() -> None:
    rows = 283
    data = frame(
        [{"open": 100, "high": 101, "low": 99, "close": 100} for _ in range(rows)]
    )
    data["atr"] = 1.0
    data["trend"] = 0
    data["momentum_atr"] = 0.0
    data["delta_zscore"] = 0.0
    data["relative_volume"] = 0.0
    data["delta_proxy"] = 0.0
    data["bullish_sweep"] = 0
    data["bearish_sweep"] = 0
    data.loc[251, ["trend", "momentum_atr", "delta_zscore", "relative_volume", "delta_proxy"]] = [
        1, 1.0, 2.0, 2.0, 10.0
    ]
    ledger = build_candidate_ledger(data)
    assert ledger.loc[0, "decision"] == "REJECTED_INCOMPLETE_HORIZON"


def test_signal_on_exit_bar_can_enter_on_following_bar(monkeypatch) -> None:
    rows = 300
    data = frame(
        [{"open": 100, "high": 100, "low": 100, "close": 100} for _ in range(rows)]
    )
    data["atr"] = 1.0
    data.loc[[251, 252], "high"] = 105.0
    monkeypatch.setattr(
        run_v010_clean_baseline,
        "detect_setup",
        lambda row: "LONG" if row.name in {250, 251} else "NONE",
    )
    ledger = build_candidate_ledger(data)
    assert ledger["decision"].tolist() == ["ACCEPTED", "ACCEPTED"]
    assert ledger["entry_index"].tolist() == [251, 252]


def test_candidate_during_open_position_is_rejected(monkeypatch) -> None:
    rows = 300
    data = frame(
        [{"open": 100, "high": 100, "low": 100, "close": 100} for _ in range(rows)]
    )
    data["atr"] = 1.0
    monkeypatch.setattr(
        run_v010_clean_baseline,
        "detect_setup",
        lambda row: "LONG" if row.name in {250, 251} else "NONE",
    )
    ledger = build_candidate_ledger(data)
    assert ledger["decision"].tolist() == ["ACCEPTED", "REJECTED_POSITION_OPEN"]
