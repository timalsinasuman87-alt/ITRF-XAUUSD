from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from itrf_failure_anatomy import (
    build_competing_hazards,
    build_lifecycle_panel,
    lifecycle_phase,
    summarize_trade_paths,
)


def market_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=8, freq="15min"),
            "high": [100, 101, 102, 99, 100, 101, 102, 103],
            "low": [100, 99, 100, 97, 100, 99, 98, 97],
        }
    )


def ledger(events: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for event in events:
        rows.append(
            {
                "decision": "ACCEPTED",
                "entry_index": event["entry_index"],
                "exit_index": event["exit_index"],
                "direction": event["direction"],
                "entry": event.get("entry", 100.0),
                "risk": event.get("risk", 2.0),
                "bars_held": event["exit_index"] - event["entry_index"] + 1,
                "exit_reason": event["exit_reason"],
                "gross_r_lower": event.get("gross_r_lower", -1.0),
                "gross_r_upper": event.get("gross_r_upper", -1.0),
                "ambiguous": event.get("ambiguous", 0),
            }
        )
    return pd.DataFrame(rows)


def test_long_excursions_and_stop_terminal_exclusion() -> None:
    events = ledger([{"entry_index": 1, "exit_index": 3, "direction": "LONG", "exit_reason": "STOP"}])
    panel = build_lifecycle_panel(market_frame(), events)
    paths = summarize_trade_paths(panel, events)
    assert panel["bar_favorable_r"].tolist() == [0.5, 1.0, -0.5]
    assert panel["bar_adverse_r"].tolist() == [-0.5, 0.0, -1.5]
    assert paths.loc[0, "preterminal_mfe_r"] == 1.0
    assert paths.loc[0, "preterminal_mae_r"] == -0.5
    assert paths.loc[0, "full_path_mae_r"] == -1.5


def test_short_excursion_signs_and_target_terminal_exclusion() -> None:
    events = ledger([{"entry_index": 4, "exit_index": 7, "direction": "SHORT", "exit_reason": "TARGET"}])
    panel = build_lifecycle_panel(market_frame(), events)
    paths = summarize_trade_paths(panel, events)
    assert panel["bar_favorable_r"].tolist() == [0.0, 0.5, 1.0, 1.5]
    assert panel["bar_adverse_r"].tolist() == [0.0, -0.5, -1.0, -1.5]
    assert paths.loc[0, "preterminal_mfe_r"] == 1.0
    assert paths.loc[0, "full_path_mfe_r"] == 1.5


def test_timeout_includes_terminal_bar_and_entry_bar_stop_has_zero_preterminal() -> None:
    events = ledger(
        [
            {"entry_index": 1, "exit_index": 2, "direction": "LONG", "exit_reason": "TIMEOUT", "gross_r_lower": 0.0, "gross_r_upper": 0.0},
            {"entry_index": 3, "exit_index": 3, "direction": "LONG", "exit_reason": "STOP"},
        ]
    )
    panel = build_lifecycle_panel(market_frame(), events)
    paths = summarize_trade_paths(panel, events)
    assert paths.loc[0, "preterminal_mfe_r"] == 1.0
    assert paths.loc[1, "preterminal_mfe_r"] == 0.0
    assert paths.loc[1, "preterminal_mae_r"] == 0.0


def test_fixed_mfe_landmarks_are_derived_from_preterminal_path() -> None:
    events = ledger([{"entry_index": 1, "exit_index": 3, "direction": "LONG", "exit_reason": "STOP"}])
    paths = summarize_trade_paths(build_lifecycle_panel(market_frame(), events), events)
    assert paths.loc[0, "preterminal_reached_0_5r"] == 1
    assert paths.loc[0, "preterminal_reached_1_0r"] == 1
    assert paths.loc[0, "preterminal_reached_2_0r"] == 0


def test_competing_hazards_use_start_of_bar_risk_set() -> None:
    paths = pd.DataFrame(
        {
            "bars_held": [1, 2, 2, 3],
            "exit_reason": ["STOP", "STOP", "TARGET", "TIMEOUT"],
        }
    )
    hazards = build_competing_hazards(paths, maximum_holding_bars=3)
    assert hazards["at_risk"].tolist() == [4, 3, 1]
    assert hazards["stop_events"].tolist() == [1, 1, 0]
    assert hazards["target_events"].tolist() == [0, 1, 0]
    assert hazards["timeout_events"].tolist() == [0, 0, 1]
    assert hazards.loc[0, "stop_hazard"] == 0.25
    assert hazards.loc[1, "target_hazard"] == pytest.approx(1 / 3)
    assert hazards.loc[2, "survival_after"] == pytest.approx(0.0)
    incidence = sum(hazards.iloc[-1][f"{name}_cumulative_incidence"] for name in ("stop", "target", "timeout", "ambiguous_stop_target"))
    assert incidence == pytest.approx(1.0)


def test_invalid_lifecycle_inputs_fail_loudly() -> None:
    events = ledger([{"entry_index": 1, "exit_index": 2, "direction": "LONG", "exit_reason": "STOP"}])
    invalid_risk = events.copy()
    invalid_risk.loc[0, "risk"] = np.nan
    with pytest.raises(ValueError, match="risk"):
        build_lifecycle_panel(market_frame(), invalid_risk)
    invalid_interval = events.copy()
    invalid_interval.loc[0, "bars_held"] = 3
    with pytest.raises(ValueError, match="bars_held"):
        build_lifecycle_panel(market_frame(), invalid_interval)
    invalid_direction = events.copy()
    invalid_direction.loc[0, "direction"] = "FLAT"
    with pytest.raises(ValueError, match="direction"):
        build_lifecycle_panel(market_frame(), invalid_direction)
    invalid_paths = pd.DataFrame({"bars_held": [1], "exit_reason": ["UNKNOWN"]})
    with pytest.raises(ValueError, match="unsupported"):
        build_competing_hazards(invalid_paths)
    with pytest.raises(ValueError, match="outside"):
        lifecycle_phase(0)
