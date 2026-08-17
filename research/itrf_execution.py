"""Causal, event-driven execution primitives for ITRF research."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BracketPolicy:
    """Frozen bracket policy expressed in risk units and completed bars."""

    stop_atr: float = 1.5
    target_r: float = 3.0
    maximum_holding_bars: int = 32

    def validate(self) -> None:
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError("stop_atr must be positive and finite")
        if not np.isfinite(self.target_r) or self.target_r <= 0:
            raise ValueError("target_r must be positive and finite")
        if self.maximum_holding_bars < 1:
            raise ValueError("maximum_holding_bars must be at least one")


def _directional_r(price: float, entry: float, risk: float, direction: str) -> float:
    return (price - entry) / risk if direction == "LONG" else (entry - price) / risk


def simulate_bracket_trade(
    frame: pd.DataFrame,
    signal_index: int,
    direction: str,
    signal_atr: float,
    policy: BracketPolicy = BracketPolicy(),
) -> dict[str, object]:
    """Simulate a next-open bracket and preserve unknowable OHLC event order."""
    policy.validate()
    if direction not in {"LONG", "SHORT"}:
        raise ValueError("direction must be LONG or SHORT")
    if not np.isfinite(signal_atr) or signal_atr <= 0:
        raise ValueError("signal_atr must be positive and finite")
    required = {"time", "open", "high", "low", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing execution columns: {sorted(missing)}")
    entry_index = signal_index + 1
    if signal_index < 0 or entry_index >= len(frame):
        raise IndexError("signal has no next-bar entry")

    direction_sign = 1 if direction == "LONG" else -1
    entry = float(frame.iloc[entry_index]["open"])
    risk = float(signal_atr) * policy.stop_atr
    stop = entry - direction_sign * risk
    target = entry + direction_sign * risk * policy.target_r
    future = frame.iloc[entry_index : entry_index + policy.maximum_holding_bars]

    for bars_held, (exit_index, bar) in enumerate(future.iterrows(), start=1):
        if direction == "LONG":
            stop_hit = bool(bar["low"] <= stop)
            target_hit = bool(bar["high"] >= target)
        else:
            stop_hit = bool(bar["high"] >= stop)
            target_hit = bool(bar["low"] <= target)

        if stop_hit and target_hit:
            return {
                "entry_index": entry_index,
                "exit_index": int(exit_index),
                "entry_time": frame.iloc[entry_index]["time"],
                "exit_time": bar["time"],
                "entry": entry,
                "stop": stop,
                "target": target,
                "risk": risk,
                "bars_held": bars_held,
                "exit_reason": "AMBIGUOUS_STOP_TARGET",
                "ambiguous": 1,
                "gross_r_lower": -1.0,
                "gross_r_upper": policy.target_r,
            }
        if stop_hit:
            outcome = -1.0
            reason = "STOP"
            exit_price = stop
        elif target_hit:
            outcome = policy.target_r
            reason = "TARGET"
            exit_price = target
        else:
            continue
        return {
            "entry_index": entry_index,
            "exit_index": int(exit_index),
            "entry_time": frame.iloc[entry_index]["time"],
            "exit_time": bar["time"],
            "entry": entry,
            "stop": stop,
            "target": target,
            "risk": risk,
            "bars_held": bars_held,
            "exit_reason": reason,
            "exit_price": exit_price,
            "ambiguous": 0,
            "gross_r_lower": outcome,
            "gross_r_upper": outcome,
        }

    final_index = int(future.index[-1])
    final_bar = future.iloc[-1]
    timeout_r = _directional_r(float(final_bar["close"]), entry, risk, direction)
    return {
        "entry_index": entry_index,
        "exit_index": final_index,
        "entry_time": frame.iloc[entry_index]["time"],
        "exit_time": final_bar["time"],
        "entry": entry,
        "stop": stop,
        "target": target,
        "risk": risk,
        "bars_held": len(future),
        "exit_reason": "TIMEOUT",
        "exit_price": float(final_bar["close"]),
        "ambiguous": 0,
        "gross_r_lower": timeout_r,
        "gross_r_upper": timeout_r,
    }


def deduct_cost_r(outcome_r: pd.Series, round_trip_cost_r: float) -> pd.Series:
    """Apply a non-negative fixed completed-trade sensitivity in R."""
    if not np.isfinite(round_trip_cost_r) or round_trip_cost_r < 0:
        raise ValueError("round_trip_cost_r must be non-negative and finite")
    return outcome_r - round_trip_cost_r
