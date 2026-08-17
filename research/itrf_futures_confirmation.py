"""Causal CME Gold candle confirmation for XAU/USD entry research."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from itrf_event_research import block_bootstrap_mean_ci
from itrf_execution import BracketPolicy, deduct_cost_r, simulate_bracket_trade
from itrf_research import detect_setup
from run_v010_clean_baseline import (
    FROZEN_POLICY,
    MINIMUM_HISTORY,
    profit_factor,
)


GATE_REASONS = (
    "PASS",
    "MISSING_FUTURES_BAR",
    "ZERO_BODY_FUTURES",
    "FUTURES_DISAGREEMENT",
)


def load_futures_bars(path: Path) -> pd.DataFrame:
    """Load strict, unique historical futures bars without repairing timestamps."""
    data = pd.read_csv(Path(path))
    required = {"time", "open", "high", "low", "close", "volume"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"missing futures columns: {sorted(missing)}")
    data = data.loc[:, ["time", "open", "high", "low", "close", "volume"]].copy()
    data["time"] = pd.to_datetime(data["time"], errors="coerce", utc=True).dt.tz_convert(None)
    for column in ("open", "high", "low", "close", "volume"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data.isna().any().any():
        raise ValueError("futures bars contain missing or invalid values")
    if not data["time"].is_monotonic_increasing or data["time"].duplicated().any():
        raise ValueError("futures timestamps must be unique and chronological")
    numeric = data.loc[:, ["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("futures OHLCV must be finite")
    if (data["high"] < data[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("futures high is inconsistent with OHLC")
    if (data["low"] > data[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("futures low is inconsistent with OHLC")
    if (data["volume"] < 0).any():
        raise ValueError("futures volume cannot be negative")
    return data


def merge_exact_futures_bars(frame: pd.DataFrame, futures: pd.DataFrame) -> pd.DataFrame:
    """Join only equal completed-bar timestamps; missing rows remain missing."""
    if "time" not in frame.columns:
        raise ValueError("XAU/USD frame is missing time")
    required = {"time", "open", "close"}
    missing = required - set(futures.columns)
    if missing:
        raise ValueError(f"missing futures join columns: {sorted(missing)}")
    right = futures.loc[:, ["time", "open", "close"]].rename(
        columns={"open": "gc_open", "close": "gc_close"}
    )
    if right["time"].duplicated().any():
        raise ValueError("futures join timestamps must be unique")
    result = frame.merge(right, on="time", how="left", validate="one_to_one", indicator=True)
    result["futures_bar_available"] = result["_merge"].eq("both").astype(int)
    return result.drop(columns="_merge")


def classify_futures_gate(row: pd.Series, direction: str) -> tuple[bool, str]:
    if direction not in {"LONG", "SHORT"}:
        raise ValueError("direction must be LONG or SHORT")
    if int(row.get("futures_bar_available", 0)) != 1:
        return False, "MISSING_FUTURES_BAR"
    gc_open = float(row["gc_open"])
    gc_close = float(row["gc_close"])
    if not np.isfinite(gc_open) or not np.isfinite(gc_close):
        return False, "MISSING_FUTURES_BAR"
    body = gc_close - gc_open
    if body == 0:
        return False, "ZERO_BODY_FUTURES"
    aligned = (direction == "LONG" and body > 0) or (direction == "SHORT" and body < 0)
    return (True, "PASS") if aligned else (False, "FUTURES_DISAGREEMENT")


def build_futures_confirmed_ledger(
    frame: pd.DataFrame,
    policy: BracketPolicy = FROZEN_POLICY,
) -> pd.DataFrame:
    """Sequence only sign-confirmed v0.8 candidates and retain every rejection."""
    required = {"time", "open", "high", "low", "close", "atr", "gc_open", "gc_close", "futures_bar_available"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing confirmed-ledger columns: {sorted(missing)}")
    records: list[dict[str, object]] = []
    next_eligible_signal_index = MINIMUM_HISTORY
    for signal_index in range(MINIMUM_HISTORY, len(frame)):
        row = frame.iloc[signal_index]
        if pd.isna(row["atr"]):
            continue
        direction = detect_setup(row)
        if direction == "NONE":
            continue
        gate_pass, gate_reason = classify_futures_gate(row, direction)
        base = {
            "signal_index": signal_index,
            "signal_time": row["time"],
            "direction": direction,
            "signal_close": float(row["close"]),
            "signal_atr": float(row["atr"]),
            "futures_bar_available": int(row["futures_bar_available"]),
            "gc_open": row["gc_open"],
            "gc_close": row["gc_close"],
            "futures_gate_pass": int(gate_pass),
            "futures_gate_reason": gate_reason,
        }
        if signal_index + 1 >= len(frame):
            records.append({**base, "decision": "REJECTED_NO_ENTRY_BAR"})
            continue
        if signal_index + policy.maximum_holding_bars >= len(frame):
            records.append({**base, "decision": "REJECTED_INCOMPLETE_HORIZON"})
            continue
        if not gate_pass:
            records.append({**base, "decision": f"REJECTED_{gate_reason}"})
            continue
        if signal_index < next_eligible_signal_index:
            records.append({**base, "decision": "REJECTED_POSITION_OPEN"})
            continue
        trade = simulate_bracket_trade(frame, signal_index, direction, float(row["atr"]), policy)
        records.append({**base, "decision": "ACCEPTED", **trade})
        next_eligible_signal_index = int(trade["exit_index"])
    return pd.DataFrame(records)


def sequential_metrics(
    ledger: pd.DataFrame,
    cost_r: float,
    outcome_column: str = "gross_r_lower",
) -> dict[str, float | int]:
    accepted = ledger.loc[ledger["decision"] == "ACCEPTED"].copy().reset_index(drop=True)
    if accepted.empty:
        raise ValueError("strategy ledger has no accepted trades")
    if outcome_column not in {"gross_r_lower", "gross_r_upper"}:
        raise ValueError("outcome column must be a registered ambiguity bound")
    unambiguous = accepted.loc[accepted["ambiguous"] == 0]
    if not np.allclose(unambiguous["gross_r_lower"], unambiguous["gross_r_upper"]):
        raise ValueError("unambiguous outcome bounds disagree")
    values = accepted[outcome_column]
    net = deduct_cost_r(values.astype(float), cost_r).reset_index(drop=True)
    equity = net.cumsum()
    peaks = pd.concat([pd.Series([0.0]), equity], ignore_index=True).cummax().iloc[1:].reset_index(drop=True)
    drawdown = equity - peaks
    return {
        "trades": len(net),
        "average_r": float(net.mean()),
        "total_r": float(net.sum()),
        "profit_factor_r": float(profit_factor(net)),
        "maximum_drawdown_r": float(drawdown.min()),
    }


def gated_cost_interval(ledger: pd.DataFrame, cost_r: float = 0.05) -> tuple[float, float, float]:
    accepted = ledger.loc[ledger["decision"] == "ACCEPTED", "gross_r_lower"].astype(float)
    net = deduct_cost_r(accepted, cost_r)
    low, high = block_bootstrap_mean_ci(net)
    return float(net.mean()), low, high


def chronological_half_means(ledger: pd.DataFrame, cost_r: float = 0.05) -> tuple[float, float]:
    accepted = ledger.loc[ledger["decision"] == "ACCEPTED"].sort_values("signal_index").reset_index(drop=True)
    if len(accepted) < 2:
        raise ValueError("at least two accepted trades are required")
    split = len(accepted) // 2
    first = deduct_cost_r(accepted.iloc[:split]["gross_r_lower"].astype(float), cost_r)
    second = deduct_cost_r(accepted.iloc[split:]["gross_r_lower"].astype(float), cost_r)
    return float(first.mean()), float(second.mean())
