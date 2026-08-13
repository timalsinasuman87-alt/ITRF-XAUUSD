"""Causal v0.9 market-context definitions for ITRF-XAUUSD.

This module is deliberately separate from the validated v0.8 research path.
It creates context and trade-plan candidates; it does not assert an edge, place
orders, or modify the v0.8 observation database.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ContextConfig:
    """Frozen, descriptive defaults for the first v0.9 hypothesis test."""

    swing_length: int = 3
    liquidity_lookback: int = 20
    regime_lookback: int = 100
    displacement_body_ratio: float = 0.60
    displacement_range_atr: float = 1.00
    stop_atr_buffer: float = 0.10
    account_risk_fraction: float = 0.005


def _confirmed_swings(df: pd.DataFrame, length: int) -> tuple[pd.Series, pd.Series]:
    """Return last *confirmed* swing levels, never using the current future.

    A pivot at bar i becomes available only at i + ``length``.  This is the
    same confirmation timing used by Pine's ``ta.pivothigh/ta.pivotlow``.
    """
    window = 2 * length + 1
    pivot_high = df["high"].eq(df["high"].rolling(window, center=True).max())
    pivot_low = df["low"].eq(df["low"].rolling(window, center=True).min())
    confirmed_high = df["high"].where(pivot_high).shift(length)
    confirmed_low = df["low"].where(pivot_low).shift(length)
    return confirmed_high.ffill(), confirmed_low.ffill()


def _structure_events(df: pd.DataFrame) -> pd.DataFrame:
    """Classify close-confirmed BOS and CHoCH using only known swing levels."""
    result = pd.DataFrame(index=df.index)
    bullish_bos = (df["close"] > df["swing_high_level"]) & df["swing_high_level"].notna()
    bearish_bos = (df["close"] < df["swing_low_level"]) & df["swing_low_level"].notna()

    bias = []
    bullish_choch = []
    bearish_choch = []
    previous_bias = 0
    for up_break, down_break in zip(bullish_bos, bearish_bos):
        bullish_choch.append(int(up_break and previous_bias == -1))
        bearish_choch.append(int(down_break and previous_bias == 1))
        if up_break:
            previous_bias = 1
        elif down_break:
            previous_bias = -1
        bias.append(previous_bias)

    result["bullish_bos"] = bullish_bos.astype(int)
    result["bearish_bos"] = bearish_bos.astype(int)
    result["structure_bias"] = bias
    result["bullish_choch"] = bullish_choch
    result["bearish_choch"] = bearish_choch
    return result


def create_context_features(
    df: pd.DataFrame, config: ContextConfig = ContextConfig()
) -> pd.DataFrame:
    """Add causal market-regime, structure, liquidity, location and entry fields.

    Required input columns are OHLC plus ``atr``, ``ema_50``, ``ema_200``,
    ``body_ratio`` and ``range_atr`` from the v0.8 feature engine.
    """
    required = {"open", "high", "low", "close", "atr", "ema_50", "ema_200", "body_ratio", "range_atr"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing v0.8 feature columns: {sorted(missing)}")

    result = df.copy()
    swing_high, swing_low = _confirmed_swings(result, config.swing_length)
    result["swing_high_level"] = swing_high
    result["swing_low_level"] = swing_low
    result = result.join(_structure_events(result))

    # Regime is descriptive: directional EMA alignment plus an ATR-percentile
    # volatility label. It is not a calibrated trade filter.
    atr_percent = np.where(result["close"] > 0, result["atr"] / result["close"], np.nan)
    result["atr_percent"] = atr_percent
    result["atr_percent_median"] = result["atr_percent"].rolling(config.regime_lookback).median()
    result["volatility_regime"] = np.where(
        result["atr_percent"] > result["atr_percent_median"], "HIGH", "LOW"
    )
    result["market_regime"] = np.select(
        [result["ema_50"] > result["ema_200"], result["ema_50"] < result["ema_200"]],
        ["TREND_UP", "TREND_DOWN"],
        default="RANGE",
    )

    # External liquidity uses completed bars only, matching the v0.8 sweep
    # convention. A sweep must take a level and close back through it.
    result["external_high"] = result["high"].shift(1).rolling(config.liquidity_lookback).max()
    result["external_low"] = result["low"].shift(1).rolling(config.liquidity_lookback).min()
    result["sell_side_sweep"] = ((result["low"] < result["external_low"]) & (result["close"] > result["external_low"])).astype(int)
    result["buy_side_sweep"] = ((result["high"] > result["external_high"]) & (result["close"] < result["external_high"])).astype(int)

    # Location is calculated only from already confirmed structure levels.
    result["dealing_midpoint"] = (result["swing_high_level"] + result["swing_low_level"]) / 2
    valid_range = result["swing_high_level"] > result["swing_low_level"]
    result["discount"] = (valid_range & (result["close"] < result["dealing_midpoint"])).astype(int)
    result["premium"] = (valid_range & (result["close"] > result["dealing_midpoint"])).astype(int)

    result["bullish_displacement"] = ((result["close"] > result["open"]) & (result["body_ratio"] >= config.displacement_body_ratio) & (result["range_atr"] >= config.displacement_range_atr)).astype(int)
    result["bearish_displacement"] = ((result["close"] < result["open"]) & (result["body_ratio"] >= config.displacement_body_ratio) & (result["range_atr"] >= config.displacement_range_atr)).astype(int)

    # A candidate needs a sweep, location, confirmed structure event, and
    # displacement. This is a hypothesis label, not a trading recommendation.
    result["long_confirmation"] = ((result["sell_side_sweep"] == 1) & (result["discount"] == 1) & ((result["bullish_bos"] == 1) | (result["bullish_choch"] == 1)) & (result["bullish_displacement"] == 1)).astype(int)
    result["short_confirmation"] = ((result["buy_side_sweep"] == 1) & (result["premium"] == 1) & ((result["bearish_bos"] == 1) | (result["bearish_choch"] == 1)) & (result["bearish_displacement"] == 1)).astype(int)
    result["context_signal"] = np.select(
        [result["long_confirmation"] == 1, result["short_confirmation"] == 1],
        ["LONG", "SHORT"],
        default="NONE",
    )
    return result


def build_trade_plan(row: pd.Series, account_equity: float, config: ContextConfig = ContextConfig()) -> dict[str, float | str]:
    """Create a deterministic candidate plan; return NONE when no v0.9 signal exists.

    ``units`` assumes one unit moves one account currency per price unit. A
    broker contract multiplier, spread, slippage, commissions and lot limits
    must be applied separately before this is used for execution.
    """
    direction = str(row["context_signal"])
    if direction == "NONE" or account_equity <= 0 or not np.isfinite(row["atr"]):
        return {"direction": "NONE", "entry": np.nan, "stop": np.nan, "risk_per_unit": np.nan, "units": 0.0}

    entry = float(row["close"])
    buffer = float(row["atr"]) * config.stop_atr_buffer
    if direction == "LONG":
        stop = min(float(row["low"]), float(row["swing_low_level"])) - buffer
    else:
        stop = max(float(row["high"]), float(row["swing_high_level"])) + buffer
    risk_per_unit = abs(entry - stop)
    units = (account_equity * config.account_risk_fraction / risk_per_unit) if risk_per_unit > 0 else 0.0
    return {"direction": direction, "entry": entry, "stop": stop, "risk_per_unit": risk_per_unit, "units": units}
