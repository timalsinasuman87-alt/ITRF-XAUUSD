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
    confirmation_window_bars: int = 8
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
    """Classify first close-cross BOS/CHoCH events using known swing levels."""
    result = pd.DataFrame(index=df.index)
    bullish_bos = (
        (df["close"] > df["swing_high_level"])
        & (df["close"].shift(1) <= df["swing_high_level"])
        & df["swing_high_level"].notna()
    )
    bearish_bos = (
        (df["close"] < df["swing_low_level"])
        & (df["close"].shift(1) >= df["swing_low_level"])
        & df["swing_low_level"].notna()
    )

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


def apply_v091_state_machine(
    df: pd.DataFrame,
    config: ContextConfig = ContextConfig(),
) -> pd.DataFrame:
    """Sequence sweep arming and later structure/displacement confirmation."""
    if config.confirmation_window_bars < 1:
        raise ValueError("confirmation_window_bars must be at least 1.")
    required = {
        "close", "low", "high", "sell_side_sweep", "buy_side_sweep",
        "discount", "premium", "bullish_bos", "bearish_bos", "bullish_choch",
        "bearish_choch", "bullish_displacement", "bearish_displacement",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing v0.9 state-machine columns: {sorted(missing)}")

    result = df.copy()
    output = {
        "v091_long_arm": [],
        "v091_short_arm": [],
        "v091_invalidation": [],
        "v091_expiration": [],
        "v091_replacement": [],
        "v091_long_confirmation": [],
        "v091_short_confirmation": [],
        "v091_context_signal": [],
        "v091_confirmation_lag": [],
        "v091_source_sweep_time": [],
        "v091_source_sweep_extreme": [],
        "v091_state": [],
    }
    state_side: str | None = None
    armed_index: int | None = None
    sweep_extreme: float | None = None
    sweep_time = None
    event_times = result["time"].tolist() if "time" in result.columns else [pd.NaT] * len(result)

    for position, row in enumerate(result.itertuples(index=False)):
        long_arm = bool(row.sell_side_sweep == 1 and row.discount == 1)
        short_arm = bool(row.buy_side_sweep == 1 and row.premium == 1)
        invalidated = 0
        expired = 0
        replaced = 0
        long_confirmation = 0
        short_confirmation = 0
        signal = "NONE"
        confirmation_lag = pd.NA
        source_sweep_time = pd.NaT
        source_sweep_extreme = np.nan

        # Qualifying new information replaces the old context and cannot also
        # confirm on the same completed bar.
        if long_arm ^ short_arm:
            if state_side is not None:
                replaced = 1
            state_side = "LONG" if long_arm else "SHORT"
            armed_index = position
            sweep_extreme = float(row.low if long_arm else row.high)
            sweep_time = event_times[position]
        elif long_arm and short_arm:
            if state_side is not None:
                replaced = 1
            state_side = None
            armed_index = None
            sweep_extreme = None
            sweep_time = None
        elif state_side is not None and armed_index is not None and sweep_extreme is not None:
            age = position - armed_index
            breached = (
                (state_side == "LONG" and float(row.close) < sweep_extreme)
                or (state_side == "SHORT" and float(row.close) > sweep_extreme)
            )
            if breached:
                invalidated = 1
                state_side = None
            else:
                bullish_confirmation = bool(
                    (row.bullish_bos == 1 or row.bullish_choch == 1)
                    and row.bullish_displacement == 1
                )
                bearish_confirmation = bool(
                    (row.bearish_bos == 1 or row.bearish_choch == 1)
                    and row.bearish_displacement == 1
                )
                if 1 <= age <= config.confirmation_window_bars and (
                    (state_side == "LONG" and bullish_confirmation)
                    or (state_side == "SHORT" and bearish_confirmation)
                ):
                    signal = state_side
                    long_confirmation = int(state_side == "LONG")
                    short_confirmation = int(state_side == "SHORT")
                    confirmation_lag = age
                    source_sweep_time = sweep_time
                    source_sweep_extreme = sweep_extreme
                    state_side = None
                elif age >= config.confirmation_window_bars:
                    expired = 1
                    state_side = None

            if state_side is None:
                armed_index = None
                sweep_extreme = None
                sweep_time = None

        output["v091_long_arm"].append(int(long_arm))
        output["v091_short_arm"].append(int(short_arm))
        output["v091_invalidation"].append(invalidated)
        output["v091_expiration"].append(expired)
        output["v091_replacement"].append(replaced)
        output["v091_long_confirmation"].append(long_confirmation)
        output["v091_short_confirmation"].append(short_confirmation)
        output["v091_context_signal"].append(signal)
        output["v091_confirmation_lag"].append(confirmation_lag)
        output["v091_source_sweep_time"].append(source_sweep_time)
        output["v091_source_sweep_extreme"].append(source_sweep_extreme)
        output["v091_state"].append("FLAT" if state_side is None else f"{state_side}_ARMED")

    for column, values in output.items():
        result[column] = values
    result["v091_confirmation_lag"] = pd.array(result["v091_confirmation_lag"], dtype="Int64")
    result["v091_source_sweep_time"] = pd.to_datetime(result["v091_source_sweep_time"])
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
    return apply_v091_state_machine(result, config)


def build_trade_plan(row: pd.Series, account_equity: float, config: ContextConfig = ContextConfig()) -> dict[str, float | str]:
    """Create a deterministic candidate plan; return NONE when no v0.9 signal exists.

    ``units`` assumes one unit moves one account currency per price unit. A
    broker contract multiplier, spread, slippage, commissions and lot limits
    must be applied separately before this is used for execution.
    """
    direction = str(row.get("v091_context_signal", row.get("context_signal", "NONE")))
    if direction == "NONE" or account_equity <= 0 or not np.isfinite(row["atr"]):
        return {"direction": "NONE", "entry": np.nan, "stop": np.nan, "risk_per_unit": np.nan, "units": 0.0}

    entry = float(row["close"])
    buffer = float(row["atr"]) * config.stop_atr_buffer
    sweep_extreme = row.get("v091_source_sweep_extreme", np.nan)
    if direction == "LONG":
        stop_candidates = [float(row["low"]), float(row["swing_low_level"])]
        if np.isfinite(sweep_extreme):
            stop_candidates.append(float(sweep_extreme))
        stop = min(stop_candidates) - buffer
    else:
        stop_candidates = [float(row["high"]), float(row["swing_high_level"])]
        if np.isfinite(sweep_extreme):
            stop_candidates.append(float(sweep_extreme))
        stop = max(stop_candidates) + buffer
    risk_per_unit = abs(entry - stop)
    units = (account_equity * config.account_risk_fraction / risk_per_unit) if risk_per_unit > 0 else 0.0
    return {"direction": direction, "entry": entry, "stop": stop, "risk_per_unit": risk_per_unit, "units": units}
