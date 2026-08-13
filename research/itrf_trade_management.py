"""Pre-registered v0.9 exit-model comparison for existing ITRF entries.

This module compares fixed rules only. It does not search parameters, place
orders, account for costs, or resolve intrabar sequencing that OHLC cannot see.
When a stop and favourable level occur in one bar, the stop is assumed first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExitModel:
    name: str
    partial_at_r: float | None = None
    break_even_at_r: float | None = None
    atr_trail: bool = False


@dataclass(frozen=True)
class TradeCostConfig:
    """Broker-specific assumptions, expressed in the instrument price currency."""

    spread_price: float = 0.0
    slippage_price_per_side: float = 0.0
    commission_per_contract_per_side: float = 0.0
    contract_multiplier: float = 1.0

    def validate(self) -> None:
        if min(self.spread_price, self.slippage_price_per_side, self.commission_per_contract_per_side) < 0:
            raise ValueError("Cost inputs cannot be negative.")
        if self.contract_multiplier <= 0:
            raise ValueError("Contract multiplier must be positive.")


MODELS = (
    ExitModel("fixed_3r"),
    ExitModel("partial_2r_atr_trail", partial_at_r=2.0, atr_trail=True),
    ExitModel("break_even_at_1r", break_even_at_r=1.0),
    ExitModel("atr_trailing_stop", atr_trail=True),
)

TARGET_R = 3.0
ATR_TRAIL_MULTIPLIER = 1.0  # fixed v0.9 convention; no parameter search


def _favourable(candle: pd.Series, entry: float, risk: float, direction: str) -> float:
    return ((candle["high"] - entry) / risk) if direction == "LONG" else ((entry - candle["low"]) / risk)


def _stop_hit(candle: pd.Series, stop: float, direction: str) -> bool:
    return candle["low"] <= stop if direction == "LONG" else candle["high"] >= stop


def _trail_stop(extreme: float, atr: float, direction: str) -> float:
    return extreme - ATR_TRAIL_MULTIPLIER * atr if direction == "LONG" else extreme + ATR_TRAIL_MULTIPLIER * atr


def evaluate_exit_model(df: pd.DataFrame, index: int, direction: str, entry: float, atr: float, forward_bars: int, model: ExitModel) -> dict[str, float | int | str]:
    """Evaluate one fixed exit model, returning total realized R for one entry.

    The initial stop is one risk unit (the v0.8 `RISK_ATR * ATR` distance).
    Unclosed exposure is marked to the close of the final available forward bar.
    """
    if direction not in {"LONG", "SHORT"} or not np.isfinite(atr) or atr <= 0:
        raise ValueError("Direction must be LONG/SHORT and ATR must be positive.")
    risk = atr
    stop = entry - risk if direction == "LONG" else entry + risk
    future = df.iloc[index + 1:index + 1 + forward_bars]
    extreme = entry
    remaining = 1.0
    realized_r = 0.0
    partial_taken = False
    break_even_active = False
    exit_reason = "time_exit"
    exit_price = entry
    bars_held = 0

    for bars_held, (_, candle) in enumerate(future.iterrows(), start=1):
        # Conservative bar assumption: existing protective stop has priority.
        if _stop_hit(candle, stop, direction):
            stop_r = (stop - entry) / risk if direction == "LONG" else (entry - stop) / risk
            realized_r += remaining * stop_r
            remaining = 0.0
            exit_reason = "stop"
            exit_price = stop
            break

        favourable = _favourable(candle, entry, risk, direction)
        if model.partial_at_r is not None and not partial_taken and favourable >= model.partial_at_r:
            realized_r += 0.5 * model.partial_at_r
            remaining = 0.5
            partial_taken = True
            exit_reason = "partial_then_trail"

        if model.break_even_at_r is not None and not break_even_active and favourable >= model.break_even_at_r:
            stop = entry
            break_even_active = True

        if model.name in {"fixed_3r", "break_even_at_1r"} and favourable >= TARGET_R:
            realized_r += remaining * TARGET_R
            remaining = 0.0
            exit_reason = "target_3r"
            exit_price = entry + TARGET_R * risk if direction == "LONG" else entry - TARGET_R * risk
            break

        # Trailing stops are updated at the end of the bar and apply next bar.
        if model.atr_trail:
            extreme = max(extreme, candle["high"]) if direction == "LONG" else min(extreme, candle["low"])
            candidate = _trail_stop(extreme, float(candle["atr"]), direction)
            stop = max(stop, candidate) if direction == "LONG" else min(stop, candidate)

    if remaining > 0:
        if future.empty:
            mark_r = 0.0
            exit_reason = "no_forward_data"
            bars_held = 0
        else:
            final_close = float(future.iloc[-1]["close"])
            mark_r = (final_close - entry) / risk if direction == "LONG" else (entry - final_close) / risk
            exit_price = final_close
            bars_held = len(future)
        realized_r += remaining * mark_r

    return {"model": model.name, "outcome_r": realized_r, "partial_taken": int(partial_taken), "exit_reason": exit_reason, "exit_price": exit_price, "bars_held": bars_held}


def cost_in_r(risk: float, costs: TradeCostConfig) -> float:
    """Round-trip spread/slippage/commission converted to R for one contract."""
    costs.validate()
    price_cost = costs.spread_price + 2 * costs.slippage_price_per_side
    commission_cost = 2 * costs.commission_per_contract_per_side / costs.contract_multiplier
    return (price_cost + commission_cost) / risk


def summarize_models(observations: pd.DataFrame) -> pd.DataFrame:
    """Return trade count, average R and sequential-signal max drawdown in R."""
    records = []
    for model, group in observations.sort_values("timestamp").groupby("model", sort=False):
        outcome_column = "net_outcome_r" if "net_outcome_r" in group else "outcome_r"
        equity = group[outcome_column].cumsum()
        peak = equity.cummax()
        max_drawdown_r = (equity - peak).min() if not equity.empty else 0.0
        records.append({"model": model, "trades": len(group), "average_r": group[outcome_column].mean(), "max_drawdown_r": max_drawdown_r})
    return pd.DataFrame(records)
