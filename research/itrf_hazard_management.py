"""Paired, causal management-policy utilities for frozen ITRF events."""

from __future__ import annotations

import numpy as np
import pandas as pd

from itrf_event_research import block_bootstrap_mean_ci
from itrf_execution import deduct_cost_r


def _profit_factor(values: pd.Series) -> float:
    profits = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    return profits / losses if losses > 0 else float("inf")


def _maximum_drawdown(values: pd.Series) -> float:
    equity = values.cumsum().reset_index(drop=True)
    path = pd.concat([pd.Series([0.0]), equity], ignore_index=True)
    drawdown = path - path.cummax()
    return float(drawdown.min())


def build_paired_management_outcomes(
    frame: pd.DataFrame,
    ledger: pd.DataFrame,
    lifecycle_panel: pd.DataFrame,
    predictions: pd.DataFrame,
    target_r: float = 3.0,
) -> pd.DataFrame:
    """Apply the locked negative-terminal-pressure exit to paired events."""
    if not np.isfinite(target_r) or target_r <= 0:
        raise ValueError("target_r must be positive and finite")
    frame_required = {"time", "open"}
    ledger_required = {
        "decision", "direction", "entry", "risk", "exit_index", "exit_time",
        "exit_reason", "bars_held", "gross_r_lower", "gross_r_upper", "ambiguous",
    }
    panel_required = {"event_id", "holding_bar", "bar_index"}
    prediction_required = {
        "model", "event_id", "holding_bar", "outcome",
        "prob_none", "prob_stop", "prob_target",
    }
    for label, required, columns in (
        ("frame", frame_required, set(frame.columns)),
        ("ledger", ledger_required, set(ledger.columns)),
        ("lifecycle", panel_required, set(lifecycle_panel.columns)),
        ("prediction", prediction_required, set(predictions.columns)),
    ):
        missing = required - columns
        if missing:
            raise ValueError(f"missing {label} columns: {sorted(missing)}")

    accepted = ledger.loc[ledger["decision"] == "ACCEPTED"].copy().reset_index(drop=True)
    eligible = accepted.loc[accepted["ambiguous"] == 0].copy()
    unsupported_directions = set(eligible["direction"].astype(str)) - {"LONG", "SHORT"}
    if unsupported_directions:
        raise ValueError(f"unsupported event directions: {sorted(unsupported_directions)}")
    if (~np.isfinite(eligible["risk"].astype(float)) | (eligible["risk"].astype(float) <= 0)).any():
        raise ValueError("eligible event risk must be positive and finite")
    dynamic = predictions.loc[predictions["model"] == "dynamic_path"].copy()
    if dynamic.empty or eligible.empty:
        raise ValueError("eligible events and dynamic predictions must be non-empty")
    if dynamic.duplicated(["event_id", "holding_bar"]).any():
        raise ValueError("dynamic predictions contain duplicate event holding bars")
    probability_values = dynamic[["prob_none", "prob_stop", "prob_target"]].to_numpy(dtype=float)
    if not np.isfinite(probability_values).all():
        raise ValueError("dynamic predictions contain non-finite probabilities")
    if not np.allclose(probability_values.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("dynamic probabilities do not sum to one")

    panel = lifecycle_panel.loc[:, ["event_id", "holding_bar", "bar_index"]].copy()
    if panel.duplicated(["event_id", "holding_bar"]).any():
        raise ValueError("lifecycle panel contains duplicate event holding bars")
    dynamic = dynamic.merge(
        panel,
        on=["event_id", "holding_bar"],
        how="left",
        validate="one_to_one",
    )
    if dynamic["bar_index"].isna().any():
        raise ValueError("dynamic prediction is missing its lifecycle bar")
    predicted_events = set(dynamic["event_id"].astype(int))
    eligible_events = set(eligible.index.astype(int))
    if predicted_events != eligible_events:
        raise ValueError("dynamic predictions do not cover exactly the eligible events")

    records: list[dict[str, object]] = []
    for event_id, event in eligible.iterrows():
        event_predictions = dynamic.loc[dynamic["event_id"] == event_id].sort_values("holding_bar")
        triggers = event_predictions.loc[
            (event_predictions["holding_bar"] >= 2)
            & (event_predictions["prob_stop"] > target_r * event_predictions["prob_target"])
        ]
        baseline_r = float(event["gross_r_lower"])
        if not np.isclose(baseline_r, float(event["gross_r_upper"])):
            raise ValueError("ambiguous event entered the paired comparison")
        action = not triggers.empty
        if action:
            trigger = triggers.iloc[0]
            if int(trigger["holding_bar"]) > int(event["bars_held"]):
                raise ValueError("managed action occurs after the baseline terminal bar")
            exit_index = int(trigger["bar_index"])
            exit_price = float(frame.iloc[exit_index]["open"])
            direction_sign = 1.0 if str(event["direction"]) == "LONG" else -1.0
            managed_r = direction_sign * (exit_price - float(event["entry"])) / float(event["risk"])
            action_holding_bar: object = int(trigger["holding_bar"])
            managed_exit_reason = "HAZARD_DEFENSIVE_OPEN"
            managed_exit_time = frame.iloc[exit_index]["time"]
        else:
            exit_index = int(event["exit_index"])
            managed_r = baseline_r
            action_holding_bar = pd.NA
            managed_exit_reason = str(event["exit_reason"])
            managed_exit_time = event["exit_time"]
        records.append(
            {
                "event_id": int(event_id),
                "direction": str(event["direction"]),
                "baseline_exit_reason": str(event["exit_reason"]),
                "baseline_bars_held": int(event["bars_held"]),
                "managed_action": int(action),
                "action_holding_bar": action_holding_bar,
                "managed_exit_index": exit_index,
                "managed_exit_time": managed_exit_time,
                "managed_exit_reason": managed_exit_reason,
                "baseline_gross_r": baseline_r,
                "managed_gross_r": managed_r,
                "paired_improvement_r": managed_r - baseline_r,
            }
        )
    result = pd.DataFrame(records)
    result["action_holding_bar"] = pd.array(result["action_holding_bar"], dtype="Int64")
    result["managed_exit_time"] = pd.to_datetime(result["managed_exit_time"])
    return result


def paired_policy_metrics(outcomes: pd.DataFrame, cost_r: float) -> dict[str, float]:
    required = {"baseline_gross_r", "managed_gross_r", "paired_improvement_r"}
    missing = required - set(outcomes.columns)
    if missing:
        raise ValueError(f"missing paired outcome columns: {sorted(missing)}")
    baseline = deduct_cost_r(outcomes["baseline_gross_r"], cost_r)
    managed = deduct_cost_r(outcomes["managed_gross_r"], cost_r)
    return {
        "cost_r": float(cost_r),
        "events": len(outcomes),
        "baseline_average_r": float(baseline.mean()),
        "managed_average_r": float(managed.mean()),
        "paired_improvement_r": float(outcomes["paired_improvement_r"].mean()),
        "baseline_profit_factor_r": _profit_factor(baseline),
        "managed_profit_factor_r": _profit_factor(managed),
        "baseline_maximum_drawdown_r": _maximum_drawdown(baseline),
        "managed_maximum_drawdown_r": _maximum_drawdown(managed),
    }


def paired_improvement_interval(outcomes: pd.DataFrame) -> tuple[float, float, float]:
    if "paired_improvement_r" not in outcomes.columns:
        raise ValueError("missing paired_improvement_r")
    values = outcomes["paired_improvement_r"]
    low, high = block_bootstrap_mean_ci(values)
    return float(values.mean()), low, high
