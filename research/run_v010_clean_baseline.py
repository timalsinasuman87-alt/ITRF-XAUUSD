"""Replay frozen v0.8 signals through the pre-registered v0.10 clean core."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from itrf_execution import BracketPolicy, deduct_cost_r, simulate_bracket_trade
from itrf_research import FORWARD_BARS, create_features, detect_setup, load_market_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_FILE = PROJECT_ROOT / "data" / "XAUUSD.csv"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "v010_clean_baseline.csv"
MINIMUM_HISTORY = 250
FROZEN_POLICY = BracketPolicy(stop_atr=1.5, target_r=3.0, maximum_holding_bars=32)


def build_candidate_ledger(
    frame: pd.DataFrame,
    policy: BracketPolicy = FROZEN_POLICY,
) -> pd.DataFrame:
    """Store accepted and rejected v0.8 candidates under one-position sequencing."""
    records: list[dict[str, object]] = []
    next_eligible_signal_index = MINIMUM_HISTORY
    for signal_index in range(MINIMUM_HISTORY, len(frame)):
        row = frame.iloc[signal_index]
        if pd.isna(row["atr"]):
            continue
        direction = detect_setup(row)
        if direction == "NONE":
            continue
        base = {
            "signal_index": signal_index,
            "signal_time": row["time"],
            "direction": direction,
            "signal_close": float(row["close"]),
            "signal_atr": float(row["atr"]),
        }
        if signal_index + 1 >= len(frame):
            records.append({**base, "decision": "REJECTED_NO_ENTRY_BAR"})
            continue
        if signal_index + policy.maximum_holding_bars >= len(frame):
            records.append({**base, "decision": "REJECTED_INCOMPLETE_HORIZON"})
            continue
        if signal_index < next_eligible_signal_index:
            records.append({**base, "decision": "REJECTED_POSITION_OPEN"})
            continue
        trade = simulate_bracket_trade(
            frame,
            signal_index,
            direction,
            float(row["atr"]),
            policy,
        )
        records.append({**base, "decision": "ACCEPTED", **trade})
        next_eligible_signal_index = int(trade["exit_index"])
    return pd.DataFrame(records)


def profit_factor(values: pd.Series) -> float:
    profits = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    return profits / losses if losses > 0 else float("inf")


def summarize_bound(accepted: pd.DataFrame, column: str, cost_r: float) -> dict[str, float]:
    values = deduct_cost_r(accepted[column], cost_r)
    equity = values.cumsum()
    running_peak = pd.concat([pd.Series([0.0]), equity], ignore_index=True).cummax().iloc[1:]
    drawdown = equity.reset_index(drop=True) - running_peak.reset_index(drop=True)
    return {
        "average_r": float(values.mean()),
        "profit_factor_r": profit_factor(values),
        "maximum_drawdown_r": float(drawdown.min()),
    }


def print_report(ledger: pd.DataFrame) -> None:
    print("\nITRF v0.10 CLEAN CORE — FROZEN v0.8 BENCHMARK REPLAY")
    print("Already-inspected development data; not a profitability claim.")
    print("\nCandidate decisions:")
    print(ledger["decision"].value_counts().rename_axis("decision").to_string())
    accepted = ledger.loc[ledger["decision"] == "ACCEPTED"].copy()
    if accepted.empty:
        print("No accepted sequential trades.")
        return
    print("\nExit reasons:")
    print(accepted["exit_reason"].value_counts().rename_axis("exit_reason").to_string())
    direction = accepted.groupby("direction", sort=False).agg(
        trades=("gross_r_lower", "size"),
        average_gross_r=("gross_r_lower", "mean"),
        stopped_pct=("exit_reason", lambda values: 100.0 * values.eq("STOP").mean()),
        target_pct=("exit_reason", lambda values: 100.0 * values.eq("TARGET").mean()),
    ).reset_index()
    print("\nDirection diagnostic:")
    print(direction.round(3).to_string(index=False))

    accepted = accepted.reset_index(drop=True)
    quarter_codes = (pd.Series(range(len(accepted))) * 4 // len(accepted)).clip(upper=3)
    accepted["chronological_quarter"] = pd.Categorical.from_codes(
        quarter_codes.astype(int), ["Q1", "Q2", "Q3", "Q4"], ordered=True
    )
    chronological = accepted.groupby("chronological_quarter", observed=True).agg(
        trades=("gross_r_lower", "size"),
        average_gross_r=("gross_r_lower", "mean"),
    ).reset_index()
    print("\nChronological diagnostic:")
    print(chronological.round(3).to_string(index=False))
    rows = []
    for cost_r in (0.0, 0.05, 0.10):
        for label, column in (
            ("lower_stop_first", "gross_r_lower"),
            ("upper_target_first", "gross_r_upper"),
        ):
            rows.append(
                {
                    "cost_sensitivity_r": cost_r,
                    "ambiguity_bound": label,
                    "trades": len(accepted),
                    **summarize_bound(accepted, column, cost_r),
                }
            )
    print("\nSequential outcome bounds:")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(f"\nAmbiguous trades: {int(accepted['ambiguous'].sum())}/{len(accepted)}")
    print("No entry, target, stop, direction, or clock filter was selected from this replay.")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen v0.10 clean baseline replay.")
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    frame = create_features(load_market_data(arguments.data_file))
    ledger = build_candidate_ledger(frame)
    arguments.output_file.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(arguments.output_file, index=False)
    print_report(ledger)
    print(f"\nEvent ledger: {arguments.output_file}")


if __name__ == "__main__":
    main()
