"""Run the v0.9 fixed exit-model comparison on v0.8 entry candidates."""

from __future__ import annotations

import sqlite3
import argparse
from pathlib import Path

import pandas as pd

from itrf_research import FORWARD_BARS, RISK_ATR, create_features, detect_setup, load_market_data
from itrf_trade_management import MODELS, TradeCostConfig, cost_in_r, evaluate_exit_model, summarize_models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_FILE = PROJECT_ROOT / "database" / "itrf_v09_trade_management.db"
MINIMUM_HISTORY = 250
OOS_START = pd.Timestamp("2025-07-28 17:00:00")


def build_exit_observations(df: pd.DataFrame, costs: TradeCostConfig = TradeCostConfig()) -> pd.DataFrame:
    """Apply fixed exits with one-position-at-a-time handling for each model."""
    records = []
    next_available_index = {model.name: MINIMUM_HISTORY for model in MODELS}
    for index in range(MINIMUM_HISTORY, len(df) - FORWARD_BARS - 1):
        row = df.iloc[index]
        if pd.isna(row["atr"]):
            continue
        direction = detect_setup(row)
        if direction == "NONE":
            continue
        for model in MODELS:
            if index < next_available_index[model.name]:
                continue
            risk = float(row["atr"]) * RISK_ATR
            result = evaluate_exit_model(df, index, direction, float(row["close"]), risk, FORWARD_BARS, model)
            result["gross_outcome_r"] = result.pop("outcome_r")
            result["cost_r"] = cost_in_r(risk, costs)
            result["net_outcome_r"] = result["gross_outcome_r"] - result["cost_r"]
            result["entry"] = float(row["close"])
            result["risk"] = risk
            result["exit_timestamp"] = str(df.iloc[index + int(result["bars_held"])]["time"]) if result["bars_held"] else str(row["time"])
            next_available_index[model.name] = index + int(result["bars_held"]) + 1
            records.append({"timestamp": str(row["time"]), "direction": direction, **result})
    return pd.DataFrame(records, columns=["timestamp", "exit_timestamp", "direction", "model", "entry", "exit_price", "risk", "gross_outcome_r", "cost_r", "net_outcome_r", "partial_taken", "exit_reason", "bars_held"])


def write_observations(observations: pd.DataFrame, database_file: Path = DATABASE_FILE) -> None:
    with sqlite3.connect(database_file) as connection:
        observations.to_sql("v09_exit_observations", connection, if_exists="replace", index=False)


def _parse_arguments():
    parser = argparse.ArgumentParser(description="Run ITRF v0.9 fixed exit-model research.")
    parser.add_argument("--spread-price", type=float, default=0.0)
    parser.add_argument("--slippage-price-per-side", type=float, default=0.0)
    parser.add_argument("--commission-per-contract-per-side", type=float, default=0.0)
    parser.add_argument("--contract-multiplier", type=float, default=1.0)
    # Allows this runner to be called by itrf_research.py after its own flags.
    return parser.parse_known_args()[0]


def main() -> None:
    arguments = _parse_arguments()
    costs = TradeCostConfig(arguments.spread_price, arguments.slippage_price_per_side, arguments.commission_per_contract_per_side, arguments.contract_multiplier)
    costs.validate()
    df = create_features(load_market_data())
    observations = build_exit_observations(df, costs)
    write_observations(observations)
    print("\nITRF v0.9 EXIT-MODEL COMPARISON — PRE-REGISTERED, DESCRIPTIVE ONLY")
    if observations.empty:
        print("No v0.8 entry candidates were available.")
    else:
        print("\nIn-sample (before frozen v0.8 OOS boundary):")
        in_sample = observations.loc[pd.to_datetime(observations["timestamp"]) < OOS_START]
        print(summarize_models(in_sample).round(3).to_string(index=False) if not in_sample.empty else "No in-sample trades.")
        print("\nFrozen OOS (boundary onward):")
        oos = observations.loc[pd.to_datetime(observations["timestamp"]) >= OOS_START]
        print(summarize_models(oos).round(3).to_string(index=False) if not oos.empty else "No OOS trades.")
    print(f"\nDatabase: {DATABASE_FILE}")
    print(f"Costs: spread={costs.spread_price}, slippage/side={costs.slippage_price_per_side}, commission/contract/side={costs.commission_per_contract_per_side}, multiplier={costs.contract_multiplier}")
    print("Max drawdown uses net R on a one-position-at-a-time sequential trade curve.")
    print("Do not select, tune or deploy a model from this output alone.")


if __name__ == "__main__":
    main()
