"""Run the v0.9 fixed exit-model comparison on v0.8 entry candidates."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from itrf_research import FORWARD_BARS, RISK_ATR, create_features, detect_setup, load_market_data
from itrf_trade_management import MODELS, evaluate_exit_model, summarize_models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_FILE = PROJECT_ROOT / "database" / "itrf_v09_trade_management.db"
MINIMUM_HISTORY = 250


def build_exit_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every fixed exit model to the exact same v0.8 entry candidates."""
    records = []
    for index in range(MINIMUM_HISTORY, len(df) - FORWARD_BARS - 1):
        row = df.iloc[index]
        if pd.isna(row["atr"]):
            continue
        direction = detect_setup(row)
        if direction == "NONE":
            continue
        for model in MODELS:
            result = evaluate_exit_model(df, index, direction, float(row["close"]), float(row["atr"]) * RISK_ATR, FORWARD_BARS, model)
            records.append({"timestamp": str(row["time"]), "direction": direction, **result})
    return pd.DataFrame(records, columns=["timestamp", "direction", "model", "outcome_r", "partial_taken", "exit_reason"])


def write_observations(observations: pd.DataFrame, database_file: Path = DATABASE_FILE) -> None:
    with sqlite3.connect(database_file) as connection:
        observations.to_sql("v09_exit_observations", connection, if_exists="replace", index=False)


def main() -> None:
    df = create_features(load_market_data())
    observations = build_exit_observations(df)
    write_observations(observations)
    print("\nITRF v0.9 EXIT-MODEL COMPARISON — PRE-REGISTERED, DESCRIPTIVE ONLY")
    if observations.empty:
        print("No v0.8 entry candidates were available.")
    else:
        print(summarize_models(observations).round(3).to_string(index=False))
    print(f"\nDatabase: {DATABASE_FILE}")
    print("Max drawdown is in R on a sequential-signal curve; overlapping positions, costs and execution are not modelled.")
    print("Do not select, tune or deploy a model from this output alone.")


if __name__ == "__main__":
    main()
