"""Run the pre-registered v0.9.1 causal context study on development data."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from itrf_context import create_context_features
from itrf_research import FORWARD_BARS, create_features, evaluate_forward_path, load_market_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_FILE = PROJECT_ROOT / "database" / "itrf_v091_research.db"
DEFAULT_DATA_FILE = PROJECT_ROOT / "data" / "XAUUSD.csv"
MINIMUM_HISTORY = 250


def _create_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS v091_context_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            sweep_timestamp TEXT NOT NULL,
            confirmation_lag INTEGER NOT NULL,
            direction TEXT NOT NULL,
            market_regime TEXT NOT NULL,
            volatility_regime TEXT NOT NULL,
            structure_event TEXT NOT NULL,
            entry REAL NOT NULL,
            atr REAL NOT NULL,
            non_overlapping INTEGER NOT NULL,
            hit_1r INTEGER NOT NULL,
            hit_2r INTEGER NOT NULL,
            hit_3r INTEGER NOT NULL,
            stopped INTEGER NOT NULL,
            mfe REAL NOT NULL,
            mae REAL NOT NULL,
            outcome_r REAL NOT NULL
        )
        """
    )
    connection.commit()


def build_v091_observations(df: pd.DataFrame, connection: sqlite3.Connection) -> int:
    """Store causal confirmations with unchanged v0.8 forward labels."""
    _create_table(connection)
    connection.execute("DELETE FROM v091_context_observations")
    records = []
    next_available_index = MINIMUM_HISTORY
    last_index = len(df) - FORWARD_BARS - 1

    for index in range(MINIMUM_HISTORY, last_index):
        row = df.iloc[index]
        direction = row["v091_context_signal"]
        if direction == "NONE" or pd.isna(row["atr"]):
            continue
        outcome = evaluate_forward_path(df, index, direction, row["close"], row["atr"])
        non_overlapping = int(index >= next_available_index)
        if non_overlapping:
            next_available_index = index + FORWARD_BARS + 1
        structure_event = "CHoCH" if (
            (direction == "LONG" and row["bullish_choch"] == 1)
            or (direction == "SHORT" and row["bearish_choch"] == 1)
        ) else "BOS"
        records.append((
            str(row["time"]),
            str(row["v091_source_sweep_time"]),
            int(row["v091_confirmation_lag"]),
            direction,
            row["market_regime"],
            row["volatility_regime"],
            structure_event,
            float(row["close"]),
            float(row["atr"]),
            non_overlapping,
            outcome["hit_1r"],
            outcome["hit_2r"],
            outcome["hit_3r"],
            outcome["stopped"],
            outcome["mfe"],
            outcome["mae"],
            outcome["outcome_r"],
        ))

    connection.executemany(
        """
        INSERT INTO v091_context_observations (
            timestamp, sweep_timestamp, confirmation_lag, direction,
            market_regime, volatility_regime, structure_event, entry, atr,
            non_overlapping, hit_1r, hit_2r, hit_3r, stopped, mfe, mae, outcome_r
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        records,
    )
    connection.commit()
    return len(records)


def state_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "event": ["long arms", "short arms", "replacements", "invalidations", "expirations", "long confirmations", "short confirmations"],
        "count": [
            int(df["v091_long_arm"].sum()),
            int(df["v091_short_arm"].sum()),
            int(df["v091_replacement"].sum()),
            int(df["v091_invalidation"].sum()),
            int(df["v091_expiration"].sum()),
            int(df["v091_long_confirmation"].sum()),
            int(df["v091_short_confirmation"].sum()),
        ],
    })


def _read_observations(database_file: Path) -> pd.DataFrame:
    with sqlite3.connect(database_file) as connection:
        return pd.read_sql_query(
            "SELECT * FROM v091_context_observations ORDER BY timestamp",
            connection,
        )


def print_development_report(database_file: Path) -> None:
    observations = _read_observations(database_file)
    print("\nITRF v0.9.1 CAUSAL CONTEXT — DEVELOPMENT DIAGNOSTIC ONLY")
    if observations.empty:
        print("No causal confirmations were produced.")
        return
    summary = observations.groupby(["direction", "non_overlapping"]).agg(
        samples=("outcome_r", "size"),
        average_r=("outcome_r", "mean"),
        hit_1r_pct=("hit_1r", lambda values: 100.0 * values.mean()),
        hit_2r_pct=("hit_2r", lambda values: 100.0 * values.mean()),
        stopped_pct=("stopped", lambda values: 100.0 * values.mean()),
        median_confirmation_lag=("confirmation_lag", "median"),
    ).reset_index()
    print("\nDirection and overlap diagnostic:")
    print(summary.round(3).to_string(index=False))

    non_overlapping = observations.loc[observations["non_overlapping"] == 1].copy()
    if len(non_overlapping) >= 4:
        non_overlapping = non_overlapping.reset_index(drop=True)
        non_overlapping["development_quarter"] = pd.qcut(
            non_overlapping.index,
            q=4,
            labels=["Development 1", "Development 2", "Development 3", "Development 4"],
        )
        stability = non_overlapping.groupby("development_quarter", observed=True).agg(
            samples=("outcome_r", "size"),
            average_r=("outcome_r", "mean"),
            hit_1r_pct=("hit_1r", lambda values: 100.0 * values.mean()),
        ).reset_index()
        print("\nChronological non-overlapping diagnostic:")
        print(stability.round(3).to_string(index=False))
    print("\nAll periods are previously inspected development data, not v0.9.1 OOS evidence.")
    print("No threshold, direction, or confirmation window may be selected from this report.")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ITRF v0.9.1 causal context diagnostics.")
    parser.add_argument("--data-file", type=Path, default=None)
    parser.add_argument("--database-file", type=Path, default=None)
    return parser.parse_known_args()[0]


def main(data_file=None, database_file=None) -> None:
    arguments = _parse_arguments()
    selected_data_file = Path(data_file or arguments.data_file or DEFAULT_DATA_FILE)
    selected_database_file = Path(database_file or arguments.database_file or DATABASE_FILE)
    df = create_context_features(create_features(load_market_data(selected_data_file)))
    print("v0.9.1 state transitions:")
    print(state_diagnostics(df).to_string(index=False))
    selected_database_file.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(selected_database_file) as connection:
        count = build_v091_observations(df, connection)
    print(f"\nv0.9.1 causal confirmations written: {count:,}")
    print(f"Database: {selected_database_file}")
    print_development_report(selected_database_file)


if __name__ == "__main__":
    main()
