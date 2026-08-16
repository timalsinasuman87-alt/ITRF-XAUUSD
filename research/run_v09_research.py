"""Run the pre-registered v0.9 context study without altering v0.8 outputs.

The program requires the same real ``data/XAUUSD.csv`` input as v0.8 and
creates a separate SQLite database at ``database/itrf_v09_research.db``.
"""

from __future__ import annotations

import sqlite3
import argparse
from pathlib import Path

import pandas as pd

from itrf_context import create_context_features
from itrf_research import (
    DEFAULT_OOS_START,
    FORWARD_BARS,
    create_features,
    evaluate_forward_path,
    load_market_data,
    resolve_oos_split,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_FILE = PROJECT_ROOT / "database" / "itrf_v09_research.db"
MINIMUM_HISTORY = 250


def _create_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS v09_context_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            direction TEXT NOT NULL,
            market_regime TEXT NOT NULL,
            volatility_regime TEXT NOT NULL,
            structure_event TEXT NOT NULL,
            liquidity_event TEXT NOT NULL,
            location TEXT NOT NULL,
            entry REAL NOT NULL,
            atr REAL NOT NULL,
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


def build_context_observations(df: pd.DataFrame, connection: sqlite3.Connection) -> int:
    """Write v0.9 candidates and unchanged v0.8 forward labels to a separate table."""
    _create_table(connection)
    connection.execute("DELETE FROM v09_context_observations")
    records = []
    last_index = len(df) - FORWARD_BARS - 1

    for index in range(MINIMUM_HISTORY, last_index):
        row = df.iloc[index]
        direction = row["context_signal"]
        if direction == "NONE" or pd.isna(row["atr"]):
            continue
        outcome = evaluate_forward_path(df, index, direction, row["close"], row["atr"])
        structure_event = "CHoCH" if row["bullish_choch"] or row["bearish_choch"] else "BOS"
        liquidity_event = "SELL_SIDE_SWEEP" if row["sell_side_sweep"] else "BUY_SIDE_SWEEP"
        location = "DISCOUNT" if row["discount"] else "PREMIUM"
        records.append((
            str(row["time"]), direction, row["market_regime"], row["volatility_regime"],
            structure_event, liquidity_event, location, row["close"], row["atr"],
            outcome["hit_1r"], outcome["hit_2r"], outcome["hit_3r"], outcome["stopped"],
            outcome["mfe"], outcome["mae"], outcome["outcome_r"],
        ))

    connection.executemany(
        """
        INSERT INTO v09_context_observations (
            timestamp, direction, market_regime, volatility_regime, structure_event,
            liquidity_event, location, entry, atr, hit_1r, hit_2r, hit_3r,
            stopped, mfe, mae, outcome_r
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        records,
    )
    connection.commit()
    return len(records)


def _summary_query(database_file: Path = DATABASE_FILE, where: str = "", params: tuple = ()) -> pd.DataFrame:
    query = f"""
        SELECT
            direction,
            market_regime,
            volatility_regime,
            COUNT(*) AS samples,
            ROUND(100.0 * AVG(hit_1r), 2) AS hit_1r_pct,
            ROUND(100.0 * AVG(hit_2r), 2) AS hit_2r_pct,
            ROUND(100.0 * AVG(stopped), 2) AS stopped_pct,
            ROUND(AVG(outcome_r), 3) AS average_r,
            ROUND(AVG(mfe), 3) AS average_mfe,
            ROUND(AVG(mae), 3) AS average_mae
        FROM v09_context_observations
        {where}
        GROUP BY direction, market_regime, volatility_regime
        ORDER BY direction, market_regime, volatility_regime
    """
    with sqlite3.connect(database_file) as connection:
        return pd.read_sql_query(query, connection, params=params)


def print_report(
    oos_start: str,
    available_data: pd.DataFrame,
    database_file: Path = DATABASE_FILE,
) -> None:
    """Print descriptive in-sample/OOS summaries without selecting a winner."""
    print("\n" + "=" * 92)
    print("ITRF v0.9 CONTEXT STUDY — DESCRIPTIVE RESEARCH ONLY")
    print("=" * 92)
    print("\nAll candidates (descriptive; do not select rules from this table):")
    print(_summary_query(database_file).to_string(index=False))
    split_time = resolve_oos_split(
        available_data.rename(columns={"time": "timestamp"}),
        oos_start,
        "v0.9 context OOS report",
    )
    if split_time is not None:
        print(f"\nFrozen OOS boundary onward ({split_time}; descriptive; no parameter changes):")
        oos = _summary_query(database_file, "WHERE timestamp >= ?", (str(split_time),))
        print(oos.to_string(index=False) if not oos.empty else "No observations after the OOS boundary.")
    print("\nImportant: results are labels on historical bars, not proof of profitability.")
    print("Costs, slippage, contract multiplier, execution constraints, and overlapping positions are not modelled.")


def _parse_arguments():
    parser = argparse.ArgumentParser(description="Run ITRF v0.9 context research.")
    parser.add_argument("--data-file", type=Path, default=None)
    parser.add_argument("--database-file", type=Path, default=None)
    parser.add_argument(
        "--oos-start",
        default=DEFAULT_OOS_START,
        help="Frozen chronological OOS start. Invalid boundaries safely skip the split report.",
    )
    # The integrated v0.8 runner has additional flags that are irrelevant here.
    return parser.parse_known_args()[0]


def main(data_file=None, database_file=None, oos_start=None) -> None:
    arguments = _parse_arguments()
    selected_data_file = data_file or arguments.data_file
    selected_database_file = Path(database_file or arguments.database_file or DATABASE_FILE)
    selected_oos_start = oos_start or arguments.oos_start
    df = create_context_features(create_features(load_market_data(selected_data_file)))
    selected_database_file.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(selected_database_file) as connection:
        count = build_context_observations(df, connection)
    print(f"v0.9 context candidates written: {count:,}")
    print(f"Database: {selected_database_file}")
    print_report(selected_oos_start, df, selected_database_file)


if __name__ == "__main__":
    main()
