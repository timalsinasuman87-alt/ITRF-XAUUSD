"""Run the pre-registered v0.9.3 XAU/USD sequence-flow study."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from itrf_context import create_context_features
from itrf_research import FORWARD_BARS, create_features, evaluate_forward_path, load_market_data
from run_v092_orderflow_research import load_orderflow_data, merge_orderflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XAUUSD_DATA = PROJECT_ROOT / "data" / "XAUUSD.csv"
DEFAULT_ORDERFLOW_DATA = PROJECT_ROOT / "data" / "databento" / "GC_front_orderflow_15m.csv"
DEFAULT_DATABASE = PROJECT_ROOT / "database" / "itrf_v093_sequence_flow.db"
MINIMUM_HISTORY = 250


def add_sequence_flow(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate inclusive sweep-to-confirmation aggressor flow without lookahead."""
    required = {
        "time",
        "v091_context_signal",
        "v091_source_sweep_time",
        "v092_flow_available",
        "buy_volume",
        "sell_volume",
        "unknown_volume",
        "aggressor_coverage",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing v0.9.3 sequence columns: {sorted(missing)}")
    result = frame.copy()
    if result["time"].duplicated().any() or not result["time"].is_monotonic_increasing:
        raise ValueError("Sequence input timestamps must be unique and chronological.")
    time_to_position = {pd.Timestamp(value): position for position, value in enumerate(result["time"])}

    sequence_bars = np.zeros(len(result), dtype="int64")
    sequence_buy = np.full(len(result), np.nan)
    sequence_sell = np.full(len(result), np.nan)
    sequence_unknown = np.full(len(result), np.nan)
    sequence_delta = np.full(len(result), np.nan)
    sequence_ratio = np.full(len(result), np.nan)
    flow_pass = np.zeros(len(result), dtype="int64")
    reasons = np.full(len(result), "NO_V091_SIGNAL", dtype=object)

    for position, row in enumerate(result.itertuples(index=False)):
        direction = row.v091_context_signal
        if direction == "NONE":
            continue
        source_time = pd.Timestamp(row.v091_source_sweep_time)
        start = time_to_position.get(source_time)
        if start is None or start > position:
            reasons[position] = "INCOMPLETE_SEQUENCE"
            continue
        segment = result.iloc[start : position + 1]
        sequence_bars[position] = len(segment)
        complete = (
            segment["v092_flow_available"].eq(1).all()
            and segment[["buy_volume", "sell_volume", "unknown_volume"]].notna().all().all()
            and segment["aggressor_coverage"].gt(0).all()
        )
        if not complete:
            reasons[position] = "INCOMPLETE_SEQUENCE"
            continue
        buy = float(segment["buy_volume"].sum())
        sell = float(segment["sell_volume"].sum())
        unknown = float(segment["unknown_volume"].sum())
        delta = buy - sell
        specified = buy + sell
        sequence_buy[position] = buy
        sequence_sell[position] = sell
        sequence_unknown[position] = unknown
        sequence_delta[position] = delta
        sequence_ratio[position] = delta / specified if specified > 0 else np.nan
        if specified <= 0:
            reasons[position] = "NO_SPECIFIED_FLOW"
        elif delta == 0:
            reasons[position] = "ZERO_DELTA"
        elif (direction == "LONG" and delta > 0) or (direction == "SHORT" and delta < 0):
            reasons[position] = "PASS"
            flow_pass[position] = 1
        else:
            reasons[position] = "OPPOSITE_DELTA"

    result["v093_sequence_bars"] = sequence_bars
    result["v093_sequence_buy_volume"] = sequence_buy
    result["v093_sequence_sell_volume"] = sequence_sell
    result["v093_sequence_unknown_volume"] = sequence_unknown
    result["v093_sequence_delta"] = sequence_delta
    result["v093_sequence_signed_volume_ratio"] = sequence_ratio
    result["v093_flow_reason"] = reasons
    result["v093_flow_pass"] = flow_pass
    result["v093_signal"] = np.where(
        result["v093_flow_pass"].eq(1), result["v091_context_signal"], "NONE"
    )
    return result


def _create_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS v093_sequence_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            sweep_timestamp TEXT NOT NULL,
            direction TEXT NOT NULL,
            confirmation_lag INTEGER NOT NULL,
            sequence_bars INTEGER NOT NULL,
            sequence_buy_volume REAL,
            sequence_sell_volume REAL,
            sequence_unknown_volume REAL,
            sequence_delta REAL,
            sequence_signed_volume_ratio REAL,
            flow_reason TEXT NOT NULL,
            flow_pass INTEGER NOT NULL,
            base_non_overlapping INTEGER NOT NULL,
            flow_non_overlapping INTEGER NOT NULL,
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


def build_v093_observations(frame: pd.DataFrame, connection: sqlite3.Connection) -> int:
    _create_table(connection)
    connection.execute("DELETE FROM v093_sequence_observations")
    records = []
    next_base_index = MINIMUM_HISTORY
    next_flow_index = MINIMUM_HISTORY
    last_index = len(frame) - FORWARD_BARS - 1
    for index in range(MINIMUM_HISTORY, last_index):
        row = frame.iloc[index]
        direction = row["v091_context_signal"]
        if direction == "NONE" or pd.isna(row["atr"]):
            continue
        outcome = evaluate_forward_path(frame, index, direction, row["close"], row["atr"])
        base_non_overlapping = int(index >= next_base_index)
        if base_non_overlapping:
            next_base_index = index + FORWARD_BARS + 1
        flow_pass = int(row["v093_flow_pass"])
        flow_non_overlapping = int(flow_pass == 1 and index >= next_flow_index)
        if flow_non_overlapping:
            next_flow_index = index + FORWARD_BARS + 1

        def optional_float(column: str):
            return None if pd.isna(row[column]) else float(row[column])

        records.append(
            (
                str(row["time"]),
                str(row["v091_source_sweep_time"]),
                direction,
                int(row["v091_confirmation_lag"]),
                int(row["v093_sequence_bars"]),
                optional_float("v093_sequence_buy_volume"),
                optional_float("v093_sequence_sell_volume"),
                optional_float("v093_sequence_unknown_volume"),
                optional_float("v093_sequence_delta"),
                optional_float("v093_sequence_signed_volume_ratio"),
                row["v093_flow_reason"],
                flow_pass,
                base_non_overlapping,
                flow_non_overlapping,
                outcome["hit_1r"], outcome["hit_2r"], outcome["hit_3r"],
                outcome["stopped"], outcome["mfe"], outcome["mae"], outcome["outcome_r"],
            )
        )
    connection.executemany(
        """
        INSERT INTO v093_sequence_observations (
            timestamp, sweep_timestamp, direction, confirmation_lag,
            sequence_bars, sequence_buy_volume, sequence_sell_volume,
            sequence_unknown_volume, sequence_delta,
            sequence_signed_volume_ratio, flow_reason, flow_pass,
            base_non_overlapping, flow_non_overlapping, hit_1r, hit_2r,
            hit_3r, stopped, mfe, mae, outcome_r
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        records,
    )
    connection.commit()
    return len(records)


def read_observations(database_file: Path) -> pd.DataFrame:
    with sqlite3.connect(database_file) as connection:
        return pd.read_sql_query(
            "SELECT * FROM v093_sequence_observations ORDER BY timestamp", connection
        )


def summarize(data: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "samples": len(data),
        "average_r": data["outcome_r"].mean(),
        "hit_1r_pct": 100.0 * data["hit_1r"].mean(),
        "hit_2r_pct": 100.0 * data["hit_2r"].mean(),
        "stopped_pct": 100.0 * data["stopped"].mean(),
    })


def print_report(database_file: Path) -> None:
    observations = read_observations(database_file)
    print("\nITRF v0.9.3 SEQUENCE FLOW — DEVELOPMENT DIAGNOSTIC ONLY")
    print("Historical futures flow validates XAU/USD context only; it is not traded.")
    gate = observations.groupby(["direction", "flow_reason"]).size().rename("samples").reset_index()
    print("\nGate decisions:")
    print(gate.to_string(index=False))
    base = observations.loc[observations["base_non_overlapping"] == 1]
    accepted = observations.loc[observations["flow_non_overlapping"] == 1]
    print("\nSame-period v0.9.1 non-overlapping comparator:")
    base_summary = base.groupby("direction").apply(summarize, include_groups=False).reset_index()
    print(base_summary.round(3).to_string(index=False))
    print("\nv0.9.3 sequence-flow accepted non-overlapping observations:")
    accepted_summary = accepted.groupby("direction").apply(summarize, include_groups=False).reset_index()
    print(accepted_summary.round(3).to_string(index=False) if not accepted.empty else "No accepted observations.")
    if not accepted.empty:
        costs = pd.DataFrame({
            "assumed_round_trip_cost_r": [0.0, 0.05, 0.10],
            "samples": [len(accepted)] * 3,
            "average_net_r": [accepted["outcome_r"].mean() - value for value in [0.0, 0.05, 0.10]],
        })
        print("\nFrozen cost-sensitivity diagnostic:")
        print(costs.round(3).to_string(index=False))
        if len(accepted) >= 4:
            chronological = accepted.reset_index(drop=True)
            chronological["development_quarter"] = pd.qcut(
                chronological.index, 4,
                labels=["Development 1", "Development 2", "Development 3", "Development 4"],
            )
            stability = chronological.groupby("development_quarter", observed=True).apply(
                summarize, include_groups=False
            ).reset_index()
            print("\nChronological accepted-signal diagnostic:")
            print(stability.round(3).to_string(index=False))
    print("\nThis is already-inspected development data, not proof of profitability.")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ITRF v0.9.3 sequence-flow research.")
    parser.add_argument("--data-file", type=Path, default=DEFAULT_XAUUSD_DATA)
    parser.add_argument("--orderflow-file", type=Path, default=DEFAULT_ORDERFLOW_DATA)
    parser.add_argument("--database-file", type=Path, default=DEFAULT_DATABASE)
    return parser.parse_known_args()[0]


def main(data_file=None, orderflow_file=None, database_file=None) -> None:
    args = _parse_arguments()
    context = create_context_features(create_features(load_market_data(Path(data_file or args.data_file))))
    merged = merge_orderflow(context, load_orderflow_data(Path(orderflow_file or args.orderflow_file)))
    sequenced = add_sequence_flow(merged)
    selected_database = Path(database_file or args.database_file)
    selected_database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(selected_database) as connection:
        count = build_v093_observations(sequenced, connection)
    print(f"v0.9.3 complete v0.9.1 confirmations written: {count:,}")
    print(f"Database: {selected_database}")
    print_report(selected_database)


if __name__ == "__main__":
    main()

