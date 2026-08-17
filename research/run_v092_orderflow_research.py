"""Run the pre-registered v0.9.2 XAU/USD order-flow validation study.

CME Gold futures trades are used only as an independent contextual validation
feed. Candidates and outcomes remain defined on the XAU/USD OHLCV series.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from itrf_context import create_context_features
from itrf_research import FORWARD_BARS, create_features, evaluate_forward_path, load_market_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XAUUSD_DATA = PROJECT_ROOT / "data" / "XAUUSD.csv"
DEFAULT_ORDERFLOW_DATA = PROJECT_ROOT / "data" / "databento" / "GC_front_orderflow_15m.csv"
DEFAULT_DATABASE = PROJECT_ROOT / "database" / "itrf_v092_orderflow.db"
MINIMUM_HISTORY = 250


def load_orderflow_data(path: Path = DEFAULT_ORDERFLOW_DATA) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing historical order-flow data: {path}")
    data = pd.read_csv(path)
    required = {
        "bar_time",
        "trade_count",
        "total_volume",
        "buy_volume",
        "sell_volume",
        "unknown_volume",
        "volume_delta",
        "signed_volume_ratio",
        "aggressor_coverage",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Order-flow data is missing columns: {sorted(missing)}")
    data["time"] = pd.to_datetime(data["bar_time"], errors="coerce", utc=True).dt.tz_convert(None)
    numeric = sorted(required - {"bar_time"})
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["time", *numeric]).sort_values("time").reset_index(drop=True)
    if data.empty:
        raise ValueError("No valid historical order-flow rows remain.")
    if data["time"].duplicated().any() or not data["time"].is_monotonic_increasing:
        raise ValueError("Order-flow timestamps must be unique and chronological.")
    if not data["signed_volume_ratio"].between(-1.0, 1.0).all():
        raise ValueError("Order-flow signed-volume ratio is outside [-1, 1].")
    if not data["aggressor_coverage"].between(0.0, 1.0).all():
        raise ValueError("Order-flow aggressor coverage is outside [0, 1].")
    return data.drop(columns="bar_time")


def merge_orderflow(context: pd.DataFrame, orderflow: pd.DataFrame) -> pd.DataFrame:
    """Join completed 15-minute bars by exact UTC start; never use nearest time."""
    flow_columns = [
        "time",
        "trade_count",
        "total_volume",
        "buy_volume",
        "sell_volume",
        "unknown_volume",
        "volume_delta",
        "signed_volume_ratio",
        "aggressor_coverage",
    ]
    result = context.merge(
        orderflow[flow_columns],
        on="time",
        how="left",
        validate="one_to_one",
        indicator="v092_flow_join",
    )
    result["v092_flow_available"] = result["v092_flow_join"].eq("both").astype(int)
    return classify_orderflow_gate(result)


def classify_orderflow_gate(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"v091_context_signal", "volume_delta", "aggressor_coverage"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing v0.9.2 gate columns: {sorted(missing)}")
    result = frame.copy()
    available = result["volume_delta"].notna() & result["aggressor_coverage"].gt(0)
    direction_matches = (
        (result["v091_context_signal"].eq("LONG") & result["volume_delta"].gt(0))
        | (result["v091_context_signal"].eq("SHORT") & result["volume_delta"].lt(0))
    )
    result["v092_flow_pass"] = (available & direction_matches).astype(int)
    result["v092_signal"] = np.where(
        result["v092_flow_pass"].eq(1),
        result["v091_context_signal"],
        "NONE",
    )
    result["v092_flow_reason"] = np.select(
        [
            result["v091_context_signal"].eq("NONE"),
            ~available,
            result["volume_delta"].eq(0),
            direction_matches,
        ],
        ["NO_V091_SIGNAL", "FLOW_UNAVAILABLE", "ZERO_DELTA", "PASS"],
        default="OPPOSITE_DELTA",
    )
    return result


def _create_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS v092_orderflow_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            sweep_timestamp TEXT NOT NULL,
            direction TEXT NOT NULL,
            market_regime TEXT NOT NULL,
            structure_event TEXT NOT NULL,
            confirmation_lag INTEGER NOT NULL,
            futures_trade_count INTEGER,
            futures_total_volume REAL,
            futures_volume_delta REAL,
            futures_signed_volume_ratio REAL,
            futures_aggressor_coverage REAL,
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


def build_v092_observations(frame: pd.DataFrame, connection: sqlite3.Connection) -> int:
    """Store all v0.9.1 candidates and their frozen v0.9.2 gate decision."""
    _create_table(connection)
    connection.execute("DELETE FROM v092_orderflow_observations")
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
        flow_pass = int(row["v092_flow_pass"])
        flow_non_overlapping = int(flow_pass == 1 and index >= next_flow_index)
        if flow_non_overlapping:
            next_flow_index = index + FORWARD_BARS + 1
        structure_event = "CHoCH" if (
            (direction == "LONG" and row["bullish_choch"] == 1)
            or (direction == "SHORT" and row["bearish_choch"] == 1)
        ) else "BOS"
        records.append(
            (
                str(row["time"]),
                str(row["v091_source_sweep_time"]),
                direction,
                row["market_regime"],
                structure_event,
                int(row["v091_confirmation_lag"]),
                None if pd.isna(row["trade_count"]) else int(row["trade_count"]),
                None if pd.isna(row["total_volume"]) else float(row["total_volume"]),
                None if pd.isna(row["volume_delta"]) else float(row["volume_delta"]),
                None if pd.isna(row["signed_volume_ratio"]) else float(row["signed_volume_ratio"]),
                None if pd.isna(row["aggressor_coverage"]) else float(row["aggressor_coverage"]),
                row["v092_flow_reason"],
                flow_pass,
                base_non_overlapping,
                flow_non_overlapping,
                outcome["hit_1r"],
                outcome["hit_2r"],
                outcome["hit_3r"],
                outcome["stopped"],
                outcome["mfe"],
                outcome["mae"],
                outcome["outcome_r"],
            )
        )

    connection.executemany(
        """
        INSERT INTO v092_orderflow_observations (
            timestamp, sweep_timestamp, direction, market_regime,
            structure_event, confirmation_lag, futures_trade_count,
            futures_total_volume, futures_volume_delta,
            futures_signed_volume_ratio, futures_aggressor_coverage,
            flow_reason, flow_pass, base_non_overlapping,
            flow_non_overlapping, hit_1r, hit_2r, hit_3r, stopped,
            mfe, mae, outcome_r
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        records,
    )
    connection.commit()
    return len(records)


def read_observations(database_file: Path) -> pd.DataFrame:
    with sqlite3.connect(database_file) as connection:
        return pd.read_sql_query(
            "SELECT * FROM v092_orderflow_observations ORDER BY timestamp", connection
        )


def _summarize(data: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "samples": len(data),
            "average_r": data["outcome_r"].mean(),
            "hit_1r_pct": 100.0 * data["hit_1r"].mean(),
            "hit_2r_pct": 100.0 * data["hit_2r"].mean(),
            "stopped_pct": 100.0 * data["stopped"].mean(),
        }
    )


def print_development_report(database_file: Path, matched_bars: int, total_bars: int) -> None:
    observations = read_observations(database_file)
    print("\nITRF v0.9.2 ORDER-FLOW STUDY — DEVELOPMENT DIAGNOSTIC ONLY")
    print("CME futures trades are validation context; the research instrument is XAU/USD.")
    print(f"Exact UTC order-flow coverage: {matched_bars:,}/{total_bars:,} bars ({100.0 * matched_bars / total_bars:.2f}%)")
    if observations.empty:
        print("No complete v0.9.1 confirmations were available.")
        return

    print("\nGate decisions:")
    gate = observations.groupby(["direction", "flow_reason"]).size().rename("samples").reset_index()
    print(gate.to_string(index=False))

    base = observations.loc[observations["base_non_overlapping"] == 1]
    accepted = observations.loc[observations["flow_non_overlapping"] == 1]
    comparator = base.groupby("direction").apply(_summarize, include_groups=False).reset_index()
    result = accepted.groupby("direction").apply(_summarize, include_groups=False).reset_index()
    print("\nSame-period v0.9.1 non-overlapping comparator:")
    print(comparator.round(3).to_string(index=False) if not comparator.empty else "No observations.")
    print("\nv0.9.2 flow-accepted non-overlapping observations:")
    print(result.round(3).to_string(index=False) if not result.empty else "No accepted observations.")

    if not accepted.empty:
        costs = pd.DataFrame(
            {
                "assumed_round_trip_cost_r": [0.0, 0.05, 0.10],
                "samples": [len(accepted)] * 3,
                "average_net_r": [accepted["outcome_r"].mean() - cost for cost in [0.0, 0.05, 0.10]],
            }
        )
        print("\nFrozen cost-sensitivity diagnostic:")
        print(costs.round(3).to_string(index=False))
        if len(accepted) >= 4:
            chronological = accepted.reset_index(drop=True)
            chronological["development_quarter"] = pd.qcut(
                chronological.index,
                q=4,
                labels=["Development 1", "Development 2", "Development 3", "Development 4"],
            )
            stability = chronological.groupby("development_quarter", observed=True).apply(
                _summarize, include_groups=False
            ).reset_index()
            print("\nChronological accepted-signal diagnostic:")
            print(stability.round(3).to_string(index=False))

    print("\nThis period was already designated as development data.")
    print("No profitability claim or parameter selection is permitted from this report.")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ITRF v0.9.2 order-flow diagnostics.")
    parser.add_argument("--data-file", type=Path, default=DEFAULT_XAUUSD_DATA)
    parser.add_argument("--orderflow-file", type=Path, default=DEFAULT_ORDERFLOW_DATA)
    parser.add_argument("--database-file", type=Path, default=DEFAULT_DATABASE)
    return parser.parse_known_args()[0]


def main(data_file=None, orderflow_file=None, database_file=None) -> None:
    arguments = _parse_arguments()
    selected_data = Path(data_file or arguments.data_file)
    selected_orderflow = Path(orderflow_file or arguments.orderflow_file)
    selected_database = Path(database_file or arguments.database_file)
    context = create_context_features(create_features(load_market_data(selected_data)))
    merged = merge_orderflow(context, load_orderflow_data(selected_orderflow))
    matched_bars = int(merged["v092_flow_available"].sum())
    selected_database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(selected_database) as connection:
        count = build_v092_observations(merged, connection)
    print(f"v0.9.2 complete v0.9.1 confirmations written: {count:,}")
    print(f"Database: {selected_database}")
    print_development_report(selected_database, matched_bars, len(merged))


if __name__ == "__main__":
    main()

