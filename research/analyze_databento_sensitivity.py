"""Measure Databento roll/degradation exposure without changing ITRF rules."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from itrf_research import FORWARD_BARS, load_market_data
from validate_market_data import audit_conditions, audit_roll_schedule, calculate_roll_gaps


def quality_flags(
    timestamps: pd.Series,
    data: pd.DataFrame,
    transition_times: set[pd.Timestamp],
    degraded_dates: set,
    end_timestamps: pd.Series | None = None,
) -> pd.DataFrame:
    positions = pd.Series(data.index, index=data["time"]).to_dict()
    roll_positions = {positions[time] for time in transition_times if time in positions}
    flags = []

    for row_number, timestamp in enumerate(pd.to_datetime(timestamps)):
        start = positions.get(timestamp)
        if start is None:
            raise ValueError(f"Observation timestamp is absent from market data: {timestamp}")
        if end_timestamps is None:
            end = min(start + FORWARD_BARS, len(data) - 1)
        else:
            end_time = pd.Timestamp(end_timestamps.iloc[row_number])
            end = positions.get(end_time)
            if end is None:
                raise ValueError(f"Exit timestamp is absent from market data: {end_time}")
        window_dates = set(data.loc[start:end, "time"].dt.date)
        flags.append({
            "roll_exposed": any(start <= position <= end for position in roll_positions),
            "degraded_exposed": bool(window_dates & degraded_dates),
        })
    result = pd.DataFrame(flags)
    result["quality_exposed"] = result["roll_exposed"] | result["degraded_exposed"]
    return result


def print_sensitivity(arguments: argparse.Namespace) -> None:
    data = load_market_data(arguments.data_file)
    rolls = audit_roll_schedule(arguments.roll_file)
    roll_gaps = calculate_roll_gaps(data, rolls)
    transition_times = set(pd.to_datetime(roll_gaps["first_bar"]))
    degraded, _ = audit_conditions(arguments.condition_file, data)
    degraded_dates = set(degraded["date"])
    oos_start = pd.Timestamp(arguments.oos_start)

    with sqlite3.connect(arguments.baseline_database) as connection:
        baseline = pd.read_sql_query(
            "SELECT timestamp, direction, outcome_r FROM feature_observations",
            connection,
        )
    baseline["timestamp"] = pd.to_datetime(baseline["timestamp"])
    baseline = pd.concat(
        [baseline.reset_index(drop=True), quality_flags(baseline["timestamp"], data, transition_times, degraded_dates)],
        axis=1,
    )
    baseline["period"] = baseline["timestamp"].ge(oos_start).map({False: "TRAIN", True: "OOS"})
    baseline["quality_group"] = baseline["quality_exposed"].map({False: "unexposed", True: "roll/degraded exposed"})

    print("DATABENTO QUALITY-EXPOSURE SENSITIVITY")
    print("\nFrozen v0.8 forward labels:")
    baseline_summary = baseline.groupby(["period", "quality_group"], sort=False).agg(
        samples=("outcome_r", "size"),
        average_r=("outcome_r", "mean"),
    ).reset_index()
    print(baseline_summary.round(3).to_string(index=False))

    with sqlite3.connect(arguments.exit_database) as connection:
        exits = pd.read_sql_query("SELECT * FROM v09_exit_observations", connection)
    exits["timestamp"] = pd.to_datetime(exits["timestamp"])
    exits["exit_timestamp"] = pd.to_datetime(exits["exit_timestamp"])
    exit_flags = quality_flags(
        exits["timestamp"],
        data,
        transition_times,
        degraded_dates,
        exits["exit_timestamp"],
    )
    exits = pd.concat([exits.reset_index(drop=True), exit_flags], axis=1)
    exits = exits.loc[exits["timestamp"] >= oos_start].copy()
    exits["quality_group"] = exits["quality_exposed"].map({False: "unexposed", True: "roll/degraded exposed"})
    print("\nFrozen v0.9 exit models, OOS and zero-cost:")
    exit_summary = exits.groupby(["model", "quality_group"], sort=False).agg(
        trades=("net_outcome_r", "size"),
        average_r=("net_outcome_r", "mean"),
    ).reset_index()
    print(exit_summary.round(3).to_string(index=False))
    print("\nThis sensitivity removes no records and changes no strategy rule; it only reports metadata exposure.")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report strategy-observation exposure to Databento quality events.")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--roll-file", type=Path, required=True)
    parser.add_argument("--condition-file", type=Path, required=True)
    parser.add_argument("--baseline-database", type=Path, required=True)
    parser.add_argument("--exit-database", type=Path, required=True)
    parser.add_argument("--oos-start", default="2025-07-28 17:00:00")
    return parser.parse_args()


def main() -> None:
    print_sensitivity(_parse_arguments())


if __name__ == "__main__":
    main()
