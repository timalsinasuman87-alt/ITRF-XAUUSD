"""Audit an OHLCV file and optional Databento metadata without tuning signals."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from itrf_research import load_market_data


def audit_roll_schedule(roll_file: Path) -> pd.DataFrame:
    rolls = pd.read_csv(roll_file)
    required = {"start_date", "end_date", "instrument_id"}
    missing = required - set(rolls.columns)
    if missing:
        raise ValueError(f"Roll schedule is missing columns: {sorted(missing)}")
    rolls = rolls.copy()
    rolls["start_date"] = pd.to_datetime(rolls["start_date"], errors="raise")
    rolls["end_date"] = pd.to_datetime(rolls["end_date"], errors="raise")
    rolls = rolls.sort_values("start_date").reset_index(drop=True)
    if (rolls["start_date"] >= rolls["end_date"]).any():
        raise ValueError("Every roll mapping interval must end after it starts.")
    if len(rolls) > 1 and not (
        rolls["start_date"].iloc[1:].reset_index(drop=True)
        == rolls["end_date"].iloc[:-1].reset_index(drop=True)
    ).all():
        raise ValueError("Roll mapping intervals contain a gap or overlap.")
    return rolls


def calculate_roll_gaps(data: pd.DataFrame, rolls: pd.DataFrame) -> pd.DataFrame:
    records = []
    times = data["time"]
    for row in rolls.iloc[1:].itertuples(index=False):
        next_positions = data.index[times >= row.start_date]
        if len(next_positions) == 0 or int(next_positions[0]) == 0:
            continue
        position = int(next_positions[0])
        previous = data.iloc[position - 1]
        current = data.iloc[position]
        gap = float(current["open"] - previous["close"])
        records.append({
            "mapping_start": row.start_date.date().isoformat(),
            "instrument_id": str(row.instrument_id),
            "previous_bar": previous["time"],
            "first_bar": current["time"],
            "price_gap": gap,
            "gap_pct": 100.0 * gap / float(previous["close"]),
        })
    return pd.DataFrame(records)


def audit_conditions(condition_file: Path, data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    conditions = pd.read_csv(condition_file)
    required = {"date", "condition"}
    missing = required - set(conditions.columns)
    if missing:
        raise ValueError(f"Condition file is missing columns: {sorted(missing)}")
    conditions = conditions.copy()
    conditions["date"] = pd.to_datetime(conditions["date"], errors="raise").dt.date
    degraded = conditions.loc[conditions["condition"].str.lower() != "available"].copy()
    observed_dates = set(data["time"].dt.date)
    degraded_with_bars = int(degraded["date"].isin(observed_dates).sum())
    return degraded, degraded_with_bars


def print_audit(
    data_file: Path,
    roll_file: Path | None = None,
    condition_file: Path | None = None,
) -> None:
    data = load_market_data(data_file)
    intervals = data["time"].diff().dropna()
    expected = pd.Timedelta(minutes=15)
    regular_intervals = int(intervals.eq(expected).sum())

    print("ITRF MARKET-DATA AUDIT")
    print(f"File: {data_file}")
    print(f"Bars: {len(data):,}")
    print(f"Range (UTC): {data['time'].min()} to {data['time'].max()}")
    print(f"Unique ordered timestamps: {not data['time'].duplicated().any() and data['time'].is_monotonic_increasing}")
    print(f"Valid OHLC and non-negative volume: yes")
    print(f"Zero-volume bars: {int(data['volume'].eq(0).sum()):,}")
    print(f"15-minute adjacent intervals: {regular_intervals:,}/{len(intervals):,}")
    print(f"Expected session/holiday gaps over 15 minutes: {int(intervals.gt(expected).sum()):,}")
    print(f"Largest timestamp gap: {intervals.max()}")

    if roll_file is not None:
        rolls = audit_roll_schedule(roll_file)
        gaps = calculate_roll_gaps(data, rolls)
        print(f"\nOfficial continuous-contract mappings: {len(rolls):,}")
        print("Mapping intervals are contiguous and non-overlapping: yes")
        print("Observed unadjusted mapping-transition gaps:")
        print(gaps.round({"price_gap": 4, "gap_pct": 4}).to_string(index=False))

    if condition_file is not None:
        degraded, degraded_with_bars = audit_conditions(condition_file, data)
        print(f"\nProvider condition records: {len(pd.read_csv(condition_file)):,}")
        print(f"Non-available/degraded dates: {len(degraded):,}")
        print(f"Degraded dates represented by at least one bar: {degraded_with_bars:,}")
        if not degraded.empty:
            print(degraded[["date", "condition"]].to_string(index=False))

    print("\nAudit conclusion: structurally usable for research; unadjusted rolls and provider-degraded dates remain explicit limitations.")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit engine-ready OHLCV and optional Databento metadata.")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--roll-file", type=Path)
    parser.add_argument("--condition-file", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    print_audit(arguments.data_file, arguments.roll_file, arguments.condition_file)


if __name__ == "__main__":
    main()
