"""Create consistent ITRF research timeframes from real XAU/USD 1-minute data.

The raw source file is never altered. This script only aggregates its real OHLCV
bars into higher timeframes and saves the 15-minute version used by v0.1.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIRECTORY = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "processed"
PRIMARY_OUTPUT = PROJECT_ROOT / "data" / "XAUUSD.csv"
TIMEFRAMES = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}


def source_file() -> Path:
    files = sorted(RAW_DIRECTORY.glob("*.csv"))
    if len(files) != 1:
        raise RuntimeError(
            "Expected exactly one raw CSV in data/raw. "
            f"Found {len(files)} file(s)."
        )
    return files[0]


def load_raw(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data.columns = [column.strip() for column in data.columns]
    time_column = next((column for column in data.columns if column.lower().startswith("time")), None)
    if time_column is None:
        raise ValueError("The raw CSV has no timestamp column.")

    data = data.rename(columns={time_column: "time"})
    required = ["time", "Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"The raw CSV is missing: {', '.join(missing)}")

    data = data.loc[:, required].rename(columns=str.lower)
    data["time"] = pd.to_datetime(data["time"], format="%Y.%m.%d %H:%M:%S", errors="coerce")
    for column in ["open", "high", "low", "close", "volume"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna().sort_values("time").drop_duplicates("time").reset_index(drop=True)
    if data.empty:
        raise ValueError("The raw CSV had no valid rows.")
    return data


def aggregate(data: pd.DataFrame, frequency: str) -> pd.DataFrame:
    indexed = data.set_index("time")
    bars = indexed.resample(frequency).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    return bars.dropna().reset_index()


def write_csv(data: pd.DataFrame, path: Path) -> None:
    output = data.copy()
    output["time"] = output["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    output.to_csv(path, index=False)


def main() -> None:
    raw_path = source_file()
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    minute_data = load_raw(raw_path)
    write_csv(minute_data, OUTPUT_DIRECTORY / "XAUUSD_1m.csv")

    for label, frequency in TIMEFRAMES.items():
        bars = aggregate(minute_data, frequency)
        path = OUTPUT_DIRECTORY / f"XAUUSD_{label}.csv"
        write_csv(bars, path)
        print(f"{label}: {len(bars):,} bars -> {path.relative_to(PROJECT_ROOT)}")
        if label == "15m":
            write_csv(bars, PRIMARY_OUTPUT)

    print(f"Raw source preserved: {raw_path.relative_to(PROJECT_ROOT)}")
    print("Source timestamps are retained as exported (EET).")


if __name__ == "__main__":
    main()
