"""Estimate and download CME Gold futures bars from Databento.

The downloader is deliberately cost-gated. It always obtains and prints an API
estimate first, and it will not request market data unless ``--download`` is
present and the estimate is no greater than ``--max-cost-usd``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "databento" / "GC_front_15m.csv"
DEFAULT_KEY_FILE = PROJECT_ROOT / ".secrets" / "databento_api_key"
DATASET = "GLBX.MDP3"
SCHEMA = "ohlcv-1m"
SYMBOL = "GC.v.0"
STYPE_IN = "continuous"
STYPE_OUT = "instrument_id"


def load_api_key(key_file: Path = DEFAULT_KEY_FILE) -> str:
    """Load the key from the environment or the ignored local secret file."""
    key = os.environ.get("DATABENTO_API_KEY", "").strip()
    if not key and key_file.exists():
        key = key_file.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError(
            "Databento API key not found. Set DATABENTO_API_KEY or create "
            f"the ignored local file {key_file}."
        )
    if not key.startswith("db-"):
        raise ValueError("The Databento API key has an unexpected format.")
    return key


def validate_date_range(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_time = pd.Timestamp(start, tz="UTC")
    end_time = pd.Timestamp(end, tz="UTC")
    if start_time >= end_time:
        raise ValueError("--start must be earlier than --end.")
    return start_time, end_time


def resample_ohlcv_to_15m(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert Databento one-minute OHLCV records to engine-ready UTC bars."""
    data = frame.copy()
    if isinstance(data.index, pd.DatetimeIndex):
        index_name = data.index.name or "ts_event"
        data = data.reset_index().rename(columns={index_name: "time"})
    elif "ts_event" in data.columns:
        data = data.rename(columns={"ts_event": "time"})
    elif "time" not in data.columns:
        raise ValueError("Databento data has no ts_event/time field.")

    required = {"time", "open", "high", "low", "close", "volume"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Databento OHLCV data is missing columns: {sorted(missing)}")

    data["time"] = pd.to_datetime(data["time"], utc=True, errors="coerce")
    for column in ["open", "high", "low", "close", "volume"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=list(required)).sort_values("time")
    if data.empty:
        raise ValueError("Databento returned no usable OHLCV records.")
    if data["time"].duplicated().any():
        raise ValueError("Databento returned overlapping records at the same timestamp.")

    indexed = data.set_index("time")
    aggregation = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    bars = indexed.resample("15min").agg(aggregation)
    bars = bars.dropna(subset=["open", "high", "low", "close"]).reset_index()

    engine_bars = bars[["time", "open", "high", "low", "close", "volume"]].copy()
    engine_bars["time"] = engine_bars["time"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    return engine_bars


def roll_schedule_from_resolution(resolution: dict, symbol: str = SYMBOL) -> pd.DataFrame:
    """Convert Databento continuous-symbol mappings to an auditable schedule."""
    mappings = resolution.get("result", {}).get(symbol, [])
    records = [
        {
            "start_date": mapping["d0"],
            "end_date": mapping["d1"],
            "instrument_id": mapping["s"],
        }
        for mapping in mappings
    ]
    if not records:
        raise ValueError(f"No continuous-contract mappings returned for {symbol}.")
    return pd.DataFrame(records)


def request_parameters(start: str, end: str) -> dict[str, object]:
    return {
        "dataset": DATASET,
        "symbols": SYMBOL,
        "schema": SCHEMA,
        "stype_in": STYPE_IN,
        "start": start,
        "end": end,
    }


def write_audit_files(client, args: argparse.Namespace) -> tuple[Path, Path, int]:
    """Save continuous mappings and provider quality conditions without price data."""
    resolution = client.symbology.resolve(
        dataset=DATASET,
        symbols=SYMBOL,
        stype_in=STYPE_IN,
        stype_out=STYPE_OUT,
        start_date=args.start,
        end_date=args.end,
    )
    rolls = roll_schedule_from_resolution(resolution)
    conditions = pd.DataFrame(
        client.metadata.get_dataset_condition(DATASET, args.start, args.end)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    roll_path = args.output.with_name(f"{args.output.stem}_rolls.csv")
    condition_path = args.output.with_name(f"{args.output.stem}_conditions.csv")
    rolls.to_csv(roll_path, index=False)
    conditions.to_csv(condition_path, index=False)
    return roll_path, condition_path, len(rolls)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cost-gated Databento CME Gold futures downloader."
    )
    parser.add_argument("--start", required=True, help="Inclusive UTC start date/time.")
    parser.add_argument("--end", required=True, help="Exclusive UTC end date/time.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--download", action="store_true", help="Request data after estimating cost.")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Refresh free symbology and quality-condition audit files without price data.",
    )
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=0.0,
        help="Hard cost ceiling. A download is refused above this estimate.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_arguments()
    validate_date_range(args.start, args.end)
    if args.max_cost_usd < 0:
        raise ValueError("--max-cost-usd cannot be negative.")
    if args.download and args.audit_only:
        raise ValueError("Choose either --download or --audit-only, not both.")
    if args.output.exists() and not args.overwrite and args.download:
        raise FileExistsError(f"Refusing to overwrite existing file: {args.output}")

    try:
        import databento as db
    except ImportError as exc:
        raise RuntimeError("Install project requirements before using Databento.") from exc

    client = db.Historical(load_api_key(args.key_file))
    parameters = request_parameters(args.start, args.end)
    estimate = float(client.metadata.get_cost(**parameters))
    record_count = int(client.metadata.get_record_count(**parameters))
    print(f"Dataset: {DATASET}")
    print(f"Instrument: CME Gold front contract ({SYMBOL}, unadjusted continuous mapping)")
    print(f"Range: {args.start} to {args.end} UTC")
    print(f"Estimated one-minute records: {record_count:,}")
    print(f"Estimated Databento cost: ${estimate:.4f} USD")

    if args.audit_only:
        roll_path, condition_path, roll_count = write_audit_files(client, args)
        print(f"Saved {roll_count:,} contract mapping intervals to {roll_path}")
        print(f"Saved provider quality conditions to {condition_path}")
        print("Audit only: no market data was requested.")
        return
    if not args.download:
        print("Estimate only: no market data was requested.")
        return
    if estimate > args.max_cost_usd:
        raise RuntimeError(
            f"Download refused: estimated cost ${estimate:.4f} exceeds "
            f"the ${args.max_cost_usd:.4f} limit."
        )

    store = client.timeseries.get_range(**parameters, stype_out=STYPE_OUT)
    bars = resample_ohlcv_to_15m(store.to_df())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bars.to_csv(args.output, index=False)
    roll_path, condition_path, roll_count = write_audit_files(client, args)
    print(f"Saved {len(bars):,} 15-minute bars to {args.output}")
    print(f"Saved {roll_count:,} contract mapping intervals to {roll_path}")
    print(f"Saved provider quality conditions to {condition_path}")
    print("The continuous series is not back-adjusted; roll gaps remain in the prices.")


if __name__ == "__main__":
    main()
