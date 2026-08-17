"""Cost-gated Databento CME Gold trade downloader and order-flow aggregator.

Raw exchange trade events are retained as compressed DBN. A separate compact
CSV contains causal 15-minute order-flow features for the research engine.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from research.download_databento import (
        DATASET,
        DEFAULT_KEY_FILE,
        STYPE_IN,
        STYPE_OUT,
        SYMBOL,
        load_api_key,
        validate_date_range,
        write_audit_files,
    )
except ModuleNotFoundError:  # Support ``python research/download_databento_trades.py``.
    from download_databento import (  # type: ignore[no-redef]
        DATASET,
        DEFAULT_KEY_FILE,
        STYPE_IN,
        STYPE_OUT,
        SYMBOL,
        load_api_key,
        validate_date_range,
        write_audit_files,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "trades"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "databento" / "GC_front_orderflow_15m.csv"
DEFAULT_RAW_OUTPUT = PROJECT_ROOT / "data" / "databento" / "GC_front_trades.dbn.zst"
BAR_FREQUENCY = "15min"


def request_parameters(start: str, end: str) -> dict[str, object]:
    return {
        "dataset": DATASET,
        "symbols": SYMBOL,
        "schema": SCHEMA,
        "stype_in": STYPE_IN,
        "start": start,
        "end": end,
    }


def _normalise_trade_frame(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "ts_event" not in data.columns:
        if isinstance(data.index, pd.DatetimeIndex):
            data["ts_event"] = data.index
        else:
            raise ValueError("Databento trade data has no ts_event timestamp.")
    if "ts_event" in data.index.names:
        data = data.reset_index(drop=True)

    required = {"ts_event", "price", "size", "side"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Databento trade data is missing columns: {sorted(missing)}")

    data["ts_event"] = pd.to_datetime(data["ts_event"], utc=True, errors="coerce")
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data["size"] = pd.to_numeric(data["size"], errors="coerce")
    data = data.dropna(subset=["ts_event", "price", "size"])
    data = data.loc[data["size"] > 0].copy()
    if data.empty:
        return data

    side = data["side"].astype("string").str.upper().str.strip()
    data["is_buy"] = side.isin(["B", "BID", "BUY"])
    data["is_sell"] = side.isin(["A", "ASK", "SELL"])
    data["is_unknown"] = ~(data["is_buy"] | data["is_sell"])
    data["bar_time"] = data["ts_event"].dt.floor(BAR_FREQUENCY)
    data["price_size"] = data["price"] * data["size"]
    data["buy_volume"] = data["size"].where(data["is_buy"], 0.0)
    data["sell_volume"] = data["size"].where(data["is_sell"], 0.0)
    data["unknown_volume"] = data["size"].where(data["is_unknown"], 0.0)
    return data


def aggregate_trade_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one trade chunk to mergeable 15-minute partial bars."""
    data = _normalise_trade_frame(frame)
    if data.empty:
        return pd.DataFrame()

    data = data.sort_values("ts_event", kind="stable")
    grouped = data.groupby("bar_time", sort=True, observed=True)
    partial = grouped.agg(
        first_ts=("ts_event", "first"),
        last_ts=("ts_event", "last"),
        first_trade_price=("price", "first"),
        high_trade_price=("price", "max"),
        low_trade_price=("price", "min"),
        last_trade_price=("price", "last"),
        trade_count=("size", "size"),
        total_volume=("size", "sum"),
        price_size=("price_size", "sum"),
        buy_trades=("is_buy", "sum"),
        sell_trades=("is_sell", "sum"),
        unknown_trades=("is_unknown", "sum"),
        buy_volume=("buy_volume", "sum"),
        sell_volume=("sell_volume", "sum"),
        unknown_volume=("unknown_volume", "sum"),
    )
    return partial.reset_index()


def combine_trade_aggregates(partials: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Merge chunk-level partial bars and calculate final causal features."""
    usable = [part for part in partials if not part.empty]
    if not usable:
        raise ValueError("Databento returned no usable trade records.")

    data = pd.concat(usable, ignore_index=True)
    first_rows = (
        data.sort_values(["bar_time", "first_ts"], kind="stable")
        .groupby("bar_time", sort=True, observed=True)
        .first()[["first_trade_price", "first_ts"]]
    )
    last_rows = (
        data.sort_values(["bar_time", "last_ts"], kind="stable")
        .groupby("bar_time", sort=True, observed=True)
        .last()[["last_trade_price", "last_ts"]]
    )
    sums = data.groupby("bar_time", sort=True, observed=True).agg(
        high_trade_price=("high_trade_price", "max"),
        low_trade_price=("low_trade_price", "min"),
        trade_count=("trade_count", "sum"),
        total_volume=("total_volume", "sum"),
        price_size=("price_size", "sum"),
        buy_trades=("buy_trades", "sum"),
        sell_trades=("sell_trades", "sum"),
        unknown_trades=("unknown_trades", "sum"),
        buy_volume=("buy_volume", "sum"),
        sell_volume=("sell_volume", "sum"),
        unknown_volume=("unknown_volume", "sum"),
    )
    bars = sums.join(first_rows).join(last_rows).reset_index()
    # Databento sizes are unsigned integers. Cast before subtraction so a
    # seller-heavy bar cannot wrap around to a value near 2**32.
    volume_columns = ["total_volume", "buy_volume", "sell_volume", "unknown_volume"]
    for column in volume_columns:
        bars[column] = bars[column].astype("int64")
    specified_volume = bars["buy_volume"] + bars["sell_volume"]
    specified_trades = bars["buy_trades"] + bars["sell_trades"]
    bars["volume_delta"] = bars["buy_volume"] - bars["sell_volume"]
    bars["signed_volume_ratio"] = np.where(
        specified_volume > 0,
        bars["volume_delta"] / specified_volume,
        np.nan,
    )
    bars["buy_trade_ratio"] = np.where(
        specified_trades > 0,
        bars["buy_trades"] / specified_trades,
        np.nan,
    )
    bars["aggressor_coverage"] = np.where(
        bars["trade_count"] > 0,
        specified_trades / bars["trade_count"],
        np.nan,
    )
    bars["vwap"] = bars["price_size"] / bars["total_volume"]
    bars["trades_per_second"] = bars["trade_count"] / (15 * 60)
    bars["volume_per_second"] = bars["total_volume"] / (15 * 60)

    integer_columns = [
        "trade_count",
        "buy_trades",
        "sell_trades",
        "unknown_trades",
    ]
    for column in integer_columns:
        bars[column] = bars[column].astype("int64")

    output_columns = [
        "bar_time",
        "first_trade_price",
        "high_trade_price",
        "low_trade_price",
        "last_trade_price",
        "vwap",
        "trade_count",
        "total_volume",
        "buy_trades",
        "sell_trades",
        "unknown_trades",
        "buy_volume",
        "sell_volume",
        "unknown_volume",
        "volume_delta",
        "signed_volume_ratio",
        "buy_trade_ratio",
        "aggressor_coverage",
        "trades_per_second",
        "volume_per_second",
        "first_ts",
        "last_ts",
    ]
    return bars[output_columns].sort_values("bar_time").reset_index(drop=True)


def validate_orderflow_bars(bars: pd.DataFrame, processed_records: int) -> None:
    if bars.empty:
        raise ValueError("No order-flow bars were produced.")
    if bars["bar_time"].duplicated().any():
        raise ValueError("Order-flow output contains duplicate bar timestamps.")
    if not bars["bar_time"].is_monotonic_increasing:
        raise ValueError("Order-flow output is not chronological.")
    if int(bars["trade_count"].sum()) != processed_records:
        raise ValueError("Aggregated trade count does not match processed records.")
    if (bars[["trade_count", "total_volume"]] <= 0).any().any():
        raise ValueError("Order-flow output contains a non-positive count or volume.")
    ratios = bars["signed_volume_ratio"].dropna()
    if not ratios.between(-1.0, 1.0).all():
        raise ValueError("Signed-volume ratio is outside [-1, 1].")
    coverage = bars["aggressor_coverage"].dropna()
    if not coverage.between(0.0, 1.0).all():
        raise ValueError("Aggressor coverage is outside [0, 1].")


def aggregate_store(store, chunk_records: int) -> tuple[pd.DataFrame, int]:
    partials: list[pd.DataFrame] = []
    processed_records = 0
    for frame in store.to_df(count=chunk_records, map_symbols=False):
        partial = aggregate_trade_frame(frame)
        if not partial.empty:
            partials.append(partial)
            processed_records += int(partial["trade_count"].sum())
    bars = combine_trade_aggregates(partials)
    validate_orderflow_bars(bars, processed_records)
    return bars, processed_records


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cost-gated Databento CME Gold trade and order-flow downloader."
    )
    parser.add_argument("--start", required=True, help="Inclusive UTC start date/time.")
    parser.add_argument("--end", required=True, help="Exclusive UTC end date/time.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_OUTPUT)
    parser.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--process-existing",
        action="store_true",
        help="Aggregate an existing raw DBN file locally without an API request.",
    )
    parser.add_argument("--max-cost-usd", type=float, default=0.0)
    parser.add_argument("--chunk-records", type=int, default=500_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_arguments()
    validate_date_range(args.start, args.end)
    if args.max_cost_usd < 0:
        raise ValueError("--max-cost-usd cannot be negative.")
    if args.chunk_records < 1:
        raise ValueError("--chunk-records must be positive.")
    if args.download and args.process_existing:
        raise ValueError("Choose either --download or --process-existing, not both.")
    if args.download and not args.overwrite:
        existing = [path for path in [args.output, args.raw_output] if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing data: {existing}")

    try:
        import databento as db
    except ImportError as exc:
        raise RuntimeError("Install project requirements before using Databento.") from exc

    if args.process_existing:
        if not args.raw_output.exists():
            raise FileNotFoundError(f"Raw DBN file does not exist: {args.raw_output}")
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing data: {args.output}")
        store = db.DBNStore.from_file(args.raw_output)
        bars, processed_records = aggregate_store(store, args.chunk_records)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        bars.to_csv(args.output, index=False)
        print("Local processing only: no Databento API request was made.")
        print(f"Read {processed_records:,} raw trades from {args.raw_output}")
        print(f"Saved {len(bars):,} order-flow bars to {args.output}")
        print(f"Aggressor-side coverage: {bars['aggressor_coverage'].mean():.2%}")
        return

    client = db.Historical(load_api_key(args.key_file))
    parameters = request_parameters(args.start, args.end)
    estimate = float(client.metadata.get_cost(**parameters))
    record_count = int(client.metadata.get_record_count(**parameters))
    print(f"Dataset: {DATASET}")
    print(f"Instrument: CME Gold front contract ({SYMBOL})")
    print(f"Schema: {SCHEMA}")
    print(f"Range: {args.start} to {args.end} UTC")
    print(f"Estimated trade records: {record_count:,}")
    print(f"Estimated Databento cost: ${estimate:.4f} USD")

    if not args.download:
        print("Estimate only: no market data was requested.")
        return
    if estimate > args.max_cost_usd:
        raise RuntimeError(
            f"Download refused: estimated cost ${estimate:.4f} exceeds "
            f"the ${args.max_cost_usd:.4f} limit."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    store = client.timeseries.get_range(
        **parameters,
        stype_out=STYPE_OUT,
        path=args.raw_output,
    )
    bars, processed_records = aggregate_store(store, args.chunk_records)
    bars.to_csv(args.output, index=False)
    roll_path, condition_path, roll_count = write_audit_files(client, args)
    print(f"Saved {processed_records:,} raw trades to {args.raw_output}")
    print(f"Saved {len(bars):,} order-flow bars to {args.output}")
    print(f"Aggressor-side coverage: {bars['aggressor_coverage'].mean():.2%}")
    print(f"Saved {roll_count:,} contract mapping intervals to {roll_path}")
    print(f"Saved provider quality conditions to {condition_path}")


if __name__ == "__main__":
    main()
