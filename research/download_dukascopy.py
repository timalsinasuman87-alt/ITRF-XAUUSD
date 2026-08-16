"""Download free Dukascopy XAU/USD ticks and aggregate them to 15-minute bars.

The resulting CSV is suitable for research only. It uses the public Dukascopy
data-feed endpoint and UTC timestamps; it is not broker execution data and its
tick-side volume is not directly comparable with a CFD broker's volume.
"""

from __future__ import annotations

import argparse
import lzma
import struct
import time
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "XAUUSD.csv"
BAR_MILLISECONDS = 15 * 60 * 1000
TICK_SIZE = struct.calcsize(">IIIff")
PRICE_SCALE = 1_000.0
USER_AGENT = "ITRF-XAUUSD-research/0.9 (historical research)"


def hour_url(symbol: str, hour: datetime) -> str:
    """Return the Dukascopy URL. Its month component is zero-indexed."""
    return (
        "https://datafeed.dukascopy.com/datafeed/"
        f"{symbol.upper()}/{hour.year}/{hour.month - 1:02d}/{hour.day:02d}/"
        f"{hour.hour:02d}h_ticks.bi5"
    )


def decode_ticks(payload: bytes) -> Iterable[tuple[int, float, float, float]]:
    """Yield millisecond offset, bid, ask and combined tick-side volume."""
    raw = lzma.decompress(payload, format=lzma.FORMAT_ALONE)
    if len(raw) % TICK_SIZE:
        raise ValueError("Dukascopy tick payload has an incomplete record.")
    for offset, ask, bid, ask_volume, bid_volume in struct.iter_unpack(">IIIff", raw):
        yield offset, bid / PRICE_SCALE, ask / PRICE_SCALE, ask_volume + bid_volume


def download_hour(symbol: str, hour: datetime, retries: int) -> bytes | None:
    request = Request(hour_url(symbol, hour), headers={"User-Agent": USER_AGENT})
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=30) as response:
                return response.read()
        except HTTPError as error:
            if error.code == 404:
                return None  # Weekend/holiday hours have no file.
            if error.code == 429 and attempt < retries:
                retry_after = error.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 10.0 * (attempt + 1)
                print(f"Source rate-limited; waiting {delay:.0f} seconds before retrying.")
                time.sleep(delay)
                continue
            raise RuntimeError(f"Dukascopy returned HTTP {error.code} for {hour_url(symbol, hour)}") from error
        except URLError as error:
            raise RuntimeError(f"Could not download {hour_url(symbol, hour)}: {error.reason}") from error
    raise AssertionError("The retry loop must return or raise.")


def aggregate_hour(payload: bytes, hour: datetime) -> list[dict[str, object]]:
    """Aggregate one compressed hour into causal 15-minute OHLCV bars."""
    bars: dict[datetime, dict[str, object]] = {}
    for offset, bid, ask, volume in decode_ticks(payload):
        timestamp = hour + timedelta(milliseconds=offset)
        bucket = timestamp.replace(minute=(timestamp.minute // 15) * 15, second=0, microsecond=0)
        price = (bid + ask) / 2
        current = bars.get(bucket)
        if current is None:
            bars[bucket] = {
                "time": bucket,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": volume,
            }
        else:
            current["high"] = max(float(current["high"]), price)
            current["low"] = min(float(current["low"]), price)
            current["close"] = price
            current["volume"] = float(current["volume"]) + volume
    return [bars[key] for key in sorted(bars)]


def download_range(symbol: str, start: datetime, end: datetime, pause_seconds: float, retries: int) -> pd.DataFrame:
    """Download [start, end) in UTC, retaining only aggregated bars in memory."""
    rows: list[dict[str, object]] = []
    hour = start
    completed = 0
    missing = 0
    while hour < end:
        payload = download_hour(symbol, hour, retries)
        if payload is None:
            missing += 1
        else:
            rows.extend(aggregate_hour(payload, hour))
        completed += 1
        if completed % 24 == 0:
            print(f"Downloaded through {hour.date()} ({len(rows):,} 15-minute bars)")
        hour += timedelta(hours=1)
        time.sleep(pause_seconds)
    if not rows:
        raise RuntimeError("No ticks were downloaded. Check the symbol and date range.")
    print(f"Skipped {missing:,} unavailable hourly file(s), usually weekends or holidays.")
    return pd.DataFrame(rows).drop_duplicates("time").sort_values("time").reset_index(drop=True)


def parse_utc_date(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use YYYY-MM-DD dates.") from error


def write_output(data: pd.DataFrame, output: Path, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}. Re-run with --overwrite if intended.")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = data.copy()
    result["time"] = result["time"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    result.to_csv(output, index=False)
    try:
        display_path = output.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = output
    print(f"Wrote {len(result):,} 15-minute XAU/USD bars to {display_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download free Dukascopy XAU/USD data for ITRF research.")
    parser.add_argument("--start", required=True, type=parse_utc_date, help="Inclusive UTC date, YYYY-MM-DD")
    parser.add_argument("--end", required=True, type=parse_utc_date, help="Exclusive UTC date, YYYY-MM-DD")
    parser.add_argument("--symbol", default="XAUUSD", help="Dukascopy symbol (default: XAUUSD)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output file")
    parser.add_argument("--pause-seconds", type=float, default=0.5, help="Delay between hourly requests (default: 0.5)")
    parser.add_argument("--retries", type=int, default=3, help="Retries after a source rate limit (default: 3)")
    args = parser.parse_args()
    if args.end <= args.start:
        parser.error("--end must be later than --start.")
    if args.pause_seconds < 0 or args.retries < 0:
        parser.error("--pause-seconds and --retries must not be negative.")

    data = download_range(args.symbol, args.start, args.end, args.pause_seconds, args.retries)
    write_output(data, args.output, args.overwrite)


if __name__ == "__main__":
    main()
