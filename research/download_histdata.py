"""Download HistData XAU/USD tick archives and create ITRF 15-minute OHLCV bars.

HistData's public XAU/USD ticks have no usable trade-volume field.  This
research importer therefore uses the number of ticks in each 15-minute bar as
``volume``.  It is a transparent activity proxy, not exchange volume, and must
not be treated as equivalent to a broker or TradingView CFD volume feed.
"""

from __future__ import annotations

import argparse
import csv
import http.cookiejar
import io
import re
import zipfile
from collections.abc import Iterable
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "XAUUSD.csv"
SELECTION_URL = "https://www.histdata.com/download-free-forex-historical-data/?/ascii/tick-data-quotes/xauusd/{year}/{month}"
DOWNLOAD_URL = "https://www.histdata.com/get.php"
TOKEN_PATTERN = re.compile(r'name="tk"\s+id="tk"\s+value="([a-f0-9]+)"')


def parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use YYYY-MM-DD dates.") from error


def months_in_range(start: date, end: date) -> Iterable[tuple[int, int]]:
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        yield year, month
        month = 1 if month == 12 else month + 1
        year += 1 if month == 1 else 0


def histdata_opener():
    return build_opener(HTTPCookieProcessor(http.cookiejar.CookieJar()))


def archive_request(opener, year: int, month: int) -> bytes:
    selection_url = SELECTION_URL.format(year=year, month=month)
    with opener.open(selection_url, timeout=60) as response:
        selection = response.read().decode("utf-8", errors="replace")
    token_match = TOKEN_PATTERN.search(selection)
    if token_match is None:
        raise RuntimeError(f"HistData did not provide a download token for {year}-{month:02d}.")

    form = urlencode(
        {
            "tk": token_match.group(1),
            "date": str(year),
            "datemonth": f"{year}{month:02d}",
            "platform": "ASCII",
            "timeframe": "T",
            "fxpair": "XAUUSD",
        }
    ).encode()
    request = Request(DOWNLOAD_URL, data=form, headers={"Referer": selection_url})
    with opener.open(request, timeout=180) as response:
        archive = response.read()
    if not archive.startswith(b"PK"):
        raise RuntimeError(f"HistData returned an invalid archive for {year}-{month:02d}.")
    return archive


def parse_tick_timestamp(date_text: str, time_text: str) -> datetime:
    """Parse HistData's YYYYMMDD and HHMMSSmmm fields without timezone conversion."""
    base = datetime.strptime(f"{date_text} {time_text[:6]}", "%Y%m%d %H%M%S")
    return base + timedelta(milliseconds=int(time_text[6:] or 0))


def aggregate_tick_rows(rows: Iterable[list[str]], start: date, end: date) -> list[dict[str, object]]:
    """Aggregate bid/ask ticks into midpoint OHLC and a tick-count activity proxy."""
    bars: dict[datetime, dict[str, object]] = {}
    for row in rows:
        if len(row) < 4:
            continue
        timestamp_parts = row[0].split()
        if len(timestamp_parts) != 2:
            raise ValueError(f"Unexpected HistData timestamp field: {row[0]!r}")
        timestamp = parse_tick_timestamp(timestamp_parts[0], timestamp_parts[1])
        if not (start <= timestamp.date() < end):
            continue
        bid, ask = float(row[1]), float(row[2])
        price = (bid + ask) / 2
        bucket = timestamp.replace(minute=(timestamp.minute // 15) * 15, second=0, microsecond=0)
        current = bars.get(bucket)
        if current is None:
            bars[bucket] = {
                "time": bucket,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 1,
            }
        else:
            current["high"] = max(float(current["high"]), price)
            current["low"] = min(float(current["low"]), price)
            current["close"] = price
            current["volume"] = int(current["volume"]) + 1
    return [bars[key] for key in sorted(bars)]


def aggregate_archive(archive_bytes: bytes, start: date, end: date) -> list[dict[str, object]]:
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise RuntimeError("HistData archive did not contain exactly one tick CSV.")
        with archive.open(csv_names[0]) as binary_file:
            reader = csv.reader(io.TextIOWrapper(binary_file, encoding="utf-8", newline=""))
            return aggregate_tick_rows(reader, start, end)


def download_range(start: date, end: date) -> pd.DataFrame:
    opener = histdata_opener()
    all_bars: list[dict[str, object]] = []
    for year, month in months_in_range(start, end - timedelta(days=1)):
        print(f"Downloading HistData XAU/USD ticks for {year}-{month:02d}...")
        archive = archive_request(opener, year, month)
        bars = aggregate_archive(archive, start, end)
        all_bars.extend(bars)
        print(f"  Aggregated {len(bars):,} 15-minute bars from that archive.")
    if not all_bars:
        raise RuntimeError("No XAU/USD ticks were available for the requested date range.")
    data = pd.DataFrame(all_bars).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    if (data["volume"] <= 0).any():
        raise RuntimeError("Tick-count activity proxy must be positive.")
    return data


def write_output(data: pd.DataFrame, output: Path, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}. Re-run with --overwrite if intended.")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = data.copy()
    result["time"] = result["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    result.to_csv(output, index=False)
    try:
        display_path = output.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = output
    print(f"Wrote {len(result):,} 15-minute bars to {display_path}")
    print("Volume is HistData tick count, not traded volume. Source timestamps are retained without conversion.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download HistData XAU/USD ticks for ITRF research.")
    parser.add_argument("--start", required=True, type=parse_date, help="Inclusive source date, YYYY-MM-DD")
    parser.add_argument("--end", required=True, type=parse_date, help="Exclusive source date, YYYY-MM-DD")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output file")
    args = parser.parse_args()
    if args.end <= args.start:
        parser.error("--end must be later than --start.")
    write_output(download_range(args.start, args.end), args.output, args.overwrite)


if __name__ == "__main__":
    main()
