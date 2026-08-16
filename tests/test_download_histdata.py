from datetime import date

from research.download_histdata import aggregate_tick_rows, months_in_range, parse_tick_timestamp


def test_months_in_range_crosses_year() -> None:
    assert list(months_in_range(date(2025, 12, 1), date(2026, 2, 1))) == [(2025, 12), (2026, 1), (2026, 2)]


def test_parse_and_aggregate_ticks_to_15_minutes() -> None:
    assert parse_tick_timestamp("20260501", "001530123").isoformat() == "2026-05-01T00:15:30.123000"
    rows = [
        ["20260501 000001000", "4600.0", "4600.2", "0"],
        ["20260501 000300000", "4599.8", "4600.0", "0"],
        ["20260501 001501000", "4601.0", "4601.2", "0"],
    ]
    bars = aggregate_tick_rows(rows, date(2026, 5, 1), date(2026, 5, 2))
    assert len(bars) == 2
    assert bars[0] == {
        "time": parse_tick_timestamp("20260501", "000000000"),
        "open": 4600.1,
        "high": 4600.1,
        "low": 4599.9,
        "close": 4599.9,
        "volume": 2,
    }
    assert bars[1]["volume"] == 1
