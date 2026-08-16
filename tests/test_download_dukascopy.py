from __future__ import annotations

from datetime import UTC, datetime
import lzma
import struct

import pytest

from research.download_dukascopy import aggregate_hour, decode_ticks, hour_url


def test_hour_url_uses_zero_indexed_month() -> None:
    hour = datetime(2026, 5, 1, 9, tzinfo=UTC)
    assert hour_url("xauusd", hour).endswith("XAUUSD/2026/04/01/09h_ticks.bi5")


def test_decode_and_aggregate_ticks() -> None:
    raw = struct.pack(">IIIff", 1_000, 4_600_100, 4_600_000, 1.5, 2.0)
    raw += struct.pack(">IIIff", 20_000, 4_600_400, 4_600_200, 1.0, 1.0)
    payload = lzma.compress(raw, format=lzma.FORMAT_ALONE)
    assert list(decode_ticks(payload))[0] == (1_000, 4600.0, 4600.1, 3.5)

    bars = aggregate_hour(payload, datetime(2026, 5, 1, 9, tzinfo=UTC))
    assert len(bars) == 1
    assert bars[0]["open"] == pytest.approx(4600.05)
    assert bars[0]["high"] == pytest.approx(4600.3)
    assert bars[0]["low"] == pytest.approx(4600.05)
    assert bars[0]["close"] == pytest.approx(4600.3)
    assert bars[0]["volume"] == pytest.approx(5.5)
