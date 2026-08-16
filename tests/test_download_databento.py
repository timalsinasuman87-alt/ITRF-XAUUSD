from pathlib import Path

import pandas as pd
import pytest

from research.download_databento import (
    load_api_key,
    resample_ohlcv_to_15m,
    roll_schedule_from_resolution,
    validate_date_range,
)


def test_resample_ohlcv_to_15m_and_record_contract_change() -> None:
    frame = pd.DataFrame(
        {
            "open": [2400.0, 2401.0, 2402.0],
            "high": [2401.0, 2403.0, 2404.0],
            "low": [2399.0, 2400.0, 2401.0],
            "close": [2400.5, 2402.5, 2403.5],
            "volume": [10, 20, 30],
            "symbol": ["GCQ6", "GCQ6", "GCV6"],
        },
        index=pd.DatetimeIndex(
            ["2026-08-03T00:00:00Z", "2026-08-03T00:01:00Z", "2026-08-03T00:15:00Z"],
            name="ts_event",
        ),
    )

    bars = resample_ohlcv_to_15m(frame)

    assert len(bars) == 2
    assert bars.iloc[0].to_dict() == {
        "time": "2026-08-03 00:00:00+0000",
        "open": 2400.0,
        "high": 2403.0,
        "low": 2399.0,
        "close": 2402.5,
        "volume": 30,
    }


def test_roll_schedule_uses_official_continuous_mapping() -> None:
    resolution = {
        "result": {
            "GC.v.0": [
                {"d0": "2026-01-01", "d1": "2026-01-30", "s": "101"},
                {"d0": "2026-01-30", "d1": "2026-03-30", "s": "202"},
            ]
        }
    }

    schedule = roll_schedule_from_resolution(resolution)

    assert schedule.to_dict("records") == [
        {"start_date": "2026-01-01", "end_date": "2026-01-30", "instrument_id": "101"},
        {"start_date": "2026-01-30", "end_date": "2026-03-30", "instrument_id": "202"},
    ]


def test_date_range_and_missing_key_validation(monkeypatch, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="earlier"):
        validate_date_range("2026-08-02", "2026-08-01")

    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="not found"):
        load_api_key(tmp_path / "missing")
