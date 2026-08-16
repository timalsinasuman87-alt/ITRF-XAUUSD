import pandas as pd

from research.itrf_research import resolve_oos_split


def test_resolve_oos_split_returns_valid_chronological_boundary() -> None:
    data = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2026-05-01", "2026-06-01", "2026-07-01"])}
    )

    split = resolve_oos_split(data, "2026-06-01", "test analysis")

    assert split == pd.Timestamp("2026-06-01")


def test_resolve_oos_split_rejects_boundary_outside_available_data(capsys) -> None:
    data = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2026-05-01", "2026-06-01", "2026-07-01"])}
    )

    split = resolve_oos_split(data, "2025-07-28", "test analysis")

    assert split is None
    assert "outside the available range" in capsys.readouterr().out
