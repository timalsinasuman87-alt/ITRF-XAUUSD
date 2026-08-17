"""Audit the frozen v0.8 signal and forward-label definitions without tuning them."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from itrf_research import (
    FORWARD_BARS,
    RISK_ATR,
    create_features,
    detect_setup,
    evaluate_forward_path,
    load_market_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_FILE = PROJECT_ROOT / "data" / "XAUUSD.csv"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "v08_failure_audit.csv"
MINIMUM_HISTORY = 250


def exact_v08_components(row: pd.Series, direction: str) -> dict[str, int]:
    """Return the exact points used by detect_setup, not a later score variant."""
    if direction not in {"LONG", "SHORT"}:
        raise ValueError("direction must be LONG or SHORT")
    sign = 1 if direction == "LONG" else -1
    sweep = row["bullish_sweep"] if direction == "LONG" else row["bearish_sweep"]
    components = {
        "component_trend": int(row["trend"] == sign),
        "component_momentum": int(sign * row["momentum_atr"] > 0),
        "component_delta": int(sign * row["delta_zscore"] > 1),
        "component_volume": int(
            row["relative_volume"] > 1.5 and sign * row["delta_proxy"] > 0
        ),
        "component_sweep": int(sweep == 1),
    }
    components["v08_score"] = (
        components["component_trend"]
        + components["component_momentum"]
        + components["component_delta"]
        + components["component_volume"]
        + 2 * components["component_sweep"]
    )
    return components


def trace_forward_path(
    frame: pd.DataFrame,
    index: int,
    direction: str,
    entry: float,
    atr: float,
) -> dict[str, object]:
    """Add event-order diagnostics while preserving the original v0.8 label."""
    risk = atr * RISK_ATR
    sign = 1 if direction == "LONG" else -1
    stop = entry - sign * risk
    targets = [entry + sign * risk * multiple for multiple in (1, 2, 3)]
    future = frame.iloc[index + 1 : index + 1 + FORWARD_BARS]

    first_target_bars: list[int | None] = [None, None, None]
    first_stop_bar: int | None = None
    ambiguous_bars = 0
    for bar_number, (_, candle) in enumerate(future.iterrows(), start=1):
        if direction == "LONG":
            target_hits = [candle["high"] >= target for target in targets]
            stop_hit = candle["low"] <= stop
        else:
            target_hits = [candle["low"] <= target for target in targets]
            stop_hit = candle["high"] >= stop
        for target_index, hit in enumerate(target_hits):
            if hit and first_target_bars[target_index] is None:
                first_target_bars[target_index] = bar_number
        if stop_hit:
            first_stop_bar = bar_number
            if any(target_hits):
                ambiguous_bars += 1
            break

    frozen = evaluate_forward_path(frame, index, direction, entry, atr)
    first_1r = first_target_bars[0]
    if first_stop_bar is None and first_1r is None:
        first_event = "NEITHER"
    elif first_stop_bar is None:
        first_event = "TARGET_1R"
    elif first_1r is None or first_stop_bar < first_1r:
        first_event = "STOP"
    elif first_stop_bar == first_1r:
        first_event = "AMBIGUOUS_SAME_BAR"
    else:
        first_event = "TARGET_1R"

    elapsed_hours = (
        (future.iloc[-1]["time"] - frame.iloc[index]["time"]).total_seconds() / 3600.0
        if len(future) else np.nan
    )
    expected_step = pd.Timedelta(minutes=15)
    path_times = pd.concat(
        [pd.Series([frame.iloc[index]["time"]]), future["time"].reset_index(drop=True)],
        ignore_index=True,
    )
    gap_crossed = int(path_times.diff().dropna().gt(expected_step).any())
    return {
        **frozen,
        "bars_to_1r": first_target_bars[0],
        "bars_to_2r": first_target_bars[1],
        "bars_to_3r": first_target_bars[2],
        "bars_to_stop": first_stop_bar,
        "first_event": first_event,
        "same_bar_stop_target": int(ambiguous_bars > 0),
        "target_then_later_stop": int(
            first_1r is not None
            and first_stop_bar is not None
            and first_1r < first_stop_bar
        ),
        "path_elapsed_hours": elapsed_hours,
        "gap_crossed": gap_crossed,
    }


def build_audit_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Reproduce every eligible v0.8 setup and attach audit-only diagnostics."""
    records: list[dict[str, object]] = []
    next_available_index = MINIMUM_HISTORY
    last_index = len(frame) - FORWARD_BARS - 1
    for index in range(MINIMUM_HISTORY, last_index):
        row = frame.iloc[index]
        if pd.isna(row["atr"]):
            continue
        direction = detect_setup(row)
        if direction == "NONE":
            continue
        non_overlapping = int(index >= next_available_index)
        if non_overlapping:
            next_available_index = index + FORWARD_BARS + 1
        components = exact_v08_components(row, direction)
        outcome = trace_forward_path(
            frame, index, direction, float(row["close"]), float(row["atr"])
        )
        records.append(
            {
                "source_index": index,
                "timestamp": row["time"],
                "direction": direction,
                "entry": float(row["close"]),
                "atr": float(row["atr"]),
                "non_overlapping": non_overlapping,
                "high_volatility": int(row["high_volatility"]),
                "trend": int(row["trend"]),
                "relative_volume": float(row["relative_volume"]),
                "delta_zscore": float(row["delta_zscore"]),
                "momentum_atr": float(row["momentum_atr"]),
                # HistData source timestamps are preserved as supplied, so this
                # is intentionally not named as a UTC or exchange session.
                "source_time_window": f"{(row['time'].hour // 6) * 6:02d}-{(row['time'].hour // 6) * 6 + 5:02d}",
                "weekday": row["time"].day_name(),
                **components,
                **outcome,
            }
        )
    result = pd.DataFrame(records)
    if not result.empty:
        result["timestamp"] = pd.to_datetime(result["timestamp"])
        result["first_touch_1r_outcome"] = result["first_event"].map(
            {
                "TARGET_1R": 1.0,
                "STOP": -1.0,
                "AMBIGUOUS_SAME_BAR": -1.0,
                "NEITHER": 0.0,
            }
        )
        quarter_codes = np.minimum(
            3,
            np.floor(np.arange(len(result)) * 4 / len(result)).astype(int),
        )
        result["chronological_quarter"] = pd.Categorical.from_codes(
            quarter_codes,
            categories=["Q1", "Q2", "Q3", "Q4"],
            ordered=True,
        )
    return result


def profit_factor(values: pd.Series) -> float:
    profits = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    return profits / losses if losses > 0 else np.inf


def maximum_drawdown_r(values: pd.Series) -> float:
    equity = values.fillna(0).cumsum()
    drawdown = equity.cummax().clip(lower=0) - equity
    return float(drawdown.max()) if len(drawdown) else np.nan


def block_bootstrap_mean_ci(
    values: pd.Series,
    *,
    seed: int = 20260817,
    samples: int = 10_000,
) -> tuple[float, float]:
    """Deterministic circular block-bootstrap interval for serial trade outcomes."""
    data = np.asarray(values.dropna(), dtype=float)
    if len(data) == 0:
        return np.nan, np.nan
    if len(data) == 1:
        return float(data[0]), float(data[0])
    block_length = max(2, int(round(np.sqrt(len(data)))))
    blocks_needed = int(np.ceil(len(data) / block_length))
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    offsets = np.arange(block_length)
    for sample in range(samples):
        starts = rng.integers(0, len(data), size=blocks_needed)
        indices = (starts[:, None] + offsets[None, :]) % len(data)
        means[sample] = data[indices.ravel()[: len(data)]].mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize(data: pd.DataFrame) -> pd.Series:
    outcomes = data["outcome_r"]
    low, high = block_bootstrap_mean_ci(outcomes)
    return pd.Series(
        {
            "samples": len(data),
            "average_r": outcomes.mean(),
            "bootstrap_low_r": low,
            "bootstrap_high_r": high,
            "profit_factor_r": profit_factor(outcomes),
            "max_drawdown_r": maximum_drawdown_r(outcomes),
            "stopped_pct": 100.0 * data["stopped"].mean(),
            "hit_1r_pct": 100.0 * data["hit_1r"].mean(),
            "hit_2r_pct": 100.0 * data["hit_2r"].mean(),
            "hit_3r_pct": 100.0 * data["hit_3r"].mean(),
            "median_mfe_r": data["mfe"].median(),
            "median_mae_r": data["mae"].median(),
        }
    )


def grouped_summary(data: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        data.groupby(column, observed=True, sort=False)
        .apply(summarize, include_groups=False)
        .reset_index()
    )


def print_report(audit: pd.DataFrame, source: pd.DataFrame) -> None:
    print("\nITRF v0.8 FROZEN FAILURE AUDIT — DIAGNOSTIC, NOT OPTIMIZATION")
    print(f"Source bars: {len(source):,} | {source['time'].min()} to {source['time'].max()}")
    print(f"Frozen v0.8 setups: {len(audit):,}")
    print("\nAll signals versus mechanically non-overlapping signals:")
    populations = pd.concat(
        [
            summarize(audit).rename("all_signals"),
            summarize(audit.loc[audit["non_overlapping"] == 1]).rename("non_overlapping"),
        ],
        axis=1,
    ).T.reset_index(names="population")
    print(populations.round(3).to_string(index=False))

    non_overlap = audit.loc[audit["non_overlapping"] == 1].copy()
    for column, title in [
        ("direction", "Direction"),
        ("component_sweep", "Sweep present"),
        ("v08_score", "Exact v0.8 score"),
        ("high_volatility", "Volatility flag"),
        ("source_time_window", "Fixed source-time window"),
        ("chronological_quarter", "Chronological quarter"),
    ]:
        print(f"\n{title} — non-overlapping only:")
        print(grouped_summary(non_overlap, column).round(3).to_string(index=False))

    stopped = audit.loc[audit["stopped"] == 1]
    print("\nForward-label semantics — all frozen signals:")
    for target in (1, 2, 3):
        overwritten = int(
            ((audit["stopped"] == 1) & (audit[f"hit_{target}r"] == 1)).sum()
        )
        print(f"Stopped after a recorded {target}R touch: {overwritten:,}")
    print(f"Same-bar target/stop ambiguity: {int(audit['same_bar_stop_target'].sum()):,}")
    print("First-event distribution:")
    print(audit["first_event"].value_counts().rename_axis("first_event").to_string())
    first_touch = pd.DataFrame(
        {
            "population": ["all_signals", "non_overlapping"],
            "samples": [len(audit), len(non_overlap)],
            "first_touch_1r_average": [
                audit["first_touch_1r_outcome"].mean(),
                non_overlap["first_touch_1r_outcome"].mean(),
            ],
            "frozen_label_average": [audit["outcome_r"].mean(), non_overlap["outcome_r"].mean()],
        }
    )
    print("\nDiagnostic first-touch 1R comparison (not a promoted exit rule):")
    print(first_touch.round(3).to_string(index=False))
    print(
        f"Stopped paths crossing a market-time gap: "
        f"{int(stopped['gap_crossed'].sum()):,}/{len(stopped):,}"
    )
    print("\nAll statistics are from already-inspected development data.")
    print("No costs are included; executable expectancy would be lower.")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the frozen ITRF v0.8 engine.")
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    source = create_features(load_market_data(arguments.data_file))
    audit = build_audit_frame(source)
    arguments.output_file.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(arguments.output_file, index=False)
    print_report(audit, source)
    print(f"\nAudit observations: {arguments.output_file}")


if __name__ == "__main__":
    main()
