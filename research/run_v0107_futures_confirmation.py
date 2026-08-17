"""Run the pre-registered v0.10.7 historical futures-confirmation test."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from itrf_futures_confirmation import (
    build_futures_confirmed_ledger,
    chronological_half_means,
    gated_cost_interval,
    load_futures_bars,
    merge_exact_futures_bars,
    sequential_metrics,
)
from itrf_research import create_features
from run_v0104_external_transport import (
    DEFAULT_DEVELOPMENT_FILE,
    DEFAULT_FULL_FILE,
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    EXTERNAL_END,
    EXTERNAL_START,
    load_locked_segments,
    sha256,
)
from run_v010_clean_baseline import build_candidate_ledger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FUTURES_FILE = PROJECT_ROOT / "data" / "databento" / "GC_front_15m.csv"
DEFAULT_LEDGERS_FILE = PROJECT_ROOT / "data" / "processed" / "v0107_candidate_ledgers.csv"
DEFAULT_METRICS_FILE = PROJECT_ROOT / "data" / "processed" / "v0107_strategy_metrics.csv"
LOCKED_FUTURES_SHA256 = "5fe1c3678c17ce063ea86718e233bc8476d81e12fb79d49f31a5a6b1ca1f02b1"


def build_interval_ledgers(
    raw_frame: pd.DataFrame,
    futures: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = merge_exact_futures_bars(create_features(raw_frame), futures)
    baseline = build_candidate_ledger(frame)
    confirmed = build_futures_confirmed_ledger(frame)
    if not baseline[["signal_index", "signal_time", "direction"]].equals(
        confirmed[["signal_index", "signal_time", "direction"]]
    ):
        raise ValueError("baseline and confirmed ledgers do not retain identical raw candidates")
    return frame, baseline, confirmed


def interval_report(
    segment: str,
    frame: pd.DataFrame,
    baseline: pd.DataFrame,
    confirmed: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    print(f"\n{segment.upper()} INTERVAL")
    print(
        f"Exact futures bars: {int(frame['futures_bar_available'].sum())}/{len(frame)}; "
        f"raw v0.8 candidates: {len(confirmed)}"
    )
    print("Futures gate diagnostics:")
    print(confirmed["futures_gate_reason"].value_counts().rename_axis("reason").to_string())
    print("Confirmed-ledger decisions:")
    print(confirmed["decision"].value_counts().rename_axis("decision").to_string())
    for policy_name, ledger in (("baseline", baseline), ("futures_confirmed", confirmed)):
        accepted = ledger.loc[ledger["decision"] == "ACCEPTED"]
        print(f"{policy_name} exit events:")
        print(accepted["exit_reason"].value_counts().rename_axis("exit_reason").to_string())
        print(f"{policy_name} ambiguous events: {int(accepted['ambiguous'].sum())}/{len(accepted)}")

    metric_rows: list[dict[str, object]] = []
    for policy_name, ledger in (("baseline", baseline), ("futures_confirmed", confirmed)):
        for bound, column in (("lower", "gross_r_lower"), ("upper", "gross_r_upper")):
            for cost in (0.0, 0.05, 0.10):
                metric_rows.append(
                    {
                        "segment": segment,
                        "policy": policy_name,
                        "ambiguity_bound": bound,
                        "cost_r": cost,
                        **sequential_metrics(ledger, cost, column),
                    }
                )
    metrics = pd.DataFrame(metric_rows)
    print("Sequential lower-bound comparison:")
    print(metrics.round(5).to_string(index=False))
    mean, low, high = gated_cost_interval(confirmed, 0.05)
    first, second = chronological_half_means(confirmed, 0.05)
    print(f"Confirmed 0.05R average: {mean:.5f}, block-bootstrap 95%=[{low:.5f}, {high:.5f}]")
    print(f"Confirmed chronological halves at 0.05R: first={first:.5f}, second={second:.5f}")
    audit = {
        "segment": segment,
        "raw_candidates": len(confirmed),
        "exact_bar_candidates": int(confirmed["futures_bar_available"].sum()),
        "gate_pass_candidates": int(confirmed["futures_gate_pass"].sum()),
        "gated_trades": int((confirmed["decision"] == "ACCEPTED").sum()),
        "ci_mean_r": mean,
        "ci_low_r": low,
        "ci_high_r": high,
        "first_half_average_r": first,
        "second_half_average_r": second,
    }
    return metrics, audit


def decision_passes(metrics: pd.DataFrame, audits: pd.DataFrame) -> bool:
    indexed = metrics.set_index(["segment", "policy", "ambiguity_bound", "cost_r"])
    minimum = {"development": 50, "backward_external": 300}
    for segment, minimum_trades in minimum.items():
        audit = audits.set_index("segment").loc[segment]
        if int(audit["gated_trades"]) < minimum_trades:
            return False
        if not (float(audit["ci_mean_r"]) > 0 and float(audit["ci_low_r"]) > 0):
            return False
        if not (
            float(audit["first_half_average_r"]) > 0
            and float(audit["second_half_average_r"]) > 0
        ):
            return False
        for cost in (0.05, 0.10):
            baseline = indexed.loc[(segment, "baseline", "lower", cost)]
            gated = indexed.loc[(segment, "futures_confirmed", "lower", cost)]
            if not float(gated["profit_factor_r"]) > 1.0:
                return False
            if not float(gated["total_r"]) > float(baseline["total_r"]):
                return False
            if not float(gated["maximum_drawdown_r"]) >= float(baseline["maximum_drawdown_r"]):
                return False
    return True


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ITRF v0.10.7 futures confirmation.")
    parser.add_argument("--development-file", type=Path, default=DEFAULT_DEVELOPMENT_FILE)
    parser.add_argument("--full-file", type=Path, default=DEFAULT_FULL_FILE)
    parser.add_argument("--futures-file", type=Path, default=DEFAULT_FUTURES_FILE)
    parser.add_argument("--ledgers-file", type=Path, default=DEFAULT_LEDGERS_FILE)
    parser.add_argument("--metrics-file", type=Path, default=DEFAULT_METRICS_FILE)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    if sha256(arguments.futures_file) != LOCKED_FUTURES_SHA256:
        raise ValueError("futures file hash differs from the pre-registered value")
    development, external = load_locked_segments(arguments.development_file, arguments.full_file)
    futures = load_futures_bars(arguments.futures_file)
    segments = {
        "development": (development, DEVELOPMENT_START, DEVELOPMENT_END),
        "backward_external": (external, EXTERNAL_START, EXTERNAL_END),
    }
    ledgers: list[pd.DataFrame] = []
    metrics: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    print("\nITRF v0.10.7 FUTURES-CONFIRMATION ENTRY TEST")
    print("Already-inspected historical data; not a profitability or deployment claim.")
    for segment, (raw, start, end) in segments.items():
        selected_futures = futures.loc[
            (futures["time"] >= start) & (futures["time"] <= end)
        ].reset_index(drop=True)
        frame, baseline, confirmed = build_interval_ledgers(raw, selected_futures)
        for policy, ledger in (("baseline", baseline), ("futures_confirmed", confirmed)):
            output = ledger.copy()
            output.insert(0, "policy", policy)
            output.insert(0, "segment", segment)
            ledgers.append(output)
        interval_metrics, audit = interval_report(segment, frame, baseline, confirmed)
        metrics.append(interval_metrics)
        audits.append(audit)
    all_ledgers = pd.concat(ledgers, ignore_index=True, sort=False)
    all_metrics = pd.concat(metrics, ignore_index=True)
    audit_frame = pd.DataFrame(audits)
    passed = decision_passes(all_metrics, audit_frame)
    print(f"\nPre-registered exploratory decision: {'PASS' if passed else 'FAIL'}")
    print("Even a pass would require an untouched future XAU/USD holdout.")
    arguments.ledgers_file.parent.mkdir(parents=True, exist_ok=True)
    all_ledgers.to_csv(arguments.ledgers_file, index=False)
    all_metrics.to_csv(arguments.metrics_file, index=False)
    print(f"\nCandidate ledgers: {arguments.ledgers_file}")
    print(f"Strategy metrics: {arguments.metrics_file}")


if __name__ == "__main__":
    main()
