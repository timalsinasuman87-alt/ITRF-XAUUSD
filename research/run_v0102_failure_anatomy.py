"""Run the pre-registered v0.10.2 failure-anatomy diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from itrf_failure_anatomy import (
    build_competing_hazards,
    build_lifecycle_panel,
    lifecycle_phase,
    summarize_trade_paths,
)
from itrf_research import create_features, load_market_data
from run_v010_clean_baseline import build_candidate_ledger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_FILE = PROJECT_ROOT / "data" / "XAUUSD.csv"
DEFAULT_PANEL_FILE = PROJECT_ROOT / "data" / "processed" / "v0102_lifecycle_panel.csv"
DEFAULT_PATHS_FILE = PROJECT_ROOT / "data" / "processed" / "v0102_trade_paths.csv"
DEFAULT_HAZARDS_FILE = PROJECT_ROOT / "data" / "processed" / "v0102_competing_hazards.csv"


def print_report(paths: pd.DataFrame, hazards: pd.DataFrame) -> None:
    print("\nITRF v0.10.2 FAILURE ANATOMY — FROZEN CLEAN-CORE EVENTS")
    print("Already-inspected development data; no trade-management rule may be selected.")
    print(f"\nAccepted events: {len(paths)}")
    outcome = paths.groupby("exit_reason", sort=False).agg(
        events=("event_id", "size"),
        median_bars=("bars_held", "median"),
        median_preterminal_mfe_r=("preterminal_mfe_r", "median"),
        median_preterminal_mae_r=("preterminal_mae_r", "median"),
    ).reset_index()
    print("\nOutcome path anatomy:")
    print(outcome.round(3).to_string(index=False))

    terminal = paths[["bars_held", "exit_reason"]].copy()
    terminal["phase"] = terminal["bars_held"].map(lifecycle_phase)
    phase = terminal.groupby(["phase", "exit_reason"], sort=False).size().unstack(fill_value=0)
    print("\nTerminal events by pre-declared lifecycle phase:")
    print(phase.to_string())

    stopped = paths.loc[paths["exit_reason"] == "STOP"]
    if not stopped.empty:
        landmark_columns = [
            "preterminal_reached_0_5r",
            "preterminal_reached_1_0r",
            "preterminal_reached_2_0r",
        ]
        landmarks = 100.0 * stopped[landmark_columns].mean()
        print("\nStopped trades reaching fixed MFE landmarks before the terminal bar (%):")
        print(landmarks.round(2).to_string())

    final = hazards.iloc[-1]
    print("\nFinal competing-event cumulative incidence:")
    for label in ("stop", "target", "timeout", "ambiguous_stop_target"):
        print(f"{label}: {100 * final[f'{label}_cumulative_incidence']:.2f}%")
    print("\nTerminal-bar extremes were excluded from pre-terminal stop/target excursions.")
    print("These diagnostics can motivate a future hypothesis but do not validate an edge.")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ITRF v0.10.2 failure anatomy.")
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--panel-file", type=Path, default=DEFAULT_PANEL_FILE)
    parser.add_argument("--paths-file", type=Path, default=DEFAULT_PATHS_FILE)
    parser.add_argument("--hazards-file", type=Path, default=DEFAULT_HAZARDS_FILE)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    frame = create_features(load_market_data(arguments.data_file))
    ledger = build_candidate_ledger(frame)
    panel = build_lifecycle_panel(frame, ledger)
    paths = summarize_trade_paths(panel, ledger)
    hazards = build_competing_hazards(paths)
    for output in (arguments.panel_file, arguments.paths_file, arguments.hazards_file):
        output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(arguments.panel_file, index=False)
    paths.to_csv(arguments.paths_file, index=False)
    hazards.to_csv(arguments.hazards_file, index=False)
    print_report(paths, hazards)
    print(f"\nLifecycle panel: {arguments.panel_file}")
    print(f"Trade paths: {arguments.paths_file}")
    print(f"Competing hazards: {arguments.hazards_file}")


if __name__ == "__main__":
    main()
