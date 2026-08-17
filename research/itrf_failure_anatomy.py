"""Causal lifecycle diagnostics for the frozen v0.10 event ledger."""

from __future__ import annotations

import numpy as np
import pandas as pd


TERMINAL_EVENTS = ("STOP", "TARGET", "TIMEOUT", "AMBIGUOUS_STOP_TARGET")
MFE_LANDMARKS = (0.5, 1.0, 2.0)


def _accepted_events(ledger: pd.DataFrame) -> pd.DataFrame:
    ledger_required = {
        "decision", "entry_index", "exit_index", "direction", "entry", "risk",
        "bars_held", "exit_reason", "gross_r_lower", "gross_r_upper", "ambiguous",
    }
    missing_ledger = ledger_required - set(ledger.columns)
    if missing_ledger:
        raise ValueError(f"missing lifecycle ledger columns: {sorted(missing_ledger)}")
    accepted = ledger.loc[ledger["decision"] == "ACCEPTED"].copy().reset_index(drop=True)
    if accepted.empty:
        return accepted
    if not accepted["entry_index"].is_monotonic_increasing:
        raise ValueError("accepted events must be chronological")
    return accepted


def _validate_inputs(frame: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    frame_required = {"time", "high", "low"}
    missing_frame = frame_required - set(frame.columns)
    if missing_frame:
        raise ValueError(f"missing lifecycle frame columns: {sorted(missing_frame)}")
    return _accepted_events(ledger)


def build_lifecycle_panel(frame: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    """Expand accepted events into entry-to-terminal OHLC lifecycle rows."""
    accepted = _validate_inputs(frame, ledger)
    records: list[dict[str, object]] = []
    for event_id, event in accepted.iterrows():
        direction = str(event["direction"])
        if direction not in {"LONG", "SHORT"}:
            raise ValueError("accepted event direction must be LONG or SHORT")
        entry_index = int(event["entry_index"])
        exit_index = int(event["exit_index"])
        bars_held = int(event["bars_held"])
        risk = float(event["risk"])
        entry = float(event["entry"])
        if not np.isfinite(risk) or risk <= 0:
            raise ValueError("accepted event risk must be positive and finite")
        if entry_index < 0 or exit_index < entry_index or exit_index >= len(frame):
            raise ValueError("accepted event interval is outside the market frame")
        if exit_index - entry_index + 1 != bars_held:
            raise ValueError("bars_held does not match the inclusive event interval")
        exit_reason = str(event["exit_reason"])
        if exit_reason not in TERMINAL_EVENTS:
            raise ValueError(f"unsupported terminal event: {exit_reason}")

        running_mfe = 0.0
        running_mae = 0.0
        for holding_bar, bar_index in enumerate(range(entry_index, exit_index + 1), start=1):
            bar = frame.iloc[bar_index]
            if direction == "LONG":
                favorable = (float(bar["high"]) - entry) / risk
                adverse = (float(bar["low"]) - entry) / risk
            else:
                favorable = (entry - float(bar["low"])) / risk
                adverse = (entry - float(bar["high"])) / risk
            running_mfe = max(running_mfe, favorable)
            running_mae = min(running_mae, adverse)
            is_terminal = bar_index == exit_index
            preterminal_observable = exit_reason == "TIMEOUT" or not is_terminal
            records.append(
                {
                    "event_id": event_id,
                    "bar_index": bar_index,
                    "bar_time": bar["time"],
                    "holding_bar": holding_bar,
                    "direction": direction,
                    "entry": entry,
                    "risk": risk,
                    "bar_favorable_r": favorable,
                    "bar_adverse_r": adverse,
                    "running_mfe_r": running_mfe,
                    "running_mae_r": running_mae,
                    "preterminal_observable": int(preterminal_observable),
                    "terminal_event": exit_reason if is_terminal else "NONE",
                }
            )
    panel = pd.DataFrame(records)
    if not panel.empty:
        panel["bar_time"] = pd.to_datetime(panel["bar_time"])
    return panel


def summarize_trade_paths(panel: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    """Create one failure-anatomy row per accepted event."""
    accepted = _accepted_events(ledger)
    if accepted.empty:
        return accepted
    required_panel = {
        "event_id", "holding_bar", "running_mfe_r", "running_mae_r",
        "preterminal_observable", "terminal_event",
    }
    missing = required_panel - set(panel.columns)
    if missing:
        raise ValueError(f"missing lifecycle panel columns: {sorted(missing)}")
    records: list[dict[str, object]] = []
    for event_id, event in accepted.iterrows():
        path = panel.loc[panel["event_id"] == event_id].sort_values("holding_bar")
        if len(path) != int(event["bars_held"]):
            raise ValueError("lifecycle panel does not match accepted event")
        terminal_rows = path.loc[path["terminal_event"] != "NONE", "terminal_event"]
        if len(terminal_rows) != 1 or terminal_rows.iloc[0] != str(event["exit_reason"]):
            raise ValueError("lifecycle panel terminal event does not match ledger")
        observable = path.loc[path["preterminal_observable"] == 1]
        pre_mfe = float(observable["running_mfe_r"].max()) if not observable.empty else 0.0
        pre_mae = float(observable["running_mae_r"].min()) if not observable.empty else 0.0
        record: dict[str, object] = {
            "event_id": event_id,
            "direction": str(event["direction"]),
            "entry_index": int(event["entry_index"]),
            "exit_index": int(event["exit_index"]),
            "bars_held": int(event["bars_held"]),
            "exit_reason": str(event["exit_reason"]),
            "ambiguous": int(event["ambiguous"]),
            "gross_r_lower": float(event["gross_r_lower"]),
            "gross_r_upper": float(event["gross_r_upper"]),
            "preterminal_mfe_r": max(0.0, pre_mfe),
            "preterminal_mae_r": min(0.0, pre_mae),
            "full_path_mfe_r": max(0.0, float(path["running_mfe_r"].max())),
            "full_path_mae_r": min(0.0, float(path["running_mae_r"].min())),
        }
        for landmark in MFE_LANDMARKS:
            label = str(landmark).replace(".", "_")
            record[f"preterminal_reached_{label}r"] = int(record["preterminal_mfe_r"] >= landmark)
        records.append(record)
    return pd.DataFrame(records)


def build_competing_hazards(paths: pd.DataFrame, maximum_holding_bars: int = 32) -> pd.DataFrame:
    """Calculate discrete cause-specific hazards and cumulative incidence."""
    required = {"bars_held", "exit_reason"}
    missing = required - set(paths.columns)
    if missing:
        raise ValueError(f"missing path-summary columns: {sorted(missing)}")
    if maximum_holding_bars < 1:
        raise ValueError("maximum_holding_bars must be at least one")
    if paths.empty:
        return pd.DataFrame()
    unsupported = set(paths["exit_reason"].astype(str)) - set(TERMINAL_EVENTS)
    if unsupported:
        raise ValueError(f"unsupported terminal events: {sorted(unsupported)}")
    if (paths["bars_held"] < 1).any() or (paths["bars_held"] > maximum_holding_bars).any():
        raise ValueError("bars_held is outside the frozen lifecycle")

    survival = 1.0
    cumulative = {event: 0.0 for event in TERMINAL_EVENTS}
    records: list[dict[str, object]] = []
    for holding_bar in range(1, maximum_holding_bars + 1):
        at_risk = int((paths["bars_held"] >= holding_bar).sum())
        counts = {
            event: int(((paths["bars_held"] == holding_bar) & (paths["exit_reason"] == event)).sum())
            for event in TERMINAL_EVENTS
        }
        if at_risk == 0:
            break
        survival_before = survival
        total_events = sum(counts.values())
        record: dict[str, object] = {
            "holding_bar": holding_bar,
            "at_risk": at_risk,
            "survival_before": survival_before,
        }
        for event, count in counts.items():
            prefix = event.lower()
            hazard = count / at_risk
            cumulative[event] += survival_before * hazard
            record[f"{prefix}_events"] = count
            record[f"{prefix}_hazard"] = hazard
            record[f"{prefix}_cumulative_incidence"] = cumulative[event]
        survival = survival_before * (1.0 - total_events / at_risk)
        record["survival_after"] = survival
        records.append(record)
    return pd.DataFrame(records)


def lifecycle_phase(holding_bar: int) -> str:
    if 1 <= holding_bar <= 4:
        return "ENTRY_1_4"
    if 5 <= holding_bar <= 16:
        return "DEVELOPMENT_5_16"
    if 17 <= holding_bar <= 31:
        return "LATE_17_31"
    if holding_bar == 32:
        return "FORCED_CLOSE_32"
    raise ValueError("holding_bar is outside the frozen 32-bar lifecycle")
