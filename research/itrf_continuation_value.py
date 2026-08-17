"""Causal continuation-value modeling for frozen ITRF event paths."""

from __future__ import annotations

import numpy as np
import pandas as pd

from itrf_competing_risk import DYNAMIC_FEATURES, PROBABILITY_COLUMNS


DECISION_FEATURES = DYNAMIC_FEATURES + PROBABILITY_COLUMNS + ("exit_now_r",)


class FixedWeightedRidge:
    """Deterministic event-weighted ridge with train-only preprocessing."""

    def __init__(self, l2_strength: float = 1.0):
        if not np.isfinite(l2_strength) or l2_strength < 0:
            raise ValueError("l2_strength must be non-negative and finite")
        self.l2_strength = float(l2_strength)
        self.medians: np.ndarray | None = None
        self.means: np.ndarray | None = None
        self.scales: np.ndarray | None = None
        self.coefficients: np.ndarray | None = None

    def _prepare(
        self,
        features: pd.DataFrame,
        fit: bool,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        matrix = features.to_numpy(dtype=float)
        if fit:
            if weights is None:
                raise ValueError("fit preprocessing requires weights")
            finite_matrix = np.where(np.isfinite(matrix), matrix, np.nan)
            self.medians = np.nanmedian(finite_matrix, axis=0)
            self.medians = np.where(np.isfinite(self.medians), self.medians, 0.0)
        if self.medians is None:
            raise RuntimeError("ridge preprocessing is not fitted")
        missing = ~np.isfinite(matrix)
        matrix[missing] = np.take(self.medians, np.where(missing)[1])
        if fit:
            assert weights is not None
            self.means = np.average(matrix, axis=0, weights=weights)
            variance = np.average((matrix - self.means) ** 2, axis=0, weights=weights)
            self.scales = np.where(variance > 0, np.sqrt(variance), 1.0)
        if self.means is None or self.scales is None:
            raise RuntimeError("ridge preprocessing is not fitted")
        standardized = (matrix - self.means) / self.scales
        return np.column_stack([np.ones(len(standardized)), standardized])

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        weights: pd.Series,
    ) -> "FixedWeightedRidge":
        y = target.to_numpy(dtype=float)
        raw_weights = weights.to_numpy(dtype=float)
        if len(features) == 0 or len(features) != len(y) or len(y) != len(raw_weights):
            raise ValueError("features, target, and weights must have equal non-zero length")
        if not np.isfinite(y).all():
            raise ValueError("ridge target must be finite")
        if not np.isfinite(raw_weights).all() or (raw_weights <= 0).any():
            raise ValueError("ridge weights must be positive and finite")
        normalized_weights = raw_weights * len(raw_weights) / raw_weights.sum()
        design = self._prepare(features, fit=True, weights=normalized_weights)
        columns = design.shape[1]
        gram = np.zeros((columns, columns), dtype=float)
        right_hand_side = np.zeros(columns, dtype=float)
        weighted_target = normalized_weights * y
        # Explicit reductions avoid platform-specific small-matrix BLAS status
        # flag anomalies while preserving the exact weighted ridge objective.
        for left in range(columns):
            right_hand_side[left] = np.sum(design[:, left] * weighted_target)
            for right in range(left + 1):
                value = np.sum(
                    normalized_weights * design[:, left] * design[:, right]
                )
                gram[left, right] = value
                gram[right, left] = value
        for column in range(1, columns):
            gram[column, column] += self.l2_strength
        coefficients = self._cholesky_solve(gram, right_hand_side)
        if not np.isfinite(coefficients).all():
            raise FloatingPointError("ridge solver produced non-finite coefficients")
        self.coefficients = coefficients
        return self

    @staticmethod
    def _cholesky_solve(matrix: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Solve a positive-definite system with deterministic scalar reductions."""
        size = len(values)
        lower = np.zeros_like(matrix, dtype=float)
        for row in range(size):
            for column in range(row + 1):
                remainder = matrix[row, column] - np.sum(
                    lower[row, :column] * lower[column, :column]
                )
                if row == column:
                    if not np.isfinite(remainder) or remainder <= 0:
                        raise FloatingPointError("ridge system is not positive definite")
                    lower[row, column] = np.sqrt(remainder)
                else:
                    lower[row, column] = remainder / lower[column, column]
        forward = np.zeros(size, dtype=float)
        for row in range(size):
            forward[row] = (
                values[row] - np.sum(lower[row, :row] * forward[:row])
            ) / lower[row, row]
        solution = np.zeros(size, dtype=float)
        for row in range(size - 1, -1, -1):
            solution[row] = (
                forward[row]
                - np.sum(lower[row + 1 :, row] * solution[row + 1 :])
            ) / lower[row, row]
        return solution

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("ridge model is not fitted")
        design = self._prepare(features, fit=False)
        prediction = np.sum(design * self.coefficients, axis=1)
        if not np.isfinite(prediction).all():
            raise FloatingPointError("ridge prediction is non-finite")
        return prediction


def event_balanced_row_weights(rows: pd.DataFrame) -> pd.Series:
    if "event_id" not in rows.columns or rows.empty:
        raise ValueError("non-empty rows with event_id are required")
    counts = rows.groupby("event_id")["event_id"].transform("size")
    weights = 1.0 / counts.astype(float)
    return weights / weights.mean()


def build_causal_decision_rows(
    frame: pd.DataFrame,
    lifecycle_panel: pd.DataFrame,
    risk_rows: pd.DataFrame,
    hazard_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Join causal start-of-bar state, dynamic hazards, and current open R."""
    frame_required = {"open"}
    panel_required = {"event_id", "holding_bar", "bar_index", "direction", "entry", "risk"}
    risk_required = {"event_id", "holding_bar", *DYNAMIC_FEATURES}
    prediction_required = {"model", "event_id", "holding_bar", *PROBABILITY_COLUMNS}
    for label, required, columns in (
        ("frame", frame_required, set(frame.columns)),
        ("lifecycle", panel_required, set(lifecycle_panel.columns)),
        ("risk", risk_required, set(risk_rows.columns)),
        ("hazard", prediction_required, set(hazard_predictions.columns)),
    ):
        missing = required - columns
        if missing:
            raise ValueError(f"missing {label} columns: {sorted(missing)}")

    dynamic = hazard_predictions.loc[
        hazard_predictions["model"] == "dynamic_path",
        ["event_id", "holding_bar", *PROBABILITY_COLUMNS],
    ].copy()
    if dynamic.empty or dynamic.duplicated(["event_id", "holding_bar"]).any():
        raise ValueError("dynamic hazard rows must be non-empty and unique")
    rows = risk_rows.merge(
        dynamic,
        on=["event_id", "holding_bar"],
        how="inner",
        validate="one_to_one",
    )
    panel = lifecycle_panel.loc[
        :, ["event_id", "holding_bar", "bar_index", "direction", "entry", "risk"]
    ].copy()
    if panel.duplicated(["event_id", "holding_bar"]).any():
        raise ValueError("lifecycle rows must be unique")
    rows = rows.merge(panel, on=["event_id", "holding_bar"], how="left", validate="one_to_one")
    if rows["bar_index"].isna().any():
        raise ValueError("decision row is missing its lifecycle bar")
    rows = rows.loc[rows["holding_bar"] >= 2].copy()
    if rows.empty:
        raise ValueError("no post-entry decision rows are available")
    unsupported = set(rows["direction"].astype(str)) - {"LONG", "SHORT"}
    if unsupported:
        raise ValueError(f"unsupported directions: {sorted(unsupported)}")
    if (~np.isfinite(rows["risk"].astype(float)) | (rows["risk"].astype(float) <= 0)).any():
        raise ValueError("decision-row risk must be positive and finite")
    bar_indices = rows["bar_index"].astype(int).to_numpy()
    if (bar_indices < 0).any() or (bar_indices >= len(frame)).any():
        raise ValueError("decision-row bar index is outside the market frame")
    exit_prices = frame.iloc[bar_indices]["open"].to_numpy(dtype=float)
    signs = np.where(rows["direction"].astype(str).to_numpy() == "LONG", 1.0, -1.0)
    rows["exit_now_r"] = signs * (
        exit_prices - rows["entry"].to_numpy(dtype=float)
    ) / rows["risk"].to_numpy(dtype=float)
    rows = rows.sort_values(["event_id", "holding_bar"]).reset_index(drop=True)
    if not np.isfinite(rows.loc[:, DECISION_FEATURES].to_numpy(dtype=float)).all():
        # NaNs in the pre-entry fields are allowed and imputed by the model.
        probabilities_and_exit = rows.loc[:, [*PROBABILITY_COLUMNS, "exit_now_r"]]
        if not np.isfinite(probabilities_and_exit.to_numpy(dtype=float)).all():
            raise ValueError("hazard probabilities and exit-now R must be finite")
    return rows


def attach_continuation_labels(
    decision_rows: pd.DataFrame,
    ledger: pd.DataFrame,
) -> pd.DataFrame:
    """Attach development-only full-path continuation advantage labels."""
    required = {"decision", "gross_r_lower", "gross_r_upper", "ambiguous"}
    missing = required - set(ledger.columns)
    if missing:
        raise ValueError(f"missing ledger columns: {sorted(missing)}")
    accepted = ledger.loc[ledger["decision"] == "ACCEPTED"].copy().reset_index(drop=True)
    eligible = accepted.loc[accepted["ambiguous"] == 0, ["gross_r_lower", "gross_r_upper"]].copy()
    if not np.allclose(
        eligible["gross_r_lower"].to_numpy(dtype=float),
        eligible["gross_r_upper"].to_numpy(dtype=float),
    ):
        raise ValueError("ambiguous outcomes entered continuation labels")
    terminal = eligible["gross_r_lower"].rename("baseline_terminal_gross_r")
    result = decision_rows.merge(
        terminal.rename_axis("event_id").reset_index(),
        on="event_id",
        how="left",
        validate="many_to_one",
    )
    if result["baseline_terminal_gross_r"].isna().any():
        raise ValueError("decision row is missing its terminal development label")
    result["continuation_advantage_r"] = (
        result["baseline_terminal_gross_r"] - result["exit_now_r"]
    )
    return result


def build_continuation_policy_outcomes(
    ledger: pd.DataFrame,
    external_decision_rows: pd.DataFrame,
    predicted_advantage: np.ndarray,
) -> pd.DataFrame:
    """Exit at the earliest post-entry state with negative predicted advantage."""
    required_ledger = {
        "decision", "direction", "exit_index", "exit_time", "exit_reason", "bars_held",
        "gross_r_lower", "gross_r_upper", "ambiguous",
    }
    missing = required_ledger - set(ledger.columns)
    if missing:
        raise ValueError(f"missing ledger columns: {sorted(missing)}")
    rows = external_decision_rows.copy()
    if len(rows) != len(predicted_advantage):
        raise ValueError("predicted continuation values do not align with decision rows")
    if not np.isfinite(predicted_advantage).all():
        raise ValueError("predicted continuation values must be finite")
    rows["predicted_continuation_advantage_r"] = predicted_advantage
    if rows.duplicated(["event_id", "holding_bar"]).any():
        raise ValueError("external decision rows contain duplicates")

    accepted = ledger.loc[ledger["decision"] == "ACCEPTED"].copy().reset_index(drop=True)
    eligible = accepted.loc[accepted["ambiguous"] == 0].copy()
    row_events = set(rows["event_id"].astype(int))
    eligible_events = set(eligible.index.astype(int))
    if not row_events.issubset(eligible_events):
        raise ValueError("external decision rows contain an ineligible event")
    missing_actionable = [
        event_id
        for event_id in eligible_events - row_events
        if int(eligible.loc[event_id, "bars_held"]) > 1
    ]
    if missing_actionable:
        raise ValueError("external decision rows omit an actionable event")

    records: list[dict[str, object]] = []
    for event_id, event in eligible.iterrows():
        event_rows = rows.loc[rows["event_id"] == event_id].sort_values("holding_bar")
        triggers = event_rows.loc[event_rows["predicted_continuation_advantage_r"] < 0]
        baseline_r = float(event["gross_r_lower"])
        if not np.isclose(baseline_r, float(event["gross_r_upper"])):
            raise ValueError("ambiguous event entered continuation policy")
        action = not triggers.empty
        if action:
            trigger = triggers.iloc[0]
            if int(trigger["holding_bar"]) > int(event["bars_held"]):
                raise ValueError("continuation action occurs after baseline terminal")
            managed_r = float(trigger["exit_now_r"])
            action_holding_bar: object = int(trigger["holding_bar"])
            managed_exit_index = int(trigger["bar_index"])
            predicted_at_action = float(trigger["predicted_continuation_advantage_r"])
            managed_exit_reason = "NEGATIVE_CONTINUATION_VALUE"
        else:
            managed_r = baseline_r
            action_holding_bar = pd.NA
            managed_exit_index = int(event["exit_index"])
            predicted_at_action = np.nan
            managed_exit_reason = str(event["exit_reason"])
        records.append(
            {
                "event_id": int(event_id),
                "direction": str(event["direction"]),
                "baseline_exit_reason": str(event["exit_reason"]),
                "baseline_bars_held": int(event["bars_held"]),
                "managed_action": int(action),
                "action_holding_bar": action_holding_bar,
                "managed_exit_index": managed_exit_index,
                "managed_exit_reason": managed_exit_reason,
                "predicted_advantage_at_action_r": predicted_at_action,
                "baseline_gross_r": baseline_r,
                "managed_gross_r": managed_r,
                "paired_improvement_r": managed_r - baseline_r,
            }
        )
    result = pd.DataFrame(records)
    result["action_holding_bar"] = pd.array(result["action_holding_bar"], dtype="Int64")
    return result
