"""Leakage-resistant discrete competing-risk calibration utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from itrf_event_research import (
    BASELINE_FEATURES,
    WalkForwardConfig,
    block_bootstrap_mean_ci,
    purged_walk_forward_folds,
)


CLASSES = ("NONE", "STOP", "TARGET")
PROBABILITY_COLUMNS = ("prob_none", "prob_stop", "prob_target")
TIME_FEATURES = (
    "holding_fraction",
    "entry_phase",
    "development_phase",
    "late_phase",
    "forced_close_phase",
)
PREENTRY_FEATURES = TIME_FEATURES + BASELINE_FEATURES
PATH_STATE_FEATURES = (
    "prior_mfe_r",
    "prior_mae_r",
    "prior_close_r",
    "prior_bar_range_r",
    "prior_close_change_r",
)
DYNAMIC_FEATURES = PREENTRY_FEATURES + PATH_STATE_FEATURES


def build_risk_set_rows(
    frame: pd.DataFrame,
    lifecycle_panel: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Create start-of-bar causal rows for discrete competing events."""
    if "close" not in frame.columns:
        raise ValueError("market frame requires close")
    panel_required = {
        "event_id", "bar_index", "holding_bar", "direction", "entry", "risk",
        "bar_favorable_r", "bar_adverse_r", "running_mfe_r", "running_mae_r",
        "terminal_event",
    }
    event_required = {"event_id", "signal_index", "exit_index", "ambiguous", *BASELINE_FEATURES}
    missing_panel = panel_required - set(lifecycle_panel.columns)
    missing_events = event_required - set(events.columns)
    if missing_panel:
        raise ValueError(f"missing lifecycle columns: {sorted(missing_panel)}")
    if missing_events:
        raise ValueError(f"missing event columns: {sorted(missing_events)}")

    event_lookup = events.set_index("event_id", verify_integrity=True)
    records: list[dict[str, object]] = []
    for event_id, path in lifecycle_panel.groupby("event_id", sort=True):
        if event_id not in event_lookup.index:
            raise ValueError("lifecycle event is absent from event dataset")
        event = event_lookup.loc[event_id]
        if int(event["ambiguous"]) == 1:
            continue
        path = path.sort_values("holding_bar")
        prior_mfe = 0.0
        prior_mae = 0.0
        prior_close_r = 0.0
        prior_bar_range_r = 0.0
        prior_close_change_r = 0.0
        previous_close_r = 0.0
        for row_number, row in enumerate(path.itertuples(index=False)):
            holding_bar = int(row.holding_bar)
            terminal = str(row.terminal_event)
            outcome = terminal if terminal in {"STOP", "TARGET"} else "NONE"
            record: dict[str, object] = {
                "event_id": int(event_id),
                "signal_index": int(event["signal_index"]),
                "exit_index": int(event["exit_index"]),
                "holding_bar": holding_bar,
                "outcome": outcome,
                "holding_fraction": (holding_bar - 1) / 32.0,
                "entry_phase": int(1 <= holding_bar <= 4),
                "development_phase": int(5 <= holding_bar <= 16),
                "late_phase": int(17 <= holding_bar <= 31),
                "forced_close_phase": int(holding_bar == 32),
                "prior_mfe_r": prior_mfe,
                "prior_mae_r": prior_mae,
                "prior_close_r": prior_close_r,
                "prior_bar_range_r": prior_bar_range_r,
                "prior_close_change_r": prior_close_change_r,
            }
            for feature in BASELINE_FEATURES:
                record[feature] = event[feature]
            records.append(record)

            direction_sign = 1.0 if str(row.direction) == "LONG" else -1.0
            close_r = direction_sign * (
                float(frame.iloc[int(row.bar_index)]["close"]) - float(row.entry)
            ) / float(row.risk)
            prior_mfe = max(0.0, float(row.running_mfe_r))
            prior_mae = min(0.0, float(row.running_mae_r))
            prior_close_r = close_r
            prior_bar_range_r = float(row.bar_favorable_r) - float(row.bar_adverse_r)
            prior_close_change_r = close_r - previous_close_r
            previous_close_r = close_r
        if row_number + 1 != len(path):
            raise RuntimeError("risk-set path expansion failed")
    return pd.DataFrame(records)


class FixedMultinomialModel:
    """Deterministic L2 softmax model with NONE as the reference category."""

    def __init__(self, l2_strength: float = 1.0, maximum_iterations: int = 20_000):
        if not np.isfinite(l2_strength) or l2_strength < 0:
            raise ValueError("l2_strength must be non-negative and finite")
        self.l2_strength = float(l2_strength)
        self.maximum_iterations = int(maximum_iterations)
        self.medians: np.ndarray | None = None
        self.means: np.ndarray | None = None
        self.scales: np.ndarray | None = None
        self.coefficients: np.ndarray | None = None
        self.converged = False
        self.iterations = 0

    def _prepare(self, features: pd.DataFrame, fit: bool) -> np.ndarray:
        matrix = features.to_numpy(dtype=float)
        if fit:
            self.medians = np.nanmedian(matrix, axis=0)
            self.medians = np.where(np.isfinite(self.medians), self.medians, 0.0)
        if self.medians is None:
            raise RuntimeError("model preprocessing is not fitted")
        missing = ~np.isfinite(matrix)
        matrix[missing] = np.take(self.medians, np.where(missing)[1])
        if fit:
            self.means = matrix.mean(axis=0)
            scales = matrix.std(axis=0)
            self.scales = np.where(scales > 0, scales, 1.0)
        if self.means is None or self.scales is None:
            raise RuntimeError("model preprocessing is not fitted")
        standardized = (matrix - self.means) / self.scales
        return np.column_stack([np.ones(len(standardized)), standardized])

    @staticmethod
    def _probabilities(design: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
        cause_logits = np.sum(design[:, :, None] * coefficients[None, :, :], axis=1)
        logits = np.column_stack([np.zeros(len(design)), cause_logits])
        logits = logits - logits.max(axis=1, keepdims=True)
        exponentials = np.exp(np.clip(logits, -35.0, 0.0))
        return exponentials / exponentials.sum(axis=1, keepdims=True)

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "FixedMultinomialModel":
        design = self._prepare(features, fit=True)
        labels = target.astype(str).to_numpy()
        unknown = set(labels) - set(CLASSES)
        if unknown:
            raise ValueError(f"unsupported competing-risk classes: {sorted(unknown)}")
        if any(not np.any(labels == label) for label in CLASSES):
            raise ValueError("every competing-risk class must occur in training")
        coefficients = np.zeros((design.shape[1], 2))
        counts = np.array([(labels == label).sum() for label in CLASSES], dtype=float)
        coefficients[0, :] = np.log(counts[1:] / counts[0])
        penalty = np.full(design.shape[1], self.l2_strength)
        penalty[0] = 0.0
        lipschitz = float(0.5 * np.sum(design * design) + 2.0 * np.sum(penalty))
        if not np.isfinite(lipschitz) or lipschitz <= 0:
            raise FloatingPointError("invalid multinomial curvature bound")
        step_size = 1.0 / lipschitz
        indicators = np.column_stack([labels == "STOP", labels == "TARGET"]).astype(float)
        for iteration in range(1, self.maximum_iterations + 1):
            probabilities = self._probabilities(design, coefficients)
            residual = indicators - probabilities[:, 1:]
            gradient = np.sum(
                design[:, :, None] * residual[:, None, :], axis=0
            ) - penalty[:, None] * coefficients
            update = step_size * gradient
            coefficients = coefficients + update
            self.iterations = iteration
            if np.max(np.abs(update)) < 1e-9:
                self.converged = True
                break
        if not self.converged:
            raise RuntimeError("multinomial solver did not converge")
        if not np.isfinite(coefficients).all():
            raise FloatingPointError("multinomial solver produced non-finite coefficients")
        self.coefficients = coefficients
        return self

    def predict_probability(self, features: pd.DataFrame) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("model is not fitted")
        return self._probabilities(self._prepare(features, fit=False), self.coefficients)


class EmpiricalHoldingHazard:
    """Training-only exact holding-bar hazards with fixed Jeffreys smoothing."""

    def __init__(self, smoothing: float = 0.5):
        if not np.isfinite(smoothing) or smoothing <= 0:
            raise ValueError("smoothing must be positive and finite")
        self.smoothing = float(smoothing)
        self.probabilities: dict[int, np.ndarray] = {}
        self.fallback: np.ndarray | None = None

    def fit(self, rows: pd.DataFrame) -> "EmpiricalHoldingHazard":
        required = {"holding_bar", "outcome"}
        missing = required - set(rows.columns)
        if missing:
            raise ValueError(f"missing empirical hazard columns: {sorted(missing)}")
        counts = np.array([(rows["outcome"] == label).sum() for label in CLASSES], dtype=float)
        self.fallback = (counts + self.smoothing) / (counts.sum() + 3 * self.smoothing)
        for holding_bar, group in rows.groupby("holding_bar"):
            bar_counts = np.array([(group["outcome"] == label).sum() for label in CLASSES], dtype=float)
            self.probabilities[int(holding_bar)] = (
                bar_counts + self.smoothing
            ) / (bar_counts.sum() + 3 * self.smoothing)
        return self

    def predict_probability(self, rows: pd.DataFrame) -> np.ndarray:
        if self.fallback is None:
            raise RuntimeError("empirical hazard is not fitted")
        return np.vstack([
            self.probabilities.get(int(holding_bar), self.fallback)
            for holding_bar in rows["holding_bar"]
        ])


def walk_forward_hazard_predictions(
    risk_rows: pd.DataFrame,
    events: pd.DataFrame,
    config: WalkForwardConfig = WalkForwardConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate fixed competing-risk predictions with event-level purging."""
    eligible = events.loc[events["ambiguous"] == 0].sort_values("signal_index").reset_index(drop=True)
    folds = purged_walk_forward_folds(eligible, config)
    predictions: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for fold in folds:
        train_events = eligible.iloc[fold.train_positions]
        test_events = eligible.iloc[fold.test_positions]
        train_ids = set(train_events["event_id"].astype(int))
        test_ids = set(test_events["event_id"].astype(int))
        train = risk_rows.loc[risk_rows["event_id"].isin(train_ids)].copy()
        test = risk_rows.loc[risk_rows["event_id"].isin(test_ids)].copy()
        model_specs = {
            "empirical_hazard": (EmpiricalHoldingHazard().fit(train), None),
            "preentry_time": (
                FixedMultinomialModel().fit(train.loc[:, PREENTRY_FEATURES], train["outcome"]),
                PREENTRY_FEATURES,
            ),
            "dynamic_path": (
                FixedMultinomialModel().fit(train.loc[:, DYNAMIC_FEATURES], train["outcome"]),
                DYNAMIC_FEATURES,
            ),
        }
        for model_name, (model, feature_names) in model_specs.items():
            probabilities = (
                model.predict_probability(test)
                if feature_names is None
                else model.predict_probability(test.loc[:, feature_names])
            )
            for (_, row), probability in zip(test.iterrows(), probabilities):
                predictions.append(
                    {
                        "fold": fold.fold,
                        "model": model_name,
                        "event_id": int(row["event_id"]),
                        "holding_bar": int(row["holding_bar"]),
                        "outcome": str(row["outcome"]),
                        **dict(zip(PROBABILITY_COLUMNS, probability)),
                    }
                )
        audits.append(
            {
                "fold": fold.fold,
                "train_events": len(train_events),
                "train_rows": len(train),
                "test_events": len(test_events),
                "test_rows": len(test),
                "train_last_exit_bar": int(train_events["exit_index"].max()),
                "test_start_bar": fold.test_start_bar,
                "realized_embargo_bars": fold.test_start_bar - int(train_events["exit_index"].max()),
            }
        )
    return pd.DataFrame(predictions), pd.DataFrame(audits)


def competing_risk_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    probabilities = predictions.loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=float)
    labels = predictions["outcome"].astype(str).to_numpy()
    class_index = {label: index for index, label in enumerate(CLASSES)}
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(labels)), [class_index[label] for label in labels]] = 1.0
    row_errors = np.sum((probabilities - one_hot) ** 2, axis=1)
    chosen = probabilities[np.arange(len(labels)), [class_index[label] for label in labels]]
    event_errors = pd.Series(row_errors, index=predictions.index).groupby(predictions["event_id"]).mean()
    return {
        "rows": len(predictions),
        "events": predictions["event_id"].nunique(),
        "multiclass_brier": float(row_errors.mean()),
        "event_balanced_brier": float(event_errors.mean()),
        "log_loss": float(-np.log(np.clip(chosen, 1e-8, 1.0)).mean()),
        "observed_stop": float(np.mean(labels == "STOP")),
        "predicted_stop": float(probabilities[:, 1].mean()),
        "observed_target": float(np.mean(labels == "TARGET")),
        "predicted_target": float(probabilities[:, 2].mean()),
    }


def event_balanced_brier_improvement(
    predictions: pd.DataFrame,
    model: str,
) -> pd.Series:
    indexed = predictions.set_index(["event_id", "holding_bar", "model"])
    rows: list[tuple[int, float]] = []
    class_index = {label: index for index, label in enumerate(CLASSES)}
    for event_id in sorted(predictions["event_id"].unique()):
        event = indexed.loc[event_id]
        null = event.xs("empirical_hazard", level="model")
        candidate = event.xs(model, level="model")
        if not null.index.equals(candidate.index):
            raise ValueError("model risk rows do not align")
        labels = null["outcome"].to_numpy()
        one_hot = np.zeros((len(labels), 3))
        one_hot[np.arange(len(labels)), [class_index[label] for label in labels]] = 1.0
        null_error = np.sum((null.loc[:, PROBABILITY_COLUMNS].to_numpy() - one_hot) ** 2, axis=1)
        model_error = np.sum((candidate.loc[:, PROBABILITY_COLUMNS].to_numpy() - one_hot) ** 2, axis=1)
        rows.append((int(event_id), float(np.mean(null_error - model_error))))
    return pd.Series(dict(rows)).sort_index()


def improvement_interval(predictions: pd.DataFrame, model: str) -> tuple[float, float, float]:
    improvements = event_balanced_brier_improvement(predictions, model)
    low, high = block_bootstrap_mean_ci(improvements)
    return float(improvements.mean()), low, high
