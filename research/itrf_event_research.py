"""Leakage-resistant event-dataset and probability-evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


BASELINE_FEATURES = (
    "direction_long",
    "trend_aligned",
    "signed_momentum_atr",
    "signed_delta_zscore",
    "relative_volume",
    "high_volatility",
)

CONTEXT_FEATURES = BASELINE_FEATURES + (
    "regime_aligned",
    "structure_aligned",
    "directional_sweep",
    "location_aligned",
    "directional_break",
    "displacement_aligned",
    "signed_midpoint_distance_atr",
)


@dataclass(frozen=True)
class WalkForwardConfig:
    folds: int = 3
    initial_train_fraction: float = 0.40
    embargo_bars: int = 32

    def validate(self) -> None:
        if self.folds < 1:
            raise ValueError("folds must be at least one")
        if not 0 < self.initial_train_fraction < 1:
            raise ValueError("initial_train_fraction must be between zero and one")
        if self.embargo_bars < 0:
            raise ValueError("embargo_bars cannot be negative")


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train_positions: np.ndarray
    test_positions: np.ndarray
    test_start_bar: int


def build_event_dataset(context: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    """Join signal-time causal features to accepted clean-core outcomes."""
    required_context = {
        "trend", "momentum_atr", "delta_zscore", "relative_volume",
        "high_volatility", "market_regime", "structure_bias",
        "sell_side_sweep", "buy_side_sweep", "discount", "premium",
        "bullish_bos", "bearish_bos", "bullish_choch", "bearish_choch",
        "bullish_displacement", "bearish_displacement", "dealing_midpoint",
        "close", "atr",
    }
    missing_context = required_context - set(context.columns)
    if missing_context:
        raise ValueError(f"missing causal context columns: {sorted(missing_context)}")
    required_ledger = {
        "signal_index", "signal_time", "direction", "decision", "exit_index",
        "exit_time", "exit_reason", "gross_r_lower", "gross_r_upper",
        "ambiguous",
    }
    missing_ledger = required_ledger - set(ledger.columns)
    if missing_ledger:
        raise ValueError(f"missing clean ledger columns: {sorted(missing_ledger)}")
    accepted = ledger.loc[ledger["decision"] == "ACCEPTED"].copy()
    accepted = accepted.sort_values("signal_index").reset_index(drop=True)
    records: list[dict[str, object]] = []
    for event_id, event in accepted.iterrows():
        signal_index = int(event["signal_index"])
        row = context.iloc[signal_index]
        direction = str(event["direction"])
        sign = 1 if direction == "LONG" else -1
        directional_sweep = row["sell_side_sweep"] if direction == "LONG" else row["buy_side_sweep"]
        location = row["discount"] if direction == "LONG" else row["premium"]
        directional_break = (
            max(row["bullish_bos"], row["bullish_choch"])
            if direction == "LONG"
            else max(row["bearish_bos"], row["bearish_choch"])
        )
        displacement = row["bullish_displacement"] if direction == "LONG" else row["bearish_displacement"]
        regime_sign = 1 if row["market_regime"] == "TREND_UP" else -1 if row["market_regime"] == "TREND_DOWN" else 0
        midpoint_distance = (
            sign * (float(row["dealing_midpoint"]) - float(row["close"])) / float(row["atr"])
            if pd.notna(row["dealing_midpoint"]) and float(row["atr"]) > 0
            else np.nan
        )
        records.append(
            {
                "event_id": event_id,
                "signal_index": signal_index,
                "exit_index": int(event["exit_index"]),
                "signal_time": event["signal_time"],
                "exit_time": event["exit_time"],
                "direction": direction,
                "target_success": int(event["exit_reason"] == "TARGET"),
                "gross_r_lower": float(event["gross_r_lower"]),
                "gross_r_upper": float(event["gross_r_upper"]),
                "ambiguous": int(event["ambiguous"]),
                "direction_long": int(direction == "LONG"),
                "trend_aligned": int(int(row["trend"]) == sign),
                "signed_momentum_atr": sign * float(row["momentum_atr"]),
                "signed_delta_zscore": sign * float(row["delta_zscore"]),
                "relative_volume": float(row["relative_volume"]),
                "high_volatility": int(row["high_volatility"]),
                "regime_aligned": int(regime_sign == sign),
                "structure_aligned": int(int(row["structure_bias"]) == sign),
                "directional_sweep": int(directional_sweep),
                "location_aligned": int(location),
                "directional_break": int(directional_break),
                "displacement_aligned": int(displacement),
                "signed_midpoint_distance_atr": midpoint_distance,
            }
        )
    result = pd.DataFrame(records)
    if not result.empty:
        result["signal_time"] = pd.to_datetime(result["signal_time"])
        result["exit_time"] = pd.to_datetime(result["exit_time"])
    return result


def purged_walk_forward_folds(
    events: pd.DataFrame,
    config: WalkForwardConfig = WalkForwardConfig(),
) -> list[WalkForwardFold]:
    """Return expanding past-only folds with label purge and pre-test embargo."""
    config.validate()
    required = {"signal_index", "exit_index"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"missing event interval columns: {sorted(missing)}")
    ordered = events.reset_index(drop=True)
    if not ordered["signal_index"].is_monotonic_increasing:
        raise ValueError("events must be chronological")
    total = len(ordered)
    initial = int(np.floor(total * config.initial_train_fraction))
    if initial < 1 or total - initial < config.folds:
        raise ValueError("not enough events for requested walk-forward folds")
    test_chunks = np.array_split(np.arange(initial, total), config.folds)
    folds: list[WalkForwardFold] = []
    for fold_number, test_positions in enumerate(test_chunks, start=1):
        if len(test_positions) == 0:
            continue
        test_start_bar = int(ordered.loc[int(test_positions[0]), "signal_index"])
        train_cutoff = test_start_bar - config.embargo_bars
        candidate_train = np.arange(0, int(test_positions[0]))
        train_positions = candidate_train[
            ordered.loc[candidate_train, "exit_index"].to_numpy() < train_cutoff
        ]
        if len(train_positions) == 0:
            raise ValueError(f"fold {fold_number} has no training events after purge")
        folds.append(
            WalkForwardFold(
                fold=fold_number,
                train_positions=train_positions,
                test_positions=np.asarray(test_positions, dtype=int),
                test_start_bar=test_start_bar,
            )
        )
    return folds


class FixedLogisticModel:
    """Small deterministic L2-logistic model with train-only preprocessing."""

    def __init__(self, l2_strength: float = 1.0, maximum_iterations: int = 100):
        if l2_strength < 0 or not np.isfinite(l2_strength):
            raise ValueError("l2_strength must be non-negative and finite")
        self.l2_strength = float(l2_strength)
        self.maximum_iterations = int(maximum_iterations)
        self.medians: np.ndarray | None = None
        self.means: np.ndarray | None = None
        self.scales: np.ndarray | None = None
        self.coefficients: np.ndarray | None = None

    def _prepare(self, values: pd.DataFrame, fit: bool) -> np.ndarray:
        matrix = values.to_numpy(dtype=float)
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
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "FixedLogisticModel":
        design = self._prepare(features, fit=True)
        y = target.to_numpy(dtype=float)
        if len(np.unique(y)) < 2:
            probability = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
            self.coefficients = np.zeros(design.shape[1])
            self.coefficients[0] = np.log(probability / (1.0 - probability))
            return self
        coefficients = np.zeros(design.shape[1])
        penalty = np.eye(design.shape[1]) * self.l2_strength
        penalty[0, 0] = 0.0
        for _ in range(self.maximum_iterations):
            probability = self._sigmoid(design @ coefficients)
            weights = np.clip(probability * (1.0 - probability), 1e-8, None)
            gradient = design.T @ (y - probability) - penalty @ coefficients
            information = design.T @ (weights[:, None] * design) + penalty
            update = np.linalg.solve(information, gradient)
            coefficients += update
            if np.max(np.abs(update)) < 1e-8:
                break
        self.coefficients = coefficients
        return self

    def predict_probability(self, features: pd.DataFrame) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("model is not fitted")
        design = self._prepare(features, fit=False)
        return self._sigmoid(design @ self.coefficients)


def probability_metrics(target: pd.Series, probability: pd.Series) -> dict[str, float]:
    y = target.to_numpy(dtype=float)
    p = np.clip(probability.to_numpy(dtype=float), 1e-8, 1 - 1e-8)
    brier = float(np.mean((p - y) ** 2))
    log_loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    auc = np.nan
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives > 0 and negatives > 0:
        ranks = pd.Series(p).rank(method="average").to_numpy()
        auc = float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))
    return {
        "events": len(y),
        "observed_success": float(y.mean()),
        "mean_probability": float(p.mean()),
        "brier_score": brier,
        "log_loss": log_loss,
        "roc_auc": auc,
    }


def block_bootstrap_mean_ci(
    values: pd.Series,
    seed: int = 20260818,
    samples: int = 10_000,
) -> tuple[float, float]:
    data = values.dropna().to_numpy(dtype=float)
    if len(data) == 0:
        return np.nan, np.nan
    if len(data) == 1:
        return float(data[0]), float(data[0])
    block = max(2, int(round(np.sqrt(len(data)))))
    blocks = int(np.ceil(len(data) / block))
    offsets = np.arange(block)
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for sample in range(samples):
        starts = rng.integers(0, len(data), size=blocks)
        indices = (starts[:, None] + offsets) % len(data)
        means[sample] = data[indices.ravel()[: len(data)]].mean()
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def walk_forward_predictions(
    events: pd.DataFrame,
    config: WalkForwardConfig = WalkForwardConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit fixed models per fold and return event predictions plus fold audit."""
    eligible = events.loc[events["ambiguous"] == 0].sort_values("signal_index").reset_index(drop=True)
    folds = purged_walk_forward_folds(eligible, config)
    predictions: list[dict[str, object]] = []
    fold_records: list[dict[str, object]] = []
    for fold in folds:
        train = eligible.iloc[fold.train_positions]
        test = eligible.iloc[fold.test_positions]
        null_probability = float(np.clip(train["target_success"].mean(), 1e-6, 1 - 1e-6))
        model_specs = {
            "constant_null": None,
            "price_activity": BASELINE_FEATURES,
            "causal_context": CONTEXT_FEATURES,
        }
        for model_name, feature_names in model_specs.items():
            if feature_names is None:
                probabilities = np.full(len(test), null_probability)
            else:
                model = FixedLogisticModel(l2_strength=1.0)
                model.fit(train.loc[:, feature_names], train["target_success"])
                probabilities = model.predict_probability(test.loc[:, feature_names])
            for (_, event), probability in zip(test.iterrows(), probabilities):
                predictions.append(
                    {
                        "fold": fold.fold,
                        "model": model_name,
                        "event_id": int(event["event_id"]),
                        "signal_index": int(event["signal_index"]),
                        "target_success": int(event["target_success"]),
                        "gross_r": float(event["gross_r_lower"]),
                        "probability": float(probability),
                    }
                )
        fold_records.append(
            {
                "fold": fold.fold,
                "train_events": len(train),
                "test_events": len(test),
                "train_last_exit_bar": int(train["exit_index"].max()),
                "test_start_bar": fold.test_start_bar,
                "realized_embargo_bars": fold.test_start_bar - int(train["exit_index"].max()),
            }
        )
    return pd.DataFrame(predictions), pd.DataFrame(fold_records)
