from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from itrf_context import ContextConfig, build_trade_plan, create_context_features
from itrf_research import create_features
from run_v09_research import build_context_observations


def _base_frame():
    rows = 80
    close = [100 + (i * 0.1) for i in range(rows)]
    return pd.DataFrame({
        "open": close,
        "high": [value + 0.25 for value in close],
        "low": [value - 0.25 for value in close],
        "close": close,
        "atr": [1.0] * rows,
        "ema_50": [101.0] * rows,
        "ema_200": [100.0] * rows,
        "body_ratio": [0.7] * rows,
        "range_atr": [1.2] * rows,
    })


class ContextFeatureTests(unittest.TestCase):
    def test_swing_is_not_available_until_confirmation_bar(self):
        frame = _base_frame()
        frame.loc[10, ["high", "low"]] = [120.0, 99.0]
        result = create_context_features(frame, ContextConfig(swing_length=3))
        self.assertTrue(pd.isna(result.loc[12, "swing_high_level"]))
        self.assertEqual(result.loc[13, "swing_high_level"], 120.0)

    def test_trade_plan_uses_fixed_fractional_risk_and_stop_below_long_low(self):
        row = pd.Series({"context_signal": "LONG", "close": 100.0, "low": 98.0, "high": 101.0, "swing_low_level": 99.0, "swing_high_level": 105.0, "atr": 2.0})
        plan = build_trade_plan(row, account_equity=10_000, config=ContextConfig(account_risk_fraction=0.01, stop_atr_buffer=0.10))
        self.assertEqual(plan["direction"], "LONG")
        self.assertAlmostEqual(plan["stop"], 97.8)
        self.assertAlmostEqual(plan["units"], 100 / 2.2)

    def test_missing_v08_features_is_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "Missing v0.8 feature columns"):
            create_context_features(pd.DataFrame({"close": [1.0]}))

    def test_context_layer_accepts_v08_feature_output(self):
        frame = _base_frame()
        frame["volume"] = 100.0
        frame["time"] = pd.date_range("2025-01-01", periods=len(frame), freq="15min")
        v08 = create_features(frame)
        context = create_context_features(v08)
        self.assertIn("context_signal", context.columns)
        self.assertIn("market_regime", context.columns)

    def test_research_runner_writes_separate_context_table(self):
        frame = _base_frame()
        frame["volume"] = 100.0
        frame["time"] = pd.date_range("2025-01-01", periods=len(frame), freq="15min")
        context = create_context_features(create_features(frame))
        import sqlite3
        with sqlite3.connect(":memory:") as connection:
            count = build_context_observations(context, connection)
            stored = connection.execute("SELECT COUNT(*) FROM v09_context_observations").fetchone()[0]
        self.assertEqual(count, stored)
