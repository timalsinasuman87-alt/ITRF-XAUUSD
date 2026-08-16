from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from itrf_context import (
    ContextConfig,
    apply_v091_state_machine,
    build_trade_plan,
    create_context_features,
)
from itrf_research import (
    add_frozen_order_flow_score,
    create_features,
    load_market_data,
    validate_market_data,
)
from run_v09_research import build_context_observations, context_gate_funnel
from run_v091_research import build_v091_observations
from itrf_trade_management import ExitModel, TradeCostConfig, cost_in_r, evaluate_exit_model, summarize_models
from run_v09_trade_management import build_exit_observations


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


def _state_frame(rows=12):
    return pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=rows, freq="15min"),
        "close": [100.0] * rows,
        "low": [99.0] * rows,
        "high": [101.0] * rows,
        "sell_side_sweep": [0] * rows,
        "buy_side_sweep": [0] * rows,
        "discount": [0] * rows,
        "premium": [0] * rows,
        "bullish_bos": [0] * rows,
        "bearish_bos": [0] * rows,
        "bullish_choch": [0] * rows,
        "bearish_choch": [0] * rows,
        "bullish_displacement": [0] * rows,
        "bearish_displacement": [0] * rows,
    })


class ContextFeatureTests(unittest.TestCase):
    def test_pine_reference_locks_v091_window_and_first_cross_structure(self):
        pine = (Path(__file__).resolve().parents[1] / "tradingview" / "ITRF_XAUUSD_v09_reference.pine").read_text()
        self.assertIn("confirmationWindowBars = 8", pine)
        self.assertIn("close[1] <= lastSwingHigh", pine)
        self.assertIn("close[1] >= lastSwingLow", pine)
        self.assertIn("barsSinceSweep >= 1", pine)

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

    def test_v091_never_confirms_on_the_arming_bar(self):
        frame = _state_frame()
        frame.loc[0, ["sell_side_sweep", "discount", "bullish_bos", "bullish_displacement"]] = 1
        frame.loc[1, ["bullish_bos", "bullish_displacement"]] = 1

        result = apply_v091_state_machine(frame)

        self.assertEqual(result.loc[0, "v091_context_signal"], "NONE")
        self.assertEqual(result.loc[1, "v091_context_signal"], "LONG")
        self.assertEqual(result.loc[1, "v091_confirmation_lag"], 1)
        self.assertEqual(result.loc[1, "v091_source_sweep_time"], frame.loc[0, "time"])
        self.assertEqual(result.loc[1, "v091_source_sweep_extreme"], frame.loc[0, "low"])

    def test_v091_expires_after_eight_confirmation_bars(self):
        frame = _state_frame()
        frame.loc[0, ["sell_side_sweep", "discount"]] = 1
        frame.loc[9, ["bullish_bos", "bullish_displacement"]] = 1

        result = apply_v091_state_machine(frame)

        self.assertEqual(int(result["v091_long_confirmation"].sum()), 0)
        self.assertEqual(result.loc[8, "v091_expiration"], 1)

    def test_v091_sweep_extreme_invalidates_before_confirmation(self):
        frame = _state_frame()
        frame.loc[0, ["sell_side_sweep", "discount"]] = 1
        frame.loc[1, ["close", "bullish_bos", "bullish_displacement"]] = [98.5, 1, 1]

        result = apply_v091_state_machine(frame)

        self.assertEqual(result.loc[1, "v091_invalidation"], 1)
        self.assertEqual(result.loc[1, "v091_context_signal"], "NONE")

    def test_v091_opposite_sweep_replaces_active_state(self):
        frame = _state_frame()
        frame.loc[0, ["sell_side_sweep", "discount"]] = 1
        frame.loc[1, ["buy_side_sweep", "premium"]] = 1
        frame.loc[2, ["bearish_bos", "bearish_displacement"]] = 1

        result = apply_v091_state_machine(frame)

        self.assertEqual(result.loc[1, "v091_replacement"], 1)
        self.assertEqual(result.loc[2, "v091_context_signal"], "SHORT")
        self.assertEqual(result.loc[2, "v091_confirmation_lag"], 1)

    def test_v091_confirmation_clears_state_and_emits_once(self):
        frame = _state_frame()
        frame.loc[0, ["sell_side_sweep", "discount"]] = 1
        frame.loc[1:2, ["bullish_bos", "bullish_displacement"]] = 1

        result = apply_v091_state_machine(frame)

        self.assertEqual(int(result["v091_long_confirmation"].sum()), 1)
        self.assertEqual(result.loc[1, "v091_state"], "FLAT")

    def test_v091_trade_plan_places_stop_beyond_source_sweep(self):
        row = pd.Series({
            "v091_context_signal": "LONG",
            "close": 100.0,
            "low": 99.0,
            "high": 101.0,
            "swing_low_level": 98.5,
            "swing_high_level": 105.0,
            "v091_source_sweep_extreme": 97.5,
            "atr": 2.0,
        })

        plan = build_trade_plan(row, account_equity=10_000)

        self.assertAlmostEqual(plan["stop"], 97.3)
        self.assertAlmostEqual(plan["units"], 50.0 / 2.7)

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

    def test_context_funnel_final_gate_matches_context_signals(self):
        frame = pd.concat([_base_frame()] * 5, ignore_index=True)
        frame["volume"] = 100.0
        frame["time"] = pd.date_range("2025-01-01", periods=len(frame), freq="15min")
        context = create_context_features(create_features(frame))
        funnel = context_gate_funnel(context).set_index("side")
        eligible = context.iloc[250: len(context) - 32 - 1]

        self.assertEqual(funnel.loc["LONG", "all_confirmation_gates"], int(eligible["long_confirmation"].sum()))
        self.assertEqual(funnel.loc["SHORT", "all_confirmation_gates"], int(eligible["short_confirmation"].sum()))

    def test_v091_runner_marks_overlapping_fixed_horizon_candidates(self):
        frame = pd.concat([_base_frame()] * 5, ignore_index=True)
        frame["volume"] = 100.0
        frame["time"] = pd.date_range("2025-01-01", periods=len(frame), freq="15min")
        context = create_context_features(create_features(frame))
        for index in (250, 251):
            context.loc[index, "v091_context_signal"] = "LONG"
            context.loc[index, "v091_source_sweep_time"] = context.loc[index - 1, "time"]
            context.loc[index, "v091_confirmation_lag"] = 1
            context.loc[index, "bullish_bos"] = 1

        import sqlite3
        with sqlite3.connect(":memory:") as connection:
            count = build_v091_observations(context, connection)
            overlap_flags = [row[0] for row in connection.execute(
                "SELECT non_overlapping FROM v091_context_observations ORDER BY timestamp"
            )]

        self.assertEqual(count, 2)
        self.assertEqual(overlap_flags, [1, 0])

    def test_break_even_stop_applies_on_the_bar_after_one_r_is_reached(self):
        frame = pd.DataFrame({"high": [101.1, 100.2], "low": [100.0, 99.8], "close": [100.8, 100.0], "atr": [1.0, 1.0]})
        result = evaluate_exit_model(frame, -1, "LONG", 100.0, 1.0, 2, ExitModel("break_even_at_1r", break_even_at_r=1.0))
        self.assertEqual(result["outcome_r"], 0.0)
        self.assertEqual(result["exit_reason"], "stop")

    def test_fixed_target_marks_three_r_when_reached_without_stop(self):
        frame = pd.DataFrame({"high": [103.1], "low": [100.1], "close": [103.0], "atr": [1.0]})
        result = evaluate_exit_model(frame, -1, "LONG", 100.0, 1.0, 1, ExitModel("fixed_3r"))
        self.assertEqual(result["outcome_r"], 3.0)

    def test_summary_drawdown_is_reported_in_r(self):
        observations = pd.DataFrame({"timestamp": ["1", "2", "3"], "model": ["fixed_3r"] * 3, "outcome_r": [1.0, -2.0, 1.0]})
        summary = summarize_models(observations).iloc[0]
        self.assertEqual(summary["trades"], 3)
        self.assertEqual(summary["max_drawdown_r"], -2.0)

    def test_costs_convert_to_r_and_reject_invalid_multiplier(self):
        costs = TradeCostConfig(spread_price=0.2, slippage_price_per_side=0.1, commission_per_contract_per_side=1.0, contract_multiplier=10.0)
        self.assertAlmostEqual(cost_in_r(2.0, costs), 0.3)
        with self.assertRaisesRegex(ValueError, "multiplier"):
            cost_in_r(2.0, TradeCostConfig(contract_multiplier=0))

    def test_one_position_rule_skips_overlapping_v08_candidates(self):
        frame = pd.concat([_base_frame()] * 4, ignore_index=True)
        frame["volume"] = 100.0
        frame["time"] = pd.date_range("2025-01-01", periods=len(frame), freq="15min")
        features = create_features(frame)
        # Force a valid v0.8 long setup on adjacent eligible bars.
        for index in (250, 251):
            if index < len(features):
                features.loc[index, ["trend", "momentum_atr", "delta_zscore", "bullish_sweep"]] = [1, 1, 2, 1]
        observations = build_exit_observations(features)
        self.assertTrue((observations.groupby("model")["timestamp"].count() <= 1).all())

    def test_invalid_ohlc_is_rejected(self):
        invalid = pd.DataFrame({"time": pd.to_datetime(["2025-01-01"]), "open": [10.0], "high": [9.0], "low": [8.0], "close": [10.0], "volume": [1.0]})
        with self.assertRaisesRegex(ValueError, "Invalid OHLC"):
            validate_market_data(invalid)

    def test_custom_data_file_is_loaded_and_normalized_to_utc_naive(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            data_file = Path(directory) / "custom.csv"
            pd.DataFrame({
                "time": ["2026-01-01 00:00:00+0000"],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [10.0],
            }).to_csv(data_file, index=False)

            loaded = load_market_data(data_file)

        self.assertEqual(loaded.loc[0, "time"], pd.Timestamp("2026-01-01 00:00:00"))
        self.assertIsNone(loaded.loc[0, "time"].tzinfo)

    def test_frozen_score_requires_v05_thresholds_and_includes_sweep(self):
        rows = pd.DataFrame({
            "direction": ["LONG", "LONG", "SHORT"],
            "delta_zscore": [1.0, 0.99, -1.0],
            "delta_change": [1.0, 1.0, -1.0],
            "momentum_atr": [1.0, 1.0, -1.0],
            "candle_efficiency": [0.60, 0.59, 0.60],
            "bullish_sweep": [1, 1, 0],
            "bearish_sweep": [0, 0, 1],
            "relative_volume": [1.5, 1.5, 1.5],
        })

        scored = add_frozen_order_flow_score(rows)

        self.assertEqual(scored["order_flow_score"].tolist(), [7, 3, 7])
