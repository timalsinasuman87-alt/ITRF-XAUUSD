from pathlib import Path
import sqlite3
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from run_v092_orderflow_research import (
    build_v092_observations,
    classify_orderflow_gate,
    merge_orderflow,
)


class V092OrderFlowTests(unittest.TestCase):
    def test_gate_uses_only_directional_delta_sign(self):
        frame = pd.DataFrame(
            {
                "v091_context_signal": ["LONG", "LONG", "SHORT", "SHORT", "NONE"],
                "volume_delta": [1.0, -1.0, -1.0, 1.0, 1.0],
                "aggressor_coverage": [0.9] * 5,
            }
        )
        result = classify_orderflow_gate(frame)
        self.assertEqual(result["v092_flow_pass"].tolist(), [1, 0, 1, 0, 0])
        self.assertEqual(result["v092_signal"].tolist(), ["LONG", "NONE", "SHORT", "NONE", "NONE"])

    def test_missing_flow_and_zero_delta_fail_closed(self):
        frame = pd.DataFrame(
            {
                "v091_context_signal": ["LONG", "SHORT", "LONG"],
                "volume_delta": [None, 0.0, 1.0],
                "aggressor_coverage": [None, 0.9, 0.0],
            }
        )
        result = classify_orderflow_gate(frame)
        self.assertEqual(result["v092_flow_pass"].tolist(), [0, 0, 0])
        self.assertEqual(
            result["v092_flow_reason"].tolist(),
            ["FLOW_UNAVAILABLE", "ZERO_DELTA", "FLOW_UNAVAILABLE"],
        )

    def test_merge_requires_an_exact_bar_timestamp(self):
        context = pd.DataFrame(
            {
                "time": pd.to_datetime(["2026-07-01 00:00", "2026-07-01 00:15"]),
                "v091_context_signal": ["LONG", "LONG"],
            }
        )
        flow = pd.DataFrame(
            {
                "time": pd.to_datetime(["2026-07-01 00:00", "2026-07-01 00:16"]),
                "trade_count": [10, 20],
                "total_volume": [10, 20],
                "buy_volume": [6, 11],
                "sell_volume": [4, 9],
                "unknown_volume": [0, 0],
                "volume_delta": [2, 2],
                "signed_volume_ratio": [0.2, 0.1],
                "aggressor_coverage": [1.0, 1.0],
            }
        )
        merged = merge_orderflow(context, flow)
        self.assertEqual(merged["v092_flow_available"].tolist(), [1, 0])
        self.assertEqual(merged["v092_flow_pass"].tolist(), [1, 0])

    def test_flow_non_overlap_is_calculated_on_accepted_signals(self):
        rows = 400
        frame = pd.DataFrame(
            {
                "time": pd.date_range("2026-01-01", periods=rows, freq="15min"),
                "open": [100.0] * rows,
                "high": [101.0] * rows,
                "low": [99.0] * rows,
                "close": [100.0] * rows,
                "atr": [1.0] * rows,
                "v091_context_signal": ["NONE"] * rows,
                "v091_source_sweep_time": pd.date_range("2025-12-31 23:45", periods=rows, freq="15min"),
                "v091_confirmation_lag": [1] * rows,
                "bullish_choch": [0] * rows,
                "bearish_choch": [0] * rows,
                "market_regime": ["RANGE"] * rows,
                "trade_count": [10] * rows,
                "total_volume": [10] * rows,
                "volume_delta": [1.0] * rows,
                "signed_volume_ratio": [0.1] * rows,
                "aggressor_coverage": [1.0] * rows,
                "v092_flow_reason": ["NO_V091_SIGNAL"] * rows,
                "v092_flow_pass": [0] * rows,
            }
        )
        for index, passed in [(250, 1), (251, 0), (284, 1)]:
            frame.loc[index, "v091_context_signal"] = "LONG"
            frame.loc[index, "v092_flow_pass"] = passed
            frame.loc[index, "v092_flow_reason"] = "PASS" if passed else "OPPOSITE_DELTA"
        with sqlite3.connect(":memory:") as connection:
            build_v092_observations(frame, connection)
            stored = connection.execute(
                "SELECT flow_pass, base_non_overlapping, flow_non_overlapping "
                "FROM v092_orderflow_observations ORDER BY timestamp"
            ).fetchall()
        self.assertEqual(stored, [(1, 1, 1), (0, 0, 0), (1, 1, 1)])


if __name__ == "__main__":
    unittest.main()

