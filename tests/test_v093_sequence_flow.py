from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from run_v093_sequence_flow_research import add_sequence_flow


class V093SequenceFlowTests(unittest.TestCase):
    def _frame(self):
        times = pd.date_range("2026-07-01", periods=5, freq="15min")
        return pd.DataFrame({
            "time": times,
            "v091_context_signal": ["NONE", "NONE", "LONG", "NONE", "SHORT"],
            "v091_source_sweep_time": [pd.NaT, pd.NaT, times[0], pd.NaT, times[3]],
            "v092_flow_available": [1, 1, 1, 1, 1],
            "buy_volume": [2, 3, 1, 2, 1],
            "sell_volume": [1, 1, 3, 1, 4],
            "unknown_volume": [0, 0, 0, 0, 0],
            "aggressor_coverage": [1.0] * 5,
        })

    def test_sequence_uses_inclusive_sweep_to_confirmation_flow(self):
        result = add_sequence_flow(self._frame())
        self.assertEqual(result.loc[2, "v093_sequence_bars"], 3)
        self.assertEqual(result.loc[2, "v093_sequence_delta"], 1)
        self.assertEqual(result.loc[2, "v093_flow_pass"], 1)
        self.assertEqual(result.loc[4, "v093_sequence_delta"], -2)
        self.assertEqual(result.loc[4, "v093_flow_pass"], 1)

    def test_any_missing_bar_fails_the_complete_sequence(self):
        frame = self._frame()
        frame.loc[1, "v092_flow_available"] = 0
        result = add_sequence_flow(frame)
        self.assertEqual(result.loc[2, "v093_flow_pass"], 0)
        self.assertEqual(result.loc[2, "v093_flow_reason"], "INCOMPLETE_SEQUENCE")

    def test_opposite_cumulative_delta_fails(self):
        frame = self._frame()
        frame.loc[:2, "buy_volume"] = 0
        result = add_sequence_flow(frame)
        self.assertEqual(result.loc[2, "v093_flow_reason"], "OPPOSITE_DELTA")


if __name__ == "__main__":
    unittest.main()

