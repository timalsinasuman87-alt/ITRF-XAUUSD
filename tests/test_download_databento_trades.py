import unittest

import pandas as pd

from research.download_databento_trades import (
    aggregate_trade_frame,
    combine_trade_aggregates,
    validate_orderflow_bars,
)


class DatabentoTradeAggregationTests(unittest.TestCase):
    def _frame(self, rows):
        return pd.DataFrame(rows)

    def test_aggregates_aggressor_flow_without_using_unknown_side(self):
        frame = self._frame(
            [
                {"ts_event": "2026-07-01T00:00:01Z", "price": 2400.0, "size": 2, "side": "B"},
                {"ts_event": "2026-07-01T00:00:02Z", "price": 2400.1, "size": 1, "side": "A"},
                {"ts_event": "2026-07-01T00:00:03Z", "price": 2400.2, "size": 3, "side": "N"},
            ]
        )
        bars = combine_trade_aggregates([aggregate_trade_frame(frame)])
        bar = bars.iloc[0]
        self.assertEqual(bar["trade_count"], 3)
        self.assertEqual(bar["total_volume"], 6)
        self.assertEqual(bar["buy_volume"], 2)
        self.assertEqual(bar["sell_volume"], 1)
        self.assertEqual(bar["unknown_volume"], 3)
        self.assertEqual(bar["volume_delta"], 1)
        self.assertAlmostEqual(bar["signed_volume_ratio"], 1 / 3)
        self.assertAlmostEqual(bar["aggressor_coverage"], 2 / 3)

    def test_combines_a_bar_split_across_download_chunks(self):
        first = self._frame(
            [
                {"ts_event": "2026-07-01T00:00:01Z", "price": 2400.0, "size": 2, "side": "B"},
                {"ts_event": "2026-07-01T00:14:00Z", "price": 2401.0, "size": 1, "side": "B"},
            ]
        )
        second = self._frame(
            [
                {"ts_event": "2026-07-01T00:14:30Z", "price": 2399.0, "size": 4, "side": "A"},
                {"ts_event": "2026-07-01T00:15:00Z", "price": 2398.0, "size": 1, "side": "A"},
            ]
        )
        bars = combine_trade_aggregates(
            [aggregate_trade_frame(first), aggregate_trade_frame(second)]
        )
        self.assertEqual(len(bars), 2)
        first_bar = bars.iloc[0]
        self.assertEqual(first_bar["first_trade_price"], 2400.0)
        self.assertEqual(first_bar["last_trade_price"], 2399.0)
        self.assertEqual(first_bar["high_trade_price"], 2401.0)
        self.assertEqual(first_bar["low_trade_price"], 2399.0)
        self.assertEqual(first_bar["trade_count"], 3)
        self.assertEqual(first_bar["volume_delta"], -1)

    def test_uses_datetime_index_when_ts_event_is_not_a_column(self):
        frame = pd.DataFrame(
            [{"price": 2400.0, "size": 1, "side": "B"}],
            index=pd.DatetimeIndex(["2026-07-01T00:00:01Z"], name="ts_event"),
        )
        bars = combine_trade_aggregates([aggregate_trade_frame(frame)])
        self.assertEqual(bars.iloc[0]["buy_trades"], 1)

    def test_validator_rejects_an_incorrect_processed_count(self):
        frame = self._frame(
            [{"ts_event": "2026-07-01T00:00:01Z", "price": 2400.0, "size": 1, "side": "B"}]
        )
        bars = combine_trade_aggregates([aggregate_trade_frame(frame)])
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_orderflow_bars(bars, processed_records=2)


if __name__ == "__main__":
    unittest.main()

