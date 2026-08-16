# ITRF-XAUUSD Quantitative Research Roadmap

## Objective

Develop a research-grade XAU/USD strategy through reproducible data, explicit
definitions, chronological validation, and realistic execution assumptions.
The objective is not to maximise a short backtest or claim profitability from
in-sample results.

## Development sequence

1. Preserve the validated v0.8 baseline as the comparison point.
2. Define and test market regime, structure (BOS/CHoCH), liquidity sweeps,
   premium/discount location, and entry confirmation independently.
3. Acquire longer, independent historical data and record source limitations.
4. Pre-declare a small number of hypotheses and test them chronologically,
   without changing parameters after seeing later-period outcomes.
5. Include realistic spread, slippage, commission, contract multiplier, and
   one-position-at-a-time assumptions before evaluating trade management.
6. Retain only rules that are stable across independent periods and data
   sources; discard or quarantine rules that fail.
7. Keep the overlapping Python and Pine definitions aligned for chart-level
   verification.
8. Consider paper trading only after the research evidence is robust; no live
   deployment follows from a backtest alone.

## Current checkpoint

The v0.8 baseline and v0.9 research architecture are retained. The short
HistData sample did not validate a general v0.8 entry edge. A longer Databento
CME Gold futures check produced a slightly positive aggregate OOS forward-label
average before costs, but the newest chronological quarter was negative and
the executable exit-model margins were small. This is not a validated edge.

The strict v0.9 same-bar context conjunction produced zero candidates because
a sweep-plus-location bar never also produced the required structure break.
The next increment is a pre-registered causal event sequence: map liquidity,
arm a setup after a sweep, then wait for later BOS/CHoCH and entry confirmation.
The time horizon must be frozen before an unseen validation period is reserved;
it must not be selected from the Databento result.
