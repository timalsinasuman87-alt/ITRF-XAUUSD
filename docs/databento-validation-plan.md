# Databento GC validation plan

## Purpose

This experiment tests whether the frozen ITRF definitions behave consistently
on a regulated CME Gold futures feed. It does not replace the preserved XAUUSD
spot/CFD baseline, and it is not a parameter search.

## Pre-registered inputs

- Dataset: Databento `GLBX.MDP3`
- Symbol: unadjusted continuous front contract `GC.v.0`
- Schema: one-minute OHLCV resampled to 15-minute UTC bars
- Available range: 2025-01-01 through 2026-08-07
- Frozen OOS boundary: 2025-07-28 17:00 UTC
- Entry, feature, forward-label, and v0.9 context definitions: unchanged
- Primary first pass: zero execution costs, interpreted only as an upper bound

The OOS boundary predates this download and must not be moved after reviewing
results. No parameter or direction may be selected from this one feed.

## Data-quality protocol

Run the structural audit before the engine:

```bash
python research/validate_market_data.py \
  --data-file data/databento/GC_front_15m.csv \
  --roll-file data/databento/GC_front_15m_rolls.csv \
  --condition-file data/databento/GC_front_15m_conditions.csv
```

The primary run retains the provider-delivered unadjusted prices. Roll mapping
transitions and degraded dates are reported, not silently repaired. Any later
exclusion sensitivity must be labelled separately and must not redefine the
primary result.

## Interpretation rules

- Treat all results as historical research, not evidence of profitability.
- Do not optimize thresholds, lookbacks, direction, or exit parameters here.
- Require adequate sample size and chronological stability before advancing.
- Apply realistic commission, slippage, spread, and contract sizing only after
  broker/execution assumptions are sourced and frozen.
- A positive in-sample or OOS result on this dataset alone is insufficient for
  deployment; independent walk-forward and cross-feed confirmation are still
  required.
