# Databento GC validation results

## Status

Research checkpoint only. These results do not demonstrate profitability and
do not authorize paper or live deployment.

## Data audit

- 37,756 15-minute bars from 2025-01-01 23:00 UTC through 2026-08-07 20:45 UTC
- Unique, chronological timestamps; valid OHLC relationships; no zero-volume bars
- 10 contiguous official contract-mapping intervals
- Nine visible unadjusted mapping-transition gaps, approximately 0.68% to 1.76%
- 537 provider condition records and 10 degraded dates; eight degraded dates contain bars
- 426 gaps longer than 15 minutes, consistent with session and holiday closures

The primary study retained the delivered data. No roll adjustment, degraded-day
deletion, or post-result cleaning was used.

## Frozen v0.8 labels

The unchanged v0.8 feature and forward-label engine created 3,221 setup
observations. The pre-existing OOS boundary was 2025-07-28 17:00 UTC.

| Period | Samples | Average R |
| --- | ---: | ---: |
| Training | 1,170 | +0.045 |
| OOS | 2,051 | +0.048 |
| Latest OOS quarter | 513 | -0.168 |

The frozen Score 5–7 OOS subset contained 1,292 observations with average
+0.113R and an IID-bootstrap 95% interval of +0.025R to +0.204R. Its latest
chronological quarter averaged -0.124R with an interval of -0.288R to +0.043R.
The deterioration in the newest period prevents interpreting the aggregate as
a stable edge. These labels also overlap and do not represent an executable
one-position portfolio.

During validation, the v0.8 robustness report was found to reconstruct the
v0.5 score incorrectly: it had loosened the delta and efficiency thresholds and
omitted the sweep component. The report now calls one tested canonical v0.5
definition. Entry candidates and stored forward outcomes were not changed. A
regression test locks the corrected definition.

## Data-quality exposure sensitivity

This is descriptive and changes no rule. It flags a setup when its 32-bar
forward window touches a roll transition or degraded provider date.

| Period | Quality group | Samples | Average R |
| --- | --- | ---: | ---: |
| Training | Unexposed | 1,163 | +0.041 |
| Training | Roll/degraded exposed | 7 | +0.714 |
| OOS | Unexposed | 1,991 | +0.046 |
| OOS | Roll/degraded exposed | 60 | +0.133 |

The small exposed group is not the source of the broad OOS average. The
unadjusted continuous series remains unsuitable for treating individual roll
gaps as genuine market moves.

## Frozen v0.9 context study

The strict same-bar context definition produced zero candidates. The gate
funnel explains why:

| Side | Eligible bars | Sweep | Location | Structure | Displacement | Sweep + location | Plus same-bar structure | All gates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Long | 37,473 | 1,907 | 16,856 | 7,411 | 3,349 | 1,827 | 0 | 0 |
| Short | 37,473 | 2,370 | 20,283 | 5,467 | 3,122 | 2,263 | 0 | 0 |

The null result is an architecture diagnosis, not a reason to loosen thresholds
on this dataset. A future version should pre-register a causal state machine in
which a liquidity sweep arms a setup and a later bar confirms structure. Its
confirmation horizon must be fixed before testing on a reserved unseen period.

## Frozen v0.9 trade management

The following OOS results use one position at a time for each model and zero
costs. They are gross upper bounds.

| Exit model | OOS trades | Average R | Max drawdown (R) |
| --- | ---: | ---: | ---: |
| Fixed 3R | 848 | -0.008 | -49.500 |
| Partial 2R + ATR trail | 1,252 | +0.032 | -19.187 |
| Break-even at 1R | 927 | +0.006 | -36.151 |
| ATR trailing stop | 1,252 | +0.036 | -19.451 |

The best gross average is only +0.036R per trade. Any all-in execution cost
above that removes the mean before considering uncertainty, so broker-specific
commission, slippage, contract multiplier, and fill assumptions are required
before this module can advance.

## Decision

No deployable edge is validated. Preserve v0.8 as the baseline, retain the
Databento run as independent evidence, and advance v0.9 through a pre-registered
multi-bar event sequence rather than parameter optimisation.
