# 2026-08-16 Local Validation Checkpoint

## Scope

This is a descriptive, reproducible research checkpoint. It is not a
profitability claim, a recommendation, or permission to trade the strategy.
No parameters were tuned after viewing these results.

## Dataset integrity

The local `data/XAUUSD.csv` file was built from HistData XAU/USD tick archives
using midpoint bid/ask OHLC and tick count per 15-minute bar as an activity
proxy. It contains 6,466 bars from 2026-05-01 00:00 through 2026-08-07 16:45.

| Check | Result |
| --- | ---: |
| Missing required columns | 0 |
| Invalid timestamps | 0 |
| Duplicate timestamps | 0 |
| Non-positive price bars | 0 |
| Invalid OHLC bars | 0 |
| Bars off the 15-minute grid | 0 |

The file is structurally valid for this research engine. Tick count is not
broker volume, and this data feed is not interchangeable with TradingView or a
live broker feed.

## Automated checks

All 17 automated tests passed. The suite covers the two data importers, OOS
split protection, and v0.9 market-context calculations.

The engine was run with a chronological split at 2026-06-20. The runner now
requires the requested split to fall inside the data range; it skips a split
report rather than mislabelling all observations if that condition is not met.

## Entry and context results

| Measure | Earlier period | Later period |
| --- | ---: | ---: |
| Baseline observations | 199 | 185 |
| Baseline average R | -0.161 | -0.200 |
| Long average R | — | -0.187 |
| Short average R | — | -0.213 |
| Score 5–7 average R | — | -0.128 |

The score 5–7 later-period bootstrap 95% interval was -0.407R to +0.120R, so
it crosses zero. The stricter v0.9 context definition generated zero candidates
on this dataset. Neither result supports selecting an entry rule.

## Fixed exit-model results

All four pre-registered exit models were checked with one position at a time
and zero configured costs. The two trailing variants had slightly positive
later-period average R, but that is not an edge: the numbers are small,
pre-cost, drawn from a short single-source sample, and accompanied by material
sequential drawdown.

| Exit model | Earlier average R | Later average R | Later max drawdown (R) |
| --- | ---: | ---: | ---: |
| Fixed 3R | -0.014 | -0.267 | -33.764 |
| Partial 2R + ATR trail | +0.026 | +0.020 | -8.807 |
| Break-even at 1R | +0.013 | -0.187 | -27.448 |
| ATR trailing stop | +0.031 | +0.026 | -7.908 |

## Decision

No entry or exit model is promoted. The next valid research step is to acquire
a longer independent history, set the split before viewing its results, include
realistic broker costs, and retest only pre-declared hypotheses. A larger
HistData download was attempted but the public source was temporarily
unreachable; the existing local data is unaffected.
