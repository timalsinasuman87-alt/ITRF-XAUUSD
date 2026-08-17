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

Version 0.9.1 implemented that frozen sequence with an eight-bar confirmation
window and matching Python/Pine state transitions. It produced workable sample
counts, but both directions were negative on Databento and every chronological
Databento development quarter was negative. A small long/uptrend cohort was
positive in aggregate across two development feeds but failed in the newest
Databento quarter, so it is not promoted. The next research module must add a
genuinely different information source rather than tune this state machine.

Version 0.9.2 added that new information through historical CME Gold trade
aggressor flow while retaining XAU/USD as the research instrument. The frozen
confirmation-bar delta-sign gate produced 13 non-overlapping accepted
development observations with a -0.154R gross average and failed. More of the
same history will not be purchased to rescue it. Any next test must use a new,
pre-registered economic mechanism, such as flow across the complete
sweep-to-confirmation sequence, and the later untouched holdout requirement
remains unchanged.

Version 0.9.3 tested that complete sequence mechanism using the already
downloaded file at no additional data cost. Fifteen accepted non-overlapping
observations averaged 0.000R gross and became negative under both frozen cost
burdens; the chronological quarters were unstable. It is not promoted. The
project now has evidence that simple directional delta signs, whether measured
on one bar or across the causal sequence, are insufficient. The next step
should be an explicit research-design review before another paid download or
feature test, not another threshold search.

Version 0.10 then replaced overlapping forward labels with a clean event core:
next-open entry, one position at a time, explicit stop/target/timeout events,
same-bar ambiguity bounds, cost sensitivity, and right-censor protection. The
212 accepted sequential events averaged -0.179R gross with a 0.758 R profit
factor, so the v0.8 signal remained negative under the corrected architecture.

Version 0.10.1 used the clean events in three purged, expanding walk-forward
folds. The fixed expanded causal-context logistic model achieved a 0.643 pooled
ROC AUC, but its 0.1214 Brier score was worse than the training-frequency null's
0.1169. Its paired Brier improvement was -0.00445 with a block-bootstrap 95%
interval of [-0.02205, 0.01442], and it degraded in two of three folds. The
context family is not promoted. The next increment is failure anatomy and event
lifecycle measurement, not a probability-threshold or feature-subset search.

Version 0.10.2 expanded all 212 clean events into a conserved 2,601-row
lifecycle panel and measured competing stop, target and timeout events. Stops
were 59 of 65 terminals in bars 1-4 and 125 of 145 terminals through bar 16.
However, 79 of 155 eventual stops first reached +0.5R and 44 first reached +1R
on fully observed pre-terminal bars. The failure set therefore mixes early
adverse selection with favorable-excursion giveback. Twenty-two of 25 timeouts
were positive, but timeouts were too rare to explain the dominant 73.1% stop
incidence. No exit or time rule is promoted. A future increment may build a
calibration-only competing-risk framework against empirical holding-bar
hazards, followed by untouched chronological validation.

Version 0.10.3 implemented that causal competing-risk framework. It generated
2,601 start-of-bar risk rows, with dynamic features shifted so current-bar OHLC
could not predict its own stop or target. Across 1,594 purged walk-forward test
rows, the dynamic model reduced pooled log loss from 0.29931 to 0.24174 and
event-balanced Brier error from 0.42507 to 0.41963 versus empirical holding-bar
hazards. However, its +0.00544 mean event-balanced improvement had a wide
block-bootstrap 95% interval of [-0.01689, 0.02571] and was negative in two of
three folds. Path state is promising but not validated, and no management rule
is promoted. The frozen pipeline now requires a genuinely untouched
chronological XAU/USD period rather than further tuning on this sample.
