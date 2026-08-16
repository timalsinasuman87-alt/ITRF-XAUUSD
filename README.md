
This project is for researching real historical XAU/USD price data. It does not generate market data or trading probabilities without evidence.

## Folders

- `data/` — put your real XAU/USD CSV here (not included in Git).
- `research/` — the Python research program.
- `database/` — the SQLite research database generated from your data (not included in Git).
- `tests/` — future automated checks.
- `research/itrf_context.py` — v0.9 causal market-context definitions, kept separate from the validated v0.8 baseline.
- `tradingview/` — Pine visual parity references; these are not automated strategies.

## Research versions

`main` preserves the validated v0.8 baseline. The v0.9 contextual framework is
developed on a feature branch and is deliberately research-only: it adds market
regime, structure, liquidity, location, confirmation and position-sizing
definitions without using them to claim profitability. See
[`docs/v0.9-hypotheses.md`](docs/v0.9-hypotheses.md) for the frozen hypotheses,
Pine parity, limitations and pre-registered next validation step.

With real data in `data/XAUUSD.csv`, run the isolated v0.9 study with:

```bash
python research/run_v09_research.py
```

It writes `database/itrf_v09_research.db`, separately from the v0.8 database.

The pre-registered v0.9 exit-model comparison is separate as well:

```bash
python research/run_v09_trade_management.py
```

It compares four fixed management models over the existing v0.8 entry set and
writes `database/itrf_v09_trade_management.db`. It is not parameter tuning and
does not establish a profitable or executable strategy.

Before interpreting an exit comparison, pass your broker's actual cost inputs.
For example (values shown are placeholders, not recommendations):

```bash
python research/run_v09_trade_management.py \
  --spread-price 0.00 --slippage-price-per-side 0.00 \
  --commission-per-contract-per-side 0.00 --contract-multiplier 1
```

## Integrated research commands

The default main-engine command remains the validated v0.8 workflow:

```bash
python research/itrf_research.py
```

Run either v0.9 study after that baseline with `--v09-context` or
`--v09-trade-management`; use `--v09-all` for both. Each v0.9 study maintains
its own database and does not change v0.8 tables or reports.

## TradingView

Paste [`tradingview/ITRF_XAUUSD_v09_research_strategy.pine`](tradingview/ITRF_XAUUSD_v09_research_strategy.pine)
into a new Pine Editor strategy on an XAUUSD 15-minute chart. It translates the
v0.8 entry score and lets you select one fixed v0.9 exit model. Configure the
broker's costs and fill settings in TradingView Strategy Properties; compare its
output against the Python research, not as evidence that the strategy is ready
for live trading.

## First run

From this folder in Terminal:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python research/itrf_research.py
```

The final command deliberately stops with clear instructions until you add a real CSV file as `data/XAUUSD.csv`.

### Free-data option

If you do not have a broker or TradingView CSV export, the project can download
and aggregate public Dukascopy XAU/USD ticks into its required 15-minute CSV.
For example, the following creates `data/XAUUSD.csv` in UTC:

```bash
python research/download_dukascopy.py --start 2026-05-01 --end 2026-08-15
```

The downloader is reproducible and refuses to overwrite an existing data file
unless `--overwrite` is supplied. It is research data only: its tick-side
volume and fills will not match a broker's CFD feed, so results must not be
treated as execution or profitability evidence.

## Data format

The CSV must contain these columns (capitalization does not matter):

`time, open, high, low, close, volume`

Do not use fabricated, random, or demo data for research results.
=======
# ITRF-XAUUSD
Institutional Trading Research Framework for XAU/USD
>>>>>>> 0e162c6ae61f28e876bfa0b2a463b9d47b5bf319
