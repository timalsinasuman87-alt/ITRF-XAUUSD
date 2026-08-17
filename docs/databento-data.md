# Databento CME Gold Research Feed

Databento is used as a separate exchange-traded futures research feed. It does
not replace or masquerade as the OTC XAU/USD spot/CFD feed.

The importer requests `GC.v.0` from `GLBX.MDP3` with continuous symbology and
the one-minute OHLCV schema, then resamples to 15-minute bars in UTC. Databento
continuous prices are original and unadjusted, so rollover gaps are retained.
Official continuous-symbol instrument-ID intervals and provider quality
conditions are saved beside the bar file for audit.

## Local API key

The importer first checks `DATABENTO_API_KEY`. If the desktop environment does
not expose it, place the key alone in `.secrets/databento_api_key` and restrict
that file to the current user. The complete `.secrets/` folder is ignored by
Git and must never be committed.

## Estimate before downloading

This command performs metadata calls only and prints the provider's estimated
cost and record count:

```bash
python research/download_databento.py \
  --start 2025-01-01 --end 2026-01-01
```

No market data is requested without `--download`. A download also requires an
explicit maximum cost:

```bash
python research/download_databento.py \
  --start 2025-01-01 --end 2026-01-01 \
  --download --max-cost-usd 5.00
```

Use `--audit-only` to refresh the free continuous-contract mapping and provider
quality-condition files without downloading price data.

The output defaults to `data/databento/GC_front_15m.csv`. It remains separate
from `data/XAUUSD.csv`; cross-feed research must not combine their volume or
price series as if they represented the same instrument.

Before running research, use `research/validate_market_data.py` with the bar,
roll, and condition files. The frozen experiment and interpretation rules are
documented in `docs/databento-validation-plan.md`.

## v0.9.2 trade-level order flow

The v0.9.2 downloader requests the separate Databento `trades` schema and
retains the compressed raw DBN file for audit. It also creates compact
15-minute features containing actual aggressor buy/sell volume, volume delta,
aggressor-side coverage, trade intensity, and trade-price VWAP.

Always estimate first:

```bash
python research/download_databento_trades.py \
  --start 2026-05-01 --end 2026-08-08
```

The approved bounded download uses a hard ceiling slightly above the provider
estimate:

```bash
python research/download_databento_trades.py \
  --start 2026-05-01 --end 2026-08-08 \
  --download --max-cost-usd 7.00
```

The default local outputs are `data/databento/GC_front_trades.dbn.zst` and
`data/databento/GC_front_orderflow_15m.csv`. Both remain ignored by Git. The
frozen feature and evaluation definitions are in `docs/v0.9.2-hypothesis.md`.
