# Databento CME Gold Research Feed

Databento is used as a separate exchange-traded futures research feed. It does
not replace or masquerade as the OTC XAU/USD spot/CFD feed.

The importer requests `GC.v.0` from `GLBX.MDP3` with continuous symbology and
the one-minute OHLCV schema, then resamples to 15-minute bars in UTC. Databento
continuous prices are original and unadjusted, so rollover gaps are retained.
Contract changes are saved beside the bar file for audit.

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

The output defaults to `data/databento/GC_front_15m.csv`. It remains separate
from `data/XAUUSD.csv`; cross-feed research must not combine their volume or
price series as if they represented the same instrument.
