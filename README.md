
This project is for researching real historical XAU/USD price data. It does not generate market data or trading probabilities without evidence.

## Folders

- `data/` — put your real XAU/USD CSV here (not included in Git).
- `research/` — the Python research program.
- `database/` — the SQLite research database generated from your data (not included in Git).
- `tests/` — future automated checks.

## First run

From this folder in Terminal:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python research/itrf_research.py
```

The final command deliberately stops with clear instructions until you add a real CSV file as `data/XAUUSD.csv`.

## Data format

The CSV must contain these columns (capitalization does not matter):

`time, open, high, low, close, volume`

Do not use fabricated, random, or demo data for research results.
=======
# ITRF-XAUUSD
Institutional Trading Research Framework for XAU/USD
>>>>>>> 0e162c6ae61f28e876bfa0b2a463b9d47b5bf319
