# StatStock

A personal project for pulling stock data from Yahoo Finance, storing it in MongoDB, and visualising it in Power BI.

## What it does

Fetches three types of data for ~100 S&P 500 tickers and stores them in a local MongoDB database:

- **Fundamentals** — company name, sector, market cap, beta
- **Quarterly financials** — revenue, margins, EPS, debt, free cash flow, etc.
- **Daily price history** — OHLCV data going back to 2020, plus valuation ratios (P/E, P/B, P/S)

There's also a correlation module that calculates pairwise daily return correlations across all tickers and exports the result as a long-format CSV for use in Power BI.

## Project structure

```
StatStock/
├── loadData/
│   ├── db.py               # MongoDB connection
│   ├── load_yfinance.py    # All data fetching logic
│   ├── mongo_helpers.py    # Upsert and merge helpers
│   └── tickers.py          # Ticker lists
├── calculations/
│   └── correlation.py      # Return correlation matrix + CSV export
├── main.py                 # Entry point
└── requirements.txt
```

## Setup

**Requirements:** Python 3.11+, MongoDB running locally on port 27017

```bash
pip install -r requirements.txt
```

Then just run:

```bash
python main.py
```

## Update modes

You don't have to re-fetch everything every time. Set `UPDATE_MODE` in `main.py`:

| Mode | What it updates |
|------|----------------|
| 0 | Everything |
| 1 | Fundamentals only |
| 2 | Quarterly financials only |
| 3 | Price history only (incremental) |

Price history is fetched incrementally by default — it checks the latest date in the database and only pulls what's missing.

## Stack

- `yfinance` for data
- `pymongo` for storage
- `pandas` for data wrangling
- MongoDB as the database
- Power BI for visualisation