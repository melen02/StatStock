"""Shared read access to stored prices, so every analysis sees the same universe."""
import pandas as pd

from loadData.tickers import TICKERS

# A one-day move this large is far more often an unadjusted split than real news.
SUSPICIOUS_DAILY_MOVE = 0.50


def find_suspicious_moves(history, threshold=SUSPICIOUS_DAILY_MOVE):
    """Return rows whose daily return looks like a split that was never adjusted."""
    returns = history.groupby("Ticker")["AdjClose"].pct_change(fill_method=None)
    return history.assign(Return=returns)[returns.abs() > threshold][
        ["Ticker", "Date", "AdjClose", "Return"]
    ]


def load_history(db):
    """Load prices for the configured universe, sorted by ticker and date.

    Other tickers in the database are ignored on purpose: mixing exchanges brings
    in other currencies and trading calendars.
    """
    cursor = db["Stock_History"].find(
        {"Ticker": {"$in": TICKERS}},
        {"_id": 0, "Ticker": 1, "Date": 1, "Close": 1, "Adj Close": 1},
    )
    history = pd.DataFrame(list(cursor))
    if history.empty:
        raise RuntimeError("Stock_History is empty. Run: python main.py fetch")

    history = history.rename(columns={"Adj Close": "AdjClose"})
    history["Date"] = pd.to_datetime(history["Date"])
    history = history.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    suspicious = find_suspicious_moves(history)
    if not suspicious.empty:
        print(
            f"WARNING: daily moves above {SUSPICIOUS_DAILY_MOVE:.0%} may be unadjusted splits. "
            "Run 'python main.py fetch --mode 3 --full' to rebuild prices:"
        )
        print(suspicious.to_string(index=False))

    return history


def wide_prices(history, column="AdjClose"):
    """Pivot one price column to a dates-by-tickers table."""
    return history.pivot(index="Date", columns="Ticker", values=column).sort_index()
