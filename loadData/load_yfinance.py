from datetime import datetime

import pandas as pd
import yfinance as yf

from loadData.db import get_database
from loadData.mongo_helpers import merge_quarterly_rows, upsert_collection
from loadData.tickers import TICKERS

HISTORY_START = datetime(2020, 1, 1)

# Incremental fetches re-read this many days so they can be compared with stored prices.
OVERLAP_DAYS = 7
# Yahoo rewrites old adjusted prices after a split or dividend. A larger gap than this
# between stored and fresh prices means the ticker has to be reloaded in full.
ADJUSTMENT_TOLERANCE = 0.001

# Reported figures for recent quarters are often filled in or restated later.
QUARTERS_TO_REFRESH = 2


def get_company_data(ticker):
    """Return a snapshot of company data as of today.

    The ratios here are today's values. They belong to the company, not to any
    historical date, so they must not be used as historical regression inputs.
    """
    try:
        info = yf.Ticker(ticker).info
    except Exception as e:
        print(f"Error fetching fundamentals for {ticker}: {e}")
        return None

    return {
        "Ticker": ticker,
        "SnapshotDate": datetime.today().replace(hour=0, minute=0, second=0, microsecond=0),
        "CompanyName": info.get("longName"),
        "Sector": info.get("sector"),
        "Industry": info.get("industry"),
        "MarketCap": info.get("marketCap"),
        "Beta": info.get("beta"),
        "EPS_Forward": info.get("forwardEps"),
        "P_E_Trailing": info.get("trailingPE"),
        "P_E_Forward": info.get("forwardPE"),
        "P_B": info.get("priceToBook"),
        "P_S": info.get("priceToSalesTrailing12Months"),
        "DividendYield": info.get("dividendYield"),
    }


def get_quarterly_data(ticker):
    stock = yf.Ticker(ticker)

    try:
        financials = stock.quarterly_financials
        cashflow = stock.quarterly_cashflow
        balance = stock.quarterly_balance_sheet
        info = stock.info
    except Exception as e:
        print(f"Error fetching quarterly data for {ticker}: {e}")
        return None

    def safe_get_by_date(df, row, q):
        if df is None or df.empty or row not in df.index or q not in df.columns:
            return None
        value = df.loc[row, q]
        return None if pd.isna(value) else value

    def first_available(df, rows, q):
        """Return the first label with a value. An actual 0 counts as a value."""
        for row in rows:
            value = safe_get_by_date(df, row, q)
            if value is not None:
                return value
        return None

    def ratio(numerator, denominator):
        if numerator is None or denominator in (None, 0):
            return None
        return numerator / denominator

    columns = [set(df.columns) for df in (financials, cashflow, balance) if df is not None and not df.empty]
    all_quarters = sorted(set().union(*columns)) if columns else []
    shares_outstanding = info.get("sharesOutstanding")

    data_list = []

    for q in all_quarters:
        q_date = pd.to_datetime(q)

        revenue = safe_get_by_date(financials, "Total Revenue", q)
        gross_profit = safe_get_by_date(financials, "Gross Profit", q)
        operating_income = safe_get_by_date(financials, "Operating Income", q)
        net_income = safe_get_by_date(financials, "Net Income", q)

        data_list.append({
            "Ticker": ticker,
            "Quarter": f"{q_date.year}-Q{((q_date.month - 1) // 3) + 1}",
            "QuarterDate": q_date.to_pydatetime(),
            "Revenue": revenue,
            "GrossProfit": gross_profit,
            "OperatingIncome": operating_income,
            "NetIncome": net_income,
            "EBITDA": safe_get_by_date(financials, "EBITDA", q),
            # Uses today's share count, so it is only an approximation for old quarters.
            "EPS_Trailing": ratio(net_income, shares_outstanding),
            "TotalAssets": safe_get_by_date(balance, "Total Assets", q),
            "TotalCash": first_available(balance, ["Cash And Cash Equivalents", "Cash"], q),
            # Long-term debt only, which matches the rows already stored.
            "TotalDebt": safe_get_by_date(balance, "Long Term Debt", q),
            # Yahoo serves about six quarters of these, so history builds up over time.
            "StockholdersEquity": safe_get_by_date(balance, "Stockholders Equity", q),
            "SharesOutstanding": safe_get_by_date(balance, "Ordinary Shares Number", q),
            "FreeCashFlow": safe_get_by_date(cashflow, "Free Cash Flow", q),
            "CapEx": safe_get_by_date(cashflow, "Capital Expenditure", q),
            "DividendsPaid": first_available(cashflow, ["Cash Dividends Paid", "Dividends Paid"], q),
            "GrossMargin": ratio(gross_profit, revenue),
            "OperatingMargin": ratio(operating_income, revenue),
            "NetMargin": ratio(net_income, revenue),
        })

    return data_list


def download_history(ticker, start_date):
    # The end date is exclusive, so today's unfinished trading day is left out.
    hist = yf.download(
        ticker,
        start=start_date,
        end=datetime.today().replace(hour=0, minute=0, second=0, microsecond=0),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if hist.empty:
        return hist

    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    hist.columns.name = None

    hist = hist.reset_index()
    hist["Ticker"] = ticker
    hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
    return hist


def prices_were_readjusted(history_col, ticker, fresh):
    """Check whether Yahoo has changed prices we already stored."""
    stored = pd.DataFrame(list(history_col.find(
        {"Ticker": ticker, "Date": {"$gte": fresh["Date"].min()}},
        {"_id": 0, "Date": 1, "Adj Close": 1},
    )))
    if stored.empty:
        return False

    merged = stored.merge(fresh[["Date", "Adj Close"]], on="Date", suffixes=("_stored", "_fresh"))
    difference = (merged["Adj Close_fresh"] / merged["Adj Close_stored"] - 1).abs()
    return bool((difference > ADJUSTMENT_TOLERANCE).any())


def fetch_stock_history(ticker, history_col, full_reload=False):
    """Fetch prices for one ticker, reloading everything if old prices have changed.

    Appending new rows after a split would otherwise leave old and new prices on
    different scales, which appears as a huge one-day loss.
    """
    try:
        last_entry = None if full_reload else history_col.find_one({"Ticker": ticker}, sort=[("Date", -1)])

        if last_entry is None:
            return download_history(ticker, HISTORY_START)

        hist = download_history(ticker, last_entry["Date"] - pd.Timedelta(days=OVERLAP_DAYS))
        if not hist.empty and prices_were_readjusted(history_col, ticker, hist):
            print(f"{ticker}: stored prices are out of date, reloading full history")
            return download_history(ticker, HISTORY_START)
        return hist

    except Exception as e:
        print(f"Error fetching history for {ticker}: {e}")
        return None


def latest_quarters(df, count=QUARTERS_TO_REFRESH):
    return set(sorted(df["Quarter"].unique())[-count:]) if not df.empty else set()


def load_raw_data(update_mode=0, full_reload=False):
    """Fetch data from Yahoo Finance. Modes: 0 all, 1 fundamentals, 2 quarterly, 3 prices."""
    db = get_database()

    if update_mode in [0, 1]:
        fundamentals = [get_company_data(ticker) for ticker in TICKERS]
        df_fundamentals = pd.DataFrame([d for d in fundamentals if d])
        upsert_collection(db["Fundamentals"], df_fundamentals, ["Ticker"], "Fundamentals")

    if update_mode in [0, 2]:
        rows = []
        for ticker in TICKERS:
            rows.extend(get_quarterly_data(ticker) or [])
        df_quarterly = pd.DataFrame(rows)

        merge_quarterly_rows(
            db["CompanyQuarterlyData"],
            df_quarterly,
            ["Ticker", "Quarter"],
            "CompanyQuarterlyData",
            refresh_quarters=latest_quarters(df_quarterly),
        )

    if update_mode in [0, 3]:
        for ticker in TICKERS:
            df_history = fetch_stock_history(ticker, db["Stock_History"], full_reload)
            if df_history is not None and not df_history.empty:
                upsert_collection(db["Stock_History"], df_history, ["Ticker", "Date"], f"Stock_History ({ticker})")
