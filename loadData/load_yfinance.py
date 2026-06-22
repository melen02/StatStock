import yfinance as yf
import pandas as pd
from datetime import datetime

from loadData.db import get_database
from loadData.tickers import TICKERS
from loadData.mongo_helpers import upsert_collection, merge_quarterly_rows


FULL_RELOAD_HISTORY = False 


CHOSEN_QUARTERS = {"2026-Q1", "2026-Q2"}


def get_company_data(ticker):
    stock = yf.Ticker(ticker)

    try:
        info = stock.info
    except Exception as e:
        print(f"Error fetching fundamentals for {ticker}: {e}")
        return None

    return {
        "Ticker": ticker,
        "CompanyName": info.get("longName"),
        "Sector": info.get("sector"),
        "Industry": info.get("industry"),
        "MarketCap": info.get("marketCap"),
        "Beta": info.get("beta")
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
        try:
            if df is None or df.empty:
                return None
            if row not in df.index:
                return None
            if q not in df.columns:
                return None

            value = df.loc[row, q]
            return None if pd.isna(value) else value
        except Exception:
            return None

    financial_cols = set(financials.columns) if financials is not None and not financials.empty else set()
    cashflow_cols = set(cashflow.columns) if cashflow is not None and not cashflow.empty else set()
    balance_cols = set(balance.columns) if balance is not None and not balance.empty else set()

    all_quarters = sorted(financial_cols | cashflow_cols | balance_cols)
    shares_outstanding = info.get("sharesOutstanding")

    data_list = []

    for q in all_quarters:
        q_date = pd.to_datetime(q)
        quarter_label = f"{q_date.year}-Q{((q_date.month - 1) // 3) + 1}"

        revenue = safe_get_by_date(financials, "Total Revenue", q)
        gross_profit = safe_get_by_date(financials, "Gross Profit", q)
        operating_income = safe_get_by_date(financials, "Operating Income", q)
        net_income = safe_get_by_date(financials, "Net Income", q)
        ebitda = safe_get_by_date(financials, "EBITDA", q)

        total_assets = safe_get_by_date(balance, "Total Assets", q)
        total_cash = safe_get_by_date(balance, "Cash", q) or safe_get_by_date(balance, "Cash & Cash Equivalents", q)
        total_debt = safe_get_by_date(balance, "Long Term Debt", q) or safe_get_by_date(balance, "Total Liab", q)

        free_cash_flow = safe_get_by_date(cashflow, "Free Cash Flow", q)
        capex = safe_get_by_date(cashflow, "Capital Expenditure", q)
        dividends_paid = safe_get_by_date(cashflow, "Dividends Paid", q)

        gross_margin = gross_profit / revenue if gross_profit is not None and revenue not in [None, 0] else None
        operating_margin = operating_income / revenue if operating_income is not None and revenue not in [None, 0] else None
        net_margin = net_income / revenue if net_income is not None and revenue not in [None, 0] else None
        eps_trailing = net_income / shares_outstanding if net_income is not None and shares_outstanding else None

        data_list.append({
            "Ticker": ticker,
            "Quarter": quarter_label,
            "QuarterDate": q_date.to_pydatetime(),
            "Revenue": revenue,
            "GrossProfit": gross_profit,
            "OperatingIncome": operating_income,
            "NetIncome": net_income,
            "EBITDA": ebitda,
            "EPS_Trailing": eps_trailing,
            "TotalAssets": total_assets,
            "TotalCash": total_cash,
            "TotalDebt": total_debt,
            "FreeCashFlow": free_cash_flow,
            "CapEx": capex,
            "DividendsPaid": dividends_paid,
            "GrossMargin": gross_margin,
            "OperatingMargin": operating_margin,
            "NetMargin": net_margin
        })

    return data_list


def fetch_stock_history(ticker, history_col):
    try:
        if FULL_RELOAD_HISTORY:
            start_date = datetime(2020, 1, 1)
        else:
            last_entry = history_col.find_one({"Ticker": ticker}, sort=[("Date", -1)])
            start_date = last_entry["Date"] + pd.Timedelta(days=1) if last_entry else datetime(2020, 1, 1)

        end_date = datetime.today()

        if start_date >= end_date:
            print(f"No new history for {ticker}")
            return None

        hist = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=False
        )

        if hist.empty:
            print(f"No new data for {ticker}")
            return None

        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = ["_".join([str(i) for i in col if i]).strip() for col in hist.columns]

        hist.reset_index(inplace=True)
        hist["Ticker"] = ticker
        hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
        hist.columns = [col.split("_")[0] if "_" in col else col for col in hist.columns]

        stock = yf.Ticker(ticker)
        info = stock.info

        hist["EPS_Forward"] = info.get("forwardEps")
        hist["P_E_Trailing"] = info.get("trailingPE")
        hist["P_E_Forward"] = info.get("forwardPE")
        hist["P_B"] = info.get("priceToBook")
        hist["P_S"] = info.get("priceToSalesTrailing12Months")
        hist["DividendYield"] = info.get("dividendYield")

        return hist

    except Exception as e:
        print(f"Error fetching history for {ticker}: {e}")
        return None


def load_raw_data(update_mode=0):
    db = get_database()

    fundamentals_col = db["Fundamentals"]
    quarterly_col = db["CompanyQuarterlyData"]
    history_col = db["Stock_History"]

    if update_mode in [0, 1]:
        fundamentals_list = [get_company_data(ticker) for ticker in TICKERS]
        df_fundamentals = pd.DataFrame([d for d in fundamentals_list if d])
        upsert_collection(fundamentals_col, df_fundamentals, ["Ticker"], "Fundamentals")

    if update_mode in [0, 2]:
        quarterly_frames = []

        for ticker in TICKERS:
            qdata = get_quarterly_data(ticker)
            if qdata:
                quarterly_frames.extend(qdata)

        df_quarterly = pd.DataFrame(quarterly_frames)

        merge_quarterly_rows(
            quarterly_col,
            df_quarterly,
            ["Ticker", "Quarter"],
            "CompanyQuarterlyData",
            refresh_quarters=CHOSEN_QUARTERS
        )

    if update_mode in [0, 3]:
        for ticker in TICKERS:
            df_history = fetch_stock_history(ticker, history_col)
            if df_history is not None:
                upsert_collection(history_col, df_history, ["Ticker", "Date"], "Stock_History")

    print("\n--- Sample Documents from MongoDB ---")
    print("Fundamentals sample:", fundamentals_col.find_one())
    print("CompanyQuarterlyData sample:", quarterly_col.find_one())
    print("Stock_History sample:", history_col.find_one())
