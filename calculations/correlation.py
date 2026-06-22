import os
import pandas as pd
from loadData.db import get_database


def calculate_correlation():
    db = get_database()

    df = pd.DataFrame(list(db["Stock_History"].find()))
    df = df.drop(columns=["_id"], errors="ignore")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"])

    # Calculate daily returns per stock
    df["DailyReturn"] = df.groupby("Ticker")["Close"].pct_change()

    # Create table: rows = dates, columns = tickers, values = daily returns
    returns_table = df.pivot(
        index="Date",
        columns="Ticker",
        values="DailyReturn"
    )

    # Calculate correlation between stocks
    correlation_matrix = returns_table.corr()

    return correlation_matrix


def export_correlation():
    correlation_matrix = calculate_correlation()

    # Avoid duplicate column name errors
    correlation_matrix.index.name = None
    correlation_matrix.columns.name = None

    # Convert from wide matrix to Power BI-friendly long format
    correlation_long = (
        correlation_matrix
        .stack()
        .reset_index()
    )

    correlation_long.columns = ["Ticker1", "Ticker2", "Correlation"]

    # Remove self-correlations like AAPL-AAPL = 1.0
    correlation_long = correlation_long[
        correlation_long["Ticker1"] != correlation_long["Ticker2"]
    ]

    os.makedirs("output", exist_ok=True)

    correlation_long.to_csv(
        "output/stock_correlation_long.csv",
        index=False
    )

    print("✅ Correlation exported to output/stock_correlation_long.csv")