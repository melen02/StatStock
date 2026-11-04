import os

from loadData.db import OUTPUT_DIR, get_database
from loadData.market_data import load_history, wide_prices


def calculate_correlation():
    returns = wide_prices(load_history(get_database())).pct_change(fill_method=None)
    return returns.corr()


def export_correlation():
    correlation_matrix = calculate_correlation()

    # Avoid duplicate column name errors
    correlation_matrix.index.name = None
    correlation_matrix.columns.name = None

    # Long format suits Power BI
    correlation_long = correlation_matrix.stack().reset_index()
    correlation_long.columns = ["Ticker1", "Ticker2", "Correlation"]

    # Remove self-correlations like AAPL-AAPL = 1.0
    correlation_long = correlation_long[correlation_long["Ticker1"] != correlation_long["Ticker2"]]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "stock_correlation_long.csv")
    correlation_long.to_csv(path, index=False)
    print(f"Correlation exported to {path}")
