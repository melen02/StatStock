import os
import pandas as pd
import numpy as np
from loadData.db import get_database


# ── Config ────────────────────────────────────────────────────────────────────

N_SIMULATIONS  = 1000    # number of price paths per stock
N_DAYS         = 252     # trading days to simulate (~1 year)
CONFIDENCE     = [0.05, 0.25, 0.50, 0.75, 0.95]  # percentile bands for fan chart
LOOKBACK_DAYS  = 252     # how many days of history to estimate drift and volatility


# ── Load price history ────────────────────────────────────────────────────────

def load_history(db):
    history = pd.DataFrame(list(db["Stock_History"].find()))
    history.drop(columns=["_id"], errors="ignore", inplace=True)
    history["Date"] = pd.to_datetime(history["Date"])
    history = history.sort_values(["Ticker", "Date"])
    history["DailyReturn"] = history.groupby("Ticker")["Close"].pct_change(fill_method=None)
    return history


# ── Estimate parameters per stock ─────────────────────────────────────────────

def estimate_parameters(history):
    """
    For each stock, estimate annualised drift (mu) and volatility (sigma)
    from the most recent LOOKBACK_DAYS of daily returns.
    GBM assumption: daily log returns are normally distributed.
    """
    params = []

    for ticker, group in history.groupby("Ticker"):
        recent = group.tail(LOOKBACK_DAYS).copy()
        returns = recent["DailyReturn"].dropna()

        if len(returns) < 30:
            continue

        # Use log returns for GBM
        log_returns = np.log(1 + returns)

        mu_daily    = log_returns.mean()
        sigma_daily = log_returns.std()

        # Annualise
        mu_annual    = mu_daily * 252
        sigma_annual = sigma_daily * np.sqrt(252)

        last_price = recent["Close"].iloc[-1]
        last_date  = recent["Date"].iloc[-1]

        params.append({
            "Ticker":       ticker,
            "LastPrice":    last_price,
            "LastDate":     last_date,
            "MuDaily":      mu_daily,
            "SigmaDaily":   sigma_daily,
            "MuAnnual":     mu_annual,
            "SigmaAnnual":  sigma_annual,
            "SharpeRatio":  mu_annual / sigma_annual if sigma_annual > 0 else np.nan,
        })

    return pd.DataFrame(params)


# ── Monte Carlo simulation ─────────────────────────────────────────────────────

def run_gbm(S0, mu, sigma, n_days, n_sims):
    """
    Geometric Brownian Motion:
        S(t) = S(0) * exp((mu - 0.5*sigma^2)*t + sigma*W(t))
    where W(t) is a Wiener process (cumulative random normal).

    Returns matrix of shape (n_days, n_sims) — each column is one price path.
    """
    dt = 1  # one trading day

    # Random shocks: shape (n_days, n_sims)
    Z = np.random.standard_normal((n_days, n_sims))

    # Daily log returns
    daily_log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    # Cumulative price paths
    price_paths = S0 * np.exp(np.cumsum(daily_log_returns, axis=0))

    return price_paths


def simulate_all(params_df):
    """
    Run GBM for every stock and collect percentile bands.
    Returns two DataFrames:
    - percentiles_df: fan chart data (one row per stock per day per percentile)
    - terminal_df: distribution of final prices after N_DAYS
    """
    percentile_rows = []
    terminal_rows   = []

    print(f"Running {N_SIMULATIONS} simulations × {N_DAYS} days for {len(params_df)} stocks...")

    for _, row in params_df.iterrows():
        ticker  = row["Ticker"]
        S0      = row["LastPrice"]
        mu      = row["MuDaily"]
        sigma   = row["SigmaDaily"]

        paths = run_gbm(S0, mu, sigma, N_DAYS, N_SIMULATIONS)

        # Percentile bands per day (for fan chart)
        for day in range(0, N_DAYS, 5):  # every 5 days to keep data manageable
            day_prices = paths[day, :]
            p_row = {"Ticker": ticker, "Day": day + 1}
            for p in CONFIDENCE:
                p_row[f"P{int(p*100)}"] = np.percentile(day_prices, p * 100)
            percentile_rows.append(p_row)

        # Terminal distribution (day N_DAYS)
        terminal_prices = paths[-1, :]
        terminal_return = terminal_prices / S0 - 1

        terminal_rows.append({
            "Ticker":           ticker,
            "LastPrice":        S0,
            "MuAnnual":         row["MuAnnual"],
            "SigmaAnnual":      row["SigmaAnnual"],
            "SharpeRatio":      row["SharpeRatio"],
            "MedianPrice":      np.percentile(terminal_prices, 50),
            "MedianReturn":     np.percentile(terminal_return, 50),
            "P5_Price":         np.percentile(terminal_prices, 5),
            "P95_Price":        np.percentile(terminal_prices, 95),
            "P5_Return":        np.percentile(terminal_return, 5),
            "P95_Return":       np.percentile(terminal_return, 95),
            "ProbPositive":     (terminal_return > 0).mean(),
            "ProbUp10pct":      (terminal_return > 0.10).mean(),
            "ProbDown20pct":    (terminal_return < -0.20).mean(),
            "ExpectedReturn":   terminal_return.mean(),
            "VaR_5pct":         np.percentile(terminal_return, 5),  # Value at Risk
        })

        if len(terminal_rows) % 20 == 0:
            print(f"  {len(terminal_rows)}/{len(params_df)} done...")

    return pd.DataFrame(percentile_rows), pd.DataFrame(terminal_rows)


# ── Portfolio simulation ───────────────────────────────────────────────────────

def simulate_equal_weight_portfolio(params_df, history):
    """
    Simulate an equal-weight portfolio of all stocks.
    Uses correlated returns based on the historical covariance matrix.
    """
    print("\nSimulating equal-weight portfolio (correlated)...")

    tickers = params_df["Ticker"].tolist()

    # Build return matrix
    pivot = (
        history.pivot(index="Date", columns="Ticker", values="DailyReturn")
        [tickers]
        .dropna()
        .tail(LOOKBACK_DAYS)
    )

    if pivot.empty or len(pivot) < 30:
        print("  Not enough data for portfolio simulation, skipping.")
        return None

    # Log return covariance matrix
    log_pivot = np.log(1 + pivot)
    cov_matrix = log_pivot.cov().values
    mu_vector  = log_pivot.mean().values

    n_stocks = len(tickers)
    weights  = np.ones(n_stocks) / n_stocks  # equal weight

    portfolio_paths = []

    for _ in range(N_SIMULATIONS):
        # Correlated daily shocks using Cholesky decomposition
        try:
            L = np.linalg.cholesky(cov_matrix + 1e-8 * np.eye(n_stocks))
        except np.linalg.LinAlgError:
            L = np.diag(np.sqrt(np.diag(cov_matrix)))

        Z = np.random.standard_normal((N_DAYS, n_stocks))
        correlated_shocks = Z @ L.T

        daily_log_returns = (mu_vector - 0.5 * np.diag(cov_matrix)) + correlated_shocks
        stock_paths = np.exp(np.cumsum(daily_log_returns, axis=0))

        portfolio_path = stock_paths @ weights
        portfolio_paths.append(portfolio_path)

    portfolio_paths = np.array(portfolio_paths).T  # shape (N_DAYS, N_SIMULATIONS)

    portfolio_rows = []
    for day in range(0, N_DAYS, 5):
        day_values = portfolio_paths[day, :]
        p_row = {"Day": day + 1, "Ticker": "PORTFOLIO_EW"}
        for p in CONFIDENCE:
            p_row[f"P{int(p*100)}"] = np.percentile(day_values, p * 100)
        portfolio_rows.append(p_row)

    terminal = portfolio_paths[-1, :]
    terminal_return = terminal - 1

    portfolio_summary = {
        "Ticker":        "PORTFOLIO_EW",
        "LastPrice":     1.0,
        "MuAnnual":      float((mu_vector * 252 * weights).sum()),
        "SigmaAnnual":   float(np.sqrt(weights @ (cov_matrix * 252) @ weights)),
        "SharpeRatio":   np.nan,
        "MedianPrice":   float(np.percentile(terminal, 50)),
        "MedianReturn":  float(np.percentile(terminal_return, 50)),
        "P5_Price":      float(np.percentile(terminal, 5)),
        "P95_Price":     float(np.percentile(terminal, 95)),
        "P5_Return":     float(np.percentile(terminal_return, 5)),
        "P95_Return":    float(np.percentile(terminal_return, 95)),
        "ProbPositive":  float((terminal_return > 0).mean()),
        "ProbUp10pct":   float((terminal_return > 0.10).mean()),
        "ProbDown20pct": float((terminal_return < -0.20).mean()),
        "ExpectedReturn":float(terminal_return.mean()),
        "VaR_5pct":      float(np.percentile(terminal_return, 5)),
    }

    return pd.DataFrame(portfolio_rows), portfolio_summary


# ── Save to MongoDB ───────────────────────────────────────────────────────────

def save_to_mongo(db, collection_name, df):
    col = db[collection_name]
    col.drop()
    records = df.copy()
    for c in records.select_dtypes(include="bool").columns:
        records[c] = records[c].astype(bool)
    # Convert dates to string for MongoDB compatibility
    for c in records.select_dtypes(include="datetime64").columns:
        records[c] = records[c].astype(str)
    records = records.to_dict("records")
    if records:
        col.insert_many(records)
    print(f"✅ Saved {len(records)} rows to MongoDB:{collection_name}")


def export_csvs(params_df, percentiles_df, terminal_df):
    os.makedirs("output", exist_ok=True)
    params_df.to_csv("output/mc_parameters.csv",   index=False)
    percentiles_df.to_csv("output/mc_fan_chart.csv", index=False)
    terminal_df.to_csv("output/mc_terminal.csv",   index=False)
    print("✅ Monte Carlo CSVs exported to output/")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_monte_carlo():
    np.random.seed(42)  # reproducible results
    db = get_database()

    print("Loading price history...")
    history = load_history(db)
    print(f"Loaded {len(history)} rows across {history['Ticker'].nunique()} tickers")

    print("\nEstimating GBM parameters per stock...")
    params_df = estimate_parameters(history)
    print(f"Parameters estimated for {len(params_df)} stocks")
    print(f"\nTop 5 by Sharpe Ratio:")
    print(params_df.nlargest(5, "SharpeRatio")[["Ticker", "MuAnnual", "SigmaAnnual", "SharpeRatio"]].to_string(index=False))

    # Individual stock simulations
    print("\nRunning individual stock simulations...")
    percentiles_df, terminal_df = simulate_all(params_df)

    # Portfolio simulation
    portfolio_result = simulate_equal_weight_portfolio(params_df, history)
    if portfolio_result is not None:
        portfolio_fan, portfolio_summary = portfolio_result
        percentiles_df = pd.concat([percentiles_df, portfolio_fan], ignore_index=True)
        terminal_df = pd.concat([terminal_df, pd.DataFrame([portfolio_summary])], ignore_index=True)

    # Save parameter estimates
    params_df["LastDate"] = params_df["LastDate"].astype(str)

    # Save to MongoDB
    save_to_mongo(db, "MC_Parameters",  params_df)
    save_to_mongo(db, "MC_FanChart",    percentiles_df)
    save_to_mongo(db, "MC_Terminal",    terminal_df)

    # Export CSV backup
    export_csvs(params_df, percentiles_df, terminal_df)

    print("\n✅ Done.")
    print("MongoDB collections written:")
    print("  MC_Parameters  ← mu, sigma, Sharpe per stock")
    print("  MC_FanChart    ← percentile bands per day (fan chart)")
    print("  MC_Terminal    ← 1-year outcome distribution per stock")
    print(f"\nSample terminal results:")
    print(terminal_df[["Ticker", "MedianReturn", "P5_Return", "P95_Return",
                        "ProbPositive", "VaR_5pct"]].head(10).to_string(index=False))


if __name__ == "__main__":
    run_monte_carlo()