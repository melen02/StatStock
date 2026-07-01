import os
import pandas as pd
import numpy as np
from loadData.db import get_database


# ── Config ────────────────────────────────────────────────────────────────────

N_SIMULATIONS  = 1000
N_DAYS         = 252
CONFIDENCE     = [0.05, 0.25, 0.50, 0.75, 0.95]
LOOKBACK_DAYS  = 252

# Jump diffusion parameters (Merton model)
JUMP_INTENSITY  = 5      # expected number of jumps per year (lambda)
JUMP_MEAN       = -0.03  # average jump size (negative = crashes more common)
JUMP_STD        = 0.08   # std of jump size

# Mean reversion (not used in GBM, but used to cap drift)
MAX_ANNUAL_DRIFT = 1.5   # cap drift at 150% annualised — prevents runaway optimism

# Fundamentals adjustment
FM_WEIGHT = 0.3          # how much weight to give fundamentals vs historical drift


# ── Load data ─────────────────────────────────────────────────────────────────

def load_history(db):
    history = pd.DataFrame(list(db["Stock_History"].find()))
    history.drop(columns=["_id"], errors="ignore", inplace=True)
    history["Date"] = pd.to_datetime(history["Date"])
    history = history.sort_values(["Ticker", "Date"])
    history["DailyReturn"] = history.groupby("Ticker")["Close"].pct_change(fill_method=None)
    return history


def load_fundamentals_signal(db):
    """
    Load Fama-MacBeth summary to get which features significantly predict returns.
    Also load latest quarterly data to score each stock.
    """
    try:
        fm_summary = pd.DataFrame(list(db["Regression_FM_Summary"].find()))
        fm_summary.drop(columns=["_id"], errors="ignore", inplace=True)

        quarterly = pd.DataFrame(list(db["CompanyQuarterlyData"].find()))
        quarterly.drop(columns=["_id"], errors="ignore", inplace=True)
        quarterly["QuarterDate"] = pd.to_datetime(quarterly["QuarterDate"])

        # Get most recent quarter per ticker
        latest = (
            quarterly.sort_values("QuarterDate")
            .groupby("Ticker")
            .last()
            .reset_index()
        )

        return fm_summary, latest

    except Exception as e:
        print(f"  Could not load fundamentals signal: {e}")
        return None, None


def compute_fundamentals_score(fm_summary, latest_quarterly):
    """
    For each stock, compute a fundamentals score based on significant FM predictors.
    Score is a weighted sum of standardised feature values × FM coefficient.
    Positive score → fundamentals suggest above-average return.
    Negative score → fundamentals suggest below-average return.
    """
    if fm_summary is None or latest_quarterly is None:
        return {}

    # Only use features with reasonable t-stat (|T_Stat| > 1.0)
    significant = fm_summary[fm_summary["T_Stat"].abs() > 1.0].copy()

    if significant.empty:
        return {}

    features = significant["Feature"].tolist()
    coefs    = significant.set_index("Feature")["MeanCoef"]

    scores = {}
    for _, row in latest_quarterly.iterrows():
        ticker = row["Ticker"]
        score  = 0.0
        for feat in features:
            if feat in row and pd.notna(row[feat]):
                # Standardise using cross-sectional mean/std
                col_vals = latest_quarterly[feat].dropna()
                if col_vals.std() > 0:
                    z = (row[feat] - col_vals.mean()) / col_vals.std()
                    score += coefs[feat] * z
        scores[ticker] = score

    # Normalise scores to [-0.02, 0.02] daily drift adjustment
    if scores:
        vals = np.array(list(scores.values()))
        lo, hi = vals.min(), vals.max()
        if hi > lo:
            for t in scores:
                scores[t] = (scores[t] - lo) / (hi - lo) * 0.04 - 0.02
        else:
            scores = {t: 0.0 for t in scores}

    return scores


# ── Parameter estimation ──────────────────────────────────────────────────────

def estimate_parameters(history, fundamentals_scores):
    params = []

    for ticker, group in history.groupby("Ticker"):
        recent  = group.tail(LOOKBACK_DAYS).copy()
        returns = recent["DailyReturn"].dropna()

        if len(returns) < 30:
            continue

        log_returns = np.log(1 + returns)
        mu_daily    = log_returns.mean()
        sigma_daily = log_returns.std()

        # Cap annualised drift to prevent runaway optimism
        mu_annual_raw = mu_daily * 252
        mu_annual_cap = np.clip(mu_annual_raw, -MAX_ANNUAL_DRIFT, MAX_ANNUAL_DRIFT)

        if mu_annual_raw != mu_annual_cap:
            mu_daily = mu_annual_cap / 252

        # Fundamentals adjustment — blend historical drift with fundamentals signal
        fm_adjustment = fundamentals_scores.get(ticker, 0.0)
        mu_daily_adj  = (1 - FM_WEIGHT) * mu_daily + FM_WEIGHT * (mu_daily + fm_adjustment)

        sigma_annual = sigma_daily * np.sqrt(252)

        # Estimate jump parameters from historical large moves
        # Define jumps as daily returns beyond 3 sigma
        threshold   = 3 * sigma_daily
        jump_days   = returns[returns.abs() > threshold]
        jump_freq   = len(jump_days) / len(returns) * 252  # annualised jump frequency
        jump_mean   = jump_days.mean() if len(jump_days) > 0 else JUMP_MEAN
        jump_std    = jump_days.std()  if len(jump_days) > 1 else JUMP_STD

        # Use estimated jumps but bound them sensibly
        jump_freq = np.clip(jump_freq, 1, 20)
        jump_mean = np.clip(jump_mean, -0.15, 0.05)
        jump_std  = np.clip(jump_std,  0.02, 0.20)

        last_price = recent["Close"].iloc[-1]
        last_date  = recent["Date"].iloc[-1]

        params.append({
            "Ticker":          ticker,
            "LastPrice":       last_price,
            "LastDate":        last_date,
            "MuDaily_Raw":     log_returns.mean(),
            "MuDaily_Adj":     mu_daily_adj,
            "FMadjustment":    fm_adjustment,
            "SigmaDaily":      sigma_daily,
            "MuAnnual":        mu_daily_adj * 252,
            "SigmaAnnual":     sigma_annual,
            "JumpFreq":        jump_freq,
            "JumpMean":        jump_mean,
            "JumpStd":         jump_std,
            "SharpeRatio":     (mu_daily_adj * 252) / sigma_annual if sigma_annual > 0 else np.nan,
        })

    return pd.DataFrame(params)


# ── Jump diffusion simulation (Merton model) ──────────────────────────────────

def run_jump_diffusion(S0, mu, sigma, jump_freq, jump_mean, jump_std, n_days, n_sims):
    """
    Merton Jump Diffusion Model:
        S(t) = S(0) * exp(diffusion + jumps)

    Diffusion: standard GBM component
    Jumps: Poisson-distributed number of jumps per day,
           each jump drawn from N(jump_mean, jump_std)

    The drift is adjusted for the expected jump contribution:
        mu_adj = mu - lambda * (exp(jump_mean + 0.5*jump_std^2) - 1)
    so that the expected return equals mu regardless of jumps.
    """
    dt          = 1
    lambda_dt   = jump_freq / 252  # daily jump probability

    # Drift adjustment for jump component
    expected_jump = np.exp(jump_mean + 0.5 * jump_std**2) - 1
    mu_adj        = mu - lambda_dt * expected_jump

    # Diffusion component
    Z = np.random.standard_normal((n_days, n_sims))
    diffusion = (mu_adj - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    # Jump component — Poisson number of jumps per day
    N_jumps     = np.random.poisson(lambda_dt, (n_days, n_sims))
    jump_shocks = np.random.normal(jump_mean, jump_std, (n_days, n_sims))
    jumps       = N_jumps * jump_shocks

    daily_log_returns = diffusion + jumps
    price_paths = S0 * np.exp(np.cumsum(daily_log_returns, axis=0))

    return price_paths


# ── Simulate all stocks ───────────────────────────────────────────────────────

def simulate_all(params_df):
    percentile_rows = []
    terminal_rows   = []

    print(f"Running {N_SIMULATIONS} simulations × {N_DAYS} days for {len(params_df)} stocks...")
    print("Model: Merton Jump Diffusion + Fundamentals-adjusted drift\n")

    for _, row in params_df.iterrows():
        ticker    = row["Ticker"]
        S0        = row["LastPrice"]
        mu        = row["MuDaily_Adj"]
        sigma     = row["SigmaDaily"]
        jump_freq = row["JumpFreq"]
        jump_mean = row["JumpMean"]
        jump_std  = row["JumpStd"]

        paths = run_jump_diffusion(
            S0, mu, sigma, jump_freq, jump_mean, jump_std, N_DAYS, N_SIMULATIONS
        )

        # Fan chart — every 5 days
        for day in range(0, N_DAYS, 5):
            day_prices = paths[day, :]
            p_row = {"Ticker": ticker, "Day": day + 1}
            for p in CONFIDENCE:
                p_row[f"P{int(p*100)}"] = float(np.percentile(day_prices, p * 100))
            percentile_rows.append(p_row)

        # Terminal distribution
        terminal_prices = paths[-1, :]
        terminal_return = terminal_prices / S0 - 1

        terminal_rows.append({
            "Ticker":           ticker,
            "LastPrice":        float(S0),
            "MuAnnual":         float(row["MuAnnual"]),
            "SigmaAnnual":      float(row["SigmaAnnual"]),
            "FMadjustment":     float(row["FMadjustment"]),
            "JumpFreq":         float(jump_freq),
            "SharpeRatio":      float(row["SharpeRatio"]) if pd.notna(row["SharpeRatio"]) else None,
            "MedianReturn":     float(np.percentile(terminal_return, 50)),
            "P5_Return":        float(np.percentile(terminal_return, 5)),
            "P95_Return":       float(np.percentile(terminal_return, 95)),
            "P5_Price":         float(np.percentile(terminal_prices, 5)),
            "P95_Price":        float(np.percentile(terminal_prices, 95)),
            "ProbPositive":     float((terminal_return > 0).mean()),
            "ProbUp10pct":      float((terminal_return > 0.10).mean()),
            "ProbDown20pct":    float((terminal_return < -0.20).mean()),
            "ExpectedReturn":   float(terminal_return.mean()),
            "VaR_5pct":         float(np.percentile(terminal_return, 5)),
            "CVaR_5pct":        float(terminal_return[terminal_return <= np.percentile(terminal_return, 5)].mean()),
        })

        if len(terminal_rows) % 20 == 0:
            print(f"  {len(terminal_rows)}/{len(params_df)} done...")

    return pd.DataFrame(percentile_rows), pd.DataFrame(terminal_rows)


# ── Portfolio simulation ──────────────────────────────────────────────────────

def simulate_equal_weight_portfolio(params_df, history):
    print("\nSimulating equal-weight portfolio (correlated jump diffusion)...")

    tickers = params_df["Ticker"].tolist()

    pivot = (
        history.pivot(index="Date", columns="Ticker", values="DailyReturn")
        [tickers]
        .dropna()
        .tail(LOOKBACK_DAYS)
    )

    if pivot.empty or len(pivot) < 30:
        print("  Not enough data for portfolio simulation, skipping.")
        return None

    log_pivot  = np.log(1 + pivot)
    cov_matrix = log_pivot.cov().values
    mu_vector  = params_df.set_index("Ticker").loc[tickers, "MuDaily_Adj"].values

    n_stocks = len(tickers)
    weights  = np.ones(n_stocks) / n_stocks

    # Average jump parameters across stocks
    avg_jump_freq = params_df["JumpFreq"].mean()
    avg_jump_mean = params_df["JumpMean"].mean()
    avg_jump_std  = params_df["JumpStd"].mean()
    lambda_dt     = avg_jump_freq / 252
    expected_jump = np.exp(avg_jump_mean + 0.5 * avg_jump_std**2) - 1

    portfolio_paths = []

    for _ in range(N_SIMULATIONS):
        try:
            L = np.linalg.cholesky(cov_matrix + 1e-8 * np.eye(n_stocks))
        except np.linalg.LinAlgError:
            L = np.diag(np.sqrt(np.diag(cov_matrix)))

        Z = np.random.standard_normal((N_DAYS, n_stocks))
        correlated_shocks = Z @ L.T

        sigma_vec = np.sqrt(np.diag(cov_matrix))
        mu_adj    = mu_vector - lambda_dt * expected_jump

        diffusion = (mu_adj - 0.5 * sigma_vec**2) + correlated_shocks

        # Portfolio-level jumps
        N_jumps     = np.random.poisson(lambda_dt, (N_DAYS, 1))
        jump_shocks = np.random.normal(avg_jump_mean, avg_jump_std, (N_DAYS, 1))
        jumps       = N_jumps * jump_shocks

        daily_log_returns = diffusion + jumps
        stock_paths = np.exp(np.cumsum(daily_log_returns, axis=0))
        portfolio_path = stock_paths @ weights
        portfolio_paths.append(portfolio_path)

    portfolio_paths = np.array(portfolio_paths).T

    portfolio_rows = []
    for day in range(0, N_DAYS, 5):
        day_values = portfolio_paths[day, :]
        p_row = {"Ticker": "PORTFOLIO_EW", "Day": day + 1}
        for p in CONFIDENCE:
            p_row[f"P{int(p*100)}"] = float(np.percentile(day_values, p * 100))
        portfolio_rows.append(p_row)

    terminal        = portfolio_paths[-1, :]
    terminal_return = terminal - 1

    portfolio_summary = {
        "Ticker":        "PORTFOLIO_EW",
        "LastPrice":     1.0,
        "MuAnnual":      float((mu_vector * 252 * weights).sum()),
        "SigmaAnnual":   float(np.sqrt(weights @ (cov_matrix * 252) @ weights)),
        "FMadjustment":  0.0,
        "JumpFreq":      float(avg_jump_freq),
        "SharpeRatio":   None,
        "MedianReturn":  float(np.percentile(terminal_return, 50)),
        "P5_Return":     float(np.percentile(terminal_return, 5)),
        "P95_Return":    float(np.percentile(terminal_return, 95)),
        "P5_Price":      float(np.percentile(terminal, 5)),
        "P95_Price":     float(np.percentile(terminal, 95)),
        "ProbPositive":  float((terminal_return > 0).mean()),
        "ProbUp10pct":   float((terminal_return > 0.10).mean()),
        "ProbDown20pct": float((terminal_return < -0.20).mean()),
        "ExpectedReturn":float(terminal_return.mean()),
        "VaR_5pct":      float(np.percentile(terminal_return, 5)),
        "CVaR_5pct":     float(terminal_return[terminal_return <= np.percentile(terminal_return, 5)].mean()),
    }

    return pd.DataFrame(portfolio_rows), portfolio_summary


# ── Save to MongoDB ───────────────────────────────────────────────────────────

def save_to_mongo(db, collection_name, df):
    col = db[collection_name]
    col.drop()
    records = df.copy()
    for c in records.select_dtypes(include="bool").columns:
        records[c] = records[c].astype(bool)
    for c in records.select_dtypes(include="datetime64").columns:
        records[c] = records[c].astype(str)
    records = records.to_dict("records")
    if records:
        col.insert_many(records)
    print(f"✅ Saved {len(records)} rows to MongoDB:{collection_name}")


def export_csvs(params_df, percentiles_df, terminal_df):
    os.makedirs("output", exist_ok=True)
    params_df.to_csv("output/mc_parameters.csv",    index=False)
    percentiles_df.to_csv("output/mc_fan_chart.csv", index=False)
    terminal_df.to_csv("output/mc_terminal.csv",    index=False)
    print("✅ Monte Carlo CSVs exported to output/")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_monte_carlo():
    np.random.seed(42)
    db = get_database()

    print("Loading price history...")
    history = load_history(db)
    print(f"Loaded {len(history)} rows across {history['Ticker'].nunique()} tickers")

    print("\nLoading fundamentals signal from Fama-MacBeth results...")
    fm_summary, latest_quarterly = load_fundamentals_signal(db)
    fundamentals_scores = compute_fundamentals_score(fm_summary, latest_quarterly)
    print(f"Fundamentals scores computed for {len(fundamentals_scores)} tickers")

    print("\nEstimating parameters per stock...")
    params_df = estimate_parameters(history, fundamentals_scores)
    print(f"Parameters estimated for {len(params_df)} stocks")

    print(f"\nTop 5 by Sharpe Ratio (FM-adjusted drift):")
    print(params_df.nlargest(5, "SharpeRatio")[
        ["Ticker", "MuAnnual", "SigmaAnnual", "FMadjustment", "JumpFreq", "SharpeRatio"]
    ].to_string(index=False))

    print(f"\nTop 5 most jump-prone stocks:")
    print(params_df.nlargest(5, "JumpFreq")[
        ["Ticker", "JumpFreq", "JumpMean", "JumpStd"]
    ].to_string(index=False))

    print("\nRunning individual stock simulations...")
    percentiles_df, terminal_df = simulate_all(params_df)

    portfolio_result = simulate_equal_weight_portfolio(params_df, history)
    if portfolio_result is not None:
        portfolio_fan, portfolio_summary = portfolio_result
        percentiles_df = pd.concat([percentiles_df, portfolio_fan], ignore_index=True)
        terminal_df    = pd.concat([terminal_df, pd.DataFrame([portfolio_summary])], ignore_index=True)

    params_df["LastDate"] = params_df["LastDate"].astype(str)

    save_to_mongo(db, "MC_Parameters", params_df)
    save_to_mongo(db, "MC_FanChart",   percentiles_df)
    save_to_mongo(db, "MC_Terminal",   terminal_df)
    export_csvs(params_df, percentiles_df, terminal_df)

    print("\n✅ Done.")
    print("MongoDB collections written:")
    print("  MC_Parameters  ← mu, sigma, jump params, FM adjustment per stock")
    print("  MC_FanChart    ← percentile bands per day (fan chart)")
    print("  MC_Terminal    ← 1-year outcome distribution + VaR + CVaR")

    print(f"\nSample terminal results:")
    print(terminal_df[["Ticker", "MedianReturn", "P5_Return", "P95_Return",
                        "ProbPositive", "VaR_5pct", "CVaR_5pct"]].head(10).to_string(index=False))


if __name__ == "__main__":
    run_monte_carlo()