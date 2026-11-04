"""Correlated Merton jump-diffusion simulation with a VaR backtest.

Every stock is simulated in one correlated run, so stock and portfolio results are
consistent with each other. Daily log return of stock i:

    drift_i + diffusion_i + idiosyncratic jumps_i + beta_i * market jumps

drift_i includes the jump compensators, so that E[S_T] = S_0 * exp(mu_i * T).
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

from loadData.db import OUTPUT_DIR, get_database
from loadData.market_data import load_history, wide_prices
from loadData.mongo_helpers import replace_collection

# ── Config ────────────────────────────────────────────────────────────────────

SEED = 42
N_SIMULATIONS = 5000
N_DAYS = 252
LOOKBACK_DAYS = 252
FAN_STEP_DAYS = 5
PERCENTILES = [5, 25, 50, 75, 95]
VAR_LEVEL = 0.05

# A year of returns says almost nothing about expected return: with 30% volatility
# the standard error of the mean is 30 percentage points. Drift is therefore set
# from CAPM with these assumed values, and is not estimated from recent performance.
RISK_FREE_RATE = 0.04
EQUITY_PREMIUM = 0.05
MAX_ANNUAL_TILT = 0.05  # largest drift change allowed from the regression signal

JUMP_THRESHOLD = 3.0     # jump = move beyond this many robust standard deviations
MIN_OWN_JUMPS = 5        # with fewer, a stock borrows jump sizes from the whole universe
MIN_JUMP_STD = 0.02

BACKTEST_HORIZON = 63
BACKTEST_SIMULATIONS = 2000
BACKTEST_MIN_HISTORY = 504
BOOTSTRAP_RESAMPLES = 5000


# ── Parameter estimation ──────────────────────────────────────────────────────

def robust_sigma(x):
    """Estimate standard deviation per column from the MAD, so jumps don't inflate it."""
    return 1.4826 * np.median(np.abs(x - np.median(x, axis=0, keepdims=True)), axis=0)


def jump_mask(x):
    """Flag values far from the median, column by column."""
    return np.abs(x - np.median(x, axis=0, keepdims=True)) > JUMP_THRESHOLD * robust_sigma(x)


def compensator(mean, std, loading=1.0):
    """E[exp(loading * J)] - 1 for a normally distributed jump J."""
    return np.exp(loading * mean + 0.5 * (loading * std) ** 2) - 1


def jump_size_stats(sizes):
    if len(sizes) < 2:
        return 0.0, MIN_JUMP_STD
    return float(sizes.mean()), max(float(sizes.std(ddof=1)), MIN_JUMP_STD)


def estimate_parameters(log_returns, tilts=None):
    """Estimate model parameters using only the history passed in.

    log_returns is a complete dates-by-tickers table. Market jumps are rare, so they
    are estimated from all of it. Everything else uses the last LOOKBACK_DAYS.
    """
    tickers = log_returns.columns
    all_returns = log_returns.values
    market_all = all_returns.mean(axis=1, keepdims=True)
    market_jumps_all = jump_mask(market_all).ravel()

    window = all_returns[-LOOKBACK_DAYS:]
    market = market_all[-LOOKBACK_DAYS:]
    market_jump_days = market_jumps_all[-LOOKBACK_DAYS:]

    market_lambda = market_jumps_all.mean() * 252
    market_jump_mean, market_jump_std = jump_size_stats(market_all[market_jumps_all].ravel())

    calm = window[~market_jump_days]
    market_dev = market[~market_jump_days] - market[~market_jump_days].mean()
    betas = ((calm - calm.mean(axis=0)).T @ market_dev / (market_dev.T @ market_dev)).ravel()

    own_jumps = jump_mask(calm)
    pooled_mean, pooled_std = jump_size_stats(calm[own_jumps])
    jump_lambda = own_jumps.mean(axis=0) * 252
    jump_mean = np.full(len(tickers), pooled_mean)
    jump_std = np.full(len(tickers), pooled_std)
    for i in np.flatnonzero(own_jumps.sum(axis=0) >= MIN_OWN_JUMPS):
        jump_mean[i], jump_std[i] = jump_size_stats(calm[own_jumps[:, i], i])

    # Jump moves are already counted through the jump terms, so they are replaced with
    # the median here. Replacing single values instead of deleting whole days keeps
    # the matrix positive semi-definite.
    diffusion_cov = np.cov(np.where(own_jumps, np.median(calm, axis=0), calm), rowvar=False)

    tilt = np.zeros(len(tickers)) if tilts is None else tilts.reindex(tickers).fillna(0.0).values

    return {
        "tickers": tickers,
        "mu_annual": RISK_FREE_RATE + betas * EQUITY_PREMIUM + tilt,
        "beta": betas,
        "tilt_annual": tilt,
        "diffusion_cov": diffusion_cov,
        "total_cov": np.cov(window, rowvar=False),  # includes jump days, used by the GBM baseline
        "jump_lambda": jump_lambda,
        "jump_mean": jump_mean,
        "jump_std": jump_std,
        "market_lambda": market_lambda,
        "market_jump_mean": market_jump_mean,
        "market_jump_std": market_jump_std,
    }


def as_gbm(params):
    """Same drift and total variance, but no jumps: the baseline the jump model must beat."""
    return {
        **params,
        "diffusion_cov": params["total_cov"],
        "jump_lambda": np.zeros(len(params["tickers"])),
        "market_lambda": 0.0,
    }


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(params, n_days, n_sims, seed=SEED, sample_every=FAN_STEP_DAYS):
    """Simulate all stocks jointly.

    Returns (sampled days, price relatives with shape [days, sims, stocks]). Each path
    starts at 1 and the final day is always included.
    """
    rng = np.random.default_rng(seed)
    beta = params["beta"]
    n_stocks = len(beta)
    lam = params["jump_lambda"] / 252
    jump_mean, jump_std = params["jump_mean"], params["jump_std"]
    market_lam = params["market_lambda"] / 252
    market_mean, market_std = params["market_jump_mean"], params["market_jump_std"]
    cov = params["diffusion_cov"]

    try:
        chol = np.linalg.cholesky(cov + 1e-10 * np.eye(n_stocks))
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        chol = eigenvectors * np.sqrt(np.clip(eigenvalues, 0, None))

    drift = (
        params["mu_annual"] / 252
        - 0.5 * np.diag(cov)
        - lam * compensator(jump_mean, jump_std)
        - market_lam * compensator(market_mean, market_std, beta)
    )

    sample_days = sorted(set(range(sample_every, n_days + 1, sample_every)) | {n_days})
    relatives = np.empty((len(sample_days), n_sims, n_stocks), dtype=np.float32)
    cumulative = np.zeros((n_sims, n_stocks))
    position = 0

    for day in range(1, n_days + 1):
        cumulative += drift + rng.standard_normal((n_sims, n_stocks)) @ chol.T

        # A sum of N normal jumps is normal with mean N*m and variance N*s^2.
        n_jumps = rng.poisson(lam, (n_sims, n_stocks))
        cumulative += n_jumps * jump_mean + np.sqrt(n_jumps) * jump_std * rng.standard_normal(n_jumps.shape)

        if market_lam > 0:
            n_market = rng.poisson(market_lam, (n_sims, 1))
            market_shock = n_market * market_mean + np.sqrt(n_market) * market_std * rng.standard_normal(n_market.shape)
            cumulative += market_shock * beta

        if day == sample_days[position]:
            relatives[position] = np.exp(cumulative)
            position += 1

    return sample_days, relatives


def risk_metrics(terminal_return):
    """Summarize a return distribution. VaR is the return at the quantile, so losses are negative."""
    var = np.percentile(terminal_return, VAR_LEVEL * 100)
    return {
        "ExpectedReturn": float(terminal_return.mean()),
        "MedianReturn": float(np.percentile(terminal_return, 50)),
        "P5_Return": float(var),
        "P95_Return": float(np.percentile(terminal_return, 95)),
        "ProbPositive": float((terminal_return > 0).mean()),
        "ProbUp10pct": float((terminal_return > 0.10).mean()),
        "ProbDown20pct": float((terminal_return < -0.20).mean()),
        "VaR_5pct": float(var),
        "CVaR_5pct": float(terminal_return[terminal_return <= var].mean()),
    }


def summarize_simulation(params, last_prices, sample_days, relatives):
    """Return (fan chart rows, terminal rows) for each stock and the equal-weight portfolio."""
    fan = []
    terminal = []
    columns = [f"P{p}" for p in PERCENTILES]

    def add_fan(ticker, values, scale):
        bands = np.percentile(values, PERCENTILES, axis=1).T * scale
        fan.append(pd.DataFrame(bands, columns=columns).assign(Ticker=ticker, Day=sample_days))

    for i, ticker in enumerate(params["tickers"]):
        price = float(last_prices[ticker])
        add_fan(ticker, relatives[:, :, i], price)
        terminal.append({"Ticker": ticker, "LastPrice": price, **risk_metrics(relatives[-1, :, i].astype(float) - 1)})

    # Buy and hold: equal weights at the start, which then drift with prices.
    portfolio = relatives.mean(axis=2)
    add_fan("PORTFOLIO_EW", portfolio, 1.0)
    terminal.append({"Ticker": "PORTFOLIO_EW", "LastPrice": 1.0, **risk_metrics(portfolio[-1].astype(float) - 1)})

    return pd.concat(fan, ignore_index=True)[["Ticker", "Day"] + columns], pd.DataFrame(terminal)


# ── Backtest ──────────────────────────────────────────────────────────────────

def kupiec_pvalue(breaches, n, level=VAR_LEVEL):
    """Kupiec's test of whether the breach rate equals the VaR level."""
    if n == 0:
        return np.nan
    statistic = -2 * (stats.binom.logpmf(breaches, n, level) - stats.binom.logpmf(breaches, n, breaches / n))
    return float(stats.chi2.sf(max(statistic, 0.0), df=1))


def backtest(adj_close, horizon=BACKTEST_HORIZON, n_sims=BACKTEST_SIMULATIONS, min_history=BACKTEST_MIN_HISTORY):
    """Check how often realized returns fall outside the forecast, without future data.

    At each non-overlapping origin, parameters use prior prices only. Drift has no
    regression tilt here, so this tests the risk model on its own.
    """
    log_returns = np.log(adj_close).diff().iloc[1:]
    rows = []

    for origin in range(min_history, len(log_returns) - horizon + 1, horizon):
        past = log_returns.iloc[:origin]
        future = log_returns.iloc[origin:origin + horizon]
        # A stock needs a full year of data and a complete forecast window.
        usable = past.iloc[-LOOKBACK_DAYS:].notna().all() & future.notna().all()
        tickers = usable[usable].index
        realized_stocks = np.exp(future[tickers].sum()).values - 1
        realized_portfolio = realized_stocks.mean()
        params = estimate_parameters(past[tickers].fillna(0.0))

        for model, model_params in [("Jump diffusion", params), ("GBM baseline", as_gbm(params))]:
            _, relatives = simulate(model_params, horizon, n_sims, seed=SEED + origin, sample_every=horizon)
            terminal = relatives[-1].astype(float) - 1
            lower, upper = np.percentile(terminal, [5, 95], axis=0)
            portfolio = relatives[-1].mean(axis=1).astype(float) - 1
            p_lower, p_upper = np.percentile(portfolio, [5, 95])
            date = future.index[0]

            rows.extend(
                {"Model": model, "Level": "Single stocks", "Origin": date, "Breach": r < lo, "Inside90": lo <= r <= hi}
                for r, lo, hi in zip(realized_stocks, lower, upper)
            )
            rows.append({
                "Model": model, "Level": "EW portfolio", "Origin": date,
                "Breach": realized_portfolio < p_lower, "Inside90": p_lower <= realized_portfolio <= p_upper,
            })

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["Model", "Level"])
        .agg(N=("Breach", "size"), N_Origins=("Origin", "nunique"), Breaches=("Breach", "sum"), Inside90=("Inside90", "mean"))
        .reset_index()
    )
    summary["BreachRate"] = summary["Breaches"] / summary["N"]
    summary["ExpectedRate"] = VAR_LEVEL

    intervals = [bootstrap_breach_interval(group) for _, group in detail.groupby(["Model", "Level"])]
    summary[["BreachRate_Low", "BreachRate_High"]] = intervals

    # Kupiec assumes independent observations. Stocks in the same window move together,
    # so the test is only valid for the portfolio, which has one observation per window.
    summary["KupiecP"] = [
        kupiec_pvalue(b, n) if level == "EW portfolio" else np.nan
        for b, n, level in zip(summary["Breaches"], summary["N"], summary["Level"])
    ]
    return summary


def bootstrap_breach_interval(detail, n_resamples=BOOTSTRAP_RESAMPLES, seed=SEED):
    """Return a 95% interval for the breach rate, resampling whole forecast windows.

    Resampling windows instead of single observations respects the fact that stocks
    tend to breach together in a market-wide selloff.
    """
    by_origin = detail.groupby("Origin")["Breach"].agg(["sum", "size"]).values.astype(float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(by_origin), (n_resamples, len(by_origin)))
    rates = by_origin[picks, 0].sum(axis=1) / by_origin[picks, 1].sum(axis=1)
    return np.percentile(rates, [2.5, 97.5])


# ── Main ──────────────────────────────────────────────────────────────────────

def load_tilts(db):
    """Load the regression signal, if one passed its out-of-sample check."""
    rows = pd.DataFrame(list(db["Regression_Latest_Signal"].find({}, {"_id": 0})))
    if rows.empty:
        print("No validated regression signal, so drift is pure CAPM.")
        return None
    print(f"Tilting drift with the '{rows['Spec'].iloc[0]}' signal.")
    annual = rows.set_index("Ticker")["PredictedExcessQuarterly"] * 4
    return annual.clip(-MAX_ANNUAL_TILT, MAX_ANNUAL_TILT)


def run_monte_carlo():
    db = get_database()
    history = load_history(db)
    adj_close = wide_prices(history, "AdjClose")

    latest = adj_close.iloc[-LOOKBACK_DAYS - 1:]
    tickers = latest.columns[latest.notna().all()]
    log_returns = np.log(adj_close[tickers]).diff().iloc[1:].fillna(0.0)
    last_prices = wide_prices(history, "Close")[tickers].iloc[-1]

    params = estimate_parameters(log_returns, load_tilts(db))
    print(f"Simulating {N_SIMULATIONS} paths x {N_DAYS} days x {len(tickers)} stocks...")
    sample_days, relatives = simulate(params, N_DAYS, N_SIMULATIONS)
    fan, terminal = summarize_simulation(params, last_prices, sample_days, relatives)

    jump_variance = params["jump_lambda"] * (params["jump_mean"] ** 2 + params["jump_std"] ** 2)
    market_variance = (
        params["market_lambda"] * params["beta"] ** 2
        * (params["market_jump_mean"] ** 2 + params["market_jump_std"] ** 2)
    )
    sigma_annual = np.sqrt(np.diag(params["diffusion_cov"]) * 252 + jump_variance + market_variance)
    parameters = pd.DataFrame({
        "Ticker": tickers,
        "LastPrice": last_prices.values,
        "LastDate": adj_close.index[-1],
        "Beta": params["beta"],
        "MuAnnual": params["mu_annual"],
        "SignalTiltAnnual": params["tilt_annual"],
        "SigmaAnnual": sigma_annual,
        "JumpFreq": params["jump_lambda"],
        "JumpMean": params["jump_mean"],
        "JumpStd": params["jump_std"],
        "MarketJumpFreq": params["market_lambda"],
        "SharpeRatio": (params["mu_annual"] - RISK_FREE_RATE) / sigma_annual,
    })

    print("\nBacktesting VaR on past data...")
    backtest_summary = backtest(adj_close)
    print(backtest_summary.to_string(index=False))

    print("\nOne-year equal-weight portfolio:")
    print(terminal[terminal["Ticker"] == "PORTFOLIO_EW"].T.to_string(header=False))

    tables = {"Parameters": parameters, "FanChart": fan, "Terminal": terminal, "Backtest": backtest_summary}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name, table in tables.items():
        replace_collection(db, f"MC_{name}", table)
        table.to_csv(os.path.join(OUTPUT_DIR, f"mc_{name.lower()}.csv"), index=False)
    print(f"Monte Carlo results exported to {OUTPUT_DIR}/")


if __name__ == "__main__":
    run_monte_carlo()
