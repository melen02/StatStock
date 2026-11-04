"""Cross-sectional return regressions on point-in-time data.

Every feature in a panel row was knowable on that row's formation date:

* Formation dates are shared by all stocks (last trading day of Feb, May, Aug and
  Nov), so each cross-section compares returns over the same window.
* A quarter's fundamentals only count once REPORT_LAG_DAYS have passed.
* Ratios are computed from the price on the formation date, not today's values.
* Features are scaled within each date, so nothing leaks in from other periods.
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

from loadData.db import OUTPUT_DIR, get_database
from loadData.market_data import load_history, wide_prices
from loadData.mongo_helpers import replace_collection
from loadData.tickers import TICKERS

# ── Config ────────────────────────────────────────────────────────────────────

FUNDAMENTAL_FEATURES = [
    "GrossMargin",
    "OperatingMargin",
    "DebtToAssets",
    "RevenueGrowth",
    "EarningsChange",
    "FCFYield",
    "EarningsYield",
    "SalesToPrice",
]
PRICE_FEATURES = [
    "Beta",
    "LogMarketCap",
    "Momentum_3M",
    "Momentum_6M",
    "Volatility_3M",
]

# Yahoo only serves about six quarters of fundamentals, while prices go back to 2020.
# The price-only model therefore has far more cross-sections to test on.
SPECS = {
    "full": PRICE_FEATURES + FUNDAMENTAL_FEATURES,
    "price_only": PRICE_FEATURES,
}

FORMATION_MONTHS = (2, 5, 8, 11)
REPORT_LAG_DAYS = 45       # time allowed for a quarter's results to become public
MAX_FUNDAMENTAL_AGE = 200  # days before a report is too old to use
MAX_QUARTER_GAP = 100      # growth is only computed between adjacent quarters
TRADING_DAYS_3M = 63
TRADING_DAYS_6M = 126
BETA_WINDOW = 252

WINSOR_LIMITS = (0.01, 0.99)
MIN_FEATURE_COVERAGE = 0.5   # skip a date where a feature exists for fewer stocks
MIN_TRAIN_WINDOWS = 3        # cross-sections required before a walk-forward prediction
MIN_SIGNAL_WINDOWS = 8       # walk-forward windows required to trust a signal
SIGNAL_MIN_IC_T = 2.0        # IC t-statistic required to trust a signal
MIN_OBS_PER_FEATURE = 5      # sector regressions with fewer are not worth reporting


# ── Features ──────────────────────────────────────────────────────────────────

def formation_dates(index):
    """Return the last trading day of each formation month, skipping an unfinished month."""
    dates = pd.Series(index, index=index)
    periods = index.to_period("M")
    month_ends = dates.groupby(periods).max()
    month_ends = month_ends[month_ends.index < periods.max()]
    return pd.DatetimeIndex(month_ends[month_ends.index.month.isin(FORMATION_MONTHS)].values)


def compute_price_features(adj_close, close, market_cap):
    """Return a dict of dates-by-tickers tables, each using only data up to that date."""
    returns = adj_close.pct_change(fill_method=None)
    market = returns.mean(axis=1)

    # Today's share count scaled by historical price. Ignores buybacks and issuance,
    # because Yahoo's share history is too patchy to use across the whole universe.
    shares = market_cap.reindex(close.columns) / close.ffill().iloc[-1]

    min_periods = BETA_WINDOW // 2
    market_var = market.rolling(BETA_WINDOW, min_periods=min_periods).var()
    beta = returns.rolling(BETA_WINDOW, min_periods=min_periods).cov(market).div(market_var, axis=0)

    return {
        "Momentum_3M": adj_close / adj_close.shift(TRADING_DAYS_3M) - 1,
        "Momentum_6M": adj_close / adj_close.shift(TRADING_DAYS_6M) - 1,
        "Volatility_3M": returns.rolling(TRADING_DAYS_3M, min_periods=TRADING_DAYS_3M - 10).std(),
        "Beta": beta,
        "MarketCapValue": close * shares,
    }


def compute_fundamental_features(quarterly):
    """Add per-quarter features and the date on which each report counts as public."""
    q = quarterly.copy()
    quarter_ends = pd.Series(pd.PeriodIndex(q["Quarter"], freq="Q").end_time.normalize(), index=q.index)
    q["QuarterDate"] = pd.to_datetime(q["QuarterDate"]).fillna(quarter_ends)
    q = q.dropna(subset=["Revenue"]).sort_values(["Ticker", "QuarterDate"]).reset_index(drop=True)

    by_ticker = q.groupby("Ticker")
    adjacent = by_ticker["QuarterDate"].diff().dt.days <= MAX_QUARTER_GAP
    previous_revenue = by_ticker["Revenue"].shift(1)
    previous_assets = by_ticker["TotalAssets"].shift(1)

    # Quarter-over-quarter, so seasonal businesses add noise. Year-over-year would
    # cost four of the few quarters we have.
    q["RevenueGrowth"] = (q["Revenue"] / previous_revenue - 1).where(adjacent & (previous_revenue > 0))

    # Scaled by assets because a percentage change is meaningless around zero or
    # negative earnings. This is earnings change, not a surprise against analyst
    # expectations: Yahoo's estimate history stops too early to use here.
    q["EarningsChange"] = (by_ticker["NetIncome"].diff() / previous_assets).where(adjacent & (previous_assets > 0))

    q["DebtToAssets"] = (q["TotalDebt"] / q["TotalAssets"]).replace([np.inf, -np.inf], np.nan)

    # Annualised from up to four quarters, to divide by market value on the formation date.
    for column, target in [("NetIncome", "AnnualEarnings"), ("Revenue", "AnnualSales"), ("FreeCashFlow", "AnnualFCF")]:
        q[target] = by_ticker[column].transform(lambda s: s.rolling(4, min_periods=2).mean() * 4)

    q["AvailableDate"] = q["QuarterDate"] + pd.Timedelta(days=REPORT_LAG_DAYS)
    return q


def build_panel(quarterly, fundamentals, history):
    """Build one row per stock and formation date, with features and forward return.

    The newest date has a missing ForwardReturn. It is kept so that a current signal
    can be computed, and estimation functions drop it.
    """
    adj_close = wide_prices(history, "AdjClose")
    close = wide_prices(history, "Close")
    market_cap = fundamentals.set_index("Ticker")["MarketCap"].astype(float)
    price_features = compute_price_features(adj_close, close, market_cap)

    dates = formation_dates(adj_close.index)
    # Forward returns run to the next formation date, so windows never overlap.
    prices_at_formation = adj_close.loc[dates]
    forward_return = prices_at_formation.shift(-1) / prices_at_formation - 1

    tables = {name: table.loc[dates] for name, table in price_features.items()}
    tables["ForwardReturn"] = forward_return
    panel = pd.concat(
        {name: table.stack(future_stack=True) for name, table in tables.items()}, axis=1
    ).rename_axis(["Date", "Ticker"]).reset_index()
    panel = panel.dropna(subset=["MarketCapValue"])

    q = compute_fundamental_features(quarterly)
    panel = pd.merge_asof(
        panel.sort_values("Date"),
        q.sort_values("AvailableDate"),
        left_on="Date",
        right_on="AvailableDate",
        by="Ticker",
        direction="backward",
        tolerance=pd.Timedelta(days=MAX_FUNDAMENTAL_AGE),
    )

    value = panel["MarketCapValue"].where(panel["MarketCapValue"] > 0)
    panel["LogMarketCap"] = np.log(value)
    panel["EarningsYield"] = panel["AnnualEarnings"] / value
    panel["SalesToPrice"] = panel["AnnualSales"] / value
    panel["FCFYield"] = panel["AnnualFCF"] / value

    sectors = fundamentals.set_index("Ticker")["Sector"]
    panel["Sector"] = panel["Ticker"].map(sectors)
    return panel.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def standardize(panel, features):
    """Winsorize and z-score each feature within its date, then set missing values to 0.

    A coefficient then reads as return per one cross-sectional standard deviation,
    so features become comparable. Zero means "average", which is how banks are kept
    in despite having no gross margin. Dates where a feature is mostly missing are
    dropped, since that feature would carry no information there.
    """
    coverage = panel.groupby("Date")[features].apply(lambda g: g.notna().mean().min())
    usable = coverage[coverage >= MIN_FEATURE_COVERAGE].index
    out = panel[panel["Date"].isin(usable)].copy()

    def zscore(s):
        clipped = s.clip(s.quantile(WINSOR_LIMITS[0]), s.quantile(WINSOR_LIMITS[1]))
        std = clipped.std()
        return (clipped - clipped.mean()) / std if std > 0 else clipped * 0.0

    for feature in features:
        out[feature] = out[feature].replace([np.inf, -np.inf], np.nan)
        out[feature] = out.groupby("Date")[feature].transform(zscore).fillna(0.0)
    return out


def estimation_rows(panel):
    return panel.dropna(subset=["ForwardReturn"])


# ── Model 1: pooled OLS ───────────────────────────────────────────────────────

def run_pooled_ols(panel, features):
    """Regress returns in excess of the date's average, clustering errors by date.

    Removing the date average takes out market-wide moves, and clustering allows
    stocks to stay correlated within a date. Using t(G-1) keeps the inference honest
    when there are few dates.
    """
    data = estimation_rows(panel)
    excess = data["ForwardReturn"] - data.groupby("Date")["ForwardReturn"].transform("mean")
    X = data[features]
    model = sm.OLS(excess, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["Date"].factorize()[0]},
        use_t=True,
    )
    conf = model.conf_int()
    X_const = sm.add_constant(X).values

    coefficients = pd.DataFrame({
        "Feature": features,
        "Coefficient": model.params.values,
        "PValue": model.pvalues.values,
        "CI_Lower": conf[0].values,
        "CI_Upper": conf[1].values,
        "Significant": model.pvalues.values < 0.05,
        "VIF": [variance_inflation_factor(X_const, i + 1) for i in range(len(features))],
        "R2": model.rsquared,
        "AdjR2": model.rsquared_adj,
        "N": int(model.nobs),
        "N_Dates": data["Date"].nunique(),
    })

    predictions = data[["Ticker", "Date", "Sector"]].copy()
    predictions["PredictedExcess"] = model.fittedvalues
    predictions["ActualExcess"] = excess
    predictions["Residual"] = model.resid
    return coefficients, predictions


# ── Model 2: Fama-MacBeth ─────────────────────────────────────────────────────

def cross_section_coefficients(panel, features):
    """Fit one cross-sectional OLS per date. Dates with too few stocks are skipped."""
    rows = []
    for date, group in estimation_rows(panel).groupby("Date"):
        if len(group) < 2 * len(features):
            print(f"  Skipping {date.date()}: only {len(group)} stocks")
            continue
        model = sm.OLS(group["ForwardReturn"], sm.add_constant(group[features])).fit()
        rows.append({"Date": date, "R2": model.rsquared, "N": len(group), **model.params[features].to_dict()})
    return pd.DataFrame(rows, columns=["Date", "R2", "N"] + features)


def run_fama_macbeth(panel, features):
    """Average the per-date coefficients and t-test them over time.

    Returns (per-date coefficients, summary). The summary is empty if fewer than two
    dates have enough stocks. Windows don't overlap, so no autocorrelation correction
    is applied.
    """
    coefs = cross_section_coefficients(panel, features)
    if len(coefs) < 2:
        columns = ["Feature", "MeanCoef", "StdCoef", "T_Stat", "PValue", "QValue", "CI_Lower", "CI_Upper", "N_Dates", "Significant"]
        return coefs, pd.DataFrame(columns=columns)

    n = len(coefs)
    mean, std = coefs[features].mean(), coefs[features].std()
    standard_error = std / np.sqrt(n)
    t_stat = mean / standard_error
    p_value = 2 * stats.t.sf(t_stat.abs(), df=n - 1)
    margin = stats.t.ppf(0.975, n - 1) * standard_error

    summary = pd.DataFrame({
        "Feature": features,
        "MeanCoef": mean.values,
        "StdCoef": std.values,
        "T_Stat": t_stat.values,
        "PValue": p_value,
        # Benjamini-Hochberg: with many features, some look significant by chance.
        "QValue": multipletests(p_value, method="fdr_bh")[1],
        "CI_Lower": (mean - margin).values,
        "CI_Upper": (mean + margin).values,
        "N_Dates": n,
    })
    summary["Significant"] = summary["QValue"] < 0.05
    return coefs, summary.sort_values("T_Stat", key=np.abs, ascending=False).reset_index(drop=True)


# ── Out-of-sample evaluation ──────────────────────────────────────────────────

def walk_forward(panel, features, min_train=MIN_TRAIN_WINDOWS):
    """Predict each cross-section with coefficients from earlier windows only.

    A window's return is only known at the next formation date, and the coefficients
    used at a date come from windows that had ended by then.
    """
    coefs = cross_section_coefficients(panel, features).sort_values("Date").reset_index(drop=True)
    data = estimation_rows(panel)
    rows = []

    for i in range(min_train, len(coefs)):
        date = coefs.loc[i, "Date"]
        beta = coefs.loc[: i - 1, features].mean()
        group = data[data["Date"] == date]
        actual = group["ForwardReturn"] - group["ForwardReturn"].mean()
        predicted = group[features].mul(beta).sum(axis=1)

        top = actual[predicted >= predicted.quantile(0.8)].mean()
        bottom = actual[predicted <= predicted.quantile(0.2)].mean()
        rows.append({
            "Date": date,
            "N": len(group),
            "RankIC": stats.spearmanr(predicted, actual)[0],
            "QuintileSpread": top - bottom,
            "SSE_Model": ((actual - predicted) ** 2).sum(),
            "SSE_Zero": (actual ** 2).sum(),
        })

    return pd.DataFrame(rows, columns=["Date", "N", "RankIC", "QuintileSpread", "SSE_Model", "SSE_Zero"])


def summarize_walk_forward(windows, in_sample_r2):
    """Summarize a walk-forward run. OOS R² compares against predicting no excess return."""
    n = len(windows)
    if n < 2:
        return {"N_Windows": n}

    def t_stat(s):
        return s.mean() / (s.std() / np.sqrt(n)) if s.std() > 0 else np.nan

    return {
        "N_Windows": n,
        "InSample_R2": in_sample_r2,
        "OOS_R2": 1 - windows["SSE_Model"].sum() / windows["SSE_Zero"].sum(),
        "MeanRankIC": windows["RankIC"].mean(),
        "RankIC_T": t_stat(windows["RankIC"]),
        "IC_HitRate": (windows["RankIC"] > 0).mean(),
        "MeanQuintileSpread": windows["QuintileSpread"].mean(),
        "Spread_T": t_stat(windows["QuintileSpread"]),
    }


def signal_is_trustworthy(oos_summary):
    return (
        oos_summary.get("N_Windows", 0) >= MIN_SIGNAL_WINDOWS
        and oos_summary.get("RankIC_T", 0) >= SIGNAL_MIN_IC_T
    )


def latest_signal(panel, features, fm_summary):
    """Predict next-window excess return for the newest date, on the training scale."""
    latest = panel[panel["Date"] == panel["Date"].max()]
    coefficients = fm_summary.set_index("Feature")["MeanCoef"].reindex(features)
    return pd.DataFrame({
        "Ticker": latest["Ticker"],
        "Date": latest["Date"],
        "PredictedExcessQuarterly": latest[features].mul(coefficients).sum(axis=1),
    })


# ── Model 3: sector regressions ───────────────────────────────────────────────

def run_sector_regressions(panel, features):
    """Fit one regression per sector, skipping sectors with too little data to mean anything."""
    data = estimation_rows(panel)
    data = data.assign(Excess=data["ForwardReturn"] - data.groupby("Date")["ForwardReturn"].transform("mean"))
    results = []

    for sector, group in data.groupby("Sector"):
        if len(group) < MIN_OBS_PER_FEATURE * len(features):
            print(f"  Skipping {sector}: {len(group)} observations for {len(features)} features")
            continue
        model = sm.OLS(group["Excess"], sm.add_constant(group[features])).fit()
        t_values = model.tvalues[features]
        strongest = t_values.abs().idxmax()
        results.append({
            "Sector": sector,
            "R2": model.rsquared,
            "AdjR2": model.rsquared_adj,
            "N": len(group),
            "StrongestFeature": strongest,
            "StrongestT": t_values[strongest],
        })

    columns = ["Sector", "R2", "AdjR2", "N", "StrongestFeature", "StrongestT"]
    return pd.DataFrame(results, columns=columns).sort_values("AdjR2", ascending=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def load_inputs(db):
    def read(name):
        return pd.DataFrame(list(db[name].find({"Ticker": {"$in": TICKERS}}, {"_id": 0})))

    quarterly, fundamentals = read("CompanyQuarterlyData"), read("Fundamentals")
    if quarterly.empty or fundamentals.empty:
        raise RuntimeError("Fundamental data is missing. Run: python main.py fetch")
    return quarterly, fundamentals, load_history(db)


def run_regression():
    db = get_database()
    quarterly, fundamentals, history = load_inputs(db)
    raw_panel = build_panel(quarterly, fundamentals, history)

    outputs = {
        name: []
        for name in ["OLS_Coefficients", "OLS_Predictions", "FM_QuarterlyCoefs", "FM_Summary", "OOS_Windows", "OOS_Summary"]
    }
    oos_by_spec, summaries, panels = {}, {}, {}

    for spec, features in SPECS.items():
        panel = standardize(raw_panel, features)
        data = estimation_rows(panel)
        print(f"\n── Spec '{spec}': {len(data)} stock-quarters, {data['Date'].nunique()} dates, {len(features)} features ──")

        ols_coef, ols_pred = run_pooled_ols(panel, features)
        fm_coefs, fm_summary = run_fama_macbeth(panel, features)
        windows = walk_forward(panel, features)
        oos = summarize_walk_forward(windows, fm_coefs["R2"].mean())

        print("Fama-MacBeth (return per 1 SD of the feature):")
        print(fm_summary[["Feature", "MeanCoef", "T_Stat", "PValue", "QValue"]].to_string(index=False))
        print("Walk-forward:", {k: round(float(v), 4) for k, v in oos.items()})

        for name, frame in [
            ("OLS_Coefficients", ols_coef), ("OLS_Predictions", ols_pred), ("FM_QuarterlyCoefs", fm_coefs),
            ("FM_Summary", fm_summary), ("OOS_Windows", windows), ("OOS_Summary", pd.DataFrame([oos])),
        ]:
            outputs[name].append(frame.assign(Spec=spec))
        oos_by_spec[spec], summaries[spec], panels[spec] = oos, fm_summary, panel

    print("\n── Sector regressions (full spec) ──")
    sectors = run_sector_regressions(panels["full"], SPECS["full"])
    print(sectors.to_string(index=False))

    # Monte Carlo only uses a signal that has held up out of sample.
    signal = pd.DataFrame(columns=["Ticker", "Date", "PredictedExcessQuarterly", "Spec"])
    for spec in SPECS:
        if signal_is_trustworthy(oos_by_spec[spec]):
            signal = latest_signal(panels[spec], SPECS[spec], summaries[spec]).assign(Spec=spec)
            print(f"\nSpec '{spec}' passed the out-of-sample check, so it will tilt Monte Carlo drift.")
            break
    else:
        print("\nNo spec passed the out-of-sample check, so Monte Carlo drift will not be tilted.")

    tables = {name: pd.concat(frames, ignore_index=True) for name, frames in outputs.items()}
    tables["Sector_R2"] = sectors
    tables["Latest_Signal"] = signal

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name, table in tables.items():
        replace_collection(db, f"Regression_{name}", table)
        table.to_csv(os.path.join(OUTPUT_DIR, f"regression_{name.lower()}.csv"), index=False)
    print(f"Regression results exported to {OUTPUT_DIR}/")


if __name__ == "__main__":
    run_regression()
