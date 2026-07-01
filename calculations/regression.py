import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from loadData.db import get_database


# ── Config ────────────────────────────────────────────────────────────────────

FEATURES = [
    "GrossMargin",
    "OperatingMargin",
    "DebtToAssets",
    "RevenueGrowth",
    "EPS_Surprise",
    "Beta",
    "MarketCap",
    "FreeCashFlow",
    "Momentum_3M",
    "Momentum_6M",
    "Volatility_3M",
    "PE_Trailing",
    "PB",
]

RETURN_HORIZON = 63
MOMENTUM_3M    = 63
MOMENTUM_6M    = 126
VOLATILITY_WIN = 63


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(db):
    quarterly    = pd.DataFrame(list(db["CompanyQuarterlyData"].find()))
    fundamentals = pd.DataFrame(list(db["Fundamentals"].find()))
    history      = pd.DataFrame(list(db["Stock_History"].find()))

    for df in [quarterly, fundamentals, history]:
        df.drop(columns=["_id"], errors="ignore", inplace=True)

    # ── Fundamental features ──
    quarterly = quarterly.sort_values(["Ticker", "QuarterDate"])

    quarterly["RevenueGrowth"] = (
        quarterly.groupby("Ticker")["Revenue"]
        .pct_change(fill_method=None)
    )
    quarterly["EPS_Surprise"] = (
        quarterly.groupby("Ticker")["EPS_Trailing"]
        .pct_change(fill_method=None)
    )
    quarterly["DebtToAssets"] = (
        quarterly["TotalDebt"] / quarterly["TotalAssets"]
    ).replace([np.inf, -np.inf], np.nan)

    # ── Price-based features ──
    history["Date"] = pd.to_datetime(history["Date"])
    history = history.sort_values(["Ticker", "Date"])
    history["DailyReturn"] = history.groupby("Ticker")["Close"].pct_change(fill_method=None)

    history["ForwardReturn"] = (
        history.groupby("Ticker")["Close"]
        .transform(lambda x: x.shift(-RETURN_HORIZON) / x - 1)
    )
    history["Momentum_3M"] = (
        history.groupby("Ticker")["Close"]
        .transform(lambda x: x / x.shift(MOMENTUM_3M) - 1)
    )
    history["Momentum_6M"] = (
        history.groupby("Ticker")["Close"]
        .transform(lambda x: x / x.shift(MOMENTUM_6M) - 1)
    )
    history["Volatility_3M"] = (
        history.groupby("Ticker")["DailyReturn"]
        .transform(lambda x: x.rolling(VOLATILITY_WIN).std())
    )
    history = history.rename(columns={"P_E_Trailing": "PE_Trailing", "P_B": "PB"})

    # ── Snapshot price features at each quarter end ──
    quarterly["QuarterDate"] = pd.to_datetime(quarterly["QuarterDate"])
    price_cols = ["ForwardReturn", "Momentum_3M", "Momentum_6M",
                  "Volatility_3M", "PE_Trailing", "PB"]

    def get_price_snapshot(row):
        ticker_hist = history[history["Ticker"] == row["Ticker"]]
        if ticker_hist.empty:
            return pd.Series({c: None for c in price_cols})
        quarter_date = pd.to_datetime(row["QuarterDate"]).tz_localize(None)
        diff = (ticker_hist["Date"] - quarter_date).abs()
        if diff.dropna().empty:
            return pd.Series({c: None for c in price_cols})
        idx = diff.idxmin()
        return ticker_hist.loc[idx, price_cols]

    print("Snapshotting price features per quarter...")
    price_snapshot = quarterly.apply(get_price_snapshot, axis=1)
    quarterly = pd.concat([quarterly, price_snapshot], axis=1)

    df = quarterly.merge(
        fundamentals[["Ticker", "Beta", "MarketCap", "Sector"]],
        on="Ticker",
        how="left"
    )

    return df


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset(df):
    df = df.reset_index(drop=True)

    available = [c for c in FEATURES + ["ForwardReturn", "Ticker", "Quarter", "Sector", "QuarterDate"]
                 if c in df.columns]
    df = df[available].copy()

    required = ["GrossMargin", "OperatingMargin", "Beta", "Volatility_3M", "ForwardReturn"]
    required = [c for c in required if c in df.columns]
    df = df.dropna(subset=required)

    # Fill non-required features with global median
    for col in FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    df = df.dropna(subset=[c for c in FEATURES if c in df.columns] + ["ForwardReturn"])

    # Clip outliers
    for col in [c for c in FEATURES if c in df.columns] + ["ForwardReturn"]:
        lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(lo, hi)

    df = df.sort_values("QuarterDate").reset_index(drop=True)

    X    = df[FEATURES].reset_index(drop=True)
    y    = df["ForwardReturn"].reset_index(drop=True)
    meta = df[["Ticker", "Quarter", "Sector", "QuarterDate"]].reset_index(drop=True)

    return X, y, meta


# ── Model 1: OLS ──────────────────────────────────────────────────────────────

def run_ols(X, y):
    X_const = sm.add_constant(X)
    model   = sm.OLS(y, X_const).fit()
    print(model.summary())
    return model


def ols_results_for_mongo(model, feature_names):
    params  = model.params[1:]
    pvalues = model.pvalues[1:]
    conf    = model.conf_int().iloc[1:]

    results = pd.DataFrame({
        "Feature":     feature_names,
        "Coefficient": params.values,
        "PValue":      pvalues.values,
        "CI_Lower":    conf[0].values,
        "CI_Upper":    conf[1].values,
        "Significant": (pvalues < 0.05).values,
        "R2":          model.rsquared,
        "AdjR2":       model.rsquared_adj,
    })
    return results


# ── Model 2: Fama-MacBeth ─────────────────────────────────────────────────────

def run_fama_macbeth(X, y, meta):
    """
    Fama-MacBeth regression:
    1. For each quarter, run a cross-sectional OLS across all stocks
    2. Collect the coefficients from each quarter
    3. Average them and t-test whether they're significantly different from zero

    This is the correct approach for panel data with few time periods.
    Each quarterly coefficient tells you: in that quarter, which fundamentals
    separated winners from losers?
    """
    df = X.copy()
    df["ForwardReturn"] = y.values
    df["Quarter"]       = meta["Quarter"].values
    df["Ticker"]        = meta["Ticker"].values

    quarters = sorted(df["Quarter"].unique())
    quarterly_coefs = []

    print(f"\nRunning cross-sectional regression for {len(quarters)} quarters...")

    for quarter in quarters:
        q_df = df[df["Quarter"] == quarter].copy()

        if len(q_df) < len(FEATURES) + 2:
            print(f"  Skipping {quarter} — too few stocks ({len(q_df)})")
            continue

        X_q = q_df[FEATURES].dropna()
        y_q = q_df.loc[X_q.index, "ForwardReturn"]

        if len(X_q) < len(FEATURES) + 2:
            continue

        try:
            model = sm.OLS(y_q, sm.add_constant(X_q)).fit()
            coef_row = {"Quarter": quarter, "R2": model.rsquared, "N": len(y_q)}
            for feat in FEATURES:
                coef_row[feat] = model.params.get(feat, np.nan)
            quarterly_coefs.append(coef_row)
            print(f"  {quarter}: R²={model.rsquared:.3f}, N={len(y_q)}")
        except Exception as e:
            print(f"  Skipping {quarter}: {e}")

    if not quarterly_coefs:
        print("❌ No quarters had enough data for Fama-MacBeth.")
        return None, None, None

    coef_df = pd.DataFrame(quarterly_coefs)

    # ── Step 2: Average coefficients and t-test ──
    summary_rows = []
    for feat in FEATURES:
        if feat not in coef_df.columns:
            continue
        vals = coef_df[feat].dropna()
        if len(vals) < 2:
            continue

        mean_coef = vals.mean()
        std_coef  = vals.std()
        n         = len(vals)
        t_stat    = mean_coef / (std_coef / np.sqrt(n))
        p_value   = 2 * (1 - sm.stats.stattools.durbin_watson([0]))  # placeholder
        # Proper p-value using t-distribution
        from scipy import stats
        p_value = 2 * stats.t.sf(abs(t_stat), df=n - 1)

        summary_rows.append({
            "Feature":     feat,
            "MeanCoef":    mean_coef,
            "StdCoef":     std_coef,
            "T_Stat":      t_stat,
            "PValue":      p_value,
            "Significant": p_value < 0.05,
            "N_Quarters":  n,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("T_Stat", key=abs, ascending=False)

    print("\n── Fama-MacBeth Summary ──")
    print(summary_df[["Feature", "MeanCoef", "T_Stat", "PValue", "Significant"]].to_string(index=False))

    return coef_df, summary_df


# ── Model 3: Sector OLS ───────────────────────────────────────────────────────

def run_sector_regressions(X, y, meta):
    results = []
    df = X.copy()
    df["ForwardReturn"] = y.values
    df["Sector"]        = meta["Sector"].values

    for sector, group in df.groupby("Sector"):
        if len(group) < 10:
            continue
        X_s = group[FEATURES].dropna()
        y_s = group.loc[X_s.index, "ForwardReturn"]
        if X_s.empty:
            continue
        try:
            model = sm.OLS(y_s, sm.add_constant(X_s)).fit()
            best  = FEATURES[np.argmax(np.abs(model.params[1:].values))]
            results.append({
                "Sector":        sector,
                "R2":            model.rsquared,
                "AdjR2":         model.rsquared_adj,
                "N":             len(y_s),
                "BestPredictor": best,
            })
        except Exception as e:
            print(f"Skipping {sector}: {e}")

    return pd.DataFrame(results).sort_values("R2", ascending=False)


# ── Save to MongoDB ───────────────────────────────────────────────────────────

def save_to_mongo(db, collection_name, df):
    col = db[collection_name]
    col.drop()
    records = df.copy()
    # Convert bool columns to plain Python bool for MongoDB
    for c in records.select_dtypes(include="bool").columns:
        records[c] = records[c].astype(bool)
    records = records.to_dict("records")
    if records:
        col.insert_many(records)
    print(f"✅ Saved {len(records)} rows to MongoDB:{collection_name}")


def export_csvs(ols_coef, ols_pred, fm_quarterly, fm_summary, sector_results):
    os.makedirs("output", exist_ok=True)
    ols_coef.to_csv("output/ols_coefficients.csv",       index=False)
    ols_pred.to_csv("output/ols_predictions.csv",        index=False)
    fm_quarterly.to_csv("output/fm_quarterly_coefs.csv", index=False)
    fm_summary.to_csv("output/fm_summary.csv",           index=False)
    sector_results.to_csv("output/sector_r2.csv",        index=False)
    print("✅ CSVs exported to output/")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_regression():
    db = get_database()

    print("Loading data...")
    df = load_data(db)

    print(f"Dataset size before cleaning: {len(df)}")
    X, y, meta = build_dataset(df)
    print(f"Dataset size after cleaning:  {len(X)}")

    if len(X) < 20:
        print("❌ Not enough data. Run UPDATE_MODE 2 and 3 first.")
        return

    # ── OLS ──
    print("\n── OLS Regression ──")
    ols_model = run_ols(X, y)
    ols_coef  = ols_results_for_mongo(ols_model, FEATURES)

    X_const   = sm.add_constant(X)
    ols_pred  = meta.copy()
    ols_pred["PredictedReturn"] = ols_model.predict(X_const)
    ols_pred["ActualReturn"]    = y.values
    ols_pred["Residual"]        = ols_pred["ActualReturn"] - ols_pred["PredictedReturn"]
    ols_pred["QuarterDate"]     = ols_pred["QuarterDate"].astype(str)

    # ── Fama-MacBeth ──
    print("\n── Fama-MacBeth Regression ──")
    fm_quarterly, fm_summary = run_fama_macbeth(X, y, meta)

    # ── Sector ──
    print("\n── Sector Regressions ──")
    sector_results = run_sector_regressions(X, y, meta)
    print(sector_results)

    # ── Save to MongoDB ──
    save_to_mongo(db, "Regression_OLS_Coefficients",  ols_coef)
    save_to_mongo(db, "Regression_OLS_Predictions",   ols_pred)
    save_to_mongo(db, "Regression_FM_QuarterlyCoefs", fm_quarterly)
    save_to_mongo(db, "Regression_FM_Summary",        fm_summary)
    save_to_mongo(db, "Regression_Sector_R2",         sector_results)

    # ── Export CSV as backup ──
    if fm_quarterly is not None and fm_summary is not None:
        export_csvs(ols_coef, ols_pred, fm_quarterly, fm_summary, sector_results)

    print("\n✅ Done.")
    print("MongoDB collections written:")
    print("  Regression_OLS_Coefficients")
    print("  Regression_OLS_Predictions")
    print("  Regression_FM_QuarterlyCoefs  ← coefficient per quarter")
    print("  Regression_FM_Summary         ← averaged + t-tested")
    print("  Regression_Sector_R2")


if __name__ == "__main__":
    run_regression()