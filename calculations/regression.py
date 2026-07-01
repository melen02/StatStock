import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm
from loadData.db import get_database


# ── Config ────────────────────────────────────────────────────────────────────

FEATURES = [
    # Fundamentals
    "GrossMargin",
    "OperatingMargin",
    "DebtToAssets",         # TotalDebt / TotalAssets (new)
    # "CashRatio",            # TotalCash / TotalDebt (new)
    "RevenueGrowth",        # QoQ revenue growth (new)
    "EPS_Surprise",         # EPS change vs previous quarter (new)
    "Beta",
    "MarketCap",
    "FreeCashFlow",
    # Price-based
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

    # ── New fundamental features ──
    quarterly = quarterly.sort_values(["Ticker", "QuarterDate"])

    # Revenue growth QoQ
    quarterly["RevenueGrowth"] = (
        quarterly.groupby("Ticker")["Revenue"]
        .pct_change(fill_method=None)
    )

    # EPS surprise: change in EPS vs previous quarter
    quarterly["EPS_Surprise"] = (
        quarterly.groupby("Ticker")["EPS_Trailing"]
        .pct_change(fill_method=None)
    )

    # Debt to assets
    quarterly["DebtToAssets"] = (
        quarterly["TotalDebt"] / quarterly["TotalAssets"]
    ).replace([np.inf, -np.inf], np.nan)

    # Cash ratio
    quarterly["CashRatio"] = (
        quarterly["TotalCash"] / quarterly["TotalDebt"]
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

    # ── Merge fundamentals ──
    df = quarterly.merge(
        fundamentals[["Ticker", "Beta", "MarketCap", "Sector"]],
        on="Ticker",
        how="left"
    )

    return df


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset(df):
    # Reset index first to avoid concat index issues
    df = df.reset_index(drop=True)

    # Only keep columns that actually exist
    available = [c for c in FEATURES + ["ForwardReturn", "Ticker", "Quarter", "Sector", "QuarterDate"] if c in df.columns]
    df = df[available].copy()

    # These must exist — drop rows where they're missing
    required = ["GrossMargin", "OperatingMargin", "Beta",
                "Volatility_3M", "ForwardReturn"]
    required = [c for c in required if c in df.columns]
    df = df.dropna(subset=required)
    print("NaN counts after required dropna:")
    for col in FEATURES:
        if col in df.columns:
            n = df[col].isna().sum()
            print(f"  {col}: {n} / {len(df)}")

    print(f"After required dropna: {len(df)} rows")

    # Fill remaining missing features with global median (skip sector groupby)
    for col in FEATURES:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    df = df.dropna(subset=[c for c in FEATURES if c in df.columns] + ["ForwardReturn"])

    print(f"After full dropna: {len(df)} rows")

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


def ols_results_for_powerbi(model, feature_names):
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
    })
    results["R2"]    = model.rsquared
    results["AdjR2"] = model.rsquared_adj
    return results


# ── Model 2: Ridge with time series CV ───────────────────────────────────────

def run_ridge_cv(X, y):
    """
    Ridge regression with time series cross-validation.
    Trains on older quarters, tests on newer ones — more honest than in-sample R².
    """
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=5)

    cv_r2_scores = []
    for train_idx, test_idx in tscv.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        cv_r2_scores.append(score)

    mean_cv_r2 = np.mean(cv_r2_scores)
    print(f"Ridge CV R² scores per fold: {[round(s, 3) for s in cv_r2_scores]}")
    print(f"Mean out-of-sample R²: {mean_cv_r2:.3f}")

    # Refit on all data for coefficients
    final_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    final_model.fit(X_scaled, y)

    coef_df = pd.DataFrame({
        "Feature":     FEATURES,
        "Coefficient": final_model.coef_,
        "MeanCV_R2":   mean_cv_r2,
    }).sort_values("Coefficient", key=abs, ascending=False)

    return final_model, scaler, coef_df, cv_r2_scores


def ridge_cv_summary_for_powerbi(cv_r2_scores):
    return pd.DataFrame({
        "Fold":  [f"Fold {i+1}" for i in range(len(cv_r2_scores))],
        "CV_R2": cv_r2_scores,
    })


# ── Model 3: Logistic ─────────────────────────────────────────────────────────

def run_logistic(X, y, market_return=0.0):
    y_binary = (y > market_return).astype(int)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model    = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_scaled, y_binary)

    # Time series CV accuracy
    tscv = TimeSeriesSplit(n_splits=5)
    cv_acc = []
    for train_idx, test_idx in tscv.split(X_scaled):
        m = LogisticRegression(max_iter=1000, class_weight="balanced")
        m.fit(X_scaled[train_idx], y_binary.iloc[train_idx])
        cv_acc.append(m.score(X_scaled[test_idx], y_binary.iloc[test_idx]))

    print(f"Logistic CV accuracy per fold: {[round(a, 3) for a in cv_acc]}")
    print(f"Mean out-of-sample accuracy:   {np.mean(cv_acc):.3f}")

    return model, scaler, y_binary, cv_acc


def logistic_results_for_powerbi(model, scaler, X, y_binary, meta, feature_names, cv_acc):
    X_scaled = scaler.transform(X)

    coef_df = pd.DataFrame({
        "Feature":    feature_names,
        "Coefficient": model.coef_[0],
        "OddsRatio":  np.exp(model.coef_[0]),
        "MeanCV_Acc": np.mean(cv_acc),
    })

    pred_df = meta.copy()
    pred_df["ProbOutperform"] = model.predict_proba(X_scaled)[:, 1]
    pred_df["Predicted"]      = model.predict(X_scaled)
    pred_df["Actual"]         = y_binary.values
    pred_df["Correct"]        = (pred_df["Predicted"] == pred_df["Actual"]).astype(int)
    return coef_df, pred_df


# ── Model 4: Sector OLS ───────────────────────────────────────────────────────

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


# ── Export ────────────────────────────────────────────────────────────────────

def export_csvs(ols_coef, ols_pred, ridge_coef, ridge_cv,
                log_coef, log_pred, sector_results):
    os.makedirs("output", exist_ok=True)
    ols_coef.to_csv("output/ols_coefficients.csv",      index=False)
    ols_pred.to_csv("output/ols_predictions.csv",       index=False)
    ridge_coef.to_csv("output/ridge_coefficients.csv",  index=False)
    ridge_cv.to_csv("output/ridge_cv_scores.csv",       index=False)
    log_coef.to_csv("output/logistic_coefficients.csv", index=False)
    log_pred.to_csv("output/logistic_predictions.csv",  index=False)
    sector_results.to_csv("output/sector_r2.csv",       index=False)
    print("✅ CSVs exported to output/")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_regression():
    db = get_database()

    print("Loading data...")
    df = load_data(db)
    print(f"Columns: {list(df.columns)}")
    print(f"GrossMargin NaN: {df['GrossMargin'].isna().sum()} / {len(df)}")
    print(f"Volatility_3M NaN: {df['Volatility_3M'].isna().sum()} / {len(df)}")
    print(f"Beta NaN: {df['Beta'].isna().sum()} / {len(df)}")
    print(f"ForwardReturn NaN: {df['ForwardReturn'].isna().sum()} / {len(df)}")

    print(f"Dataset size before cleaning: {len(df)}")
    X, y, meta = build_dataset(df)
    print(f"Dataset size after cleaning:  {len(X)}")

    if len(X) < 20:
        print("❌ Not enough data. Run UPDATE_MODE 2 and 3 first.")
        return

    # OLS
    print("\n── OLS Regression ──")
    ols_model = run_ols(X, y)
    ols_coef  = ols_results_for_powerbi(ols_model, FEATURES)
    X_const   = sm.add_constant(X)
    ols_pred  = meta.copy()
    ols_pred["PredictedReturn"] = ols_model.predict(X_const)
    ols_pred["ActualReturn"]    = y.values
    ols_pred["Residual"]        = ols_pred["ActualReturn"] - ols_pred["PredictedReturn"]
    ols_pred["QuarterDate"]     = ols_pred["QuarterDate"].astype(str)

    # Ridge CV
    print("\n── Ridge Regression with Time Series CV ──")
    ridge_model, ridge_scaler, ridge_coef, cv_r2 = run_ridge_cv(X, y)
    ridge_cv = ridge_cv_summary_for_powerbi(cv_r2)

    # Logistic
    print("\n── Logistic Regression with Time Series CV ──")
    log_model, log_scaler, y_binary, cv_acc = run_logistic(X, y)
    log_coef, log_pred = logistic_results_for_powerbi(
        log_model, log_scaler, X, y_binary, meta, FEATURES, cv_acc
    )
    log_pred["QuarterDate"] = log_pred["QuarterDate"].astype(str)

    # Sector
    print("\n── Sector Regressions ──")
    sector_results = run_sector_regressions(X, y, meta)
    print(sector_results)

    # Export
    export_csvs(ols_coef, ols_pred, ridge_coef, ridge_cv,
                log_coef, log_pred, sector_results)

    print("\n✅ Done. Features used:")
    for f in FEATURES:
        print(f"   {f}")

    print(f"Columns: {list(df.columns)}")
    print(f"GrossMargin NaN: {df['GrossMargin'].isna().sum()} / {len(df)}")
    print(f"Volatility_3M NaN: {df['Volatility_3M'].isna().sum()} / {len(df)}")
    print(f"Beta NaN: {df['Beta'].isna().sum()} / {len(df)}")
    print(f"ForwardReturn NaN: {df['ForwardReturn'].isna().sum()} / {len(df)}")


if __name__ == "__main__":
    run_regression()