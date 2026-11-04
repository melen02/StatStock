import numpy as np
import pandas as pd
import pytest

from calculations.regression import (
    REPORT_LAG_DAYS,
    build_panel,
    compute_fundamental_features,
    formation_dates,
    run_fama_macbeth,
    standardize,
    walk_forward,
)
from loadData.market_data import find_suspicious_moves

TICKERS = [f"T{i:02d}" for i in range(40)]


def make_history(seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", "2025-12-31")
    frames = []
    for ticker in TICKERS:
        price = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(dates))))
        frames.append(pd.DataFrame({"Ticker": ticker, "Date": dates, "Close": price, "AdjClose": price}))
    return pd.concat(frames, ignore_index=True)


def make_quarterly(seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for ticker in TICKERS:
        for period in pd.period_range("2023Q1", "2025Q3", freq="Q"):
            rows.append({
                "Ticker": ticker,
                "Quarter": f"{period.year}-Q{period.quarter}",
                "QuarterDate": period.end_time.normalize(),
                "Revenue": rng.uniform(50, 150),
                "NetIncome": rng.uniform(-10, 30),
                "FreeCashFlow": rng.uniform(-10, 30),
                "TotalAssets": rng.uniform(300, 500),
                "TotalDebt": rng.uniform(50, 150),
                "GrossMargin": rng.uniform(0.2, 0.6),
                "OperatingMargin": rng.uniform(0.05, 0.3),
            })
    return pd.DataFrame(rows)


def make_fundamentals():
    return pd.DataFrame({"Ticker": TICKERS, "MarketCap": 1e9, "Sector": "Tech"})


def make_panel():
    return build_panel(make_quarterly(), make_fundamentals(), make_history())


def test_formation_dates_skip_the_unfinished_month():
    index = pd.bdate_range("2024-01-01", "2024-11-15")
    dates = formation_dates(index)
    assert list(dates.month) == [2, 5, 8]
    assert dates[0] == pd.Timestamp("2024-02-29")


def test_a_report_is_not_used_before_it_is_public():
    panel = make_panel()
    assert (panel.dropna(subset=["AvailableDate"])["QuarterDate"] + pd.Timedelta(days=REPORT_LAG_DAYS) <= panel.dropna(subset=["AvailableDate"])["Date"]).all()


def test_changing_later_reports_does_not_change_earlier_features():
    cutoff = pd.Timestamp("2024-08-30")
    quarterly = make_quarterly()
    changed = quarterly.copy()
    changed.loc[changed["QuarterDate"] > cutoff - pd.Timedelta(days=REPORT_LAG_DAYS), ["Revenue", "NetIncome"]] *= 7

    before = build_panel(quarterly, make_fundamentals(), make_history())
    after = build_panel(changed, make_fundamentals(), make_history())

    columns = ["RevenueGrowth", "EarningsYield", "SalesToPrice"]
    mask = before["Date"] <= cutoff
    pd.testing.assert_frame_equal(before.loc[mask, columns], after.loc[mask, columns])


def test_prices_after_the_forward_window_do_not_change_features_or_return():
    history = make_history()
    date = pd.Timestamp("2024-08-30")
    next_date = pd.Timestamp("2024-11-29")
    changed = history.copy()
    changed.loc[changed["Date"] > next_date, ["Close", "AdjClose"]] *= 3

    fundamentals = make_fundamentals()
    before = build_panel(make_quarterly(), fundamentals, history)
    after = build_panel(make_quarterly(), fundamentals, changed)

    columns = ["ForwardReturn", "Momentum_3M", "Momentum_6M", "Volatility_3M", "Beta"]
    row = before["Date"] == date
    pd.testing.assert_frame_equal(before.loc[row, columns], after.loc[row, columns])


def test_growth_is_not_computed_across_a_missing_quarter():
    quarterly = make_quarterly()
    quarterly = quarterly[quarterly["Quarter"] != "2024-Q2"]
    features = compute_fundamental_features(quarterly)
    assert features.loc[features["Quarter"] == "2024-Q3", "RevenueGrowth"].isna().all()


def test_missing_quarter_date_is_filled_from_the_label():
    quarterly = make_quarterly()
    quarterly.loc[quarterly["Quarter"] == "2024-Q2", "QuarterDate"] = pd.NaT
    features = compute_fundamental_features(quarterly)
    assert (features.loc[features["Quarter"] == "2024-Q2", "QuarterDate"] == pd.Timestamp("2024-06-30")).all()


def test_standardize_scales_within_each_date_and_treats_missing_as_average():
    panel = pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-31"] * 6 + ["2024-02-29"] * 6),
        "A": [1.0, 2, 3, 4, 5, np.nan, 100, 200, 300, 400, 500, 600],
    })
    out = standardize(panel, ["A"])
    assert out["A"].iloc[5] == 0.0
    assert out[out["Date"] == "2024-02-29"]["A"].mean() == pytest.approx(0.0)
    assert out[out["Date"] == "2024-02-29"]["A"].std() == pytest.approx(1.0)


def test_standardize_drops_dates_where_a_feature_is_mostly_missing():
    panel = pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-31"] * 4 + ["2024-02-29"] * 4),
        "A": [np.nan, np.nan, np.nan, 1.0, 1.0, 2.0, 3.0, 4.0],
    })
    assert list(standardize(panel, ["A"])["Date"].unique()) == [pd.Timestamp("2024-02-29")]


def synthetic_panel(n_dates=12, n_stocks=80, slope=0.02, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-31", periods=n_dates, freq="QE")
    rows = []
    for date in dates:
        x = rng.standard_normal(n_stocks)
        z = rng.standard_normal(n_stocks)
        y = slope * x + rng.normal(0, 0.05, n_stocks) + rng.normal(0, 0.05)
        rows.append(pd.DataFrame({"Date": date, "Ticker": range(n_stocks), "X": x, "Z": z, "ForwardReturn": y}))
    return pd.concat(rows, ignore_index=True)


def test_fama_macbeth_recovers_a_known_slope():
    _, summary = run_fama_macbeth(synthetic_panel(), ["X", "Z"])
    summary = summary.set_index("Feature")
    assert summary.loc["X", "MeanCoef"] == pytest.approx(0.02, abs=0.005)
    assert summary.loc["X", "Significant"]
    assert not summary.loc["Z", "Significant"]


def test_fama_macbeth_always_returns_two_values():
    empty = synthetic_panel(n_dates=1, n_stocks=3)
    coefs, summary = run_fama_macbeth(empty, ["X", "Z"])
    assert coefs.empty and summary.empty


def test_walk_forward_does_not_use_later_returns():
    panel = synthetic_panel()
    changed = panel.copy()
    last_date = panel["Date"].max()
    changed.loc[changed["Date"] == last_date, "ForwardReturn"] *= -50

    before = walk_forward(panel, ["X", "Z"])
    after = walk_forward(changed, ["X", "Z"])
    earlier = before["Date"] < last_date
    assert earlier.sum() > 0
    pd.testing.assert_series_equal(before.loc[earlier, "RankIC"], after.loc[earlier, "RankIC"])


def test_walk_forward_finds_a_real_signal_and_no_signal_in_noise():
    assert walk_forward(synthetic_panel(), ["X"])["RankIC"].mean() > 0.2
    assert abs(walk_forward(synthetic_panel(slope=0.0), ["X"])["RankIC"].mean()) < 0.1


def test_suspicious_moves_flag_an_unadjusted_split():
    history = pd.DataFrame({
        "Ticker": ["A"] * 4,
        "Date": pd.bdate_range("2024-01-01", periods=4),
        "AdjClose": [1000.0, 1010.0, 101.0, 102.0],
    })
    flagged = find_suspicious_moves(history)
    assert list(flagged["Date"]) == [pd.Timestamp("2024-01-03")]
