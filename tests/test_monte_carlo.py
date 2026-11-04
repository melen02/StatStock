import numpy as np
import pandas as pd
import pytest

from calculations.monte_carlo import (
    as_gbm,
    backtest,
    bootstrap_breach_interval,
    compensator,
    estimate_parameters,
    kupiec_pvalue,
    risk_metrics,
    simulate,
)


def make_params(sigma=0.015, jump_lambda=6.0, market_lambda=3.0):
    tickers = pd.Index(["A", "B"])
    daily_cov = sigma ** 2 * np.array([[1.0, 0.5], [0.5, 1.0]])
    return {
        "tickers": tickers,
        "mu_annual": np.array([0.10, 0.06]),
        "beta": np.array([1.2, 0.8]),
        "tilt_annual": np.zeros(2),
        "diffusion_cov": daily_cov,
        "total_cov": daily_cov * 1.5,
        "jump_lambda": np.full(2, jump_lambda),
        "jump_mean": np.array([-0.03, -0.02]),
        "jump_std": np.array([0.05, 0.04]),
        "market_lambda": market_lambda,
        "market_jump_mean": -0.03,
        "market_jump_std": 0.02,
    }


def make_returns(n_days=600, n_stocks=6, seed=0):
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0004, 0.008, n_days)
    noise = rng.normal(0, 0.012, (n_days, n_stocks))
    columns = [f"S{i}" for i in range(n_stocks)]
    index = pd.bdate_range("2022-01-03", periods=n_days)
    return pd.DataFrame(market[:, None] + noise, columns=columns, index=index)


def test_compensator_matches_a_sampled_expectation():
    rng = np.random.default_rng(1)
    sample = rng.normal(-0.03, 0.05, 2_000_000)
    assert compensator(-0.03, 0.05, 1.5) == pytest.approx(np.exp(1.5 * sample).mean() - 1, abs=1e-3)


@pytest.mark.parametrize("model", [make_params(), as_gbm(make_params())])
def test_expected_price_grows_at_mu_despite_jumps(model):
    _, relatives = simulate(model, 252, 40000, seed=7, sample_every=252)
    terminal = relatives[-1].astype(float)
    standard_error = terminal.std(axis=0) / np.sqrt(len(terminal))
    assert np.all(np.abs(terminal.mean(axis=0) - np.exp(model["mu_annual"])) < 4 * standard_error)


def test_simulated_daily_correlation_matches_the_input():
    params = make_params(jump_lambda=0.0, market_lambda=0.0)
    _, relatives = simulate(params, 2, 50000, seed=2, sample_every=1)
    one_day = np.log(relatives[0].astype(float))
    assert np.corrcoef(one_day, rowvar=False)[0, 1] == pytest.approx(0.5, abs=0.02)


def test_simulation_is_reproducible_with_a_seed():
    _, first = simulate(make_params(), 20, 100, seed=11)
    _, second = simulate(make_params(), 20, 100, seed=11)
    assert np.array_equal(first, second)


def test_final_day_is_always_sampled():
    days, relatives = simulate(make_params(), 12, 10, sample_every=5)
    assert days == [5, 10, 12]
    assert relatives.shape == (3, 10, 2)


def test_jump_model_has_fatter_tails_than_gbm_at_equal_variance():
    params = make_params(jump_lambda=10.0, market_lambda=0.0)
    params["jump_mean"] = np.zeros(2)
    params["jump_std"] = np.array([0.08, 0.08])
    jump_variance = 10.0 / 252 * 0.08 ** 2
    params["total_cov"] = params["diffusion_cov"] + np.diag([jump_variance] * 2)

    def excess_kurtosis(model):
        _, relatives = simulate(model, 1, 200000, seed=5, sample_every=1)
        x = np.log(relatives[0, :, 0].astype(float))
        return ((x - x.mean()) ** 4).mean() / x.var() ** 2 - 3

    assert excess_kurtosis(params) > 2
    assert abs(excess_kurtosis(as_gbm(params))) < 0.2


def test_outliers_become_jumps_instead_of_inflating_diffusion():
    # Enough stocks that one outlier doesn't also register as a market-wide jump.
    returns = make_returns(n_stocks=40)
    clean_var = np.diag(estimate_parameters(returns)["diffusion_cov"])

    shocked = returns.copy()
    shocked.iloc[-100, 0] = -0.40
    params = estimate_parameters(shocked)

    assert np.diag(params["diffusion_cov"])[0] == pytest.approx(clean_var[0], rel=0.05)
    assert params["jump_lambda"][0] > 0


def test_jump_days_are_not_counted_in_both_diffusion_and_jumps():
    rng = np.random.default_rng(9)
    index = pd.bdate_range("2022-01-03", periods=1000)
    columns = [f"S{i}" for i in range(40)]
    returns = rng.standard_t(3, (1000, 40)) * 0.01 + rng.normal(0, 0.005, (1000, 1))
    params = estimate_parameters(pd.DataFrame(returns, index=index, columns=columns))

    jump_variance = params["jump_lambda"] / 252 * (params["jump_mean"] ** 2 + params["jump_std"] ** 2)
    model_variance = np.diag(params["diffusion_cov"]) + jump_variance
    sample_variance = np.diag(params["total_cov"])
    assert np.median(model_variance / sample_variance) == pytest.approx(1.0, abs=0.15)


def test_breach_interval_widens_when_breaches_cluster_in_one_window():
    origins = pd.date_range("2022-01-31", periods=20, freq="ME")
    evenly = pd.DataFrame({"Origin": np.repeat(origins, 50), "Breach": ([True] * 5 + [False] * 45) * 20})
    clustered = evenly.copy()
    clustered["Breach"] = False
    clustered.loc[clustered["Origin"] == origins[0], "Breach"] = True

    evenly_low, evenly_high = bootstrap_breach_interval(evenly)
    clustered_low, clustered_high = bootstrap_breach_interval(clustered)
    assert evenly_high - evenly_low == pytest.approx(0.0)
    assert clustered_high - clustered_low > 0.05


def test_drift_does_not_follow_recent_performance():
    returns = make_returns()
    rally = returns + 0.003  # about 75 percentage points a year on every stock
    assert estimate_parameters(rally)["mu_annual"] == pytest.approx(estimate_parameters(returns)["mu_annual"], abs=1e-9)


def test_tilt_moves_drift_only_for_named_stocks():
    returns = make_returns()
    plain = estimate_parameters(returns)["mu_annual"]
    tilted = estimate_parameters(returns, pd.Series({"S0": 0.03}))["mu_annual"]
    assert tilted[0] - plain[0] == pytest.approx(0.03)
    assert tilted[1] == pytest.approx(plain[1])


def test_estimation_recovers_beta():
    assert estimate_parameters(make_returns(n_days=1500, n_stocks=40))["beta"].mean() == pytest.approx(1.0, abs=0.01)


def test_risk_metrics_on_a_known_distribution():
    metrics = risk_metrics(np.linspace(-1, 1, 2001))
    assert metrics["VaR_5pct"] == pytest.approx(-0.9, abs=1e-6)
    assert metrics["CVaR_5pct"] == pytest.approx(-0.95, abs=1e-3)
    assert metrics["CVaR_5pct"] < metrics["VaR_5pct"]


def test_kupiec_accepts_the_right_rate_and_rejects_a_wrong_one():
    assert kupiec_pvalue(50, 1000) == pytest.approx(1.0)
    assert kupiec_pvalue(100, 1000) < 0.001
    assert kupiec_pvalue(0, 1000) < 0.001


def test_backtest_is_roughly_calibrated_on_data_from_the_model():
    returns = make_returns(n_days=1100, n_stocks=10, seed=4)
    prices = 100 * np.exp(returns.cumsum())
    summary = backtest(prices, horizon=21, n_sims=500, min_history=300)
    stocks = summary[(summary["Model"] == "GBM baseline") & (summary["Level"] == "Single stocks")].iloc[0]
    assert 0.0 < stocks["BreachRate"] < 0.15
    assert stocks["N_Origins"] > 20
