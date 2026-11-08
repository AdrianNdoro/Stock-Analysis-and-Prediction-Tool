import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from stock_analyzer.analysis.risk import (
    TRADING_DAYS,
    annualised_volatility,
    beta_from_returns,
    cagr,
    compute_beta,
    compute_risk,
    daily_rate,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    summarize_risk,
    value_at_risk,
)


def px(values, start="2022-01-03"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def test_volatility_hand_computed_and_constant():
    r = pd.Series([0.01, -0.01, 0.02, -0.02])
    assert annualised_volatility(r) == pytest.approx(math.sqrt(0.001 / 3) * math.sqrt(252))
    assert annualised_volatility(pd.Series([0.01] * 10)) == pytest.approx(0.0, abs=1e-12)


def test_volatility_recovers_known_sigma():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.02, 100_000))
    assert annualised_volatility(r) == pytest.approx(0.02 * math.sqrt(252), rel=0.02)


def test_max_drawdown_unrecovered():
    dd = max_drawdown(px([100, 120, 90, 110, 130, 80]))
    assert dd.max_drawdown == pytest.approx(80 / 130 - 1)
    assert dd.recovery_date is None and dd.currently_in_drawdown == pytest.approx(80 / 130 - 1)


def test_max_drawdown_recovered():
    p = px([100, 120, 90, 125])
    dd = max_drawdown(p)
    assert dd.max_drawdown == pytest.approx(-0.25)
    assert dd.peak_date == p.index[1].date() and dd.trough_date == p.index[2].date()
    assert dd.recovery_date == p.index[3].date() and dd.currently_in_drawdown == 0.0


def test_cagr():
    p = px([100.0] + [0.0] * 252)
    p[:] = np.linspace(100, 121, 253)
    assert cagr(p) == pytest.approx(0.21)


def test_sharpe_hand_computed_and_rf_effect():
    r = pd.Series([0.01, 0.02, 0.03])
    assert sharpe_ratio(r, 0.0) == pytest.approx(2 * math.sqrt(252))
    rf = 0.0252
    expected = (r.mean() - daily_rate(rf)) / r.std(ddof=1) * math.sqrt(252)
    assert sharpe_ratio(r, rf) == pytest.approx(expected) and expected < sharpe_ratio(r, 0.0)
    assert math.isnan(sharpe_ratio(pd.Series([0.01] * 5), 0.0))


def test_sortino_hand_computed_and_no_downside():
    r = pd.Series([0.02, -0.01, 0.03, -0.02])
    downside = math.sqrt((0.01**2 + 0.02**2) / 4)
    assert sortino_ratio(r, 0.0) == pytest.approx(0.005 / downside * math.sqrt(252))
    assert math.isnan(sortino_ratio(pd.Series([0.01, 0.02, 0.03]), 0.0))


def test_beta_exact_on_constructed_returns():
    bench = pd.Series(np.random.default_rng(2).normal(0, 0.01, 500))
    beta, r2, n = beta_from_returns(1.5 * bench, bench)
    assert beta == pytest.approx(1.5) and r2 == pytest.approx(1.0) and n == 500
    assert beta_from_returns(bench, bench)[0] == pytest.approx(1.0)
    noise = pd.Series(np.random.default_rng(3).normal(0, 0.01, 20_000))
    assert abs(beta_from_returns(noise, bench.reindex(noise.index, fill_value=0.0))[0]) < 0.1


def test_beta_aligns_dates_and_requires_overlap():
    rng = np.random.default_rng(4)
    bench_r = pd.Series(rng.normal(0, 0.01, 600), index=pd.bdate_range("2022-01-03", periods=600))
    bench = 100 * (1 + bench_r).cumprod()
    stock = 50 * (1 + 2.0 * bench_r).cumprod()
    res = compute_beta(stock.iloc[5:], bench.iloc[:-5], freq="D", lookback_years=5)
    assert res.beta == pytest.approx(2.0, abs=0.01) and res.n_obs == 589
    weekly = compute_beta(stock, bench, freq="W", lookback_years=5)
    assert weekly.beta == pytest.approx(2.0, abs=0.25) and weekly.frequency == "W"
    far = px(np.linspace(10, 20, 300), start="2010-01-04")
    no_overlap = compute_beta(far, bench, freq="D")
    assert math.isnan(no_overlap.beta) and "overlapping" in no_overlap.note


def test_historical_var_exact_on_uniform_returns():
    r = pd.Series(np.linspace(-0.05, 0.05, 1001))
    v = value_at_risk(r, 0.95)
    assert v.historical == pytest.approx(0.045)
    assert v.expected_shortfall > v.historical
    assert value_at_risk(r, 0.95, horizon_days=4).historical == pytest.approx(0.09)


def test_parametric_var_matches_normal_formula():
    r = pd.Series(np.random.default_rng(5).normal(0.0005, 0.015, 50_000))
    mu, sd = r.mean(), r.std(ddof=1)
    z = NormalDist().inv_cdf(0.05)
    assert value_at_risk(r, 0.95).parametric == pytest.approx(-(mu + z * sd))
    h10 = value_at_risk(r, 0.95, horizon_days=10).parametric
    assert h10 == pytest.approx(-(10 * mu + z * sd * math.sqrt(10)))
    # on genuinely normal data the two methods agree
    assert value_at_risk(r, 0.99).historical == pytest.approx(
        value_at_risk(r, 0.99).parametric, rel=0.05
    )


def test_normal_var_understates_fat_tails():
    t = np.random.default_rng(6).standard_t(3, 400_000) / math.sqrt(3) * 0.01  # unit-variance t3
    v = value_at_risk(pd.Series(t), 0.99)
    assert v.historical > v.parametric * 1.05  # the model's blind spot, demonstrated


def test_var_input_validation_and_short_samples():
    assert math.isnan(value_at_risk(pd.Series([0.01] * 5)).historical)
    with pytest.raises(ValueError):
        value_at_risk(pd.Series(np.zeros(100)), confidence=1.5)


def test_compute_risk_window_warning_and_bullets():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2025-01-02", periods=400)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, 400))), index=idx)
    bench = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.008, 400))), index=idx)
    r = compute_risk(close, bench, risk_free_rate=0.04, lookback_years=3)
    assert any("years of history are available" in w for w in r.warnings)
    assert r.n_returns == 399 and r.beta is not None
    assert r.drawdown.max_drawdown == r.drawdown_full_history.max_drawdown  # window = all history
    text = " ".join(summarize_risk(r))
    assert "Value at Risk" in text and "fat tails" in text


def test_trading_days_constant():
    assert TRADING_DAYS == 252
