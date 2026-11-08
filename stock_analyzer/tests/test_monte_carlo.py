import math
import time
from datetime import date
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from stock_analyzer.exceptions import InsufficientDataError, InvalidParameterError
from stock_analyzer.models.monte_carlo import (
    MAX_PATHS,
    MIN_PATHS,
    Calibration,
    calibrate,
    parse_horizon,
    simulate,
)

SIGMA, MU, S0 = 0.30, 0.08, 100.0
N = 50_000


def make_calibration(sigma=SIGMA, mu=MU, log_returns=None, n=756, seed=0):
    if log_returns is None:
        log_returns = np.random.default_rng(seed).normal(0, sigma / math.sqrt(252), n)
    return Calibration(
        sigma_annual=sigma, mu_annual=mu, mu_hist_annual=mu, mu_hist_se=0.1,
        drift_mode="neutral", risk_free_rate=0.04, n_obs=len(log_returns),
        window_start=date(2023, 1, 1), window_end=date(2026, 1, 1),
        log_returns=np.asarray(log_returns), skew=0.0, excess_kurtosis=0.0,
    )  # fmt: skip


@pytest.fixture(scope="module")
def gbm_year():
    return simulate(make_calibration(), S0, 252, n_paths=N, seed=42)


def theory(h, sigma=SIGMA, mu=MU):
    sd_d, mu_d = sigma / math.sqrt(252), mu / 252
    m = math.log1p(mu_d) - 0.5 * sd_d**2
    return h * m, sigma * math.sqrt(h / 252)  # mean and s.d. of terminal log-return


def test_gbm_terminal_log_return_mean_and_variance_match_inputs(gbm_year):
    mean, sd = theory(252)
    log_ret = np.log(gbm_year.terminal / S0)
    se = sd / math.sqrt(N)
    assert abs(log_ret.mean() - mean) < 4 * se  # mean within 4 standard errors
    assert log_ret.var(ddof=1) == pytest.approx(sd**2, rel=0.03)


def test_gbm_arithmetic_mean_matches_expected_return(gbm_year):
    # E[S_T / S_0] = (1 + mu/252)^252 by construction (mean-matching)
    expected = (1 + MU / 252) ** 252
    assert (gbm_year.terminal / S0).mean() == pytest.approx(expected, abs=0.01)


def test_gbm_median_and_probabilities_match_lognormal_theory(gbm_year):
    mean, sd = theory(252)
    assert gbm_year.percentile(50) == pytest.approx(S0 * math.exp(mean), rel=0.01)
    z = NormalDist()
    assert gbm_year.prob_above_current() == pytest.approx(1 - z.cdf((0 - mean) / sd), abs=0.01)
    assert gbm_year.prob_loss_more_than(0.10) == pytest.approx(
        z.cdf((math.log(0.9) - mean) / sd), abs=0.01
    )
    assert gbm_year.prob_gain_more_than(0.20) == pytest.approx(
        1 - z.cdf((math.log(1.2) - mean) / sd), abs=0.01
    )
    p = gbm_year.prob_above_current()
    assert p + gbm_year.prob_below_current() == pytest.approx(1.0, abs=1e-3)


def test_percentiles_ordered_and_bands_consistent(gbm_year):
    pcts = [gbm_year.percentile(q) for q in (5, 25, 50, 75, 95)]
    assert pcts == sorted(pcts) and pcts[0] > 0
    b = gbm_year.bands
    assert b.shape == (253, 5) and (b.iloc[0] == S0).all()
    assert (b.diff(axis=1).iloc[:, 1:] >= 0).all().all()  # p5 <= p25 <= ... each day
    np.testing.assert_allclose(b.iloc[-1].to_numpy(), pcts)
    assert b["p95"].iloc[-1] - b["p5"].iloc[-1] > b["p95"].iloc[21] - b["p5"].iloc[21]  # fan widens
    assert gbm_year.sample_paths.shape == (50, 253)


def test_seed_reproducibility():
    c = make_calibration()
    a = simulate(c, S0, 63, n_paths=MIN_PATHS, seed=7)
    b = simulate(c, S0, 63, n_paths=MIN_PATHS, seed=7)
    d = simulate(c, S0, 63, n_paths=MIN_PATHS, seed=8)
    np.testing.assert_array_equal(a.terminal, b.terminal)
    assert not np.array_equal(a.terminal, d.terminal)
    e = simulate(c, S0, 63, n_paths=MIN_PATHS)  # no seed -> varies
    f = simulate(c, S0, 63, n_paths=MIN_PATHS)
    assert not np.array_equal(e.terminal, f.terminal)


def test_sigma_scale_changes_dispersion_not_centre():
    c = make_calibration()
    base = simulate(c, S0, 126, n_paths=N, seed=1)
    wide = simulate(c, S0, 126, n_paths=N, seed=1, sigma_scale=1.2)
    ratio = np.log(wide.terminal / S0).std() / np.log(base.terminal / S0).std()
    assert ratio == pytest.approx(1.2, rel=0.02)
    assert (wide.terminal / S0).mean() == pytest.approx((base.terminal / S0).mean(), abs=0.01)


def _fat_tailed(n=2000, seed=3):
    sd_d = SIGMA / math.sqrt(252)
    return np.random.default_rng(seed).standard_t(3, n) / math.sqrt(3) * sd_d


def test_bootstrap_matches_dispersion_and_mean_of_history():
    r = _fat_tailed()
    c = make_calibration(log_returns=r, sigma=float(r.std(ddof=1) * math.sqrt(252)))
    sim = simulate(c, S0, 252, method="bootstrap", n_paths=N, seed=5)
    mean, sd = theory(252, sigma=c.sigma_annual)
    log_ret = np.log(sim.terminal / S0)
    assert log_ret.std(ddof=1) == pytest.approx(sd, rel=0.05)
    assert (sim.terminal / S0).mean() == pytest.approx((1 + MU / 252) ** 252, abs=0.02)


def test_bootstrap_keeps_fat_tails_that_gbm_misses():
    r = _fat_tailed(n=5000)
    c = make_calibration(log_returns=r, sigma=float(r.std(ddof=1) * math.sqrt(252)))
    boot = np.log(simulate(c, S0, 1, method="bootstrap", n_paths=N, seed=1).terminal / S0)
    gbm = np.log(simulate(c, S0, 1, method="gbm", n_paths=N, seed=1).terminal / S0)
    assert pd.Series(boot).kurt() > 2.0 > pd.Series(gbm).kurt() + 1.0


def test_bootstrap_only_uses_observed_returns():
    r = np.array([-0.02, 0.01, 0.03] * 30)
    c = make_calibration(log_returns=r, sigma=0.3)
    sim = simulate(c, S0, 1, method="bootstrap", n_paths=MIN_PATHS, seed=0)
    assert len(np.unique(np.round(sim.terminal, 6))) <= 3  # one-day outcomes are resampled moves


@pytest.mark.parametrize(
    ("kwargs", "exc"),
    [
        ({"n_paths": MIN_PATHS - 1}, InvalidParameterError),
        ({"n_paths": MAX_PATHS + 1}, InvalidParameterError),
        ({"method": "magic"}, InvalidParameterError),
        ({"seed": -1}, InvalidParameterError),
        ({"sigma_scale": 0.0}, InvalidParameterError),
    ],
)
def test_simulate_input_limits(kwargs, exc):
    with pytest.raises(exc):
        simulate(make_calibration(), S0, 21, **kwargs)


@pytest.mark.parametrize("days", [0, 253])
def test_horizon_days_limits(days):
    with pytest.raises(InvalidParameterError):
        simulate(make_calibration(), S0, days)


@pytest.mark.parametrize("s0", [0.0, -5.0, float("nan"), float("inf")])
def test_bad_start_price_rejected(s0):
    with pytest.raises(InvalidParameterError):
        simulate(make_calibration(), s0, 21)


def test_parse_horizon():
    assert [parse_horizon(x) for x in ("1m", "3M", " 6m ", "1y", "12M")] == [21, 63, 126, 252, 252]
    with pytest.raises(InvalidParameterError) as err:
        parse_horizon("2Y")
    assert "1M, 3M, 6M or 1Y" in err.value.user_message


def test_performance_one_year_10k_paths_is_fast():
    start = time.perf_counter()
    simulate(make_calibration(), S0, 252, n_paths=10_000, seed=1)
    assert time.perf_counter() - start < 3.0  # loose bound so slow CI machines don't flake


# ----------------------------------------------------------------------------- calibration


def synthetic_close(n=800, daily_mu=0.0004, daily_sigma=0.012, seed=0):
    r = np.random.default_rng(seed).normal(daily_mu, daily_sigma, n)
    return pd.Series(100 * np.cumprod(1 + r), index=pd.bdate_range("2023-01-02", periods=n))


def test_calibrate_recovers_volatility_and_drift_modes():
    close = synthetic_close(n=2000)
    hist = calibrate(close, drift_mode="historical", lookback_years=8)
    assert hist.sigma_annual == pytest.approx(0.012 * math.sqrt(252), rel=0.05)
    assert hist.mu_annual == pytest.approx(hist.mu_hist_annual)
    neutral = calibrate(close, drift_mode="neutral", risk_free_rate=0.03, lookback_years=8)
    assert neutral.mu_annual == 0.03
    shrunk = calibrate(close, drift_mode="shrunk", risk_free_rate=0.03, lookback_years=8)
    assert shrunk.mu_annual == pytest.approx(0.5 * (0.03 + 0.05) + 0.5 * hist.mu_hist_annual)
    assert hist.mu_hist_se == pytest.approx(hist.sigma_annual / math.sqrt(hist.n_obs / 252))


def test_calibrate_caps_extreme_drift_and_warns():
    close = synthetic_close(n=800, daily_mu=0.01)  # ~+250% a year historically
    c = calibrate(close, drift_mode="historical")
    assert c.mu_annual == 0.5 and any("capped" in w for w in c.warnings)
    assert any("unusually extreme" in w for w in c.warnings)


def test_calibrate_warns_on_short_window_and_fat_tails():
    c = calibrate(synthetic_close(n=800), lookback_years=1.5)
    assert any("less reliable" in w for w in c.warnings)
    heavy = pd.Series(
        100 * np.cumprod(1 + np.random.default_rng(1).standard_t(2.5, 800) * 0.005),
        index=pd.bdate_range("2023-01-02", periods=800),
    ).abs()
    assert any("fat-tailed" in w for w in calibrate(heavy).warnings)


def test_calibrate_rejects_bad_input():
    with pytest.raises(InsufficientDataError):
        calibrate(synthetic_close(n=40))
    flat = pd.Series(100.0, index=pd.bdate_range("2024-01-01", periods=300))
    with pytest.raises(InsufficientDataError):
        calibrate(flat)
    with pytest.raises(InvalidParameterError):
        calibrate(synthetic_close(), drift_mode="moon")


def test_plain_english_states_assumptions_and_limits(gbm_year):
    text = " ".join(gbm_year.plain_english("USD"))
    assert "50,000 possible price paths" in text and "1 year" in text
    assert "not a prediction" in text and "volatility 30%" in text
    assert "uncertain by about" in text
    assert gbm_year.monte_carlo_error(0.5) == pytest.approx(0.5 / math.sqrt(N) * 1.0, rel=1e-6)
