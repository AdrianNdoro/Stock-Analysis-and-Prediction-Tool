import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from stock_analyzer.backtest import metrics as m
from stock_analyzer.backtest import service as bt_service
from stock_analyzer.backtest.distributions import (
    NormalLogReturn,
    SampleLogReturn,
    bootstrap_terminal_fast,
    empirical_distribution,
    gbm_distribution,
)
from stock_analyzer.backtest.reliability import GradeThresholds, assess, grade_metrics
from stock_analyzer.backtest.report import BacktestReport, build_report, format_report
from stock_analyzer.backtest.walkforward import (
    BacktestConfig,
    calibrate_at,
    run_walk_forward,
    vol_scale,
)
from stock_analyzer.config import Settings
from stock_analyzer.data.cache import SQLiteCache
from stock_analyzer.data.resilience import RateLimiter
from stock_analyzer.data.service import MarketDataService
from stock_analyzer.exceptions import InvalidParameterError
from stock_analyzer.models.forecast import forecast_ticker
from stock_analyzer.models.monte_carlo import calibrate, simulate
from stock_analyzer.tests.fakes import FakeClock, FakeProvider, make_prices

MU_D, SIGMA_D = 0.0003, 0.012


def synthetic_close(n=3000, seed=1, mu=MU_D, sigma=SIGMA_D, end="2026-11-03") -> pd.Series:
    """Prices whose daily log returns are iid Normal(mu, sigma): the true law is known."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=end, periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(mu, sigma, n))), index=idx, name="Close")


def cfg(**kw) -> BacktestConfig:
    base = {"horizons": ("1M",), "drift_mode": "historical"}
    return BacktestConfig(**{**base, **kw})


# ----------------------------------------------------------------------------- statistics


def test_wilson_matches_hand_values():
    lo, hi = m.wilson_interval(0, 10)
    z2 = 1.959963984540054**2
    assert lo == 0.0 and hi == pytest.approx(z2 / (10 + z2))
    lo, hi = m.wilson_interval(5, 10)
    assert lo == pytest.approx(1 - hi) and lo < 0.5 < hi
    assert math.isnan(m.wilson_interval(1, 0)[0])


def test_kupiec_hand_values():
    lr, p = m.kupiec_pof(1, 10, 0.10)  # exactly the expected number of misses
    assert lr == pytest.approx(0.0, abs=1e-12) and p == pytest.approx(1.0)
    lr, p = m.kupiec_pof(0, 10, 0.10)
    assert lr == pytest.approx(-20 * math.log(0.9))
    assert p == pytest.approx(chi2.sf(lr, 1))
    assert m.kupiec_pof(10, 10, 0.10)[1] < 1e-6  # every interval missed


def test_christoffersen_detects_clustering_and_handles_degenerate_input():
    clustered = np.array([0] * 20 + [1] * 5 + [0] * 20 + [1] * 5)
    spread = np.array(([0] * 9 + [1]) * 5)
    assert m.christoffersen_independence(clustered)[1] < 0.001
    assert m.christoffersen_independence(spread)[1] > 0.3
    assert m.christoffersen_independence(np.zeros(30, dtype=int)) is None
    assert m.christoffersen_independence(np.array([0, 1])) is None


def test_pinball_and_interval_score_hand_values():
    y = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert m.pinball_loss(y, q, 0.9).tolist() == pytest.approx([0.9, 0.1])
    # width 1 + (2/0.1) * miss distance 1 = 21
    assert m.interval_score(np.array([2.0]), np.array([0.0]), np.array([1.0]), 0.9) == (
        pytest.approx(21.0)
    )
    assert m.interval_score(np.array([0.5]), np.array([0.0]), np.array([1.0]), 0.9) == (
        pytest.approx(1.0)
    )


def test_non_overlapping_indices_and_pit_helpers():
    assert m.non_overlapping_indices(10, 63, 21).tolist() == [0, 3, 6, 9]
    assert m.non_overlapping_indices(5, 21, 21).tolist() == [0, 1, 2, 3, 4]
    assert m.non_overlapping_indices(5, 5, 21).tolist() == [0, 1, 2, 3, 4]
    pit = np.random.default_rng(0).uniform(size=400)
    assert sum(m.pit_histogram(pit)) == 400
    assert m.pit_uniformity_pvalue(pit) > 0.05
    assert m.pit_uniformity_pvalue(pit**3) < 0.001  # clearly non-uniform
    assert math.isnan(m.pit_uniformity_pvalue(pit[:3]))


# --------------------------------------------------------------------- fast paths vs simulator


def test_closed_form_gbm_matches_the_simulator():
    calib = calibrate(synthetic_close(900, seed=3).astype(float), drift_mode="historical")
    h = 63
    sim = simulate(calib, 100.0, h, method="gbm", n_paths=50_000, seed=7)
    dist = gbm_distribution(calib, h)
    logs = np.log(sim.terminal / 100.0)
    taus = (0.05, 0.25, 0.5, 0.75, 0.95)
    for tau, q in zip(taus, dist.quantile(taus), strict=True):
        assert np.quantile(logs, tau) == pytest.approx(q, abs=0.1 * dist.sd)
    assert logs.std() == pytest.approx(dist.sd, rel=0.02)
    # the same mean-matching: E[S_h/S_0] = (1 + mu_d)^h
    assert np.mean(sim.terminal / 100.0) == pytest.approx(
        (1 + calib.mu_annual / 252) ** h, rel=0.005
    )
    assert math.exp(dist.mean + dist.sd**2 / 2) == pytest.approx(
        (1 + calib.mu_annual / 252) ** h, rel=1e-9
    )


def test_fast_bootstrap_reproduces_the_simulator_with_same_seed():
    calib = calibrate(synthetic_close(900, seed=4), drift_mode="shrunk")
    h, n = 21, 10_000
    sim = simulate(calib, 100.0, h, method="bootstrap", n_paths=n, seed=11, sigma_scale=1.2)
    fast = bootstrap_terminal_fast(calib, h, n, np.random.default_rng(11), sigma_scale=1.2)
    np.testing.assert_allclose(np.sort(np.log(sim.terminal / 100.0)), fast.sorted_values, atol=1e-9)


def test_sample_distribution_cdf_quantile_and_empirical_baseline():
    d = SampleLogReturn.from_values(np.arange(100, dtype=float))
    assert d.cdf(-1) == 0.0 and d.cdf(1000) == 1.0
    assert d.cdf(49.5) == pytest.approx(0.5)
    assert d.cdf(10.0) == pytest.approx((10 + 0.5) / 100)  # mid-rank for the tie
    assert d.prob_up() == pytest.approx(0.99)
    nd = NormalLogReturn(0.0, 0.1)
    assert nd.cdf(0.0) == pytest.approx(0.5) and nd.prob_up() == pytest.approx(0.5)
    logp = np.log(np.linspace(100, 200, 300))
    emp = empirical_distribution(logp, 21)
    assert len(emp.sorted_values) == 300 - 21
    assert empirical_distribution(logp[:50], 21) is None  # too few past returns


# --------------------------------------------------------------------------- walk-forward


@pytest.mark.parametrize("method", ["gbm", "bootstrap"])
def test_no_look_ahead_changing_future_prices_cannot_change_a_forecast(method):
    a = synthetic_close(900, seed=5)
    cut = 700
    b = a.copy()
    b.iloc[cut + 1 :] *= np.random.default_rng(9).uniform(0.5, 2.0, len(b) - cut - 1)
    config = cfg(method=method, horizons=("1M", "3M"))
    ra, rb = run_walk_forward(a, config), run_walk_forward(b, config)
    key = ["origin", "horizon", "model"]
    cols = ["q05", "q25", "q50", "q75", "q95"]
    merged = ra.merge(rb, on=key, suffixes=("_a", "_b"))
    before = merged[merged["origin"] <= a.index[cut]]
    assert len(before) > 10
    for c in cols:
        np.testing.assert_array_equal(before[f"{c}_a"].to_numpy(), before[f"{c}_b"].to_numpy())
    assert not np.allclose(merged["y_a"], merged["y_b"])  # the outcomes did differ


def test_forecast_at_origin_equals_standalone_forecast_on_truncated_history():
    close = synthetic_close(900, seed=6)
    config = cfg()
    rows = run_walk_forward(close, config)
    row = rows[(rows["model"] == "gbm")].iloc[3]
    pos = close.index.get_loc(row["origin"])
    calib, scale = calibrate_at(close.iloc[: pos + 1], config)
    expected = gbm_distribution(calib, 21, scale).quantile((0.05, 0.25, 0.5, 0.75, 0.95))
    np.testing.assert_allclose(row[["q05", "q25", "q50", "q75", "q95"]].to_numpy(float), expected)


def test_rows_structure_and_expanding_window():
    close = synthetic_close(1000, seed=7)
    rows = run_walk_forward(close, cfg(window="expanding", horizons=("1M", "3M")))
    assert set(rows["model"]) == {"gbm", "random_walk", "empirical"}
    assert set(rows["horizon"]) == {"1M", "3M"}
    assert rows["pit"].between(0, 1).all()
    assert (rows["q05"] < rows["q50"]).all() and (rows["q50"] < rows["q95"]).all()
    # realised outcome is read from the real series
    r = rows.iloc[0]
    pos = close.index.get_loc(r["origin"])
    assert r["y"] == pytest.approx(math.log(close.iloc[pos + r["h"]] / close.iloc[pos]))


def test_date_filters_for_held_out_evaluation():
    close = synthetic_close(1500, seed=8)
    cut = close.index[1000].date()
    early = run_walk_forward(close, cfg(outcome_until=cut))
    late = run_walk_forward(close, cfg(origin_from=cut))
    assert early["origin"].max() < pd.Timestamp(cut)
    assert (late["origin"] >= pd.Timestamp(cut)).all()
    assert early["origin"].max() < late["origin"].min()


def test_short_history_gives_no_rows_warning_and_not_enough_history():
    close = synthetic_close(300, seed=9)
    config = cfg()
    rows = run_walk_forward(close, config)
    assert rows.empty
    report = build_report("TEST", rows, config, "2026-11-03")
    assert report.horizons == {} and report.warnings
    rel = assess(report, "1M")
    assert rel.grade == "Not enough history"
    # long horizon on a medium history: some origins exist for 1M but none for 1Y
    close2 = synthetic_close(560, seed=9)
    r2 = run_walk_forward(close2, cfg(horizons=("1M", "1Y")))
    assert set(r2["horizon"]) == {"1M"}


def test_few_independent_checks_cannot_be_graded():
    close = synthetic_close(1500, seed=10)
    config = cfg(horizons=("1Y",))
    report = build_report("T", run_walk_forward(close, config), config, "x")
    hr = report.horizons["1Y"]
    assert hr.n_origins > hr.n_independent and hr.n_independent < 15
    assert assess(report, "1Y").grade == "Not enough history"


def test_invalid_config_and_bad_series_are_rejected():
    close = synthetic_close(900)
    for bad in (
        {"method": "nope"}, {"window": "x"}, {"vol_inflation": 9.0}, {"n_paths": 10},
        {"horizons": ("2M",)}, {"min_train_days": 30}, {"vol_model": "garch"},
    ):  # fmt: skip
        with pytest.raises(InvalidParameterError):
            run_walk_forward(close, cfg(**bad))
    with pytest.raises(InvalidParameterError):
        run_walk_forward(close.iloc[::-1], cfg())
    with pytest.raises(InvalidParameterError):
        run_walk_forward(close * -1, cfg())


# ------------------------------------------------- known-truth calibration and detection


def test_correct_model_shows_near_nominal_coverage():
    close = synthetic_close(4000, seed=21)
    config = cfg()
    report = build_report("SYN", run_walk_forward(close, config), config, "x")
    mm = report.horizons["1M"].models["gbm"]
    assert mm.n_independent > 100
    # estimation error in sigma and mu makes coverage slightly low, never wildly so
    assert 0.84 <= mm.coverage90 <= 0.95
    assert 0.42 <= mm.coverage50 <= 0.58
    assert mm.coverage90_lo <= 0.90 <= mm.coverage90_hi + 0.05
    assert mm.kupiec90_p > 0.01
    assert assess(report, "1M").grade in {"Good", "Fair"}
    # PIT roughly flat
    assert min(mm.pit_hist) >= 0.4 * (mm.n_origins / 10)


def test_too_low_volatility_is_detected_and_graded_poor():
    close = synthetic_close(4000, seed=21)
    config = cfg(vol_inflation=0.5)  # deliberately mis-specified: half the true volatility
    report = build_report("SYN", run_walk_forward(close, config), config, "x")
    mm = report.horizons["1M"].models["gbm"]
    assert mm.coverage90 < 0.70
    assert mm.kupiec90_p < 0.001
    assert mm.pit_ks_p < 0.01
    rel = assess(report, "1M")
    assert rel.grade == "Poor" and "too narrow" in rel.headline
    # and it loses to the correctly sized random-walk baseline? (baseline shares the same scale)
    good = build_report("SYN", run_walk_forward(close, cfg()), cfg(), "x")
    assert mm.pinball > good.horizons["1M"].models["gbm"].pinball


def test_too_high_volatility_is_detected_as_too_wide():
    close = synthetic_close(4000, seed=21)
    config = cfg(vol_inflation=2.0)
    report = build_report("SYN", run_walk_forward(close, config), config, "x")
    mm = report.horizons["1M"].models["gbm"]
    assert mm.coverage90 > 0.99
    assert "wider than needed" in assess(report, "1M").headline


def test_bootstrap_on_known_data_is_also_near_nominal():
    close = synthetic_close(3000, seed=22)
    config = cfg(method="bootstrap", n_paths=2000)
    report = build_report("SYN", run_walk_forward(close, config), config, "x")
    mm = report.horizons["1M"].models["bootstrap"]
    assert 0.82 <= mm.coverage90 <= 0.96


def test_vol_clustering_is_detected_as_miscalibration():
    """Documented failure mode: with calm and stormy regimes, constant-volatility GBM is too
    wide in calm spells and too narrow in storms. Headline 90% coverage can look nearly fine
    while the 50% range and the PIT histogram expose the problem."""
    rng = np.random.default_rng(31)
    n = 4000
    state = np.zeros(n, dtype=int)
    for i in range(1, n):  # sticky two-regime volatility
        state[i] = state[i - 1] if rng.random() > 0.01 else 1 - state[i - 1]
    sigma = np.where(state == 0, 0.007, 0.03)
    close = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0.0002, sigma))),
        index=pd.bdate_range(end="2026-11-03", periods=n),
    )
    config = cfg()
    mm = (
        build_report("V", run_walk_forward(close, config), config, "x").horizons["1M"].models["gbm"]
    )
    assert abs(mm.coverage50 - 0.50) > 0.10
    assert max(mm.pit_hist) > 1.5 * min(mm.pit_hist)


def test_vol_scale_identity_and_ewma_reacts_to_a_volatility_spike():
    calm = synthetic_close(900, seed=40)
    calib, scale = calibrate_at(calm, cfg())
    assert scale == 1.0
    ewma = cfg(vol_model="ewma")
    assert 0.8 < vol_scale(calib, ewma) < 1.25  # stationary data: close to 1
    rng = np.random.default_rng(41)
    r = np.r_[rng.normal(0, 0.008, 700), rng.normal(0, 0.04, 20)]
    spiky = pd.Series(
        100 * np.exp(np.cumsum(r)), index=pd.bdate_range(end="2026-11-03", periods=720)
    )
    c2 = calibrate(spiky, drift_mode="neutral")
    assert vol_scale(c2, ewma) > 1.5
    assert vol_scale(c2, cfg(vol_model="ewma", ewma_weight=0.5)) < vol_scale(c2, ewma)
    assert vol_scale(c2, cfg(vol_inflation=1.3)) == pytest.approx(1.3)


# --------------------------------------------------------------- reliability, report, cache


def test_grading_thresholds_are_applied_in_one_place():
    close = synthetic_close(3000, seed=21)
    config = cfg()
    report = build_report("SYN", run_walk_forward(close, config), config, "x")
    mm = report.horizons["1M"].models["gbm"]
    assert grade_metrics(mm) in {"Good", "Fair"}
    strict = GradeThresholds(min_independent=10_000)
    assert grade_metrics(mm, strict) == "Not enough history"
    impossible = GradeThresholds(good_gap90=0.0, good_gap50=0.0, fair_gap90=0.0, fair_gap50=0.0)
    assert grade_metrics(mm, impossible) == "Poor"
    rel = assess(report, "1m")
    assert "90% range" in rel.headline and rel.n_independent == mm.n_independent
    assert assess(report, "1Y").grade == "Not enough history"  # horizon never tested


def test_report_round_trips_through_json_and_formats():
    close = synthetic_close(1200, seed=11)
    config = cfg(horizons=("1M", "3M"))
    report = build_report("SYN", run_walk_forward(close, config), config, "2026-11-03")
    again = BacktestReport.from_dict(json.loads(json.dumps(report.to_dict())))
    assert again == report
    text = "\n".join(format_report(report))
    assert "BACKTEST  SYN" in text and "== 3M" in text and "PIT histogram" in text


def test_cache_key_changes_with_parameters_version_and_data_date():
    k = bt_service.cache_key("AAPL", cfg(), "2026-11-03")
    assert k != bt_service.cache_key("AAPL", cfg(vol_inflation=1.1), "2026-11-03")
    assert k != bt_service.cache_key("AAPL", cfg(method="bootstrap"), "2026-11-03")
    assert k != bt_service.cache_key("AAPL", cfg(), "2026-11-04")
    assert k != bt_service.cache_key("MSFT", cfg(), "2026-11-03")
    assert k == bt_service.cache_key("AAPL", cfg(), "2026-11-03")
    assert f":v{bt_service.MODEL_VERSION}:" in k


def test_backtest_series_uses_the_cache_and_expires(tmp_path, monkeypatch):
    clock = FakeClock()
    cache = SQLiteCache(tmp_path / "bt.sqlite3", clock=clock)
    close = synthetic_close(900, seed=12)
    first = bt_service.backtest_series("SYN", close, cfg(), cache=cache)
    monkeypatch.setattr(bt_service, "run_walk_forward", lambda *a, **k: pytest.fail("recomputed"))
    assert bt_service.backtest_series("SYN", close, cfg(), cache=cache) == first
    clock.advance(bt_service.BACKTEST_CACHE_TTL_SECONDS + 1)
    with pytest.raises(pytest.fail.Exception):
        bt_service.backtest_series("SYN", close, cfg(), cache=cache)


def test_corrupt_cache_entry_falls_back_to_computing(tmp_path):
    cache = SQLiteCache(tmp_path / "bt.sqlite3")
    close = synthetic_close(900, seed=13)
    cache.set(bt_service.cache_key("SYN", cfg(), str(close.index[-1].date())), "{not json")
    report = bt_service.backtest_series("SYN", close, cfg(), cache=cache)
    assert "1M" in report.horizons


def _service(tmp_path, provider):
    clock = FakeClock()
    s = Settings(cache_path=tmp_path / "c.sqlite3", backoff_base_seconds=0.0)
    return MarketDataService(
        provider, SQLiteCache(s.cache_path, clock=clock), s,
        limiter=RateLimiter(10_000, 10_000), clock=clock, sleep=lambda x: None,
    ), s  # fmt: skip


def test_forecast_ticker_attaches_reliability_only_when_asked(tmp_path):
    svc, s = _service(tmp_path, FakeProvider(make_prices(1200, seed=3)))
    plain = forecast_ticker(svc, "AAPL", ["1M"], n_paths=10_000, seed=1, settings=s)
    assert plain.reliability is None
    withrel = forecast_ticker(
        svc, "AAPL", ["1M", "1Y"], n_paths=10_000, seed=1, settings=s, with_reliability=True
    )
    assert set(withrel.reliability) == {"1M", "1Y"}
    assert withrel.reliability["1M"].grade in {"Good", "Fair", "Poor"}
    assert withrel.reliability["1Y"].grade == "Not enough history"
    # the simulation itself is unchanged by asking for reliability
    np.testing.assert_allclose(
        plain.simulations["1M"].terminal, withrel.simulations["1M"].terminal, rtol=1e-9
    )
