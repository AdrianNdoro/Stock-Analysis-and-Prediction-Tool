import math

import numpy as np
import pytest

from stock_analyzer.analysis.pipeline import analyze_ticker
from stock_analyzer.config import Settings
from stock_analyzer.data.cache import SQLiteCache
from stock_analyzer.data.resilience import RateLimiter
from stock_analyzer.data.service import MarketDataService
from stock_analyzer.exceptions import InvalidParameterError
from stock_analyzer.models.forecast import forecast, forecast_ticker
from stock_analyzer.models.monte_carlo import simulate
from stock_analyzer.models.scenarios import build_scenarios
from stock_analyzer.tests.fakes import FakeClock, make_prices
from stock_analyzer.tests.test_benchmarks_and_pipeline import MultiProvider
from stock_analyzer.tests.test_monte_carlo import S0, make_calibration


def three_sims(sigma=0.3, h=126, seed=3, n=20_000):
    c = make_calibration(sigma=sigma)
    kw = {"n_paths": n, "seed": seed}
    return (
        simulate(c, S0, h, **kw),
        simulate(c, S0, h, sigma_scale=0.8, **kw),
        simulate(c, S0, h, sigma_scale=1.2, **kw),
    )


def test_probabilities_partition_and_match_direct_counts():
    base, lo, hi = three_sims()
    ss = build_scenarios(base, lo, hi)
    bear, mid, bull = ss.scenarios
    assert [s.name for s in ss.scenarios] == ["Bear", "Base", "Bull"]
    assert bear.probability + mid.probability + bull.probability == pytest.approx(1.0)
    hw = ss.threshold
    assert hw == pytest.approx(0.5 * 0.3 * math.sqrt(126 / 252))
    assert bear.probability == pytest.approx(base.prob_loss_more_than(hw))
    assert bull.probability == pytest.approx(base.prob_gain_more_than(hw))


def test_price_ranges_sit_inside_their_regions_and_are_ordered():
    base, lo, hi = three_sims()
    ss = build_scenarios(base, lo, hi)
    bear, mid, bull = ss.scenarios
    lower, upper = S0 * (1 - ss.threshold), S0 * (1 + ss.threshold)
    assert bear.price_low <= bear.target_price <= bear.price_high <= lower
    assert lower < mid.price_low <= mid.target_price <= mid.price_high < upper
    assert upper <= bull.price_low <= bull.target_price <= bull.price_high
    assert bear.target_price < mid.target_price < bull.target_price


def test_sensitivity_ranges_bracket_the_base_probability():
    base, lo, hi = three_sims()
    for s in build_scenarios(base, lo, hi).scenarios:
        assert s.probability_low <= s.probability <= s.probability_high
    bear, mid, bull = build_scenarios(base, lo, hi).scenarios
    # lower vol -> more mass in the middle; higher vol -> more in the tails
    assert mid.probability_high > mid.probability > mid.probability_low
    assert bull.probability_high > bull.probability_low


def test_minimum_threshold_applies_to_very_calm_assets():
    base, lo, hi = three_sims(sigma=0.04, h=21)
    assert build_scenarios(base, lo, hi).threshold == 0.03


def test_scenarios_describe_assumptions_without_tilting_probabilities():
    base, lo, hi = three_sims()
    plain = build_scenarios(base, lo, hi, None)
    assert all(s.indicator_context == "No technical context available." for s in plain.scenarios)
    assert any("Volatility assumed: 30%" in a for a in plain.scenarios[1].assumptions)
    assert "not forecasts" in plain.notes[0]


# ----------------------------------------------------------------------------- forecast glue


def make_service(tmp_path, frames):
    clock = FakeClock()
    s = Settings(cache_path=tmp_path / "c.sqlite3", backoff_base_seconds=0.0)
    svc = MarketDataService(
        MultiProvider(frames), SQLiteCache(s.cache_path, clock=clock), s,
        limiter=RateLimiter(10_000, 10_000), clock=clock, sleep=lambda x: None,
    )  # fmt: skip
    return svc, s


def test_forecast_ticker_end_to_end(tmp_path):
    frames = {"AAPL": make_prices(1000, seed=1), "^GSPC": make_prices(1000, seed=2)}
    svc, s = make_service(tmp_path, frames)
    f = forecast_ticker(svc, "aapl", ["1M", "3m", "6M", "1Y"], seed=11, settings=s)
    assert list(f.simulations) == ["1M", "3M", "6M", "1Y"]
    assert f.simulations["1Y"].horizon_days == 252 and f.simulations["1Y"].n_paths == 10_000
    spreads = [f.simulations[k].expected_range() for k in ("1M", "3M", "6M", "1Y")]
    widths = [hi - lo for lo, hi in spreads]
    assert widths == sorted(widths)  # longer horizon, wider range
    assert f.scenarios["6M"].s0 == pytest.approx(f.analysis.last_price)
    assert f.scenarios["3M"].scenarios[0].indicator_context.startswith("The chart picture")


def test_forecast_is_reproducible_with_a_seed(tmp_path):
    svc, s = make_service(tmp_path, {"AAPL": make_prices(900), "^GSPC": make_prices(900, seed=2)})
    a = analyze_ticker(svc, "AAPL", settings=s)
    one = forecast(a, ["3M"], seed=5)
    two = forecast(a, ["3M"], seed=5)
    np.testing.assert_array_equal(one.simulations["3M"].terminal, two.simulations["3M"].terminal)
    assert (
        one.scenarios["3M"].scenarios[0].probability == two.scenarios["3M"].scenarios[0].probability
    )


def test_forecast_rejects_unsupported_horizon_and_paths(tmp_path):
    svc, s = make_service(tmp_path, {"AAPL": make_prices(900), "^GSPC": make_prices(900, seed=2)})
    with pytest.raises(InvalidParameterError):
        forecast_ticker(svc, "AAPL", ["2Y"], settings=s)
    with pytest.raises(InvalidParameterError):
        forecast_ticker(svc, "AAPL", ["1M"], n_paths=100, settings=s)


def test_bootstrap_method_flows_through(tmp_path):
    svc, s = make_service(tmp_path, {"AAPL": make_prices(900), "^GSPC": make_prices(900, seed=2)})
    f = forecast_ticker(svc, "AAPL", ["3M"], method="bootstrap", seed=1, settings=s)
    assert f.simulations["3M"].method == "bootstrap"
    assert "bootstrapped" in " ".join(f.simulations["3M"].plain_english("USD"))
