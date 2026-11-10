"""Unit tests for the UI helpers that need no running Streamlit app."""

import math
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stock_analyzer.analysis.pipeline import analyze
from stock_analyzer.analysis.technical import adjusted_ohlc, compute_indicators
from stock_analyzer.app import glossary
from stock_analyzer.app.charts import (
    MAX_SAMPLE_PATHS_SHOWN,
    PriceChartOptions,
    fan_chart,
    price_chart,
    terminal_histogram,
)
from stock_analyzer.app.limitations import DISCLAIMER, HOW_IT_WORKS, glossary_markdown
from stock_analyzer.app.presenters import friendly_error as friendly
from stock_analyzer.app.presenters import (
    reliability_views,
    risk_rows,
    scenario_table,
)
from stock_analyzer.app.safety import RequestThrottle, parse_request
from stock_analyzer.backtest.reliability import Reliability
from stock_analyzer.data.types import MarketData
from stock_analyzer.exceptions import (
    InvalidParameterError,
    InvalidTickerError,
    ProviderUnavailableError,
    UnknownBenchmarkError,
)
from stock_analyzer.models.monte_carlo import simulate
from stock_analyzer.models.scenarios import build_scenarios
from stock_analyzer.tests.fakes import make_prices
from stock_analyzer.tests.test_monte_carlo import S0, make_calibration

APP_DIR = Path(__file__).resolve().parents[1] / "app"

# ----------------------------------------------------------------------------- glossary


def test_every_glossary_key_used_in_the_app_exists():
    used = set()
    for path in APP_DIR.rglob("*.py"):
        used |= set(
            re.findall(r"""(?:tip|title)\(\s*["'](\w+)["']\s*\)""", path.read_text("utf-8"))
        )
    used |= {key for _, _, key in risk_rows_for_real_data()}
    assert used and used <= set(glossary.GLOSSARY)


def test_glossary_covers_required_terms_and_is_plain_text():
    required = {"sharpe", "sortino", "beta", "var", "drawdown", "percentile", "rsi", "macd",
                "bollinger", "atr", "support_resistance"}  # fmt: skip
    assert required <= set(glossary.GLOSSARY)
    for name, text in glossary.GLOSSARY.values():
        assert name and len(text) > 40
    md = glossary_markdown()
    assert "Sharpe ratio" in md and "Bollinger" in md


def test_disclaimer_and_limitations_page_contain_the_required_points():
    low = DISCLAIMER.lower()
    assert "not financial advice" in low and "past performance" in low and "not predictions" in low
    page = HOW_IT_WORKS.lower()
    for needle in ("fat tails", "volatility clusters", "regime", "liquidity", "backtest",
                   "can't be verified", "survivorship", "personal data"):  # fmt: skip
        assert needle in page


# -------------------------------------------------------------------------------- safety


@pytest.mark.parametrize(
    "bad", ["", "   ", "AAPL; DROP TABLE", "../etc", "A" * 40, "ÄPL", "AA PL", "<script>", None, 5]
)
def test_bad_tickers_are_rejected(bad):
    with pytest.raises(InvalidTickerError):
        parse_request(bad, "3M")


def test_valid_request_is_normalised():
    r = parse_request(" aapl ", "6M", "bootstrap", "neutral", 20_000, 7, "FTSE100")
    assert (r.ticker, r.horizon, r.method, r.drift, r.paths, r.seed, r.benchmark) == (
        "AAPL", "6M", "bootstrap", "neutral", 20_000, 7, "FTSE100",
    )  # fmt: skip
    assert parse_request("BP.L", "1Y").benchmark is None  # auto


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon": "2Y"}, {"horizon": "1m"}, {"horizon": None},
        {"method": "monte"}, {"drift": "wild"},
        {"paths": 9_999}, {"paths": 50_001}, {"paths": True}, {"paths": "20000"}, {"paths": 1e4},
        {"seed": -1}, {"seed": "x"}, {"seed": 2**31}, {"seed": True},
    ],
)  # fmt: skip
def test_out_of_range_parameters_are_rejected_with_friendly_text(kwargs):
    base = {"ticker": "AAPL", "horizon": "3M"} | kwargs
    with pytest.raises(InvalidParameterError) as exc:
        parse_request(**base)
    assert exc.value.user_message


def test_unknown_benchmark_rejected():
    with pytest.raises(UnknownBenchmarkError):
        parse_request("AAPL", "3M", benchmark="DOGE")


def test_throttle_limits_one_session_and_recovers():
    now = [0.0]
    t = RequestThrottle(max_requests=3, window_seconds=10, clock=lambda: now[0])
    assert [t.try_acquire() for _ in range(3)] == [0.0, 0.0, 0.0]
    wait = t.try_acquire()
    assert 0 < wait <= 10
    now[0] = 10.5
    assert t.try_acquire() == 0.0  # window passed


def test_throttle_state_is_per_session():
    a, b = [], []
    ta = RequestThrottle(a, max_requests=1)
    tb = RequestThrottle(b, max_requests=1)
    assert ta.try_acquire() == 0.0 and ta.try_acquire() > 0
    assert tb.try_acquire() == 0.0  # another session is unaffected


# ---------------------------------------------------------------------------- presenters


def _rel(grade, h, n=40):
    return Reliability(grade, h, f"{grade} text for {h}", ("detail",), 0.9, 0.5, n * 3, n)


def test_reliability_policy_grades_1m_3m_and_flags_6m_1y_unverifiable():
    # even a "Good" 6M/1Y grade must NOT be presented as a grade
    rel = {h: _rel("Good", h) for h in ("1M", "3M", "6M", "1Y")}
    views = {v.horizon: v for v in reliability_views(rel)}
    assert [v.horizon for v in reliability_views(rel)] == ["1M", "3M", "6M", "1Y"]
    assert views["1M"].title == "1M: Good" and views["1M"].verifiable
    assert views["3M"].title == "3M: Good" and views["3M"].icon == "🟢"
    for h in ("6M", "1Y"):
        assert not views[h].verifiable and "can't be verified" in views[h].text
        assert "Good" not in views[h].title


def test_reliability_views_when_backtest_unavailable_or_short():
    views = reliability_views(None)
    assert all(not v.verifiable for v in views)
    assert "unavailable" in views[0].title
    short = reliability_views({"1M": _rel("Not enough history", "1M", 5)})
    assert short[0].title == "1M: Not enough history" and not short[0].verifiable
    assert short[0].icon == "⚪"


def test_friendly_error_never_leaks_internals():
    assert friendly(ProviderUnavailableError("secret stack detail")).startswith("The market data")
    msg = friendly(KeyError("internal_dict_key"))
    assert "internal_dict_key" not in msg and "wrong" in msg


def _analysis(n=900, seed=1):
    md = MarketData(
        "TEST", "USD", make_prices(n, seed=seed), None, "fake", datetime.now(UTC), False, False, ()
    )
    return analyze(md)


def risk_rows_for_real_data():
    return risk_rows(_analysis().risk, None)


def test_risk_rows_have_labels_values_and_glossary_keys():
    rows = risk_rows_for_real_data()
    labels = [r[0] for r in rows]
    assert "Volatility (yearly)" in labels and any("VaR" in x for x in labels)
    assert all(isinstance(v, str) and v for _, v, _ in rows)


def _scenarios(h=126):
    c = make_calibration(sigma=0.3)
    kw = {"n_paths": 10_000, "seed": 3}
    base = simulate(c, S0, h, **kw)
    return base, build_scenarios(
        base, simulate(c, S0, h, sigma_scale=0.8, **kw), simulate(c, S0, h, sigma_scale=1.2, **kw)
    )


def test_scenario_table_is_bull_first_and_probabilities_sum_to_100():
    _, ss = _scenarios()
    df = scenario_table(ss, "USD")
    assert list(df["Scenario"]) == ["Bull", "Base", "Bear"]
    total = sum(float(x.rstrip("%")) for x in df["Chance"])
    assert 99 <= total <= 101
    assert all("USD" in x for x in df["Middle outcome"])
    assert df["If volatility is 20% lower/higher"].str.contains(" to ").all()


def test_scenario_table_handles_empty_regions():
    _, ss = _scenarios(h=21)
    sc = ss.scenarios[0]
    nan_sc = type(sc)(**{**sc.__dict__, "price_low": math.nan, "price_high": math.nan,
                         "target_price": math.nan})  # fmt: skip
    ss2 = type(ss)(ss.horizon_days, ss.s0, ss.threshold, (nan_sc, *ss.scenarios[1:]), ss.notes)
    assert scenario_table(ss2, "USD").iloc[-1]["Typical price range"] == "n/a"


# ---------------------------------------------------------------------------------- charts


@pytest.fixture(scope="module")
def chart_data():
    ohlc = adjusted_ohlc(make_prices(700, seed=2))
    return ohlc, compute_indicators(ohlc)


def test_price_chart_traces_follow_the_options(chart_data):
    ohlc, ind = chart_data
    base = price_chart(ohlc, ind, PriceChartOptions(sma=(), volume=False, rsi=False), "USD")
    assert [t.type for t in base.data] == ["candlestick"]
    full = price_chart(
        ohlc,
        ind,
        PriceChartOptions(
            sma=(20, 50), ema=(200,), bollinger=True, volume=True, rsi=True, macd=True
        ),
        "USD",
    )
    names = [t.name for t in full.data]
    assert {"SMA 20", "SMA 50", "EMA 200", "Bollinger upper", "Bollinger lower", "Volume",
            "RSI 14", "MACD", "Signal"} <= set(names)  # fmt: skip
    line = price_chart(ohlc, ind, PriceChartOptions(style="line", sma=()), None)
    assert line.data[0].type == "scatter" and line.data[0].name == "Price"
    assert full.layout.height > base.layout.height  # more panels, taller figure


def test_price_chart_limits_the_visible_window_and_marks_levels(chart_data):
    ohlc, ind = chart_data
    fig = price_chart(ohlc, ind, PriceChartOptions(days=100, levels=(100.0, 120.0)), "USD")
    assert len(fig.data[0].x) == 100
    assert len(fig.layout.shapes) >= 2 + 2  # two levels plus the RSI 30/70 lines


def test_price_chart_survives_short_history():
    ohlc = adjusted_ohlc(make_prices(80, seed=5))
    fig = price_chart(ohlc, compute_indicators(ohlc), PriceChartOptions(sma=(20, 200)), "USD")
    assert len(fig.data) >= 1  # SMA 200 is all-NaN but must not crash


def test_fan_chart_has_bands_median_and_capped_sample_paths():
    sim, _ = _scenarios(h=63)
    hist = pd.Series(np.linspace(80, S0, 200), index=pd.bdate_range(end="2026-11-03", periods=200))
    fig = fan_chart(sim, hist, pd.Timestamp("2026-11-03"), "USD")
    names = [t.name for t in fig.data]
    assert "Median outcome" in names and "5th-95th percentile" in names
    assert "25th-75th percentile" in names and "Past price" in names
    paths = [t for t in fig.data if t.name == "Example simulated paths" or t.legendgroup == "s"]
    assert len(paths) == MAX_SAMPLE_PATHS_SHOWN
    median = next(t for t in fig.data if t.name == "Median outcome")
    assert len(median.y) == 64 and median.y[0] == pytest.approx(S0)
    none = fan_chart(sim, hist, pd.Timestamp("2026-11-03"), None, show_sample_paths=False)
    assert not any(t.legendgroup == "s" for t in none.data)


def test_terminal_histogram_marks_today_and_percentiles():
    sim, _ = _scenarios(h=63)
    fig = terminal_histogram(sim, "USD")
    assert fig.data[0].type == "histogram" and len(fig.data[0].x) == sim.n_paths
    assert len(fig.layout.shapes) == 4
    xs = sorted(s.x0 for s in fig.layout.shapes)
    assert xs[0] == pytest.approx(sim.percentile(5)) or xs[0] <= sim.s0
