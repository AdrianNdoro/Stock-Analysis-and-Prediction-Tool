import numpy as np
import pandas as pd
import pytest

from stock_analyzer.analysis.technical import (
    adjusted_ohlc,
    atr,
    bollinger,
    compute_indicators,
    ema,
    find_levels,
    macd,
    rsi,
    sma,
    summarize_technicals,
    true_range,
)
from stock_analyzer.tests.fakes import make_prices

# Closes used by many RSI tutorials (Wilder's 14-period example).
CLOSES = [
    44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61,
    46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64, 46.21, 46.25, 45.71, 46.45, 45.78, 45.35,
    44.03, 44.18, 44.22, 44.57, 43.42, 42.66, 43.13,
]  # fmt: skip


def series(values):
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)), dtype=float)


def test_sma_hand_computed():
    out = sma(series([1, 2, 3, 4, 5]), 3)
    assert out.isna().tolist() == [True, True, False, False, False]
    assert out.dropna().tolist() == [2.0, 3.0, 4.0]


def test_ema_hand_computed():
    # span=3 -> alpha=0.5, seeded with the first value: 1, 1.5, 2.25, 3.125
    out = ema(series([1, 2, 3, 4]), 3)
    assert out.isna().tolist() == [True, True, False, False]
    assert out.iloc[2] == pytest.approx(2.25) and out.iloc[3] == pytest.approx(3.125)


def reference_rsi(closes, period=14):
    """Straight-line implementation of Wilder's published recipe, independent of the code."""
    d = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains, losses = [max(x, 0) for x in d], [max(-x, 0) for x in d]
    ag, al = sum(gains[:period]) / period, sum(losses[:period]) / period
    out = [100 - 100 / (1 + ag / al)]
    for i in range(period, len(d)):
        ag = (ag * (period - 1) + gains[i]) / period
        al = (al * (period - 1) + losses[i]) / period
        out.append(100 - 100 / (1 + ag / al))
    return out


def test_rsi_matches_reference_implementation():
    got = rsi(series(CLOSES), 14)
    assert got.iloc[:14].isna().all()  # first value appears after 14 price changes
    np.testing.assert_allclose(got.iloc[14:].to_numpy(), reference_rsi(CLOSES), rtol=1e-9)
    assert got.iloc[14] == pytest.approx(70.46, abs=0.01)
    assert got.iloc[15] == pytest.approx(66.25, abs=0.01)


def test_rsi_extremes():
    assert rsi(series(range(1, 40))).dropna().eq(100).all()  # only gains
    assert rsi(series(range(40, 1, -1))).dropna().eq(0).all()  # only losses
    assert rsi(series([10.0] * 40)).isna().all()  # flat: undefined


def test_macd_identities_and_flat_series():
    close = series(make_prices(300)["Close"].to_numpy())
    m = macd(close)
    pd.testing.assert_series_equal(m["macd"], ema(close, 12) - ema(close, 26), check_names=False)
    pd.testing.assert_series_equal(m["macd_hist"], m["macd"] - m["macd_signal"], check_names=False)
    assert m["macd"].first_valid_index() == close.index[25]
    flat = macd(series([50.0] * 100))
    assert flat.dropna().abs().max().max() == pytest.approx(0.0, abs=1e-12)


def test_bollinger_hand_computed_and_constant():
    b = bollinger(series([1, 2, 3]), window=3, k=2)
    sd = np.sqrt(2 / 3)  # population std of 1,2,3
    assert b["bb_mid"].iloc[-1] == 2 and b["bb_upper"].iloc[-1] == pytest.approx(2 + 2 * sd)
    assert b["bb_pct_b"].iloc[-1] == pytest.approx((3 - (2 - 2 * sd)) / (4 * sd))
    flat = bollinger(series([7.0] * 30))
    assert (flat["bb_upper"].dropna() == 7).all() and flat["bb_pct_b"].isna().all()


def test_true_range_uses_gaps_and_atr_of_constant_range_is_that_range():
    s = lambda v: pd.Series(v, dtype=float)  # noqa: E731
    # day 2 gaps up: yesterday's close was 7, today's range is 9-10 -> true range is 10 - 7 = 3
    tr = true_range(s([10, 10]), s([9, 9]), s([7.0, 9.5]))
    assert tr.tolist() == [1.0, 3.0]
    high, low, close = s([12.0] * 40), s([10.0] * 40), s([11.0] * 40)
    assert atr(high, low, close, 14).dropna().eq(2.0).all()
    assert atr(high, low, close, 14).first_valid_index() == 13


def test_adjusted_ohlc_scales_by_adjustment_factor():
    p = make_prices(10)
    adj = adjusted_ohlc(p)
    np.testing.assert_allclose(adj["Close"], p["Adj Close"])
    np.testing.assert_allclose(adj["High"], p["High"] * 0.98)  # fake data uses factor 0.98


def test_levels_found_on_a_repeating_range():
    n = 300
    t = np.arange(n)
    close = 100 + 10 * np.sin(2 * np.pi * t / 40)  # oscillates 90..110, ~7 swings
    ohlc = pd.DataFrame(
        {"Open": close, "High": close + 0.5, "Low": close - 0.5, "Close": close, "Volume": 1.0},
        index=pd.bdate_range("2024-01-01", periods=n),
    )
    levels = find_levels(ohlc)
    res = [x for x in levels if x.kind == "resistance"]
    sup = [x for x in levels if x.kind == "support"]
    assert res and sup
    assert res[0].price == pytest.approx(110.5, abs=0.6) and res[0].touches >= 3
    assert sup[0].price == pytest.approx(89.5, abs=0.6) and sup[0].touches >= 3
    assert all(x.price < ohlc["Close"].iloc[-1] for x in sup)


def _trend(daily_drift, n=400, noise=0.0):
    rng = np.random.default_rng(1)
    close = 100 * np.exp(
        np.cumsum(daily_drift + rng.normal(0, noise, n) if noise else np.full(n, daily_drift))
    )
    return pd.DataFrame(
        {"Open": close, "High": close * 1.004, "Low": close * 0.996, "Close": close,
         "Volume": 1e6}, index=pd.bdate_range("2023-01-02", periods=n))  # fmt: skip


def test_summary_labels_a_clean_uptrend_bullish():
    ohlc = _trend(0.004)  # noise-free: every signal agrees
    snap = summarize_technicals(compute_indicators(ohlc), find_levels(ohlc), "USD")
    assert (snap.trend_label, snap.trend_score, snap.trend_max) == ("Bullish", 4, 4)
    assert snap.bullets[0].startswith("Overall chart picture")
    assert "not predictions" in snap.bullets[-1]
    assert any("above its 200-day average" in b for b in snap.bullets)


def test_summary_labels_a_clean_downtrend_bearish_side():
    # In an exponential decline the daily drops shrink, so MACD drifts back towards zero and
    # can read "improving" while the three trend signals stay bearish: still a bearish picture.
    ohlc = _trend(-0.004)
    snap = summarize_technicals(compute_indicators(ohlc), find_levels(ohlc), "USD")
    assert snap.trend_label in {"Bearish", "Leaning bearish"} and snap.trend_score <= -2
    assert any("below its 200-day average" in b for b in snap.bullets)


def test_noisy_uptrend_can_have_mixed_signals_but_leans_bullish():
    # Real charts disagree with themselves: here 3 of 4 signals are bullish (MACD cooling).
    ohlc = _trend(0.004, noise=0.004)
    snap = summarize_technicals(compute_indicators(ohlc), find_levels(ohlc), "USD")
    assert snap.trend_label in {"Bullish", "Leaning bullish"} and snap.trend_score >= 2


def test_summary_handles_short_history():
    ohlc = _trend(0.001, n=120, noise=0.004)
    snap = summarize_technicals(compute_indicators(ohlc), [], None)
    assert snap.values["sma_200"] is None
    assert any("not enough history for a 200-day" in b for b in snap.bullets)
    assert snap.trend_max == 2  # price vs 50-day and MACD only
