"""Technical indicators, support/resistance and a plain-English summary.

Statistical honesty: every indicator here is a *backward-looking transformation of past
prices*. It describes what has happened (trend, momentum, stretch) and has no proven power to
predict what happens next. The summary says so explicitly. Indicators also behave differently
in trending vs sideways markets, and signals lag by construction (moving averages) or whipsaw
(RSI/MACD in choppy markets).

Prices are split- and dividend-adjusted so that indicators are continuous across corporate
actions. Because of that, old price levels are slightly lower than what was quoted at the
time; the most recent bar always equals the actual price.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- price prep


def adjusted_ohlc(prices: pd.DataFrame) -> pd.DataFrame:
    """OHLC scaled by Adj Close / Close so splits and dividends don't create fake cliffs.

    Returns columns Open, High, Low, Close (adjusted) and Volume.
    """
    factor = prices["Adj Close"] / prices["Close"]
    return pd.DataFrame(
        {
            "Open": prices["Open"] * factor,
            "High": prices["High"] * factor,
            "Low": prices["Low"] * factor,
            "Close": prices["Adj Close"],
            "Volume": prices["Volume"],
        },
        index=prices.index,
    )


# --------------------------------------------------------------------------- indicators


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average; NaN until ``window`` observations exist."""
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average (alpha = 2/(span+1)), seeded with the first price.

    Seeding differs slightly from charting packages that seed with an SMA; the difference
    fades after a few multiples of ``span``.
    """
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _wilder_smooth(values: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing: first value = simple mean of ``period`` values, then
    ``avg = (prev_avg * (period - 1) + x) / period``. Used by RSI and ATR."""
    v = values.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    valid = np.flatnonzero(~np.isnan(v))
    if len(valid):
        seed = valid[0] + period - 1
        if seed < len(v):
            out[seed] = v[valid[0] : seed + 1].mean()
            for i in range(seed + 1, len(v)):
                out[i] = (out[i - 1] * (period - 1) + v[i]) / period
    return pd.Series(out, index=values.index)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index, 0-100.

    100 when there were gains but no losses in the window; NaN when price didn't move at all.
    """
    delta = close.diff()
    avg_gain = _wilder_smooth(delta.clip(lower=0), period)
    avg_loss = _wilder_smooth((-delta).clip(lower=0), period)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = 100 - 100 / (1 + avg_gain / avg_loss)
    value = value.where(avg_loss != 0, np.where(avg_gain > 0, 100.0, np.nan))
    return value.where(avg_gain.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line (fast EMA - slow EMA), signal line (EMA of MACD) and histogram."""
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def bollinger(close: pd.Series, window: int = 20, k: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands: SMA +/- k standard deviations (population std, as is conventional).

    ``pct_b`` is where price sits in the band (0 = lower band, 1 = upper band).
    """
    mid = sma(close, window)
    sd = close.rolling(window, min_periods=window).std(ddof=0)
    upper, lower = mid + k * sd, mid - k * sd
    width = (upper - lower).replace(0, np.nan)
    return pd.DataFrame(
        {
            "bb_mid": mid,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_pct_b": (close - lower) / width,
            "bb_bandwidth": (upper - lower) / mid,
        }
    )


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder's true range: the largest of today's range and the gaps from yesterday's close."""
    prev = close.shift(1)
    return pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (Wilder smoothing): typical daily movement in price units."""
    return _wilder_smooth(true_range(high, low, close), period)


def compute_indicators(ohlc: pd.DataFrame) -> pd.DataFrame:
    """All indicators for adjusted OHLCV data, aligned to the input index."""
    close = ohlc["Close"]
    out = pd.DataFrame({"close": close})
    for n in (20, 50, 200):
        out[f"sma_{n}"] = sma(close, n)
    for n in (20, 50, 200):
        out[f"ema_{n}"] = ema(close, n)
    out["rsi_14"] = rsi(close, 14)
    out = out.join(macd(close)).join(bollinger(close))
    out["atr_14"] = atr(ohlc["High"], ohlc["Low"], close, 14)
    out["atr_pct"] = out["atr_14"] / close
    out["vol_avg_20"] = ohlc["Volume"].rolling(20, min_periods=20).mean()
    out["vol_avg_60"] = ohlc["Volume"].rolling(60, min_periods=60).mean()
    return out


# --------------------------------------------------------------------------- support / resistance


@dataclass(frozen=True)
class Level:
    price: float
    touches: int  # how many swing points clustered here; more touches = more significant
    kind: str  # "support" (below price) or "resistance" (above price)
    distance_pct: float  # level / current price - 1


def find_levels(
    ohlc: pd.DataFrame,
    *,
    lookback: int = 252,
    order: int = 5,
    tolerance: float = 0.015,
    max_levels: int = 3,
) -> list[Level]:
    """Support/resistance from clustered swing highs and lows.

    A swing high (low) is a bar whose high (low) is the highest (lowest) within ``order`` bars
    on each side. Swing points within ``tolerance`` (1.5%) of each other are merged into one
    level whose strength is the number of touches. The nearest ``max_levels`` below and above
    the latest close are returned. This is a heuristic: levels are where price *has* turned,
    not guarantees that it will again.
    """
    window = ohlc.iloc[-lookback:]
    highs, lows = window["High"].to_numpy(), window["Low"].to_numpy()
    n = len(window)
    points: list[float] = []
    last_hi = last_lo = -(10**9)
    for i in range(order, n - order):
        if highs[i] == highs[i - order : i + order + 1].max() and i - last_hi > order:
            points.append(float(highs[i]))
            last_hi = i
        if lows[i] == lows[i - order : i + order + 1].min() and i - last_lo > order:
            points.append(float(lows[i]))
            last_lo = i
    if not points:
        return []

    clusters: list[list[float]] = [[]]
    for p in sorted(points):
        current = clusters[-1]
        if current and abs(p - np.mean(current)) / np.mean(current) > tolerance:
            clusters.append([])
        clusters[-1].append(p)

    price = float(window["Close"].iloc[-1])
    levels = [
        Level(
            price=float(np.mean(c)),
            touches=len(c),
            kind="support" if np.mean(c) < price else "resistance",
            distance_pct=float(np.mean(c) / price - 1),
        )
        for c in clusters
    ]
    below = sorted((x for x in levels if x.kind == "support"), key=lambda x: -x.price)
    above = sorted((x for x in levels if x.kind == "resistance"), key=lambda x: x.price)
    return below[:max_levels] + above[:max_levels]


# --------------------------------------------------------------------------- summary


@dataclass(frozen=True)
class TechnicalSnapshot:
    as_of: date
    price: float
    values: dict[str, float | None]  # latest value of every indicator
    levels: list[Level]
    trend_score: int  # sum of +1/-1 trend and momentum signals that could be evaluated
    trend_max: int  # number of signals that could be evaluated
    trend_label: str
    bullets: list[str] = field(default_factory=list)  # plain-English findings


def _latest(ind: pd.DataFrame) -> dict[str, float | None]:
    row = ind.iloc[-1]
    return {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}


def _label(score: int, maximum: int) -> str:
    if maximum == 0:
        return "Not enough data"
    r = score / maximum
    if r >= 0.75:
        return "Bullish"
    if r >= 0.25:
        return "Leaning bullish"
    if r > -0.25:
        return "Mixed"
    if r > -0.75:
        return "Leaning bearish"
    return "Bearish"


def summarize_technicals(
    ind: pd.DataFrame, levels: list[Level], currency: str | None = None
) -> TechnicalSnapshot:
    """Turn the latest indicator values into plain-English findings and a simple score.

    The score counts up to four signals (price vs 200-day, 50-day vs 200-day, price vs
    50-day, MACD vs its signal line): +1 each when bullish, -1 when bearish. It is a
    description of the current chart, not a forecast.
    """
    v = _latest(ind)
    price = v["close"]
    cur = f"{currency} " if currency else ""
    bullets: list[str] = []
    score = maximum = 0

    def signal(is_bullish: bool) -> None:
        nonlocal score, maximum
        score += 1 if is_bullish else -1
        maximum += 1

    s20, s50, s200 = v["sma_20"], v["sma_50"], v["sma_200"]
    if s200 is not None:
        diff = price / s200 - 1
        up = price > s200
        signal(up)
        bullets.append(
            f"Long-term trend: price ({cur}{price:,.2f}) is {abs(diff):.1%} "
            f"{'above' if up else 'below'} its 200-day average ({cur}{s200:,.2f})."
        )
    else:
        bullets.append(
            "Long-term trend: not enough history for a 200-day average yet, so long-term "
            "trend signals are left out."
        )
    if s50 is not None and s200 is not None:
        up = s50 > s200
        signal(up)
        bullets.append(
            f"The 50-day average is {'above' if up else 'below'} the 200-day average, "
            f"which {'supports an uptrend' if up else 'is typical of a downtrend'} "
            "(slow-moving, so it lags turning points)."
        )
    if s50 is not None:
        up = price > s50
        signal(up)
        bullets.append(
            f"Medium-term: price is {'above' if up else 'below'} its 50-day average "
            f"({cur}{s50:,.2f})."
        )
    elif s20 is not None:
        bullets.append(f"Short history: price is {price / s20 - 1:+.1%} versus its 20-day average.")

    m, ms = v["macd"], v["macd_signal"]
    if m is not None and ms is not None:
        up = m > ms
        signal(up)
        bullets.append(
            f"Momentum (MACD): the MACD line is {'above' if up else 'below'} its signal line, "
            f"so short-term momentum is {'improving' if up else 'weakening'}."
        )

    r = v["rsi_14"]
    if r is not None:
        if r >= 70:
            zone = "overbought territory: the recent rise has been strong and stretched"
        elif r <= 30:
            zone = "oversold territory: the recent fall has been sharp and stretched"
        else:
            zone = "a neutral zone (between 30 and 70)"
        bullets.append(
            f"RSI is {r:.0f}, in {zone}. Strong trends can stay stretched for a long time."
        )

    pb = v["bb_pct_b"]
    if pb is not None:
        where = (
            "above the upper Bollinger Band (unusually high versus the last 20 days)"
            if pb > 1
            else "below the lower Bollinger Band (unusually low versus the last 20 days)"
            if pb < 0
            else "inside its Bollinger Bands (a normal range versus the last 20 days)"
        )
        bullets.append(f"Price is {where}.")

    if v["atr_pct"] is not None:
        bullets.append(
            f"Typical daily movement (ATR) is about {v['atr_pct']:.1%} of the price "
            f"({cur}{v['atr_14']:,.2f})."
        )

    if v["vol_avg_20"] is not None and v["vol_avg_60"]:
        ratio = v["vol_avg_20"] / v["vol_avg_60"]
        if ratio > 1.25:
            vol_text = (
                f"Trading volume is elevated: the 20-day average is {ratio - 1:.0%} "
                "above the 3-month norm."
            )
        elif ratio < 0.75:
            vol_text = (
                f"Trading volume is subdued: the 20-day average is {1 - ratio:.0%} "
                "below the 3-month norm."
            )
        else:
            vol_text = "Trading volume is in line with its 3-month norm."
        bullets.append(vol_text)

    sup = next((x for x in levels if x.kind == "support"), None)
    res = next((x for x in levels if x.kind == "resistance"), None)
    if sup:
        bullets.append(
            "Nearest support (where price has previously turned up): "
            f"about {cur}{sup.price:,.2f} "
            f"({sup.distance_pct:+.1%}), touched {sup.touches} time(s) in the past year."
        )
    if res:
        bullets.append(
            "Nearest resistance (where price has previously turned down): "
            f"about {cur}{res.price:,.2f} "
            f"({res.distance_pct:+.1%}), touched {res.touches} time(s) in the past year."
        )

    label = _label(score, maximum)
    bullets.insert(
        0,
        f"Overall chart picture: {label}"
        + (f" ({score:+d} on a scale of ±{maximum})." if maximum else "."),
    )
    bullets.append(
        "These indicators describe past prices. They are not predictions and can give "
        "false signals, especially in choppy markets."
    )
    return TechnicalSnapshot(
        as_of=ind.index[-1].date(),
        price=price,
        values=v,
        levels=levels,
        trend_score=score,
        trend_max=maximum,
        trend_label=label,
        bullets=bullets,
    )
