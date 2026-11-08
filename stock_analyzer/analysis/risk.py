"""Risk metrics. Returns are simple daily returns of split/dividend-adjusted closes.

Assumptions and where they break (see also each function):
* Everything is estimated from a *past window* and assumes the future resembles it. Regime
  changes (crashes, rate shocks) break that, and volatility clusters: calm periods understate
  the risk of the next turbulent one.
* Sharpe/Sortino/volatility treat the mean and standard deviation as stable; real returns are
  fat-tailed and skewed, so these ratios flatter strategies with rare large losses.
* Annualisation uses 252 trading days and the square-root-of-time rule, which assumes
  returns are independent from day to day.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from statistics import NormalDist

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MIN_RETURNS = 30  # below this, risk statistics are not computed


def simple_returns(close: pd.Series) -> pd.Series:
    """Daily simple returns (P_t / P_t-1 - 1), first NaN dropped."""
    return close.pct_change().dropna()


def daily_rate(annual_rate: float) -> float:
    """Compound-equivalent daily rate for an annual rate."""
    return (1.0 + annual_rate) ** (1.0 / TRADING_DAYS) - 1.0


def annualised_volatility(returns: pd.Series) -> float:
    """Sample standard deviation of daily returns x sqrt(252)."""
    if len(returns) < 2:
        return math.nan
    return float(returns.std(ddof=1) * math.sqrt(TRADING_DAYS))


def cagr(close: pd.Series) -> float:
    """Compound annual growth rate over the window of ``close``."""
    n = len(close) - 1
    if n < 1 or close.iloc[0] <= 0:
        return math.nan
    return float((close.iloc[-1] / close.iloc[0]) ** (TRADING_DAYS / n) - 1.0)


@dataclass(frozen=True)
class DrawdownResult:
    max_drawdown: float  # negative fraction, e.g. -0.35 = a 35% fall from peak to trough
    peak_date: date
    trough_date: date
    recovery_date: date | None  # None = not yet recovered to the old peak
    days_peak_to_trough: int
    currently_in_drawdown: float  # today's fall from the running peak (<= 0)


def max_drawdown(close: pd.Series) -> DrawdownResult:
    """Largest peak-to-trough fall and when it happened."""
    running_peak = close.cummax()
    dd = close / running_peak - 1.0
    trough = dd.idxmin()
    peak = close.loc[:trough].idxmax()
    after = close.loc[trough:]
    recovered = after[after >= close.loc[peak]]
    return DrawdownResult(
        max_drawdown=float(dd.min()),
        peak_date=peak.date(),
        trough_date=trough.date(),
        recovery_date=recovered.index[0].date() if len(recovered) else None,
        days_peak_to_trough=int((trough - peak).days),
        currently_in_drawdown=float(dd.iloc[-1]),
    )


def sharpe_ratio(returns: pd.Series, risk_free_annual: float) -> float:
    """Annualised Sharpe: mean excess return / std of excess return x sqrt(252).

    Assumes returns are roughly symmetric; it penalises upside and downside swings equally.
    NaN when there is no variation.
    """
    if len(returns) < 2:
        return math.nan
    excess = returns - daily_rate(risk_free_annual)
    sd = excess.std(ddof=1)
    if not sd or math.isnan(sd):
        return math.nan
    return float(excess.mean() / sd * math.sqrt(TRADING_DAYS))


def sortino_ratio(returns: pd.Series, risk_free_annual: float) -> float:
    """Annualised Sortino: mean excess return / downside deviation x sqrt(252).

    The minimum acceptable return is the daily risk-free rate. Downside deviation is the root
    mean square of shortfalls below it, over *all* observations. NaN if there were no
    shortfalls (nothing to divide by).
    """
    if len(returns) < 2:
        return math.nan
    mar = daily_rate(risk_free_annual)
    excess = returns - mar
    downside = float(np.sqrt(np.mean(np.minimum(excess.to_numpy(), 0.0) ** 2)))
    if downside == 0.0:
        return math.nan
    return float(excess.mean() / downside * math.sqrt(TRADING_DAYS))


# --------------------------------------------------------------------------- beta


@dataclass(frozen=True)
class BetaResult:
    beta: float
    r_squared: float  # share of the stock's variation explained by the benchmark
    n_obs: int
    frequency: str  # "D" daily or "W" weekly
    start: date | None
    end: date | None
    note: str | None = None


def beta_from_returns(stock: pd.Series, bench: pd.Series) -> tuple[float, float, int]:
    """OLS slope of stock returns on benchmark returns -> (beta, R^2, n). Aligns on dates."""
    both = pd.concat([stock, bench], axis=1, join="inner").dropna()
    n = len(both)
    if n < 3:
        return math.nan, math.nan, n
    s, b = both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy()
    var_b = np.var(b, ddof=1)
    if var_b == 0:
        return math.nan, math.nan, n
    cov = np.cov(s, b, ddof=1)[0, 1]
    corr = np.corrcoef(s, b)[0, 1]
    return float(cov / var_b), float(corr**2), n


def compute_beta(
    stock_close: pd.Series,
    bench_close: pd.Series,
    *,
    freq: str = "W",
    lookback_years: float = 3.0,
    min_obs: int = 52,
) -> BetaResult:
    """Beta of the stock versus a benchmark, in *local currency* returns.

    Weekly returns (default) reduce the distortion from markets that trade at different
    hours (e.g. Tokyo stock vs S&P 500), which biases daily beta towards zero. Beta is a
    single-factor, linear, backward-looking estimate: it moves around over time and says
    nothing about company-specific risk (see R^2).
    """
    if freq not in {"D", "W"}:
        raise ValueError("freq must be 'D' or 'W'")
    s, b = stock_close, bench_close
    if freq == "W":
        s = s.resample("W-FRI").last().dropna()
        b = b.resample("W-FRI").last().dropna()
    common = s.index.intersection(b.index)
    if len(common):
        cutoff = common[-1] - pd.Timedelta(days=int(lookback_years * 365.25))
        common = common[common >= cutoff]
    s, b = s.loc[common], b.loc[common]
    beta, r2, n = beta_from_returns(simple_returns(s), simple_returns(b))
    if n < min_obs or math.isnan(beta):
        return BetaResult(
            math.nan, math.nan, n, freq, None, None,
            note=f"Beta not computed: only {n} overlapping observations (need {min_obs}).",
        )  # fmt: skip
    return BetaResult(beta, r2, n, freq, common[0].date(), common[-1].date())


# --------------------------------------------------------------------------- VaR


@dataclass(frozen=True)
class VaRResult:
    """Losses are positive fractions: 0.03 means a 3% loss."""

    confidence: float
    horizon_days: int
    historical: float
    parametric: float
    expected_shortfall: float  # average loss on the days beyond the historical VaR


def value_at_risk(returns: pd.Series, confidence: float = 0.95, horizon_days: int = 1) -> VaRResult:
    """Value at Risk two ways, plus Expected Shortfall.

    * historical: the empirical (1 - confidence) quantile of past returns; no distribution
      assumed, but only as good as the sample (few extreme days in the window).
    * parametric: assumes normally distributed returns. Real returns have fat tails, so this
      usually *understates* the 99% loss.
    * Multi-day horizons scale by sqrt(days), which assumes independent days: an
      approximation that fails when volatility clusters.
    VaR says nothing about how bad losses get beyond the threshold; Expected Shortfall does.
    """
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1")
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")
    r = returns.dropna().to_numpy()
    if len(r) < MIN_RETURNS:
        return VaRResult(confidence, horizon_days, math.nan, math.nan, math.nan)
    root_h = math.sqrt(horizon_days)
    q = float(np.quantile(r, 1.0 - confidence))
    tail = r[r <= q]
    mu, sigma = float(r.mean()), float(r.std(ddof=1))
    z = NormalDist().inv_cdf(1.0 - confidence)
    return VaRResult(
        confidence=confidence,
        horizon_days=horizon_days,
        historical=max(0.0, -q * root_h),
        parametric=max(0.0, -(mu * horizon_days + z * sigma * root_h)),
        expected_shortfall=max(0.0, -float(tail.mean()) * root_h),
    )


# --------------------------------------------------------------------------- bundle


@dataclass(frozen=True)
class RiskMetrics:
    window_start: date
    window_end: date
    n_returns: int
    risk_free_rate: float
    annualised_return: float
    annualised_volatility: float
    sharpe: float
    sortino: float
    drawdown: DrawdownResult  # within the window
    drawdown_full_history: DrawdownResult
    beta: BetaResult | None
    var: list[VaRResult]
    warnings: list[str]


def compute_risk(
    close: pd.Series,
    bench_close: pd.Series | None = None,
    *,
    risk_free_rate: float = 0.04,
    lookback_years: float = 3.0,
    confidences: tuple[float, ...] = (0.95, 0.99),
) -> RiskMetrics:
    """Risk bundle over the last ``lookback_years`` (or all history if shorter)."""
    warnings: list[str] = []
    end = close.index[-1]
    window = close.loc[end - pd.Timedelta(days=int(lookback_years * 365.25)) :]
    if close.index[0] > end - pd.Timedelta(days=int(lookback_years * 365.25) - 14):
        warnings.append(
            f"Only {(end - close.index[0]).days / 365.25:.1f} years of history are available, "
            f"less than the {lookback_years:g}-year risk window; metrics cover what exists."
        )
    returns = simple_returns(window)
    if len(returns) < MIN_RETURNS:
        warnings.append("Too little data for reliable risk statistics.")

    beta = None
    if bench_close is not None:
        beta = compute_beta(close, bench_close, lookback_years=lookback_years)
        if beta.note:
            warnings.append(beta.note)

    return RiskMetrics(
        window_start=window.index[0].date(),
        window_end=end.date(),
        n_returns=len(returns),
        risk_free_rate=risk_free_rate,
        annualised_return=cagr(window),
        annualised_volatility=annualised_volatility(returns),
        sharpe=sharpe_ratio(returns, risk_free_rate),
        sortino=sortino_ratio(returns, risk_free_rate),
        drawdown=max_drawdown(window),
        drawdown_full_history=max_drawdown(close),
        beta=beta,
        var=[value_at_risk(returns, c) for c in confidences],
        warnings=warnings,
    )


def summarize_risk(r: RiskMetrics) -> list[str]:
    """Plain-English reading of the risk numbers, with their limits."""
    years = (r.window_end - r.window_start).days / 365.25
    out = [
        f"Over the last {years:.1f} years the yearly ups and downs were typically around "
        f"{r.annualised_volatility:.0%} (annualised volatility): higher means a bumpier ride.",
        f"The worst peak-to-trough fall in that period was {abs(r.drawdown.max_drawdown):.0%}"
        + (
            f", and it recovered by {r.drawdown.recovery_date}."
            if r.drawdown.recovery_date
            else ", and it had not recovered to its old peak."
        ),
    ]
    v95 = next((v for v in r.var if v.confidence == 0.95), None)
    if v95 and not math.isnan(v95.historical):
        out.append(
            f"On about 1 in 20 days in this period the stock fell {v95.historical:.1%} or more "
            f"(historical 95% daily Value at Risk); on those bad days the average loss was "
            f"{v95.expected_shortfall:.1%}."
        )
    if not math.isnan(r.sharpe):
        out.append(
            f"Sharpe ratio {r.sharpe:.2f}: return earned per unit of risk above the "
            f"{r.risk_free_rate:.1%} risk-free rate (roughly, above 1 is considered good, "
            "below 0 means it lagged cash)."
        )
    if r.beta and not math.isnan(r.beta.beta):
        how = (
            "more than"
            if r.beta.beta > 1.05
            else "less than"
            if r.beta.beta < 0.95
            else "about as much as"
        )
        out.append(
            f"Beta {r.beta.beta:.2f}: it has tended to move {how} the benchmark; the benchmark "
            f"explains {r.beta.r_squared:.0%} of its ups and downs."
        )
    out.append(
        "All of these describe the past window. Real markets have fat tails and regime "
        "changes, so future losses can be larger than any figure above."
    )
    return out
