"""Pure statistics for judging probabilistic forecasts. No I/O, no pandas.

Overlapping origins: forecasts made 21 days apart for a 63-day horizon share most of their
outcome window, so their errors are strongly correlated. Point estimates (coverage, scores)
use every origin; anything that quotes uncertainty or a p-value uses the non-overlapping
subset (see ``non_overlapping_indices``), whose size is the honest "effective sample".
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import chi2, kstest, norm


def non_overlapping_indices(n: int, horizon_days: int, step_days: int) -> np.ndarray:
    """Positions of origins whose outcome windows do not overlap (every ceil(h/step)-th)."""
    stride = max(1, math.ceil(horizon_days / step_days))
    return np.arange(0, n, stride)


def wilson_interval(k: float, n: float, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. ``n`` may be an effective sample size."""
    if n <= 0:
        return (math.nan, math.nan)
    z = float(norm.ppf(0.5 + confidence / 2))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def inside_interval(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return (y >= lo) & (y <= hi)


def _xlogy(x: float, y: float) -> float:
    return 0.0 if x == 0 else x * math.log(y)


def kupiec_pof(exceedances: int, n: int, p: float) -> tuple[float, float]:
    """Kupiec unconditional-coverage test. ``p`` = expected exceedance rate (0.10 for a 90% range).

    Returns (LR statistic, p-value). A small p-value says the observed miss rate is
    inconsistent with ``p``. Assumes independent trials, hence use non-overlapping origins.
    """
    if n <= 0:
        return (math.nan, math.nan)
    x, phat = exceedances, exceedances / n
    ll_null = _xlogy(n - x, 1 - p) + _xlogy(x, p)
    ll_alt = _xlogy(n - x, 1 - phat) + _xlogy(x, phat)
    lr = max(0.0, -2.0 * (ll_null - ll_alt))
    return (lr, float(chi2.sf(lr, 1)))


def christoffersen_independence(hits: np.ndarray) -> tuple[float, float] | None:
    """Christoffersen test that misses are not clustered in time (first-order Markov).

    ``hits`` is a 0/1 sequence of misses in time order. Returns (LR, p-value) or None when the
    sequence is too short or has no variation in a state (the test is undefined).
    """
    h = np.asarray(hits, dtype=int)
    if len(h) < 3:
        return None
    a, b = h[:-1], h[1:]
    n00 = int(np.sum((a == 0) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    n11 = int(np.sum((a == 1) & (b == 1)))
    if n00 + n01 == 0 or n10 + n11 == 0 or n01 + n11 == 0:
        return None
    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll_null = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    ll_alt = _xlogy(n00, 1 - pi01) + _xlogy(n01, pi01) + _xlogy(n10, 1 - pi11) + _xlogy(n11, pi11)
    lr = max(0.0, -2.0 * (ll_null - ll_alt))
    return (lr, float(chi2.sf(lr, 1)))


def pinball_loss(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    """Quantile (pinball) loss per observation. Lower is better; a proper scoring rule."""
    d = y - q
    return np.maximum(tau * d, (tau - 1.0) * d)


def mean_pinball(y: np.ndarray, quantiles: np.ndarray, taus: tuple[float, ...]) -> float:
    """Average pinball loss over all quantile levels. ``quantiles`` is shape (n, len(taus))."""
    return float(np.mean([pinball_loss(y, quantiles[:, i], t).mean() for i, t in enumerate(taus)]))


def interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, coverage: float) -> float:
    """Gneiting-Raftery interval score for a central ``coverage`` interval (lower is better).

    Width plus a penalty of 2/alpha times the miss distance, so it rewards narrow intervals
    only when they still contain the outcome.
    """
    alpha = 1.0 - coverage
    score = (hi - lo) + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - hi, 0)
    return float(np.mean(score))


def pit_histogram(pit: np.ndarray, bins: int = 10) -> tuple[int, ...]:
    counts, _ = np.histogram(pit, bins=bins, range=(0.0, 1.0))
    return tuple(int(c) for c in counts)


def pit_uniformity_pvalue(pit: np.ndarray) -> float:
    """Kolmogorov-Smirnov p-value for 'PIT values are Uniform(0,1)'. Small = miscalibrated.

    Needs independent observations, so pass the non-overlapping subset. Low power for n < 30.
    """
    if len(pit) < 5:
        return math.nan
    return float(kstest(pit, "uniform").pvalue)


def brier_score(p_up: np.ndarray, went_up: np.ndarray) -> float:
    return float(np.mean((p_up - went_up.astype(float)) ** 2))
