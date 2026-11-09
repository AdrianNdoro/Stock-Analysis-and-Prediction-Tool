"""Forecast distributions for the backtest, expressed as h-day LOG returns.

Why not call ``models.monte_carlo.simulate``? A backtest needs hundreds of forecasts per
ticker. GBM has an exact closed form (a sum of h independent normals is normal), so it needs
no simulation at all. Bootstrap still needs sampling, but far fewer paths than the public
minimum are enough for a quantile check. That shortcut lives ONLY here (see
``bootstrap_terminal_fast``); the public limits in ``monte_carlo`` are untouched.
Tests prove both fast paths agree with the real simulator.

Assumptions and failure modes: identical to the models they mirror (see monte_carlo.py).
The baselines are deliberately simple so that "the model beats them" is a meaningful claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from stock_analyzer.analysis.risk import TRADING_DAYS
from stock_analyzer.models.monte_carlo import Calibration

QUANTILE_LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)
MIN_EMPIRICAL_SAMPLES = 40  # fewer past h-day returns than this: the empirical baseline is skipped


@dataclass(frozen=True, eq=False)
class NormalLogReturn:
    """log(S_h / S_0) ~ Normal(mean, sd): the exact GBM terminal distribution."""

    mean: float
    sd: float

    def quantile(self, taus: tuple[float, ...] | np.ndarray) -> np.ndarray:
        return self.mean + self.sd * norm.ppf(np.asarray(taus, dtype=float))

    def cdf(self, y: float) -> float:
        return float(norm.cdf((y - self.mean) / self.sd))

    def prob_up(self) -> float:
        return 1.0 - self.cdf(0.0)


@dataclass(frozen=True, eq=False)
class SampleLogReturn:
    """A forecast given by a sample of log returns (bootstrap draws or past h-day returns)."""

    sorted_values: np.ndarray

    @classmethod
    def from_values(cls, values: np.ndarray) -> SampleLogReturn:
        return cls(np.sort(np.asarray(values, dtype=float)))

    def quantile(self, taus: tuple[float, ...] | np.ndarray) -> np.ndarray:
        return np.quantile(self.sorted_values, np.asarray(taus, dtype=float))

    def cdf(self, y: float) -> float:
        """Mid-rank empirical CDF: ties count half, so PIT values stay uniform-ish."""
        v = self.sorted_values
        lo = np.searchsorted(v, y, side="left")
        hi = np.searchsorted(v, y, side="right")
        return float((lo + hi) / (2.0 * len(v)))

    def prob_up(self) -> float:
        return float(np.mean(self.sorted_values > 0.0))


def gbm_distribution(calib: Calibration, h: int, sigma_scale: float = 1.0) -> NormalLogReturn:
    """Closed-form GBM terminal law, matching ``simulate(method='gbm')`` step for step."""
    mu_d = calib.mu_annual / TRADING_DAYS
    sd_d = calib.sigma_annual * sigma_scale / math.sqrt(TRADING_DAYS)
    step_mean = math.log1p(mu_d) - 0.5 * sd_d * sd_d  # makes E[exp(step)] = 1 + mu_d
    return NormalLogReturn(mean=h * step_mean, sd=sd_d * math.sqrt(h))


def random_walk_distribution(
    calib: Calibration, h: int, sigma_scale: float = 1.0
) -> NormalLogReturn:
    """Baseline 1: zero-drift random walk. Same volatility as the model, median price unchanged."""
    sd_d = calib.sigma_annual * sigma_scale / math.sqrt(TRADING_DAYS)
    return NormalLogReturn(mean=0.0, sd=sd_d * math.sqrt(h))


def bootstrap_terminal_fast(
    calib: Calibration,
    h: int,
    n_paths: int,
    rng: np.random.Generator,
    sigma_scale: float = 1.0,
) -> SampleLogReturn:
    """BACKTEST-ONLY bootstrap with a small ``n_paths`` (bypasses ``MIN_PATHS``).

    Same maths as ``simulate(method='bootstrap')`` (centre, scale, mean-match, resample), but
    it sums log returns directly instead of building full price paths, which is faster and
    uses far less memory. With the same seed and path count it reproduces the simulator's
    terminal prices (tested).
    """
    r = calib.log_returns
    centred = (r - r.mean()) * sigma_scale
    mu_d = calib.mu_annual / TRADING_DAYS
    m = math.log1p(mu_d) - math.log(float(np.mean(np.exp(centred))))
    draws = centred[rng.integers(0, len(r), size=(n_paths, h))]
    return SampleLogReturn.from_values(draws.sum(axis=1) + h * m)


def empirical_distribution(window_log_prices: np.ndarray, h: int) -> SampleLogReturn | None:
    """Baseline 2: the stock's own past h-day log returns inside the calibration window.

    The windows overlap heavily (consecutive samples share h-1 days), so there are far fewer
    independent observations than the sample size suggests. Returns None if fewer than
    ``MIN_EMPIRICAL_SAMPLES`` are available.
    """
    if len(window_log_prices) <= h:
        return None
    past = window_log_prices[h:] - window_log_prices[:-h]
    if len(past) < MIN_EMPIRICAL_SAMPLES:
        return None
    return SampleLogReturn.from_values(past)
