"""Monte Carlo price simulation (Geometric Brownian Motion or bootstrapped returns).

THESE ARE NOT PREDICTIONS. A simulation answers "if future daily returns behaved like the
assumptions below, what spread of outcomes would result?" It cannot know about earnings,
news, crashes or regime changes.

Statistical assumptions, and where they break
----------------------------------------------
* GBM: daily log returns are independent and normal with constant volatility. Real returns
  have fat tails (extreme days are far more common than a normal curve says) and volatility
  clusters (calm and stormy spells), so GBM *understates* the odds of large moves.
* Bootstrap: resamples the stock's own past daily returns, so it keeps fat tails and
  skewness. It still treats days as independent (no volatility clustering) and assumes the
  calibration window is representative of the future.
* Drift (expected return) is the hardest input. The historical average return over a few
  years is extremely noisy (its standard error is volatility / sqrt(years), often +/-15% a
  year or more), so by default we blend it with a long-run prior (risk-free rate plus a
  5% equity premium). Both methods are then mean-matched to the chosen drift, so changing
  the method changes the *shape* of the distribution, not its centre.
* Volatility is held at its recent level. Real volatility moves, usually rising after
  shocks, so use the sensitivity range on the scenarios rather than a single number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from stock_analyzer.analysis.risk import TRADING_DAYS
from stock_analyzer.exceptions import InsufficientDataError, InvalidParameterError

HORIZONS: dict[str, int] = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252}
_HORIZON_NAMES = {21: "1 month", 63: "3 months", 126: "6 months", 252: "1 year"}
MIN_PATHS = 10_000
MAX_PATHS = 50_000  # 50,000 x 252 float64 paths is ~100 MB: the ceiling for small hosts
PERCENTILES = (5, 25, 50, 75, 95)
MIN_OBS = 60
METHODS = ("gbm", "bootstrap")
DRIFT_MODES = ("shrunk", "historical", "neutral")
MU_CLIP = 0.5  # expected return is clipped to +/-50% a year

_METHOD_TEXT = {
    "gbm": "Geometric Brownian Motion (normally distributed daily returns)",
    "bootstrap": "bootstrapped historical returns (resampling the stock's own past daily moves)",
}
_DRIFT_TEXT = {
    "shrunk": "a blend of the stock's historical average return and a long-run market assumption",
    "historical": "the stock's own historical average return",
    "neutral": "the risk-free rate (no extra reward for risk)",
}


def parse_horizon(label: str) -> int:
    """'1M'/'3M'/'6M'/'1Y' -> trading days. Anything else is rejected."""
    key = str(label).strip().upper()
    key = {"12M": "1Y"}.get(key, key)
    if key not in HORIZONS:
        raise InvalidParameterError(
            f"unsupported horizon {label!r}",
            user_message="Choose a horizon of 1M, 3M, 6M or 1Y.",
        )
    return HORIZONS[key]


def describe_horizon(days: int) -> str:
    return _HORIZON_NAMES.get(days, f"{days} trading days")


@dataclass(frozen=True, eq=False)
class Calibration:
    """Inputs estimated from price history. Can also be built by hand for testing."""

    sigma_annual: float
    mu_annual: float  # expected arithmetic return actually used by the simulation
    mu_hist_annual: float
    mu_hist_se: float  # standard error of the historical mean (annual)
    drift_mode: str
    risk_free_rate: float
    n_obs: int
    window_start: date
    window_end: date
    log_returns: np.ndarray
    skew: float
    excess_kurtosis: float
    warnings: tuple[str, ...] = ()


def calibrate(
    close: pd.Series,
    *,
    risk_free_rate: float = 0.04,
    lookback_years: float = 3.0,
    drift_mode: str = "shrunk",
    prior_weight: float = 0.5,
    equity_premium: float = 0.05,
) -> Calibration:
    """Estimate volatility and drift from adjusted closes over the last ``lookback_years``."""
    if drift_mode not in DRIFT_MODES:
        raise InvalidParameterError(f"unknown drift mode {drift_mode!r}")
    end = close.index[-1]
    window = close.loc[end - pd.Timedelta(days=int(lookback_years * 365.25)) :]
    log_r = np.log(window).diff().dropna()
    simple = window.pct_change().dropna()
    n = len(log_r)
    if n < MIN_OBS:
        raise InsufficientDataError(
            f"only {n} returns to calibrate on",
            user_message=(
                f"Only {n} days of price history are available; we need at least {MIN_OBS} "
                "to run a simulation."
            ),
        )
    sigma_d = float(log_r.std(ddof=1))
    if not sigma_d > 0:
        raise InsufficientDataError(
            "zero volatility", user_message="This price history shows no variation to simulate."
        )

    years = n / TRADING_DAYS
    mu_hist = float(simple.mean() * TRADING_DAYS)
    sigma_annual = sigma_d * math.sqrt(TRADING_DAYS)
    se = sigma_annual / math.sqrt(years)
    prior = risk_free_rate + equity_premium
    if drift_mode == "historical":
        mu = mu_hist
    elif drift_mode == "neutral":
        mu = risk_free_rate
    else:
        mu = prior_weight * prior + (1.0 - prior_weight) * mu_hist

    warnings: list[str] = []
    if years < 2:
        warnings.append(
            f"Calibrated on only {years:.1f} years of data; volatility and drift estimates "
            "are less reliable."
        )
    if abs(mu_hist) > 0.30:
        warnings.append(
            f"The historical average return ({mu_hist:+.0%} a year) is unusually extreme and "
            f"very uncertain (about ±{se:.0%}); "
            + (
                "the simulation blends it with a long-run assumption."
                if drift_mode == "shrunk"
                else "treat results built on it with great caution."
            )
        )
    clipped = min(max(mu, -MU_CLIP), MU_CLIP)
    if clipped != mu:
        warnings.append(f"Expected return was capped at {clipped:+.0%} a year.")
        mu = clipped
    kurt = float(pd.Series(log_r).kurt())
    if kurt > 3:
        warnings.append(
            f"This stock's daily returns are fat-tailed (excess kurtosis {kurt:.1f}): extreme "
            "days are more common than a normal curve implies, so GBM understates large moves. "
            "Compare with the bootstrap method."
        )
    return Calibration(
        sigma_annual=sigma_annual,
        mu_annual=mu,
        mu_hist_annual=mu_hist,
        mu_hist_se=se,
        drift_mode=drift_mode,
        risk_free_rate=risk_free_rate,
        n_obs=n,
        window_start=window.index[0].date(),
        window_end=end.date(),
        log_returns=log_r.to_numpy(),
        skew=float(pd.Series(log_r).skew()),
        excess_kurtosis=kurt,
        warnings=tuple(warnings),
    )


@dataclass(frozen=True, eq=False)
class SimulationResult:
    method: str
    horizon_days: int
    n_paths: int
    seed: int | None
    s0: float
    sigma_scale: float
    calibration: Calibration
    terminal: np.ndarray  # simulated price at the end of the horizon, one per path
    bands: pd.DataFrame  # price percentiles for every day (index 0..h) for the fan chart
    sample_paths: np.ndarray  # a few full paths for display, shape (k, h + 1)

    @property
    def returns(self) -> np.ndarray:
        return self.terminal / self.s0 - 1.0

    @property
    def sigma_used(self) -> float:
        return self.calibration.sigma_annual * self.sigma_scale

    def percentile(self, q: float) -> float:
        return float(np.percentile(self.terminal, q))

    def expected_range(self) -> tuple[float, float]:
        """The 5th-95th percentile: 9 in 10 simulated outcomes landed in here."""
        return self.percentile(5), self.percentile(95)

    def prob_above_current(self) -> float:
        return float(np.mean(self.terminal > self.s0))

    def prob_below_current(self) -> float:
        return float(np.mean(self.terminal < self.s0))

    def prob_gain_more_than(self, x: float) -> float:
        """Probability of finishing at least ``x`` (0.10 = 10%) above today's price."""
        return float(np.mean(self.terminal >= self.s0 * (1.0 + x)))

    def prob_loss_more_than(self, x: float) -> float:
        """Probability of finishing at least ``x`` (0.10 = 10%) below today's price."""
        return float(np.mean(self.terminal <= self.s0 * (1.0 - x)))

    def summary_table(self) -> pd.DataFrame:
        rows = {q: self.percentile(q) for q in PERCENTILES}
        return pd.DataFrame(
            {"price": rows.values(), "return": [p / self.s0 - 1.0 for p in rows.values()]},
            index=pd.Index([f"{q}th" for q in rows], name="percentile"),
        )

    def monte_carlo_error(self, p: float) -> float:
        """Sampling error (1 s.d.) of a probability estimate. Model error is far larger."""
        return math.sqrt(p * (1 - p) / self.n_paths)

    def plain_english(self, currency: str | None = None, thresholds=(0.10, 0.20)) -> list[str]:
        cur = f"{currency} " if currency else ""
        c = self.calibration
        lo, hi = self.expected_range()
        med = self.percentile(50)
        out = [
            f"We simulated {self.n_paths:,} possible price paths over "
            f"{describe_horizon(self.horizon_days)} using {_METHOD_TEXT[self.method]}.",
            f"Middle outcome (median): {cur}{med:,.2f} ({med / self.s0 - 1:+.1%}). In 9 of 10 "
            f"simulated paths the price ended between {cur}{lo:,.2f} and {cur}{hi:,.2f}.",
            f"Chance of finishing above today's price: {self.prob_above_current():.0%}; "
            f"below: {self.prob_below_current():.0%}.",
        ]
        for x in thresholds:
            out.append(
                f"Chance of gaining more than {x:.0%}: {self.prob_gain_more_than(x):.0%}; "
                f"of losing more than {x:.0%}: {self.prob_loss_more_than(x):.0%}."
            )
        out.append(
            f"Assumptions: volatility {self.sigma_used:.0%} a year and expected return "
            f"{c.mu_annual:+.1%} a year, based on {_DRIFT_TEXT[c.drift_mode]}. The stock's "
            f"own recent average return was {c.mu_hist_annual:+.1%} a year, but that figure "
            f"is uncertain by about ±{c.mu_hist_se:.0%}."
        )
        out.extend(c.warnings)
        out.append(
            "This is a spread of possibilities under stated assumptions, not a prediction. "
            "Real prices can move outside this range, especially during crashes or news shocks."
        )
        return out


def simulate(
    calibration: Calibration,
    s0: float,
    horizon_days: int,
    *,
    method: str = "gbm",
    n_paths: int = MIN_PATHS,
    seed: int | None = None,
    sigma_scale: float = 1.0,
    n_sample_paths: int = 50,
    max_paths: int = MAX_PATHS,
) -> SimulationResult:
    """Simulate ``n_paths`` price paths of ``horizon_days`` trading days, fully vectorised.

    Both methods are mean-matched: the expected one-day simple return equals
    ``mu_annual / 252`` exactly, so GBM and bootstrap differ in shape, not in centre.
    ``sigma_scale`` multiplies volatility (used for the scenario sensitivity range).
    ``seed`` makes a run reproducible.
    """
    if method not in METHODS:
        raise InvalidParameterError(f"unknown method {method!r}")
    if not MIN_PATHS <= n_paths <= max_paths:
        raise InvalidParameterError(
            f"n_paths {n_paths} outside [{MIN_PATHS}, {max_paths}]",
            user_message=f"Number of simulations must be between {MIN_PATHS:,} and {max_paths:,}.",
        )
    if not 1 <= horizon_days <= TRADING_DAYS:
        raise InvalidParameterError(f"horizon_days {horizon_days} outside [1, {TRADING_DAYS}]")
    if not (math.isfinite(s0) and s0 > 0):
        raise InvalidParameterError(f"invalid starting price {s0!r}")
    if not 0.1 <= sigma_scale <= 5.0:
        raise InvalidParameterError(f"sigma_scale {sigma_scale} outside [0.1, 5]")
    if seed is not None and (not isinstance(seed, int) or seed < 0):
        raise InvalidParameterError("seed must be a non-negative integer")

    rng = np.random.default_rng(seed)
    mu_d = calibration.mu_annual / TRADING_DAYS
    if method == "gbm":
        sd = calibration.sigma_annual * sigma_scale / math.sqrt(TRADING_DAYS)
        m = math.log1p(mu_d) - 0.5 * sd * sd  # so that E[exp(step)] = 1 + mu_d
        paths = rng.standard_normal((n_paths, horizon_days))
        paths *= sd
        paths += m
    else:
        r = calibration.log_returns
        centred = (r - r.mean()) * sigma_scale
        m = math.log1p(mu_d) - math.log(float(np.mean(np.exp(centred))))
        paths = centred[rng.integers(0, len(r), size=(n_paths, horizon_days))]
        paths += m
    np.cumsum(paths, axis=1, out=paths)  # log price relative to s0
    np.exp(paths, out=paths)
    paths *= s0  # now price levels (in place: keeps memory to a single big array)

    band_values = np.percentile(paths, PERCENTILES, axis=0).T
    bands = pd.DataFrame(
        np.vstack([np.full((1, len(PERCENTILES)), s0), band_values]),
        index=pd.RangeIndex(0, horizon_days + 1, name="day"),
        columns=[f"p{q}" for q in PERCENTILES],
    )
    k = min(n_sample_paths, n_paths)
    sample = np.hstack([np.full((k, 1), s0), paths[:k]])
    return SimulationResult(
        method=method,
        horizon_days=horizon_days,
        n_paths=n_paths,
        seed=seed,
        s0=float(s0),
        sigma_scale=sigma_scale,
        calibration=calibration,
        terminal=paths[:, -1].copy(),
        bands=bands,
        sample_paths=sample,
    )
