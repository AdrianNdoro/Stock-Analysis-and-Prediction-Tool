"""Walk-forward engine: at each origin date t, forecast using ONLY data up to t, then score.

No look-ahead by construction: the forecast at origin position ``pos`` is computed from
``close.iloc[: pos + 1]``; later prices are touched only to read the realised outcome. The
random seed depends on (seed, pos, horizon) and never on later data, so results are also
reproducible. A test changes every price after t and checks the forecast at t is identical.

Assumptions and where the validation itself breaks
--------------------------------------------------
* The past is a fair test of the future. Regime changes (2020 crash, 2022 rate shock) mean a
  model can look fine or terrible depending on which years happen to be in the sample.
* Origins are ``step`` days apart, so for horizons longer than the step the outcome windows
  overlap and the checks are not independent (handled in ``report.py``).
* Calibration uses a constant risk-free rate (settings), though real rates were near zero
  for much of 2016-2021. That biases the "shrunk" drift prior upward in those years.
* Prices are back-adjusted for dividends, so a "forecast" is of total return.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from datetime import date

import numpy as np
import pandas as pd

from stock_analyzer.analysis.risk import TRADING_DAYS
from stock_analyzer.backtest.distributions import (
    QUANTILE_LEVELS,
    NormalLogReturn,
    SampleLogReturn,
    bootstrap_terminal_fast,
    empirical_distribution,
    gbm_distribution,
    random_walk_distribution,
)
from stock_analyzer.exceptions import InvalidParameterError, StockAnalyzerError
from stock_analyzer.models.monte_carlo import (
    DRIFT_MODES,
    METHODS,
    MIN_OBS,
    Calibration,
    calibrate,
    parse_horizon,
)

log = logging.getLogger(__name__)

WINDOWS = ("rolling", "expanding")
VOL_MODELS = ("sample", "ewma")
BASELINES = ("random_walk", "empirical")
_EXPANDING_YEARS = 100.0  # "no lookback limit" (kept small enough for pandas timestamps)
QUANTILE_COLUMNS = tuple(f"q{int(t * 100):02d}" for t in QUANTILE_LEVELS)


@dataclass(frozen=True)
class BacktestConfig:
    """Everything that defines a backtest. Part of the cache key, so keep it small and flat."""

    method: str = "gbm"
    drift_mode: str = "shrunk"
    lookback_years: float = 3.0
    window: str = "rolling"  # "rolling" = last lookback_years; "expanding" = all data so far
    horizons: tuple[str, ...] = ("1M", "3M", "6M", "1Y")
    step_days: int = 21
    min_train_days: int = 504  # no forecast is made with less history than this
    risk_free_rate: float = 0.04
    n_paths: int = 4000  # bootstrap only; backtest-internal, below the public MIN_PATHS
    seed: int = 0
    # Volatility remedies (identity by default: vol_inflation=1, vol_model="sample")
    vol_inflation: float = 1.0
    vol_model: str = "sample"
    ewma_lambda: float = 0.94
    ewma_weight: float = 1.0  # share of EWMA variance blended with the sample variance
    # Optional date filters, used for held-out evaluation
    origin_from: date | None = None  # only origins on/after this date
    outcome_until: date | None = None  # only outcomes realised on/before this date

    def validate(self) -> None:
        if self.method not in METHODS:
            raise InvalidParameterError(f"unknown method {self.method!r}")
        if self.drift_mode not in DRIFT_MODES:
            raise InvalidParameterError(f"unknown drift mode {self.drift_mode!r}")
        if self.window not in WINDOWS:
            raise InvalidParameterError(f"unknown window {self.window!r}")
        if self.vol_model not in VOL_MODELS:
            raise InvalidParameterError(f"unknown vol model {self.vol_model!r}")
        if self.step_days < 1 or self.min_train_days <= MIN_OBS:
            raise InvalidParameterError("step_days must be >= 1 and min_train_days > 60")
        if not 0.5 <= self.vol_inflation <= 3.0:
            raise InvalidParameterError("vol_inflation must be within [0.5, 3]")
        if not 0.0 <= self.ewma_weight <= 1.0 or not 0.5 < self.ewma_lambda < 1.0:
            raise InvalidParameterError("invalid EWMA settings")
        if not 200 <= self.n_paths <= 50_000:
            raise InvalidParameterError("backtest n_paths must be within [200, 50000]")
        for label in self.horizons:
            parse_horizon(label)

    def with_(self, **changes) -> BacktestConfig:
        return replace(self, **changes)

    @property
    def effective_lookback(self) -> float:
        return _EXPANDING_YEARS if self.window == "expanding" else self.lookback_years


def vol_scale(calib: Calibration, config: BacktestConfig) -> float:
    """Multiplier applied to the calibrated volatility for this origin (1.0 = unchanged).

    ``vol_inflation`` is a flat factor. ``ewma`` swaps the equal-weight sample variance for an
    exponentially weighted one (recent days count more), optionally blended with the sample.
    EWMA reacts to volatility clustering but does not mean-revert over the horizon.
    """
    scale = config.vol_inflation
    if config.vol_model == "ewma":
        r = calib.log_returns
        ewma_var = float(
            pd.Series(r * r).ewm(alpha=1.0 - config.ewma_lambda, adjust=False).mean().iloc[-1]
        )
        sample_var = (calib.sigma_annual**2) / TRADING_DAYS
        w = config.ewma_weight
        target = w * ewma_var + (1.0 - w) * sample_var
        scale *= math.sqrt(target / sample_var)
    return scale


def _window_log_prices(close_to_t: pd.Series, lookback_years: float) -> np.ndarray:
    end = close_to_t.index[-1]
    window = close_to_t.loc[end - pd.Timedelta(days=int(lookback_years * 365.25)) :]
    return np.log(window.to_numpy())


def calibrate_at(close_to_t: pd.Series, config: BacktestConfig) -> tuple[Calibration, float]:
    """Calibration and volatility scale from data up to the origin only."""
    calib = calibrate(
        close_to_t,
        risk_free_rate=config.risk_free_rate,
        lookback_years=config.effective_lookback,
        drift_mode=config.drift_mode,
    )
    return calib, vol_scale(calib, config)


def forecast_distributions(
    close_to_t: pd.Series,
    calib: Calibration,
    scale: float,
    h: int,
    config: BacktestConfig,
    rng_seed_parts: tuple[int, ...],
) -> dict[str, NormalLogReturn | SampleLogReturn | None]:
    """All forecasts made at the end of ``close_to_t`` for horizon ``h`` (model + baselines).

    ``close_to_t`` must end at the origin date: this function cannot see anything later.
    """
    if config.method == "gbm":
        model = gbm_distribution(calib, h, scale)
    else:
        rng = np.random.default_rng(list(rng_seed_parts))
        model = bootstrap_terminal_fast(calib, h, config.n_paths, rng, scale)
    return {
        config.method: model,
        "random_walk": random_walk_distribution(calib, h, scale),
        "empirical": empirical_distribution(
            _window_log_prices(close_to_t, config.effective_lookback), h
        ),
    }


def origin_positions(n: int, config: BacktestConfig, horizon_days: list[int]) -> range:
    """Origin positions (indices into the price series) shared by all horizons."""
    first = config.min_train_days - 1
    last = n - 1 - min(horizon_days)
    return range(first, last + 1, config.step_days) if last >= first else range(0)


def run_walk_forward(close: pd.Series, config: BacktestConfig) -> pd.DataFrame:
    """Walk forward over ``close`` (adjusted close, ascending DatetimeIndex).

    Returns one row per (origin, horizon, model) with the realised log return ``y``, the
    forecast quantiles, the PIT value and the forecast probability of finishing above today.
    An origin whose calibration fails (for example zero volatility) is skipped, never filled.
    """
    config.validate()
    if not close.index.is_monotonic_increasing or not bool((close > 0).all()):
        raise InvalidParameterError("close must be ascending and strictly positive")
    horizon_days = {label.upper(): parse_horizon(label) for label in config.horizons}
    n = len(close)
    values = close.to_numpy()
    rows: list[dict] = []
    for pos in origin_positions(n, config, list(horizon_days.values())):
        origin_date = close.index[pos].date()
        if config.origin_from and origin_date < config.origin_from:
            continue
        upto = close.iloc[: pos + 1]  # <- the ONLY data the forecast may use
        try:
            calib, scale = calibrate_at(upto, config)
        except StockAnalyzerError as exc:
            log.debug("origin skipped", extra={"origin": str(origin_date), "error": str(exc)})
            continue
        for label, h in horizon_days.items():
            if pos + h > n - 1:
                continue
            if config.outcome_until and close.index[pos + h].date() > config.outcome_until:
                continue
            dists = forecast_distributions(upto, calib, scale, h, config, (config.seed, pos, h))
            y = float(math.log(values[pos + h] / values[pos]))
            for name, dist in dists.items():
                if dist is None:
                    continue
                q = dist.quantile(QUANTILE_LEVELS)
                rows.append(
                    {
                        "origin": close.index[pos],
                        "outcome": close.index[pos + h],
                        "horizon": label,
                        "h": h,
                        "model": name,
                        "y": y,
                        **dict(zip(QUANTILE_COLUMNS, q, strict=True)),
                        "pit": dist.cdf(y),
                        "p_up": dist.prob_up(),
                    }
                )
    columns = ["origin", "outcome", "horizon", "h", "model", "y", *QUANTILE_COLUMNS, "pit", "p_up"]
    return pd.DataFrame(rows, columns=columns)
