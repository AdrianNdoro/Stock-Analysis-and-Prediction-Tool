"""Glue: analysis result -> simulations and scenarios for one or more horizons."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from stock_analyzer.analysis.pipeline import AnalysisResult, analyze_ticker
from stock_analyzer.backtest.reliability import Reliability, assess
from stock_analyzer.backtest.service import backtest_series
from stock_analyzer.backtest.walkforward import BacktestConfig
from stock_analyzer.config import Settings, get_settings
from stock_analyzer.data.cache import SQLiteCache
from stock_analyzer.data.service import MarketDataService
from stock_analyzer.exceptions import StockAnalyzerError
from stock_analyzer.models.monte_carlo import (
    Calibration,
    SimulationResult,
    calibrate,
    parse_horizon,
    simulate,
)
from stock_analyzer.models.scenarios import ScenarioSet, build_scenarios

log = logging.getLogger(__name__)

SENSITIVITY = (0.8, 1.2)  # volatility multipliers for the scenario probability ranges


@dataclass(frozen=True)
class ForecastResult:
    analysis: AnalysisResult
    calibration: Calibration
    simulations: dict[str, SimulationResult]  # keyed by horizon label, e.g. "6M"
    scenarios: dict[str, ScenarioSet]
    # Phase 4: how reliable this model's ranges were in past walk-forward checks, per horizon.
    # None when no backtest was requested (the default: backtests are slow on a cold cache).
    reliability: dict[str, Reliability] | None = None


def forecast(
    analysis: AnalysisResult,
    horizons: Sequence[str] = ("6M",),
    *,
    method: str = "gbm",
    n_paths: int = 10_000,
    seed: int | None = None,
    drift_mode: str = "shrunk",
    lookback_years: float = 3.0,
    reliability: dict[str, Reliability] | None = None,
) -> ForecastResult:
    """Simulate each horizon (plus two volatility-sensitivity runs) and build scenarios.

    All runs for a horizon share the same seed (common random numbers), so differences between
    the base and sensitivity runs reflect the volatility change, not sampling noise.
    """
    calibration = calibrate(
        analysis.adj_close,
        risk_free_rate=analysis.risk.risk_free_rate,
        lookback_years=lookback_years,
        drift_mode=drift_mode,
    )
    s0 = float(analysis.adj_close.iloc[-1])
    sims: dict[str, SimulationResult] = {}
    scen: dict[str, ScenarioSet] = {}
    for label in horizons:
        key = label.strip().upper()
        days = parse_horizon(key)
        common = {"method": method, "n_paths": n_paths, "seed": seed}
        base = simulate(calibration, s0, days, **common)
        low = simulate(calibration, s0, days, sigma_scale=SENSITIVITY[0], **common)
        high = simulate(calibration, s0, days, sigma_scale=SENSITIVITY[1], **common)
        sims[key] = base
        scen[key] = build_scenarios(base, low, high, analysis.technical, currency=analysis.currency)
    return ForecastResult(analysis, calibration, sims, scen, reliability)


def forecast_ticker(
    service: MarketDataService,
    ticker: str,
    horizons: Sequence[str] = ("6M",),
    *,
    benchmark: str | None = None,
    method: str = "gbm",
    n_paths: int | None = None,
    seed: int | None = None,
    drift_mode: str | None = None,
    settings: Settings | None = None,
    with_reliability: bool = False,
) -> ForecastResult:
    """Fetch data, analyse, simulate. Defaults for paths/drift/window come from settings.

    ``with_reliability=True`` also runs (or reads from cache) a walk-forward backtest of the
    same method and drift on this stock and attaches a plain-English reliability grade per
    horizon. A backtest failure never breaks the forecast; it just leaves reliability empty.
    """
    settings = settings or get_settings()
    analysis = analyze_ticker(service, ticker, benchmark, settings)
    drift = drift_mode or settings.mc_drift_mode
    reliability = None
    if with_reliability:
        reliability = _reliability_for(analysis, horizons, method, drift, settings)
    return forecast(
        analysis,
        horizons,
        method=method,
        n_paths=n_paths or settings.mc_paths,
        seed=seed,
        drift_mode=drift,
        lookback_years=settings.mc_lookback_years,
        reliability=reliability,
    )


def _reliability_for(
    analysis: AnalysisResult,
    horizons: Sequence[str],
    method: str,
    drift_mode: str,
    settings: Settings,
) -> dict[str, Reliability] | None:
    config = BacktestConfig(
        method=method,
        drift_mode=drift_mode,
        lookback_years=settings.mc_lookback_years,
        risk_free_rate=settings.risk_free_rate,
    )
    try:
        cache = SQLiteCache(settings.cache_path)
    except StockAnalyzerError:
        cache = None
    try:
        report = backtest_series(analysis.ticker, analysis.adj_close, config, cache=cache)
    except StockAnalyzerError as exc:
        log.warning("reliability unavailable", extra={"ticker": analysis.ticker, "error": str(exc)})
        return None
    return {h.strip().upper(): assess(report, h) for h in horizons}
