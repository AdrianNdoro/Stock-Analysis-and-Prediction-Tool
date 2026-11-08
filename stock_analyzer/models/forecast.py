"""Glue: analysis result -> simulations and scenarios for one or more horizons."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from stock_analyzer.analysis.pipeline import AnalysisResult, analyze_ticker
from stock_analyzer.config import Settings, get_settings
from stock_analyzer.data.service import MarketDataService
from stock_analyzer.models.monte_carlo import (
    Calibration,
    SimulationResult,
    calibrate,
    parse_horizon,
    simulate,
)
from stock_analyzer.models.scenarios import ScenarioSet, build_scenarios

SENSITIVITY = (0.8, 1.2)  # volatility multipliers for the scenario probability ranges


@dataclass(frozen=True)
class ForecastResult:
    analysis: AnalysisResult
    calibration: Calibration
    simulations: dict[str, SimulationResult]  # keyed by horizon label, e.g. "6M"
    scenarios: dict[str, ScenarioSet]


def forecast(
    analysis: AnalysisResult,
    horizons: Sequence[str] = ("6M",),
    *,
    method: str = "gbm",
    n_paths: int = 10_000,
    seed: int | None = None,
    drift_mode: str = "shrunk",
    lookback_years: float = 3.0,
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
    return ForecastResult(analysis, calibration, sims, scen)


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
) -> ForecastResult:
    """Fetch data, analyse, simulate. Defaults for paths/drift/window come from settings."""
    settings = settings or get_settings()
    analysis = analyze_ticker(service, ticker, benchmark, settings)
    return forecast(
        analysis,
        horizons,
        method=method,
        n_paths=n_paths or settings.mc_paths,
        seed=seed,
        drift_mode=drift_mode or settings.mc_drift_mode,
        lookback_years=settings.mc_lookback_years,
    )
