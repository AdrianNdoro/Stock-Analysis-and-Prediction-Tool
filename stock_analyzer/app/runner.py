"""Run one validated request end to end. No Streamlit: callable from tests and scripts."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from stock_analyzer.analysis.fundamentals import FundamentalsSnapshot, build_snapshot
from stock_analyzer.analysis.pipeline import analyze_ticker
from stock_analyzer.analysis.technical import adjusted_ohlc, compute_indicators
from stock_analyzer.app.presenters import ALL_HORIZONS
from stock_analyzer.app.safety import AnalysisRequest
from stock_analyzer.config import Settings
from stock_analyzer.data.risk_free import RiskFreeRate
from stock_analyzer.data.service import MarketDataService
from stock_analyzer.models.forecast import ForecastResult, forecast, reliability_for

log = logging.getLogger("stock_analyzer.app")


@dataclass(frozen=True, eq=False)
class AppResult:
    request: AnalysisRequest
    forecast: ForecastResult
    ohlc: pd.DataFrame  # split/dividend-adjusted OHLCV, for the price chart
    indicators: pd.DataFrame
    data_date: str  # last price date: part of the cache key, so a new bar means a fresh run
    fundamentals: FundamentalsSnapshot | None = None  # None only if building it failed
    risk_free: RiskFreeRate | None = None
    provider: str | None = None  # data source name, for the report


def last_data_date(service: MarketDataService, ticker: str) -> str:
    """Date of the newest price bar (served from the data cache when fresh)."""
    return str(service.get_market_data(ticker).quality.last_date)


def run_request(
    service: MarketDataService,
    settings: Settings,
    req: AnalysisRequest,
    data_date: str,
    risk_free: RiskFreeRate | None = None,
) -> AppResult:
    """Fetch, analyse, simulate and attach reliability for ALL horizons.

    The reliability check always covers 1M-1Y (it is cached), because the page shows the 1M and
    3M grades whichever horizon is selected. Raises StockAnalyzerError with friendly text.
    """
    analysis = analyze_ticker(service, req.ticker, req.benchmark, settings)
    market = service.get_market_data(req.ticker)
    ohlc = adjusted_ohlc(market.prices)
    rel = reliability_for(analysis, ALL_HORIZONS, req.method, req.drift, settings)
    fc = forecast(
        analysis,
        [req.horizon],
        method=req.method,
        n_paths=req.paths,
        seed=req.seed,
        drift_mode=req.drift,
        lookback_years=settings.mc_lookback_years,
        reliability=rel,
    )
    try:  # fundamentals are a bonus: the service never raises for provider trouble, but be safe
        snapshot = build_snapshot(service.get_fundamentals(req.ticker))
    except Exception:  # noqa: BLE001
        log.warning("fundamentals failed", extra={"ticker": req.ticker})
        snapshot = None
    return AppResult(
        req, fc, ohlc, compute_indicators(ohlc), data_date, snapshot, risk_free,
        market.source,
    )  # fmt: skip
