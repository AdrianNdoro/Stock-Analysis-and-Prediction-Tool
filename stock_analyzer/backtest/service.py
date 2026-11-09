"""I/O edge for backtests: caching and the ticker wrapper. Maths stays in the pure modules."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict

import pandas as pd

from stock_analyzer.backtest.report import MODEL_VERSION, BacktestReport, build_report
from stock_analyzer.backtest.walkforward import BacktestConfig, run_walk_forward
from stock_analyzer.config import Settings, get_settings
from stock_analyzer.data.cache import SQLiteCache
from stock_analyzer.data.service import MarketDataService
from stock_analyzer.exceptions import CacheError, StockAnalyzerError

log = logging.getLogger(__name__)

BACKTEST_CACHE_TTL_SECONDS = 7 * 24 * 3600  # a backtest is expensive and moves slowly


def config_hash(config: BacktestConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def cache_key(ticker: str, config: BacktestConfig, last_date: str) -> str:
    """Includes the model version, every parameter and the last data date (new bar = new key)."""
    return f"backtest:v{MODEL_VERSION}:{ticker}:{config_hash(config)}:{last_date}"


def backtest_series(
    ticker: str,
    close: pd.Series,
    config: BacktestConfig,
    *,
    cache: SQLiteCache | None = None,
    extra_warnings: tuple[str, ...] = (),
) -> BacktestReport:
    """Run (or fetch from cache) the backtest for an adjusted-close series."""
    last = str(close.index[-1].date())
    key = cache_key(ticker, config, last)
    if cache is not None:
        try:
            entry = cache.get(key)
            if entry and cache.is_fresh(entry, BACKTEST_CACHE_TTL_SECONDS):
                return BacktestReport.from_dict(json.loads(entry.value))
        except (CacheError, ValueError, KeyError, TypeError) as exc:
            log.warning("backtest cache read failed", extra={"ticker": ticker, "error": str(exc)})
    rows = run_walk_forward(close, config)
    report = build_report(ticker, rows, config, last, extra_warnings)
    if cache is not None:
        try:
            cache.set(key, json.dumps(report.to_dict()))
        except CacheError as exc:
            log.warning("backtest cache write failed", extra={"ticker": ticker, "error": str(exc)})
    return report


def backtest_ticker(
    service: MarketDataService,
    ticker: str,
    config: BacktestConfig | None = None,
    *,
    settings: Settings | None = None,
    use_cache: bool = True,
) -> BacktestReport:
    """Fetch prices through ``service`` and backtest the forecasting approach on them."""
    from stock_analyzer.analysis.technical import adjusted_ohlc  # local: keeps import graph flat

    settings = settings or get_settings()
    md = service.get_market_data(ticker)
    close = adjusted_ohlc(md.prices)["Close"]
    config = config or BacktestConfig(
        drift_mode=settings.mc_drift_mode,
        lookback_years=settings.mc_lookback_years,
        risk_free_rate=settings.risk_free_rate,
    )
    cache = None
    if use_cache:
        try:
            cache = SQLiteCache(settings.cache_path)
        except StockAnalyzerError as exc:
            log.warning("backtest cache unavailable", extra={"error": str(exc)})
    return backtest_series(md.ticker, close, config, cache=cache, extra_warnings=md.warnings)
