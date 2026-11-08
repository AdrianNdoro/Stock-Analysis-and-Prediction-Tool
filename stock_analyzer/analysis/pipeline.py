"""Glue: MarketData in, one AnalysisResult out. ``analyze`` is pure; ``analyze_ticker`` does IO."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from stock_analyzer.analysis.benchmarks import (
    DEFAULT_BENCHMARK_KEY,
    Benchmark,
    get_benchmark,
    major_currency,
    suggest_benchmark,
)
from stock_analyzer.analysis.risk import RiskMetrics, compute_risk, summarize_risk
from stock_analyzer.analysis.technical import (
    TechnicalSnapshot,
    adjusted_ohlc,
    compute_indicators,
    find_levels,
    summarize_technicals,
)
from stock_analyzer.config import Settings, get_settings
from stock_analyzer.data.service import MarketDataService
from stock_analyzer.data.types import MarketData
from stock_analyzer.exceptions import StockAnalyzerError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisResult:
    ticker: str
    currency: str | None
    as_of: date
    last_price: float
    technical: TechnicalSnapshot
    risk: RiskMetrics
    risk_bullets: list[str]
    benchmark: Benchmark | None
    adj_close: pd.Series
    warnings: tuple[str, ...] = field(default_factory=tuple)


def analyze(
    stock: MarketData,
    benchmark_data: MarketData | None = None,
    benchmark: Benchmark | None = None,
    *,
    risk_free_rate: float = 0.04,
    lookback_years: float = 3.0,
) -> AnalysisResult:
    """Technical + risk analysis for one stock. Never mixes currencies silently."""
    warnings: list[str] = list(stock.warnings)

    ohlc = adjusted_ohlc(stock.prices)
    indicators = compute_indicators(ohlc)
    levels = find_levels(ohlc)
    technical = summarize_technicals(indicators, levels, stock.currency)

    bench_close = None
    if benchmark_data is not None and benchmark is not None:
        bench_close = adjusted_ohlc(benchmark_data.prices)["Close"]
        stock_ccy, bench_ccy = major_currency(stock.currency), major_currency(benchmark.currency)
        if stock_ccy and bench_ccy and stock_ccy != bench_ccy:
            warnings.append(
                f"{stock.ticker} is priced in {stock_ccy} but {benchmark.name} is in {bench_ccy}. "
                "Beta uses each series' own local-currency returns, so currency moves are not "
                "included. Treat it as indicative."
            )
        suggested = suggest_benchmark(stock.ticker)
        if suggested.key != benchmark.key:
            warnings.append(
                f"{benchmark.name} was used as the benchmark; {suggested.name} is usually a "
                f"closer match for {stock.ticker}."
            )
    elif benchmark is not None:
        warnings.append(f"Beta unavailable: {benchmark.name} data could not be loaded.")

    risk = compute_risk(
        ohlc["Close"],
        bench_close,
        risk_free_rate=risk_free_rate,
        lookback_years=lookback_years,
    )
    warnings.extend(risk.warnings)

    return AnalysisResult(
        ticker=stock.ticker,
        currency=stock.currency,
        as_of=technical.as_of,
        last_price=technical.price,
        technical=technical,
        risk=risk,
        risk_bullets=summarize_risk(risk),
        benchmark=benchmark,
        adj_close=ohlc["Close"],
        warnings=tuple(dict.fromkeys(warnings)),  # de-duplicate, keep order
    )


def analyze_ticker(
    service: MarketDataService,
    ticker: str,
    benchmark: str | None = None,
    settings: Settings | None = None,
) -> AnalysisResult:
    """Fetch data through ``service`` and analyse. A failed benchmark fetch only drops beta."""
    settings = settings or get_settings()
    stock = service.get_market_data(ticker)
    if benchmark:
        bench = get_benchmark(benchmark)
    else:
        bench = suggest_benchmark(stock.ticker)
        if bench.key == DEFAULT_BENCHMARK_KEY:  # no regional match: use the configured default
            bench = get_benchmark(settings.default_benchmark)
    bench_data = None
    try:
        bench_data = service.get_market_data(bench.ticker, allow_index=True)
    except StockAnalyzerError as exc:
        log.warning("benchmark unavailable", extra={"benchmark": bench.ticker, "error": str(exc)})
    return analyze(
        stock,
        bench_data,
        bench,
        risk_free_rate=settings.risk_free_rate,
        lookback_years=settings.risk_lookback_years,
    )
