import pandas as pd
import pytest

from stock_analyzer.analysis.benchmarks import (
    BENCHMARKS,
    get_benchmark,
    major_currency,
    suggest_benchmark,
)
from stock_analyzer.analysis.pipeline import analyze, analyze_ticker
from stock_analyzer.config import Settings
from stock_analyzer.data.cache import SQLiteCache
from stock_analyzer.data.resilience import RateLimiter
from stock_analyzer.data.service import MarketDataService
from stock_analyzer.exceptions import TickerNotFoundError, UnknownBenchmarkError
from stock_analyzer.tests.fakes import FakeClock, FakeProvider, make_prices


@pytest.mark.parametrize(
    ("ticker", "key"),
    [("AAPL", "SP500"), ("BP.L", "FTSE100"), ("SAP.DE", "EUROSTOXX50"), ("7203.T", "NIKKEI225"),
     ("0700.HK", "HANGSENG"), ("bp.l", "FTSE100")],
)  # fmt: skip
def test_suggest_benchmark_by_suffix(ticker, key):
    assert suggest_benchmark(ticker).key == key


def test_get_benchmark_by_key_or_ticker():
    assert get_benchmark("sp500").ticker == "^GSPC" and get_benchmark("^ftse").key == "FTSE100"
    assert {b.ticker for b in BENCHMARKS.values()} >= {"^GSPC", "^FTSE", "^N225"}
    with pytest.raises(UnknownBenchmarkError):
        get_benchmark("DOGE")


def test_major_currency():
    assert major_currency("GBp") == "GBP" and major_currency("usd") == "USD"
    assert major_currency(None) is None


class MultiProvider(FakeProvider):
    """Different prices per ticker; unknown tickers raise TickerNotFoundError."""

    def __init__(self, frames, currencies=None):
        super().__init__()
        self.frames, self.currencies = frames, currencies or {}

    def get_history(self, ticker, start, end=None):
        if ticker not in self.frames:
            raise TickerNotFoundError(ticker)
        self.prices, self.currency = self.frames[ticker], self.currencies.get(ticker, "USD")
        return super().get_history(ticker, start, end)


def service(tmp_path, provider):
    clock = FakeClock()
    s = Settings(cache_path=tmp_path / "c.sqlite3", backoff_base_seconds=0.0)
    return MarketDataService(
        provider, SQLiteCache(s.cache_path, clock=clock), s,
        limiter=RateLimiter(10_000, 10_000), clock=clock, sleep=lambda x: None,
    ), s  # fmt: skip


def test_analyze_ticker_end_to_end_with_beta(tmp_path):
    frames = {"AAPL": make_prices(900, seed=1), "^GSPC": make_prices(900, seed=2)}
    svc, s = service(tmp_path, MultiProvider(frames))
    a = analyze_ticker(svc, "aapl", settings=s)
    assert a.ticker == "AAPL" and a.benchmark.key == "SP500"
    assert a.risk.beta is not None and a.risk.beta.n_obs >= 52
    assert a.technical.trend_label and a.risk_bullets and len(a.risk.var) == 2
    assert a.last_price == pytest.approx(frames["AAPL"]["Adj Close"].iloc[-1])


def test_analyze_ticker_survives_benchmark_failure(tmp_path):
    svc, s = service(tmp_path, MultiProvider({"AAPL": make_prices(900)}))
    a = analyze_ticker(svc, "AAPL", settings=s)
    assert a.risk.beta is None and any("Beta unavailable" in w for w in a.warnings)


def test_currency_mismatch_and_benchmark_mismatch_are_flagged(tmp_path):
    frames = {"AAPL": make_prices(900), "^FTSE": make_prices(900, seed=9)}
    svc, s = service(tmp_path, MultiProvider(frames, {"^FTSE": "GBP"}))
    a = analyze_ticker(svc, "AAPL", "FTSE100", s)
    joined = " ".join(a.warnings)
    assert "priced in USD but FTSE 100 is in GBP" in joined
    assert "S&P 500 is usually a closer match" in joined


def test_pence_stock_vs_pound_index_is_not_a_mismatch(tmp_path):
    frames = {"BP.L": make_prices(900), "^FTSE": make_prices(900, seed=9)}
    svc, s = service(tmp_path, MultiProvider(frames, {"BP.L": "GBp", "^FTSE": "GBP"}))
    a = analyze_ticker(svc, "BP.L", settings=s)
    assert a.benchmark.key == "FTSE100"
    assert not any("priced in" in w for w in a.warnings)
    assert any("pence" in w for w in a.warnings)  # data-layer pence warning still shown


def test_analyze_pure_function_without_benchmark(tmp_path):
    svc, s = service(tmp_path, MultiProvider({"AAPL": make_prices(300)}))
    md = svc.get_market_data("AAPL")
    a = analyze(md)
    assert a.risk.beta is None and a.benchmark is None
    assert any("years of history" in w for w in a.warnings)  # <2y warning carried through
    assert isinstance(a.technical.as_of, pd.Timestamp.min.__class__) or a.technical.as_of
