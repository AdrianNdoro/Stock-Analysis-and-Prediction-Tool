import threading

import pytest

from stock_analyzer.config import Settings
from stock_analyzer.data.cache import SQLiteCache
from stock_analyzer.data.resilience import RateLimiter
from stock_analyzer.data.service import MarketDataService
from stock_analyzer.exceptions import (
    CacheError,
    InsufficientDataError,
    InvalidTickerError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    TickerNotFoundError,
)
from stock_analyzer.tests.fakes import FakeClock, FakeProvider, make_prices


@pytest.fixture
def clock():
    return FakeClock()


def make_service(tmp_path, provider, clock, cache=None, **overrides):
    settings = Settings(cache_path=tmp_path / "c.sqlite3", backoff_base_seconds=0.0, **overrides)
    cache = cache or SQLiteCache(settings.cache_path, clock=clock)
    limiter = RateLimiter(10_000, burst=10_000)
    return MarketDataService(
        provider, cache, settings, limiter=limiter, clock=clock, sleep=lambda s: None
    )


def test_miss_then_hit_calls_provider_once(tmp_path, clock):
    p = FakeProvider()
    svc = make_service(tmp_path, p, clock)
    first = svc.get_market_data("AAPL")
    second = svc.get_market_data("AAPL")
    assert p.history_calls == 1
    assert not first.from_cache and second.from_cache
    assert first.currency == "USD" and second.ticker == "AAPL"
    assert first.prices.shape == second.prices.shape
    assert abs(first.prices["Adj Close"].iloc[-1] - second.prices["Adj Close"].iloc[-1]) < 1e-6
    assert second.prices.index.equals(first.prices.index)


def test_ticker_variants_share_one_cache_entry(tmp_path, clock):
    p = FakeProvider()
    svc = make_service(tmp_path, p, clock)
    for t in ("aapl", " AAPL ", "AAPL"):
        svc.get_market_data(t)
    assert p.history_calls == 1


def test_cache_expires_after_ttl(tmp_path, clock):
    p = FakeProvider()
    svc = make_service(tmp_path, p, clock)
    svc.get_market_data("AAPL")
    clock.advance(23 * 3600)
    svc.get_market_data("AAPL")
    assert p.history_calls == 1
    clock.advance(2 * 3600)
    svc.get_market_data("AAPL")
    assert p.history_calls == 2


def test_invalid_ticker_never_reaches_provider(tmp_path, clock):
    p = FakeProvider()
    svc = make_service(tmp_path, p, clock)
    with pytest.raises(InvalidTickerError):
        svc.get_market_data("AAPL; DROP TABLE")
    assert p.history_calls == 0


def test_unknown_ticker_is_negatively_cached(tmp_path, clock):
    p = FakeProvider(outcomes=[TickerNotFoundError("nope")])
    svc = make_service(tmp_path, p, clock)
    for _ in range(3):
        with pytest.raises(TickerNotFoundError):
            svc.get_market_data("ZZZZ")
    assert p.history_calls == 1
    clock.advance(61 * 60)  # negative entry expires
    assert svc.get_market_data("ZZZZ").ticker == "ZZZZ"
    assert p.history_calls == 2


def test_transient_failures_are_retried(tmp_path, clock):
    p = FakeProvider(outcomes=[ProviderRateLimitError("429"), ProviderUnavailableError("down")])
    svc = make_service(tmp_path, p, clock)
    assert svc.get_market_data("AAPL").prices.shape[0] == 800
    assert p.history_calls == 3


def test_outage_serves_stale_cache_with_warning(tmp_path, clock):
    p = FakeProvider()
    svc = make_service(tmp_path, p, clock)
    svc.get_market_data("AAPL")
    clock.advance(3 * 24 * 3600)
    p.outcomes = [ProviderUnavailableError("down")] * 10
    md = svc.get_market_data("AAPL")
    assert md.stale and md.from_cache
    assert any("temporarily unavailable" in w for w in md.warnings)


def test_outage_without_cache_raises_friendly_error(tmp_path, clock):
    p = FakeProvider(outcomes=[ProviderRateLimitError("429")] * 10)
    svc = make_service(tmp_path, p, clock)
    with pytest.raises(ProviderRateLimitError) as err:
        svc.get_market_data("AAPL")
    assert "try again" in err.value.user_message.lower()
    assert p.history_calls == 4  # 1 try + 3 retries


def test_too_little_history_is_refused_and_not_cached(tmp_path, clock):
    p = FakeProvider(prices=make_prices(30))
    svc = make_service(tmp_path, p, clock)
    for _ in range(2):
        with pytest.raises(InsufficientDataError):
            svc.get_market_data("NEW")
    assert p.history_calls == 2


def test_many_concurrent_users_cause_a_single_fetch(tmp_path, clock):
    p = FakeProvider(delay=0.2)
    svc = make_service(tmp_path, p, clock)
    results, errors = [], []

    def user():
        try:
            results.append(svc.get_market_data("AAPL"))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=user) for _ in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and len(results) == 12
    assert p.history_calls == 1
    assert svc._inflight == {}  # per-key locks are cleaned up


def test_different_tickers_are_isolated(tmp_path, clock):
    svc = make_service(tmp_path, FakeProvider(), clock)
    a = svc.get_market_data("AAA")
    b = svc.get_market_data("BBB")
    assert a.ticker == "AAA" and b.ticker == "BBB"
    a.prices.iloc[0, 0] = -1  # mutating one result must not leak into the next read
    assert svc.get_market_data("AAA").prices.iloc[0, 0] != -1


def test_currency_warnings(tmp_path, clock):
    unknown = make_service(tmp_path, FakeProvider(currency=None), clock).get_market_data("X1")
    assert any("Currency could not be determined" in w for w in unknown.warnings)
    pence = make_service(tmp_path, FakeProvider(currency="GBp"), clock).get_market_data("BP.L")
    assert any("pence" in w for w in pence.warnings)


def test_stale_last_bar_warning(tmp_path, clock):
    p = FakeProvider(prices=make_prices(800, end="2026-09-01"))
    md = make_service(tmp_path, p, clock).get_market_data("OLD")
    assert any("halted or delisted" in w for w in md.warnings)


class BrokenCache(SQLiteCache):
    def get(self, key):
        raise CacheError("disk on fire")

    def set(self, key, value):
        raise CacheError("disk on fire")


def test_broken_cache_degrades_to_provider(tmp_path, clock):
    p = FakeProvider()
    cache = BrokenCache(tmp_path / "b.sqlite3")
    svc = make_service(tmp_path, p, clock, cache=cache)
    assert svc.get_market_data("AAPL").prices.shape[0] == 800
    assert svc.get_market_data("AAPL").prices.shape[0] == 800
    assert p.history_calls == 2


def test_corrupt_cache_entry_is_ignored(tmp_path, clock):
    p = FakeProvider()
    svc = make_service(tmp_path, p, clock)
    svc._cache.set("prices:fake:AAPL", "{not json")
    assert svc.get_market_data("AAPL").prices.shape[0] == 800
    assert p.history_calls == 1


def test_fundamentals_cached_and_flagged_missing(tmp_path, clock):
    p = FakeProvider(fundamentals={"pe_ratio": 28.5, "market_cap": 3e12, "currency": "USD"})
    svc = make_service(tmp_path, p, clock)
    f1, f2 = svc.get_fundamentals("AAPL"), svc.get_fundamentals("AAPL")
    assert f1.available and f1.values["pe_ratio"] == 28.5 and f1.values["pb_ratio"] is None
    assert f2.from_cache and p.fundamentals_calls == 1


def test_fundamentals_degrade_gracefully(tmp_path, clock):
    p = FakeProvider(fundamentals=ProviderUnavailableError("down"))
    f = make_service(tmp_path, p, clock).get_fundamentals("AAPL")
    assert not f.available and f.warnings
    empty = make_service(tmp_path, FakeProvider(fundamentals={}), clock).get_fundamentals("ETF1")
    assert not empty.available and any("No fundamental data" in w for w in empty.warnings)
