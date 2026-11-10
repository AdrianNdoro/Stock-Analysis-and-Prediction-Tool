"""Market data service: cache -> rate limit -> retry -> provider -> clean -> cache.

Behaviour worth knowing:
* Fresh cache hit: no provider call at all.
* Single-flight: if 50 users ask for the same uncached ticker at once, one fetches, the rest
  wait and then read the cache.
* "Ticker not found" is remembered briefly so typos/bots can't hammer the provider.
* Provider outage + older cache entry: serve the stale data with a visible warning.
* A broken cache never breaks the app: it logs and falls through to the provider.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from io import StringIO
from typing import Any

import pandas as pd

from stock_analyzer.config import Settings, get_settings
from stock_analyzer.data.cache import CacheEntry, SQLiteCache
from stock_analyzer.data.cleaning import clean_prices
from stock_analyzer.data.providers import get_provider
from stock_analyzer.data.providers.base import FUNDAMENTAL_FIELDS, DataProvider
from stock_analyzer.data.resilience import RateLimiter, call_with_retry
from stock_analyzer.data.types import DataQualityReport, Fundamentals, MarketData, RawHistory
from stock_analyzer.data.validation import normalize_ticker
from stock_analyzer.exceptions import (
    CacheError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    TickerNotFoundError,
)

log = logging.getLogger(__name__)

_PAYLOAD_VERSION = 1
# Currencies quoted in a fraction of the major unit (a classic source of 100x errors).
_SUBUNIT_CURRENCIES = {
    "GBp": "pence (100 GBp = 1 GBP)",
    "GBX": "pence (100 GBX = 1 GBP)",
    "ZAc": "cents (100 ZAc = 1 ZAR)",
    "ILA": "agorot (100 ILA = 1 ILS)",
}


class MarketDataService:
    def __init__(
        self,
        provider: DataProvider,
        cache: SQLiteCache,
        settings: Settings,
        *,
        limiter: RateLimiter | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._provider = provider
        self._cache = cache
        self._s = settings
        self._clock = clock
        self._sleep = sleep
        self._limiter = limiter or RateLimiter(
            settings.rate_limit_per_minute, settings.rate_limit_burst
        )
        self._inflight: dict[str, list[Any]] = {}
        self._inflight_guard = threading.Lock()

    # ------------------------------------------------------------------ public API

    def get_market_data(self, ticker: str, *, allow_index: bool = False) -> MarketData:
        """Clean daily prices for ``ticker``. Raises :class:`StockAnalyzerError` subclasses."""
        symbol = normalize_ticker(ticker, allow_index=allow_index)
        key = f"prices:{self._provider.name}:{symbol}"
        missing_key = f"missing:{self._provider.name}:{symbol}"
        ttl = self._s.cache_ttl_prices_hours * 3600

        entry = self._cache_get(key)
        if entry and self._fresh(entry, ttl):
            md = self._decode(entry, from_cache=True, stale=False)
            if md:
                return md
        self._raise_if_known_missing(missing_key, symbol)

        with self._single_flight(key):
            entry = self._cache_get(key)  # someone may have filled it while we waited
            if entry and self._fresh(entry, ttl):
                md = self._decode(entry, from_cache=True, stale=False)
                if md:
                    return md
            self._raise_if_known_missing(missing_key, symbol)
            try:
                raw = self._fetch_history(symbol)
            except TickerNotFoundError:
                self._cache_set(missing_key, "1")
                raise
            except (ProviderRateLimitError, ProviderUnavailableError):
                if entry:
                    md = self._decode(entry, from_cache=True, stale=True)
                    if md:
                        log.warning("serving stale cache", extra={"ticker": symbol})
                        return md
                raise
            cleaned, report = clean_prices(raw.prices, self._s)
            fetched_at = self._clock()
            self._cache_set(key, self._encode(symbol, raw, cleaned, report))
            return self._build(
                symbol, raw.currency, cleaned, report, raw.source, fetched_at, False, False
            )

    def get_fundamentals(self, ticker: str) -> Fundamentals:
        """Fundamental snapshot. Never raises for provider trouble: degrades gracefully."""
        symbol = normalize_ticker(ticker)
        # v2: units contract (fractions/ratios). Old v1 entries held Yahoo's raw units.
        key = f"fundamentals:v2:{self._provider.name}:{symbol}"
        ttl = self._s.cache_ttl_fundamentals_hours * 3600

        entry = self._cache_get(key)
        if entry and self._fresh(entry, ttl):
            f = self._decode_fundamentals(symbol, entry, from_cache=True)
            if f:
                return f
        with self._single_flight(key):
            entry = self._cache_get(key)
            if entry and self._fresh(entry, ttl):
                f = self._decode_fundamentals(symbol, entry, from_cache=True)
                if f:
                    return f
            try:
                values = call_with_retry(
                    lambda: self._limited(self._provider.get_fundamentals, symbol),
                    max_retries=self._s.max_retries,
                    base_delay=self._s.backoff_base_seconds,
                    sleep=self._sleep,
                    on_retry=self._log_retry(symbol),
                )
            except (ProviderRateLimitError, ProviderUnavailableError, TickerNotFoundError) as exc:
                log.warning("fundamentals unavailable", extra={"ticker": symbol, "error": str(exc)})
                if entry:
                    f = self._decode_fundamentals(symbol, entry, from_cache=True)
                    if f:
                        return Fundamentals(
                            f.ticker,
                            f.values,
                            f.available,
                            f.fetched_at,
                            True,
                            (
                                *f.warnings,
                                "Showing older fundamentals: the provider is unavailable.",
                            ),
                        )
                return Fundamentals(
                    symbol, {}, False, None, False,
                    ("Fundamental data is currently unavailable for this ticker.",),
                )  # fmt: skip
            clean = {name: values.get(name) for name in FUNDAMENTAL_FIELDS}
            available = any(v is not None for k, v in clean.items() if k not in {"currency"})
            self._cache_set(key, json.dumps({"v": _PAYLOAD_VERSION, "values": clean}))
            warnings = () if available else ("No fundamental data is available for this ticker.",)
            return Fundamentals(symbol, clean, available, self._now_dt(), False, warnings)

    # ------------------------------------------------------------------ fetching

    def _limited(self, fn: Callable[..., Any], *args: Any) -> Any:
        self._limiter.acquire()
        return fn(*args)

    def _log_retry(self, symbol: str) -> Callable[[int, float, Exception], None]:
        def _cb(attempt: int, delay: float, exc: Exception) -> None:
            log.warning(
                "provider call failed, retrying",
                extra={"ticker": symbol, "attempt": attempt, "delay_s": round(delay, 2),
                       "error": type(exc).__name__},
            )  # fmt: skip

        return _cb

    def _fetch_history(self, symbol: str) -> RawHistory:
        start: date = self._now_dt().date() - timedelta(days=int(self._s.history_years * 365.25))
        raw: RawHistory = call_with_retry(
            lambda: self._limited(self._provider.get_history, symbol, start),
            max_retries=self._s.max_retries,
            base_delay=self._s.backoff_base_seconds,
            sleep=self._sleep,
            on_retry=self._log_retry(symbol),
        )
        if raw.prices is None or raw.prices.empty:
            raise TickerNotFoundError(f"no data returned for {symbol}")
        return raw

    # ------------------------------------------------------------------ cache helpers

    def _now_dt(self) -> datetime:
        return datetime.fromtimestamp(self._clock(), UTC)

    def _fresh(self, entry: CacheEntry, ttl: float) -> bool:
        return entry.age_seconds(self._clock()) < ttl

    def _cache_get(self, key: str) -> CacheEntry | None:
        try:
            return self._cache.get(key)
        except CacheError as exc:
            log.error("cache read failed", extra={"error": str(exc)})
            return None

    def _cache_set(self, key: str, value: str) -> None:
        try:
            self._cache.set(key, value)
        except CacheError as exc:
            log.error("cache write failed", extra={"error": str(exc)})

    def _raise_if_known_missing(self, missing_key: str, symbol: str) -> None:
        entry = self._cache_get(missing_key)
        if entry and self._fresh(entry, self._s.cache_ttl_missing_minutes * 60):
            raise TickerNotFoundError(f"{symbol} recently reported as not found")

    @contextmanager
    def _single_flight(self, key: str) -> Iterator[None]:
        """Per-key lock that is removed once nobody uses it (bounded memory)."""
        with self._inflight_guard:
            slot = self._inflight.setdefault(key, [threading.Lock(), 0])
            slot[1] += 1
        try:
            with slot[0]:
                yield
        finally:
            with self._inflight_guard:
                slot[1] -= 1
                if slot[1] == 0:
                    self._inflight.pop(key, None)

    # ------------------------------------------------------------------ (de)serialisation

    @staticmethod
    def _encode(symbol: str, raw: RawHistory, df: pd.DataFrame, report: DataQualityReport) -> str:
        return json.dumps(
            {
                "v": _PAYLOAD_VERSION,
                "ticker": symbol,
                "currency": raw.currency,
                "source": raw.source,
                "prices": df.to_json(orient="split", date_format="iso"),
                "quality": report.to_dict(),
            }
        )

    def _decode(self, entry: CacheEntry, *, from_cache: bool, stale: bool) -> MarketData | None:
        try:
            p = json.loads(entry.value)
            if p.get("v") != _PAYLOAD_VERSION:
                return None
            df = pd.read_json(StringIO(p["prices"]), orient="split")
            df.index = pd.DatetimeIndex(pd.to_datetime(df.index), name="Date")
            df = df.sort_index().astype("float64")
            report = DataQualityReport.from_dict(p["quality"])
            return self._build(
                p["ticker"], p["currency"], df, report, p["source"], entry.stored_at,
                from_cache, stale,
            )  # fmt: skip
        except (ValueError, KeyError, TypeError) as exc:
            log.error("corrupt cache entry ignored", extra={"key": entry.key, "error": str(exc)})
            return None

    def _decode_fundamentals(
        self, symbol: str, entry: CacheEntry, *, from_cache: bool
    ) -> Fundamentals | None:
        try:
            p = json.loads(entry.value)
            if p.get("v") != _PAYLOAD_VERSION:
                return None
            values = {name: p["values"].get(name) for name in FUNDAMENTAL_FIELDS}
        except (ValueError, KeyError, AttributeError):
            return None
        available = any(v is not None for k, v in values.items() if k != "currency")
        warnings = () if available else ("No fundamental data is available for this ticker.",)
        return Fundamentals(
            symbol, values, available, datetime.fromtimestamp(entry.stored_at, UTC),
            from_cache, warnings,
        )  # fmt: skip

    def _build(
        self,
        symbol: str,
        currency: str | None,
        df: pd.DataFrame,
        report: DataQualityReport,
        source: str,
        fetched_epoch: float,
        from_cache: bool,
        stale: bool,
    ) -> MarketData:
        fetched_at = datetime.fromtimestamp(fetched_epoch, UTC)
        warnings = list(report.warnings)
        if stale:
            warnings.insert(
                0,
                f"Showing cached data from {fetched_at:%Y-%m-%d} because the data provider is "
                "temporarily unavailable.",
            )
        last_bar = df.index[-1].date()
        age_days = (self._now_dt().date() - last_bar).days
        if age_days > self._s.stale_bar_days:
            warnings.append(
                f"The latest price is from {last_bar} ({age_days} days ago). The stock may be "
                "halted or delisted, or the data may be delayed."
            )
        if not currency:
            warnings.append("Currency could not be determined; prices are shown without a label.")
        elif currency in _SUBUNIT_CURRENCIES:
            warnings.append(
                f"Prices are quoted in {currency}: {_SUBUNIT_CURRENCIES[currency]}. "
                "Do not compare them directly with prices in the major currency."
            )
        return MarketData(
            ticker=symbol,
            currency=currency,
            prices=df,
            quality=report,
            source=source,
            fetched_at=fetched_at,
            from_cache=from_cache,
            stale=stale,
            warnings=tuple(warnings),
        )


def build_default_service(settings: Settings | None = None) -> MarketDataService:
    """Wire the service from settings (create once per process, e.g. st.cache_resource)."""
    settings = settings or get_settings()
    return MarketDataService(get_provider(settings), SQLiteCache(settings.cache_path), settings)
