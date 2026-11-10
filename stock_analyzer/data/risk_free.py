"""Risk-free rate: a fixed setting by default, or (opt-in) the live US 13-week T-bill yield.

Opt in with ``RISK_FREE_SOURCE=live``. The live value comes from Yahoo's ``^IRX`` index, is
cached for 12 hours, and any failure falls back (last known value up to 7 days old, then the
fixed ``RISK_FREE_RATE``), so the app never breaks because of it. The result always says where
the number came from, and the UI and report show that label.

Known limits: it is the US rate, applied to every currency (a UK or Japanese stock would
strictly need its own country's rate); it is rounded to the nearest 0.25% so that tiny daily
moves do not invalidate cached backtests; it affects Sharpe/Sortino and the "neutral" and
"shrunk" drift assumptions, not volatility.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date

from stock_analyzer.config import Settings
from stock_analyzer.data.cache import SQLiteCache
from stock_analyzer.exceptions import CacheError

log = logging.getLogger(__name__)

LIVE_SYMBOL = "^IRX"  # 13-week US Treasury bill, quoted in percent
CACHE_KEY = "riskfree:v1:IRX"
FRESH_SECONDS = 12 * 3600
STALE_OK_SECONDS = 7 * 24 * 3600
RATE_STEP = 0.0025
MAX_PLAUSIBLE = 0.20
FETCH_TIMEOUT_SECONDS = 5.0

Fetcher = Callable[[], tuple[float, str]]  # -> (yield in percent, as-of date ISO)


@dataclass(frozen=True)
class RiskFreeRate:
    value: float  # annual fraction used by the models (0.0425 = 4.25%)
    source: str  # "fixed" | "live" | "last_known" | "fallback"
    label: str  # plain-English sentence for the UI and the report
    as_of: str | None = None


def round_rate(x: float) -> float:
    return round(round(x / RATE_STEP) * RATE_STEP, 6)


def fixed_rate(settings: Settings, *, fallback: bool = False) -> RiskFreeRate:
    if fallback:
        label = (
            f"Risk-free rate {settings.risk_free_rate:.2%}: the live Treasury rate could not be "
            "fetched, so the fixed setting is used."
        )
        return RiskFreeRate(settings.risk_free_rate, "fallback", label)
    label = (
        f"Risk-free rate {settings.risk_free_rate:.2%} (a fixed setting, not a live market rate)."
    )
    return RiskFreeRate(settings.risk_free_rate, "fixed", label)


def _live_label(value: float, as_of: str | None, last_known: bool) -> str:
    when = f" on {as_of}" if as_of else ""
    prefix = "Last known " if last_known else ""
    return (
        f"Risk-free rate {value:.2%}: {prefix}US 13-week Treasury bill yield{when}, rounded to "
        "the nearest 0.25%. It is a US rate applied to every currency, so for non-US stocks it "
        "is only an approximation."
    )


def yfinance_fetch(timeout: float = FETCH_TIMEOUT_SECONDS) -> tuple[float, str]:
    """Latest ^IRX close (percent) and its date. Raises on any problem."""
    import yfinance as yf

    df = yf.Ticker(LIVE_SYMBOL).history(period="10d", timeout=timeout)
    closes = df["Close"].dropna()
    if closes.empty:
        raise ValueError("no T-bill data returned")
    return float(closes.iloc[-1]), str(closes.index[-1].date())


def _validated(percent: float) -> float:
    if not math.isfinite(percent) or not 0.0 <= percent / 100 <= MAX_PLAUSIBLE:
        raise ValueError(f"implausible T-bill yield {percent!r}%")
    return round_rate(percent / 100)


def resolve_risk_free(
    settings: Settings,
    cache: SQLiteCache | None,
    *,
    fetcher: Fetcher = yfinance_fetch,
    clock: Callable[[], float] = time.time,
) -> RiskFreeRate:
    """The rate to use for this process. Never raises."""
    if settings.risk_free_source != "live":
        return fixed_rate(settings)
    entry = None
    try:
        entry = cache.get(CACHE_KEY) if cache else None
    except CacheError:
        log.warning("risk-free cache unreadable")
    cached = _decode(entry.value) if entry else None
    if entry and cached and entry.age_seconds(clock()) <= FRESH_SECONDS:
        return RiskFreeRate(cached[0], "live", _live_label(cached[0], cached[1], False), cached[1])
    try:
        percent, as_of = fetcher()
        value = _validated(percent)
    except Exception as exc:  # noqa: BLE001 - any failure must fall back quietly
        log.warning("live risk-free rate unavailable", extra={"error": type(exc).__name__})
        if entry and cached and entry.age_seconds(clock()) <= STALE_OK_SECONDS:
            return RiskFreeRate(
                cached[0], "last_known", _live_label(cached[0], cached[1], True), cached[1]
            )
        return fixed_rate(settings, fallback=True)
    if cache:
        try:
            cache.set(CACHE_KEY, json.dumps({"value": value, "as_of": as_of}))
        except CacheError:
            log.warning("risk-free cache write failed")
    return RiskFreeRate(value, "live", _live_label(value, as_of, False), as_of)


def _decode(raw: str) -> tuple[float, str | None] | None:
    try:
        d = json.loads(raw)
        value = float(d["value"])
        date.fromisoformat(d["as_of"])
        return (value, d["as_of"]) if 0.0 <= value <= MAX_PLAUSIBLE else None
    except (ValueError, KeyError, TypeError):
        return None


def with_rate(settings: Settings, rate: RiskFreeRate) -> Settings:
    """Settings copy whose ``risk_free_rate`` is the resolved one (everything reads that field)."""
    return replace(settings, risk_free_rate=rate.value)
