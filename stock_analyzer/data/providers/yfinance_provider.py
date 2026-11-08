"""yfinance-backed provider.

RISK: yfinance scrapes Yahoo Finance's unofficial endpoints. Its terms are aimed at personal
use, it can break or throttle without notice, and redistributing the data in a public product
may violate Yahoo's terms. Fine for development and a private demo; before opening the tool
to the public, switch to a licensed API through the :class:`DataProvider` interface.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date
from typing import Any

from stock_analyzer.data.providers.base import FUNDAMENTAL_FIELDS, DataProvider
from stock_analyzer.data.types import RawHistory
from stock_analyzer.exceptions import (
    ProviderRateLimitError,
    ProviderUnavailableError,
    StockAnalyzerError,
    TickerNotFoundError,
)

log = logging.getLogger(__name__)

# Our field name -> Yahoo ``info`` key.
# TODO(Phase 6): normalise units (Yahoo's dividendYield and debtToEquity units vary by version).
_INFO_MAP = {
    "pe_ratio": "trailingPE",
    "forward_pe": "forwardPE",
    "pb_ratio": "priceToBook",
    "eps_ttm": "trailingEps",
    "revenue_growth": "revenueGrowth",
    "earnings_growth": "earningsGrowth",
    "debt_to_equity": "debtToEquity",
    "dividend_yield": "dividendYield",
    "market_cap": "marketCap",
    "sector": "sector",
    "industry": "industry",
    "currency": "currency",
}


def _map_exception(exc: Exception) -> StockAnalyzerError:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "ratelimit" in text or "too many requests" in text or "429" in text:
        return ProviderRateLimitError(f"yfinance throttled: {exc}")
    return ProviderUnavailableError(f"yfinance failure: {type(exc).__name__}: {exc}")


class YFinanceProvider(DataProvider):
    name = "yfinance"

    def __init__(self, ticker_factory: Callable[[str], Any] | None = None) -> None:
        if ticker_factory is None:
            import yfinance as yf

            ticker_factory = yf.Ticker
        self._factory = ticker_factory

    def get_history(self, ticker: str, start: date, end: date | None = None) -> RawHistory:
        try:
            handle = self._factory(ticker)
            df = handle.history(
                start=start.isoformat(),
                end=end.isoformat() if end else None,
                auto_adjust=False,
                actions=True,
            )
            currency = self._currency(handle)
        except StockAnalyzerError:
            raise
        except Exception as exc:
            raise _map_exception(exc) from exc
        if df is None or df.empty:
            raise TickerNotFoundError(f"no data returned for {ticker}")
        return RawHistory(ticker=ticker, prices=df, currency=currency, source=self.name)

    @staticmethod
    def _currency(handle: Any) -> str | None:
        for getter in (
            lambda: handle.history_metadata.get("currency"),
            lambda: handle.fast_info["currency"],
        ):
            try:
                value = getter()
            except Exception:
                continue
            if value:
                return str(value)
        return None

    def get_fundamentals(self, ticker: str) -> dict[str, Any]:
        try:
            info = self._factory(ticker).info or {}
        except StockAnalyzerError:
            raise
        except Exception as exc:
            raise _map_exception(exc) from exc
        return {name: info.get(_INFO_MAP[name]) for name in FUNDAMENTAL_FIELDS}
