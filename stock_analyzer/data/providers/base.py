"""The swappable data provider interface.

To add a provider (Twelve Data, Polygon, FMP, Alpha Vantage...): subclass
:class:`DataProvider`, map its errors to our exceptions, register it in
``providers/__init__.py``, and read its API key from ``Settings.provider_api_key``
(env var ``DATA_PROVIDER_API_KEY``). Nothing else in the app needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Any

from stock_analyzer.data.types import RawHistory

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
OPTIONAL_COLUMNS = ["Dividends", "Stock Splits"]

FUNDAMENTAL_FIELDS = [
    "pe_ratio",
    "forward_pe",
    "pb_ratio",
    "eps_ttm",
    "revenue_growth",
    "earnings_growth",
    "debt_to_equity",
    "dividend_yield",
    "market_cap",
    "sector",
    "industry",
    "currency",
]


class DataProvider(ABC):
    """Contract every provider must honour. Implementations must be thread-safe."""

    name: str

    @abstractmethod
    def get_history(self, ticker: str, start: date, end: date | None = None) -> RawHistory:
        """Daily OHLCV from ``start`` (inclusive) to ``end`` (exclusive; None = latest).

        * ``prices``: DatetimeIndex, columns :data:`REQUIRED_COLUMNS` (``Close`` as-traded,
          ``Adj Close`` adjusted for splits and dividends).
        * ``currency``: ISO code of the quote currency if known, else None.
        * No data for the ticker -> raise ``TickerNotFoundError``.
        * Throttled -> ``ProviderRateLimitError``; outage/network -> ``ProviderUnavailableError``.
        """

    @abstractmethod
    def get_fundamentals(self, ticker: str) -> dict[str, Any]:
        """A dict containing every key in :data:`FUNDAMENTAL_FIELDS` (``None`` if unknown)."""
