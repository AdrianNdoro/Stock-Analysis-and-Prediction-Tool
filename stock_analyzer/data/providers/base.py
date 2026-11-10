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
    "financial_currency",
    "quote_type",
]

# UNITS CONTRACT for get_fundamentals (a provider converts its own conventions to these):
#   pe_ratio, forward_pe, pb_ratio  plain ratios (22.1 means 22.1x)
#   eps_ttm                         per share, in ``financial_currency`` (may differ from the
#                                   trading ``currency``, e.g. BP.L trades in GBp, reports in USD)
#   revenue_growth, earnings_growth fractions, latest quarter vs the same quarter a year earlier
#   debt_to_equity                  ratio (0.84 means debt is 84% of equity)
#   dividend_yield                  fraction (0.03 means 3%); 0.0 only if the provider says "none"
#   market_cap                      in the major unit of ``currency`` (pounds, not pence)
# Anything unknown or implausible is None. Never 0 for "unknown".


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
