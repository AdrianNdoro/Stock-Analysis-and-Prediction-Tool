"""Plain data containers shared across the data layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class RawHistory:
    """What a provider returns: untouched-by-us prices plus the quote currency."""

    ticker: str
    prices: pd.DataFrame
    currency: str | None
    source: str


@dataclass
class DataQualityReport:
    """What cleaning found and did. Shown to users so nothing is silently altered."""

    rows_raw: int
    rows_clean: int
    first_date: str
    last_date: str
    dropped: dict[str, int] = field(default_factory=dict)
    gaps: list[dict[str, Any]] = field(default_factory=list)
    outlier_dates: list[str] = field(default_factory=list)
    zero_volume_days: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataQualityReport:
        return cls(**data)


@dataclass(frozen=True)
class MarketData:
    """Clean price history for one ticker, ready for analysis.

    ``prices`` has a tz-naive daily DatetimeIndex and columns Open, High, Low, Close,
    Adj Close, Volume (+ Dividends, Stock Splits when available). Use ``Adj Close`` for
    returns (it accounts for splits and dividends). ``Close`` is the as-traded price.
    All prices are in ``currency``: never compare across currencies.
    """

    ticker: str
    currency: str | None
    prices: pd.DataFrame
    quality: DataQualityReport
    source: str
    fetched_at: datetime
    from_cache: bool
    stale: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class Fundamentals:
    """Fundamental snapshot. Missing values are ``None``; ``available`` is False if empty."""

    ticker: str
    values: dict[str, Any]
    available: bool
    fetched_at: datetime | None
    from_cache: bool
    warnings: tuple[str, ...]
