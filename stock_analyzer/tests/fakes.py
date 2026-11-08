"""Test doubles: synthetic prices, a scriptable provider, and a controllable clock."""

from __future__ import annotations

import threading
import time
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import pandas as pd

from stock_analyzer.data.providers.base import FUNDAMENTAL_FIELDS, DataProvider
from stock_analyzer.data.types import RawHistory

T0 = datetime(2026, 11, 4, 12, 0, tzinfo=UTC).timestamp()


class FakeClock:
    def __init__(self, now: float = T0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_prices(
    n: int = 800, *, end: str = "2026-11-03", seed: int = 0, start_price: float = 100.0
) -> pd.DataFrame:
    idx = pd.bdate_range(end=end, periods=n, name="Date")
    rng = np.random.default_rng(seed)
    close = start_price * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": close * 0.98,
            "Volume": rng.integers(500_000, 2_000_000, n).astype(float),
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=idx,
    )


class FakeProvider(DataProvider):
    """Scriptable: ``outcomes`` is a queue of exceptions (raised) or None (success)."""

    name = "fake"

    def __init__(
        self,
        prices: pd.DataFrame | None = None,
        currency: str | None = "USD",
        outcomes: list[Exception | None] | None = None,
        delay: float = 0.0,
        fundamentals: dict[str, Any] | Exception | None = None,
    ) -> None:
        self.prices = make_prices() if prices is None else prices
        self.currency = currency
        self.outcomes = list(outcomes or [])
        self.delay = delay
        self.fundamentals = fundamentals
        self.history_calls = 0
        self.fundamentals_calls = 0
        self._lock = threading.Lock()

    def get_history(self, ticker: str, start: date, end: date | None = None) -> RawHistory:
        with self._lock:
            self.history_calls += 1
            outcome = self.outcomes.pop(0) if self.outcomes else None
        if self.delay:
            time.sleep(self.delay)
        if outcome is not None:
            raise outcome
        return RawHistory(ticker, self.prices.copy(), self.currency, self.name)

    def get_fundamentals(self, ticker: str) -> dict[str, Any]:
        with self._lock:
            self.fundamentals_calls += 1
        if isinstance(self.fundamentals, Exception):
            raise self.fundamentals
        base = dict.fromkeys(FUNDAMENTAL_FIELDS)
        base.update(self.fundamentals or {})
        return base
