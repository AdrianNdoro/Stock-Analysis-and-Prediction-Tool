from datetime import date

import pytest

from stock_analyzer.data.providers.yfinance_provider import YFinanceProvider
from stock_analyzer.exceptions import (
    ProviderRateLimitError,
    ProviderUnavailableError,
    TickerNotFoundError,
)
from stock_analyzer.tests.fakes import make_prices


class FakeTicker:
    def __init__(self, df=None, error=None, info=None, meta=None):
        self._df, self._error, self.info = df, error, info or {}
        self.history_metadata = meta if meta is not None else {"currency": "USD"}
        self.fast_info = {"currency": "EUR"}

    def history(self, **kwargs):
        assert kwargs["auto_adjust"] is False  # we need both Close and Adj Close
        if self._error:
            raise self._error
        return self._df


def provider(**kw):
    return YFinanceProvider(ticker_factory=lambda t: FakeTicker(**kw))


def test_success_returns_prices_and_currency():
    raw = provider(df=make_prices(300)).get_history("AAPL", date(2020, 1, 1))
    assert raw.currency == "USD" and raw.source == "yfinance" and len(raw.prices) == 300


def test_currency_falls_back_to_fast_info():
    raw = provider(df=make_prices(300), meta={}).get_history("SAP.DE", date(2020, 1, 1))
    assert raw.currency == "EUR"


def test_empty_frame_means_ticker_not_found():
    import pandas as pd

    with pytest.raises(TickerNotFoundError):
        provider(df=pd.DataFrame()).get_history("NOPE", date(2020, 1, 1))


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RuntimeError("Too Many Requests. Rate limited."), ProviderRateLimitError),
        (type("YFRateLimitError", (Exception,), {})("x"), ProviderRateLimitError),
        (ConnectionError("network down"), ProviderUnavailableError),
    ],
)
def test_errors_are_mapped(error, expected):
    with pytest.raises(expected):
        provider(error=error).get_history("AAPL", date(2020, 1, 1))


def test_fundamentals_mapping_fills_missing_with_none():
    out = provider(info={"trailingPE": 25.0, "sector": "Technology"}).get_fundamentals("AAPL")
    assert out["pe_ratio"] == 25.0 and out["sector"] == "Technology" and out["pb_ratio"] is None
