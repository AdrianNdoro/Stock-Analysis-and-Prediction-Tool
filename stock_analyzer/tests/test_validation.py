import pytest

from stock_analyzer.data.validation import normalize_ticker
from stock_analyzer.exceptions import InvalidTickerError


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("aapl", "AAPL"),
        ("  NVDA ", "NVDA"),
        ("brk-b", "BRK-B"),
        ("bp.l", "BP.L"),
        ("7203.T", "7203.T"),
        ("AAPL\n", "AAPL"),
    ],
)
def test_valid_tickers_are_normalised(raw, expected):
    assert normalize_ticker(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "A B",
        "AAPL;DROP TABLE",
        "A" * 17,
        "AAPL/../x",
        "A..B",
        "ÄPPLE",
        "AA$",
        "^GSPC",
        "EURUSD=X",
        None,
        123,
        "'; --",
    ],
)
def test_invalid_tickers_are_rejected(raw):
    with pytest.raises(InvalidTickerError):
        normalize_ticker(raw)


def test_index_symbols_only_when_allowed():
    assert normalize_ticker("^gspc", allow_index=True) == "^GSPC"
    with pytest.raises(InvalidTickerError):
        normalize_ticker("^", allow_index=True)
