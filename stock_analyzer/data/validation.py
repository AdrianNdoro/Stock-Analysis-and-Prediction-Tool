"""Sanitise user-supplied ticker symbols before they touch the cache or a provider."""

from __future__ import annotations

import re

from stock_analyzer.exceptions import InvalidTickerError

_STOCK_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,14}$")  # AAPL, BRK-B, BP.L, 7203.T
_INDEX_RE = re.compile(r"^\^[A-Z0-9.\-]{1,14}$")  # ^GSPC (benchmarks only)
MAX_TICKER_LENGTH = 16


def normalize_ticker(raw: object, *, allow_index: bool = False) -> str:
    """Return an upper-case, validated ticker or raise :class:`InvalidTickerError`.

    Strict allow-list: ASCII letters, digits, '.' and '-' only (plus a leading '^' for index
    benchmarks when ``allow_index`` is set). Anything else, including spaces inside the
    symbol, SQL/shell metacharacters and non-ASCII text, is rejected.
    """
    if not isinstance(raw, str):
        raise InvalidTickerError(f"ticker must be a string, got {type(raw).__name__}")
    symbol = raw.strip().upper()
    if not symbol or len(symbol) > MAX_TICKER_LENGTH or not symbol.isascii():
        raise InvalidTickerError("ticker empty, too long or non-ASCII")
    if ".." in symbol:
        raise InvalidTickerError("ticker contains '..'")
    if _STOCK_RE.match(symbol) or (allow_index and _INDEX_RE.match(symbol)):
        return symbol
    raise InvalidTickerError("ticker contains disallowed characters")
