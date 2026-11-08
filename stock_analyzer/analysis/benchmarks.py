"""Benchmark indices and currency helpers.

Beta is only meaningful against a sensible market index, ideally one in the same region and
currency as the stock. The list is deliberately short; add entries here as needed.
"""

from __future__ import annotations

from dataclasses import dataclass

from stock_analyzer.exceptions import UnknownBenchmarkError


@dataclass(frozen=True)
class Benchmark:
    key: str
    name: str
    ticker: str
    currency: str  # ISO code of the index level's currency
    region: str


BENCHMARKS: dict[str, Benchmark] = {
    b.key: b
    for b in (
        Benchmark("SP500", "S&P 500", "^GSPC", "USD", "US"),
        Benchmark("FTSE100", "FTSE 100", "^FTSE", "GBP", "UK"),
        Benchmark("EUROSTOXX50", "Euro Stoxx 50", "^STOXX50E", "EUR", "Eurozone"),
        Benchmark("NIKKEI225", "Nikkei 225", "^N225", "JPY", "Japan"),
        Benchmark("HANGSENG", "Hang Seng", "^HSI", "HKD", "Hong Kong"),
        Benchmark("MSCIWORLD", "MSCI World (iShares ETF)", "URTH", "USD", "Global"),
    )
}

DEFAULT_BENCHMARK_KEY = "SP500"

# Exchange suffix -> benchmark key. Anything not listed falls back to the S&P 500.
_SUFFIX_TO_KEY = {
    ".L": "FTSE100",
    ".PA": "EUROSTOXX50",
    ".DE": "EUROSTOXX50",
    ".AS": "EUROSTOXX50",
    ".MI": "EUROSTOXX50",
    ".MC": "EUROSTOXX50",
    ".BR": "EUROSTOXX50",
    ".HE": "EUROSTOXX50",
    ".LS": "EUROSTOXX50",
    ".T": "NIKKEI225",
    ".HK": "HANGSENG",
}

# Currencies quoted in a fraction of the major unit -> the major currency.
_MAJOR = {"GBp": "GBP", "GBX": "GBP", "ZAc": "ZAR", "ILA": "ILS"}


def get_benchmark(ticker_or_key: str) -> Benchmark:
    """Look a benchmark up by key (``SP500``) or ticker (``^GSPC``), case-insensitively."""
    needle = (ticker_or_key or "").strip().upper()
    for b in BENCHMARKS.values():
        if needle in (b.key.upper(), b.ticker.upper()):
            return b
    raise UnknownBenchmarkError(f"unknown benchmark {ticker_or_key!r}")


def suggest_benchmark(ticker: str) -> Benchmark:
    """A sensible default index for ``ticker`` based on its exchange suffix."""
    upper = ticker.strip().upper()
    for suffix, key in _SUFFIX_TO_KEY.items():
        if upper.endswith(suffix):
            return BENCHMARKS[key]
    return BENCHMARKS[DEFAULT_BENCHMARK_KEY]


def major_currency(code: str | None) -> str | None:
    """Map sub-unit codes (GBp pence) to their major currency; upper-case otherwise."""
    if not code:
        return None
    return _MAJOR.get(code, code.upper())
