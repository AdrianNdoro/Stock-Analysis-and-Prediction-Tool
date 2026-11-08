"""Project exceptions.

Every error a user can trigger carries a ``user_message``: short, friendly, and safe to show
in the UI (no stack traces, no internals). The technical detail stays in ``str(exc)`` for logs.
"""

from __future__ import annotations


class StockAnalyzerError(Exception):
    """Base class for all expected, user-presentable errors."""

    user_message = "Something went wrong. Please try again in a moment."

    def __init__(self, message: str | None = None, *, user_message: str | None = None) -> None:
        super().__init__(message or user_message or self.user_message)
        if user_message:
            self.user_message = user_message


class ConfigError(StockAnalyzerError):
    user_message = "The application is misconfigured. Please contact the site owner."


class InvalidTickerError(StockAnalyzerError):
    user_message = (
        "That doesn't look like a valid ticker. Use letters, digits, '.' or '-' "
        "(for example AAPL, BRK-B or BP.L)."
    )


class TickerNotFoundError(StockAnalyzerError):
    user_message = (
        "We couldn't find price data for that ticker. Check the spelling and, for non-US "
        "stocks, the exchange suffix (for example BP.L for London)."
    )


class InsufficientDataError(StockAnalyzerError):
    user_message = "There isn't enough price history for this ticker to produce reliable results."


class DataValidationError(StockAnalyzerError):
    user_message = "The data provider returned data in an unexpected format. Please try later."


class CacheError(StockAnalyzerError):
    user_message = "The data cache is temporarily unavailable."


class ProviderError(StockAnalyzerError):
    """Base class for upstream data provider failures."""


class ProviderUnavailableError(ProviderError):
    user_message = "The market data provider is temporarily unavailable. Please try again soon."


class ProviderRateLimitError(ProviderError):
    user_message = "The market data provider is busy right now. Please try again in a minute."

    def __init__(
        self,
        message: str | None = None,
        *,
        user_message: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, user_message=user_message)
        self.retry_after = retry_after


class UnknownBenchmarkError(StockAnalyzerError):
    user_message = "That benchmark isn't supported. Please pick one from the list."


class InvalidParameterError(StockAnalyzerError):
    user_message = "One of the settings you chose isn't allowed. Please adjust it and try again."
