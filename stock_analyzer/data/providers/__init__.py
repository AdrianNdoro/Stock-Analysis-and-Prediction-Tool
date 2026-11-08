"""Provider registry."""

from __future__ import annotations

from stock_analyzer.config import Settings
from stock_analyzer.data.providers.base import DataProvider
from stock_analyzer.exceptions import ConfigError

AVAILABLE = ("yfinance",)


def get_provider(settings: Settings) -> DataProvider:
    """Instantiate the provider named in ``settings.provider``."""
    if settings.provider == "yfinance":
        from stock_analyzer.data.providers.yfinance_provider import YFinanceProvider

        return YFinanceProvider()
    raise ConfigError(
        f"Unknown DATA_PROVIDER {settings.provider!r}. Available: {', '.join(AVAILABLE)}. "
        "See data/providers/base.py to add one."
    )
