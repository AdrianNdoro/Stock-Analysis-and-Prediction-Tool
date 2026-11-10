"""Process-wide objects. One MarketDataService per server process (shared, thread-safe cache);
everything user-specific lives in ``st.session_state`` instead, so sessions never share state.

``build_service`` and ``get_app_settings`` are looked up at call time so tests can replace them.
"""

from __future__ import annotations

import streamlit as st

from stock_analyzer.config import Settings, get_settings
from stock_analyzer.data.cache import SQLiteCache
from stock_analyzer.data.risk_free import (
    FRESH_SECONDS as RISK_FREE_TTL_SECONDS,
)
from stock_analyzer.data.risk_free import RiskFreeRate, fixed_rate, resolve_risk_free
from stock_analyzer.data.service import MarketDataService, build_default_service
from stock_analyzer.exceptions import CacheError
from stock_analyzer.logging_setup import setup_logging


def get_app_settings() -> Settings:
    return get_settings()


def build_service() -> MarketDataService:
    settings = get_app_settings()
    setup_logging(settings.log_level, settings.log_json)
    return build_default_service(settings)


@st.cache_resource(show_spinner=False)
def get_service() -> MarketDataService:
    return build_service()


@st.cache_data(show_spinner=False, ttl=RISK_FREE_TTL_SECONDS)
def get_risk_free() -> RiskFreeRate:
    """The risk-free rate for this process (fixed unless RISK_FREE_SOURCE=live). Never raises."""
    settings = get_app_settings()
    if settings.risk_free_source != "live":
        return fixed_rate(settings)
    try:
        cache = SQLiteCache(settings.cache_path)
    except CacheError:
        cache = None
    return resolve_risk_free(settings, cache)
