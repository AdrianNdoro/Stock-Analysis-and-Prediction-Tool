"""Process-wide objects. One MarketDataService per server process (shared, thread-safe cache);
everything user-specific lives in ``st.session_state`` instead, so sessions never share state.

``build_service`` and ``get_app_settings`` are looked up at call time so tests can replace them.
"""

from __future__ import annotations

import streamlit as st

from stock_analyzer.config import Settings, get_settings
from stock_analyzer.data.service import MarketDataService, build_default_service
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
