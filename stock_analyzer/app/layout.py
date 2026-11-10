"""Shared page chrome. Every page calls ``page_chrome`` so the disclaimer is always visible."""

from __future__ import annotations

import streamlit as st

from stock_analyzer.app.limitations import DISCLAIMER


def page_chrome(title: str, icon: str = "📈") -> None:
    st.set_page_config(page_title=title, page_icon=icon, layout="centered")
    st.warning(DISCLAIMER, icon="⚠️")
