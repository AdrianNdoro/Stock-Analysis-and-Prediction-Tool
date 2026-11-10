"""How this works and its limitations, plus the glossary."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # make `stock_analyzer` importable

import streamlit as st  # noqa: E402

from stock_analyzer.app.layout import page_chrome  # noqa: E402
from stock_analyzer.app.limitations import HOW_IT_WORKS, glossary_markdown  # noqa: E402

page_chrome("How this works", "📘")
st.title("How this works and its limitations")
st.markdown(HOW_IT_WORKS)
st.markdown(glossary_markdown())
