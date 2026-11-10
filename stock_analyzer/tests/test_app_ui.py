"""Streamlit AppTest smoke tests: the real app script, a fake data provider, no network."""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from stock_analyzer.app import services
from stock_analyzer.config import Settings
from stock_analyzer.data.cache import SQLiteCache
from stock_analyzer.data.resilience import RateLimiter
from stock_analyzer.data.service import MarketDataService
from stock_analyzer.exceptions import ProviderUnavailableError
from stock_analyzer.tests.fakes import FakeClock, FakeProvider, make_prices
from stock_analyzer.tests.test_benchmarks_and_pipeline import MultiProvider

APP = Path(__file__).resolve().parents[1] / "app"
MAIN, HOW = str(APP / "main.py"), str(APP / "pages" / "1_How_this_works.py")


@pytest.fixture
def make_app(tmp_path, monkeypatch):
    """Build the real app wired to a fake provider. Call with a provider to get an AppTest."""

    def build(provider) -> AppTest:
        settings = Settings(cache_path=tmp_path / "ui.sqlite3", backoff_base_seconds=0.0)
        clock = FakeClock()
        service = MarketDataService(
            provider, SQLiteCache(settings.cache_path, clock=clock), settings,
            limiter=RateLimiter(10_000, 10_000), clock=clock, sleep=lambda s: None,
        )  # fmt: skip
        monkeypatch.setattr(services, "build_service", lambda: service)
        monkeypatch.setattr(services, "get_app_settings", lambda: settings)
        services.get_service.clear()
        st.cache_data.clear()
        return AppTest.from_file(MAIN, default_timeout=120)

    yield build
    services.get_service.clear()
    st.cache_data.clear()


def good_provider(n=900):
    return MultiProvider({"AAPL": make_prices(n, seed=1), "^GSPC": make_prices(n, seed=2)})


def submit(at: AppTest, ticker: str, **widgets) -> AppTest:
    at.run()
    at.text_input(key="ticker_input").set_value(ticker)
    for key, value in widgets.items():
        getattr(at, value[0])(key=key).set_value(value[1])
    at.button[0].click()
    return at.run()


def texts(elements) -> str:
    return "\n".join(str(e.value) for e in elements)


def test_first_load_shows_disclaimer_and_prompt_but_no_analysis(make_app):
    at = make_app(good_provider()).run()
    assert not at.exception
    assert "not financial advice" in texts(at.warning).lower()
    assert "press **Analyse**" in texts(at.info)
    assert len(at.tabs) == 0


def test_valid_ticker_renders_all_sections(make_app):
    at = submit(make_app(good_provider()), "aapl")
    assert not at.exception and not at.error
    assert [t.label for t in at.tabs] == [
        "Summary", "Price & indicators", "Possible outcomes", "Risk", "Fundamentals",
    ]  # fmt: skip
    # the fake provider has no fundamentals: the tab says so instead of showing zeros
    assert "isn't available" in texts(at.info)
    assert "risk-free rate 4.00%" in texts(at.caption).lower()
    labels = {m.label for m in at.metric}
    assert {"Volatility (yearly)", "1M horizon", "3M horizon", "Sharpe ratio"} <= labels
    # the 1M/3M grades are shown as grades; 6M and 1Y get the plain "can't be verified" note
    grades = {m.label: m.value for m in at.metric if m.label in ("1M horizon", "3M horizon")}
    assert all(any(g in v for g in ("Good", "Fair", "Poor", "Not enough")) for v in grades.values())
    info = texts(at.info)
    assert "6M ranges can't be verified" in info and "1Y ranges can't be verified" in info
    assert "not financial advice" in texts(at.warning).lower()  # disclaimer is still visible
    md = texts(at.markdown)
    assert "Bull / Base / Bear" in md and "In plain English" in md
    assert at.session_state["request"].ticker == "AAPL"
    assert len(at.get("plotly_chart")) == 3  # price, fan chart, histogram


def test_report_is_prepared_on_request_and_then_offered_for_download(make_app):
    at = submit(make_app(good_provider()), "AAPL")
    assert not at.get("download_button")  # nothing is built until asked: it is a few MB
    at.checkbox(key="report_small").check()  # small file: keeps the test light
    at.button(key="report_prepare").click()
    at.run()
    assert not at.exception and not at.error
    buttons = at.get("download_button")
    assert len(buttons) == 1
    assert at.session_state["report"][1].startswith("stock-analyzer-AAPL-")
    html_bytes = at.session_state["report"][2]
    assert (
        html_bytes.startswith(b"<!doctype html>") and b"not financial advice" in html_bytes.lower()
    )


def test_chart_toggles_rerun_without_new_analysis(make_app):
    provider = good_provider()
    at = submit(make_app(provider), "AAPL")
    calls = provider.history_calls
    at.checkbox(key="chart_macd").check()
    at.radio(key="chart_style").set_value("line")
    at.run()
    assert not at.exception and provider.history_calls == calls


def test_invalid_ticker_shows_friendly_error_and_never_reaches_provider(make_app):
    provider = good_provider()
    at = submit(make_app(provider), "AAPL; DROP TABLE")
    assert not at.exception
    assert "doesn't look like a valid ticker" in texts(at.error)
    assert provider.history_calls == 0 and "request" not in at.session_state


def test_unknown_ticker_shows_friendly_error(make_app):
    at = submit(make_app(good_provider()), "ZZZZ")
    assert not at.exception
    assert "couldn't find price data" in texts(at.error)
    assert len(at.tabs) == 0


def test_provider_outage_shows_friendly_error_without_traceback(make_app):
    provider = FakeProvider(outcomes=[ProviderUnavailableError("boom secret")] * 20)
    at = submit(make_app(provider), "AAPL")
    assert not at.exception
    msg = texts(at.error)
    assert "temporarily unavailable" in msg and "boom secret" not in msg


def test_unexpected_exception_is_hidden_from_the_user(make_app, monkeypatch):
    at = make_app(good_provider())
    monkeypatch.setattr(
        "stock_analyzer.app.runner.run_request",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("internal path C:/secret")),
    )
    at = submit(at, "AAPL")
    assert not at.exception
    assert "Something unexpected" in texts(at.error) and "secret" not in texts(at.error)


def test_short_history_still_renders_with_warnings_and_no_grade(make_app):
    provider = MultiProvider({"AAPL": make_prices(150, seed=3), "^GSPC": make_prices(150, seed=4)})
    at = submit(make_app(provider), "AAPL")
    assert not at.exception and not at.error
    grades = [m.value for m in at.metric if m.label in ("1M horizon", "3M horizon")]
    assert grades and all("Not enough history" in v for v in grades)
    assert "less than" in texts(at.warning).lower() or "history" in texts(at.warning).lower()


def test_too_little_history_is_refused_politely(make_app):
    at = submit(make_app(MultiProvider({"AAPL": make_prices(40)})), "AAPL")
    assert not at.exception
    assert "history" in texts(at.error).lower()


def test_advanced_options_are_validated_and_applied(make_app):
    at = submit(
        make_app(good_provider()),
        "AAPL",
        method_input=("selectbox", "bootstrap"),
        seed_input=("number_input", 7),
    )
    assert not at.exception and not at.error
    req = at.session_state["request"]
    assert req.method == "bootstrap" and req.seed == 7


def test_throttle_blocks_rapid_repeats_per_session(make_app):
    at = make_app(good_provider()).run()
    at.text_input(key="ticker_input").set_value("AAPL")
    for _ in range(8):
        at.button[0].click().run()
        assert not at.exception
    assert not at.warning or "going a little fast" not in texts(at.warning)
    at.button[0].click().run()
    assert "going a little fast" in texts(at.warning)


def test_how_it_works_page_has_limitations_glossary_and_disclaimer():
    at = AppTest.from_file(HOW, default_timeout=60).run()
    assert not at.exception
    body = texts(at.markdown)
    assert "fat tails" in body and "can't be verified" in body and "Sharpe ratio" in body
    assert "not financial advice" in texts(at.warning).lower()
