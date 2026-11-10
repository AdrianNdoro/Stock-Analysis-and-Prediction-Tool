"""Phase 6: fundamentals normalisation, risk-free rate, HTML report."""

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import pytest

from stock_analyzer.analysis.fundamentals import NOT_AVAILABLE, build_snapshot
from stock_analyzer.app.report import REPORT_LIMITS, build_html_report, report_filename
from stock_analyzer.app.runner import run_request
from stock_analyzer.app.safety import parse_request
from stock_analyzer.config import Settings
from stock_analyzer.data.cache import SQLiteCache
from stock_analyzer.data.providers.yfinance_provider import (
    YFinanceProvider,
    debt_to_equity_ratio,
    dividend_yield_fraction,
)
from stock_analyzer.data.resilience import RateLimiter
from stock_analyzer.data.risk_free import (
    CACHE_KEY,
    FRESH_SECONDS,
    STALE_OK_SECONDS,
    fixed_rate,
    resolve_risk_free,
    round_rate,
    with_rate,
)
from stock_analyzer.data.service import MarketDataService
from stock_analyzer.data.types import Fundamentals
from stock_analyzer.exceptions import ConfigError, InvalidParameterError
from stock_analyzer.tests.fakes import FakeClock, make_prices
from stock_analyzer.tests.test_benchmarks_and_pipeline import MultiProvider

FIXTURE = Path(__file__).parent / "fixtures" / "yahoo_info_sample.json"
RECORDED = json.loads(FIXTURE.read_text("utf-8"))["tickers"]


class _Ticker:
    def __init__(self, info):
        self.info = info


def provider_values(ticker: str) -> dict:
    """Run a RECORDED Yahoo payload through our provider, as production would."""
    return YFinanceProvider(ticker_factory=lambda t: _Ticker(RECORDED[ticker])).get_fundamentals(
        ticker
    )


def as_fundamentals(values: dict, ticker="TEST", warnings=()) -> Fundamentals:
    return Fundamentals(ticker, values, bool(values), datetime.now(UTC), False, tuple(warnings))


# ------------------------------------------------------------- units (recorded payloads)


def test_recorded_sample_exists_and_covers_the_tricky_cases():
    assert {"AAPL", "NVDA", "BP.L", "7203.T", "TSLA", "SPY"} <= set(RECORDED)


@pytest.mark.parametrize(
    ("ticker", "expected_yield"),
    [("AAPL", 0.0031), ("NVDA", 0.0056), ("BP.L", 0.0448), ("7203.T", 0.0286), ("TSLA", 0.0),
     ("SPY", 0.01)],
)  # fmt: skip
def test_dividend_yield_is_a_fraction_for_every_recorded_payload(ticker, expected_yield):
    # Yahoo (yfinance 1.8.0) reports PERCENT: AAPL "0.31" means 0.31%, not 31%.
    assert provider_values(ticker)["dividend_yield"] == pytest.approx(expected_yield, abs=1e-9)


def test_pence_quoted_stock_is_not_fooled_by_mismatched_units():
    # BP.L: dividend in pounds, price in pence, so trailingAnnualDividendYield (0.06%) is wrong.
    info = RECORDED["BP.L"]
    assert info["trailingAnnualDividendYield"] < 0.001 < info["dividendYield"] / 100
    assert provider_values("BP.L")["dividend_yield"] == pytest.approx(0.0448)


def test_debt_to_equity_is_converted_from_percent_to_a_ratio():
    assert provider_values("AAPL")["debt_to_equity"] == pytest.approx(0.84286)
    assert provider_values("TSLA")["debt_to_equity"] == pytest.approx(0.18295)


def test_missing_fields_stay_none_never_zero():
    v = provider_values("SPY")  # an ETF: no debt, growth or market cap in this payload
    assert v["debt_to_equity"] is None and v["revenue_growth"] is None and v["market_cap"] is None
    assert v["quote_type"] == "ETF"


def test_financial_currency_is_kept_separate_from_trading_currency():
    v = provider_values("BP.L")
    assert v["currency"] == "GBp" and v["financial_currency"] == "USD"


@pytest.mark.parametrize(
    ("info", "expected"),
    [
        (
            {"dividendYield": 0.0031, "trailingAnnualDividendYield": 0.00302},
            0.0031,
        ),  # old: fraction
        ({"dividendYield": 0.31, "trailingAnnualDividendYield": 0.00302}, 0.0031),  # new: percent
        ({"dividendYield": 0.31}, 0.0031),  # nothing to cross-check: assume percent
        ({"dividendYield": 45}, None),  # 45% (as percent 0.45 > 30%): data error
        ({"dividendYield": -1}, None),
        ({"dividendYield": "abc"}, None),
        ({"dividendYield": True}, None),
        ({"dividendYield": math.nan}, None),
        ({"dividendYield": math.inf}, None),
        ({}, None),
    ],
)
def test_dividend_yield_edge_cases(info, expected):
    out = dividend_yield_fraction(info)
    assert out == expected if expected is None else out == pytest.approx(expected)


def test_debt_to_equity_edge_cases():
    assert debt_to_equity_ratio({"debtToEquity": 150}) == pytest.approx(1.5)
    assert debt_to_equity_ratio({"debtToEquity": -5}) is None
    assert debt_to_equity_ratio({"debtToEquity": "x"}) is None
    assert debt_to_equity_ratio({}) is None


# ------------------------------------------------------------------------------ snapshot


def rows(snap):
    return {r.label: r for r in snap.rows}


def test_snapshot_formats_recorded_aapl_payload():
    snap = build_snapshot(as_fundamentals(provider_values("AAPL"), "AAPL"))
    r = rows(snap)
    assert snap.available and not snap.missing
    assert r["Dividend yield"].value == "0.31%"
    assert r["Debt / equity"].value == "0.84x"
    assert r["Market cap"].value == "5.06 trillion USD"
    assert r["P/E (last 12 months)"].value == "38.8x"
    assert r["Revenue growth"].value == "+12%"
    assert snap.sector == "Technology"
    assert any("sector averages" in n for n in snap.notes)  # we say what we do NOT show


def test_snapshot_warns_when_eps_currency_differs_and_uses_major_units():
    snap = build_snapshot(as_fundamentals(provider_values("BP.L"), "BP.L"))
    r = rows(snap)
    assert "USD" in r["Earnings per share"].note and "GBP" in r["Earnings per share"].note
    assert r["Market cap"].value.endswith("GBP")  # pounds, not pence
    assert r["Earnings growth"].note and "100%" in r["Earnings growth"].note  # +305%: small base
    assert r["Dividend yield"].value == "4.48%"


def test_snapshot_shows_no_dividend_as_text_and_unknown_as_not_available():
    assert rows(build_snapshot(as_fundamentals(provider_values("TSLA"))))[
        "Dividend yield"
    ].value == ("None reported")
    spy = build_snapshot(as_fundamentals(provider_values("SPY"), "SPY"))
    r = rows(spy)
    assert r["Debt / equity"].value == NOT_AVAILABLE and not r["Debt / equity"].available
    assert "Debt / equity" in spy.missing
    assert not any(row.value in {"0", "0.0", "0.00x", "0.0x", "0%"} for row in spy.rows)
    assert any("fund" in n for n in spy.notes)  # ETF caveat


def test_snapshot_negative_or_extreme_pe_is_not_shown_as_a_number():
    snap = build_snapshot(as_fundamentals({"pe_ratio": -12.0, "eps_ttm": -1.0, "currency": "USD"}))
    assert rows(snap)["P/E (last 12 months)"].value == NOT_AVAILABLE
    assert rows(snap)["P/E (last 12 months)"].note
    snap = build_snapshot(as_fundamentals({"pe_ratio": None, "eps_ttm": -1.0, "currency": "USD"}))
    assert "not positive" in rows(snap)["P/E (last 12 months)"].note


def test_snapshot_flags_debt_ratio_for_banks():
    snap = build_snapshot(as_fundamentals({"debt_to_equity": 2.1, "sector": "Financial Services"}))
    assert "banks" in rows(snap)["Debt / equity"].note


def test_snapshot_with_no_data_degrades_gracefully():
    snap = build_snapshot(
        as_fundamentals({}, warnings=("Fundamental data is currently unavailable.",))
    )
    assert not snap.available and all(not r.available for r in snap.rows)
    assert any("not available" in n.lower() or "unavailable" in n.lower() for n in snap.notes)


# --------------------------------------------------------------------------- risk-free rate


@pytest.fixture
def live_settings(tmp_path):
    return Settings(
        cache_path=tmp_path / "rf.sqlite3", risk_free_source="live", risk_free_rate=0.04
    )


def make_cache(settings, clock):
    return SQLiteCache(settings.cache_path, clock=clock)


def test_fixed_source_never_touches_the_network_or_cache(tmp_path):
    s = Settings(cache_path=tmp_path / "x.sqlite3", risk_free_rate=0.031)

    def boom():
        raise AssertionError("must not fetch")

    r = resolve_risk_free(s, None, fetcher=boom)
    assert (r.value, r.source) == (0.031, "fixed") and "fixed setting" in r.label


def test_live_rate_is_rounded_labelled_and_cached(live_settings):
    now = [1_000.0]
    cache = make_cache(live_settings, lambda: now[0])
    calls = []

    def fetch():
        calls.append(1)
        return 4.318, "2026-11-09"

    r1 = resolve_risk_free(live_settings, cache, fetcher=fetch, clock=lambda: now[0])
    assert r1.value == 0.0425 and r1.source == "live" and r1.as_of == "2026-11-09"
    assert "US 13-week Treasury bill" in r1.label and "approximation" in r1.label
    now[0] += FRESH_SECONDS - 10
    r2 = resolve_risk_free(live_settings, cache, fetcher=fetch, clock=lambda: now[0])
    assert r2.value == 0.0425 and len(calls) == 1  # served from cache
    now[0] += 20  # now stale: refetch
    resolve_risk_free(live_settings, cache, fetcher=fetch, clock=lambda: now[0])
    assert len(calls) == 2


def test_live_failure_falls_back_to_last_known_then_to_the_fixed_rate(live_settings):
    now = [1_000.0]
    cache = make_cache(live_settings, lambda: now[0])
    ok = lambda: (4.3, "2026-11-09")  # noqa: E731

    def down():
        raise TimeoutError("no network")

    first_fail = resolve_risk_free(live_settings, cache, fetcher=down, clock=lambda: now[0])
    assert (first_fail.value, first_fail.source) == (0.04, "fallback")
    assert "could not be fetched" in first_fail.label
    resolve_risk_free(live_settings, cache, fetcher=ok, clock=lambda: now[0])
    now[0] += FRESH_SECONDS + 1
    last = resolve_risk_free(live_settings, cache, fetcher=down, clock=lambda: now[0])
    assert (last.value, last.source) == (0.0425, "last_known") and "Last known" in last.label
    now[0] += STALE_OK_SECONDS
    too_old = resolve_risk_free(live_settings, cache, fetcher=down, clock=lambda: now[0])
    assert too_old.source == "fallback"


@pytest.mark.parametrize("bad", [250.0, -3.0, math.nan, math.inf])
def test_implausible_live_values_are_rejected(live_settings, bad):
    r = resolve_risk_free(live_settings, None, fetcher=lambda: (bad, "2026-11-09"))
    assert r.source == "fallback" and r.value == 0.04


def test_corrupt_cache_entry_is_ignored(live_settings):
    cache = make_cache(live_settings, lambda: 1_000.0)
    cache.set(CACHE_KEY, "{not json")
    r = resolve_risk_free(live_settings, cache, fetcher=lambda: (4.0, "2026-11-09"),
                          clock=lambda: 1_000.0)  # fmt: skip
    assert r.source == "live" and r.value == 0.04


def test_rate_is_rounded_to_quarter_percent_and_applied_to_settings(live_settings):
    assert round_rate(0.04124) == 0.04 and round_rate(0.0413) == 0.0425
    r = fixed_rate(Settings(risk_free_rate=0.0275))
    assert with_rate(live_settings, r).risk_free_rate == 0.0275
    assert live_settings.risk_free_rate == 0.04  # the original is not mutated


def test_operator_can_lower_the_path_ceiling_but_never_raise_it():
    assert parse_request("AAPL", "3M", paths=20_000, max_paths=20_000).paths == 20_000
    with pytest.raises(InvalidParameterError) as exc:
        parse_request("AAPL", "3M", paths=30_000, max_paths=20_000)
    assert "20,000" in exc.value.user_message
    with pytest.raises(InvalidParameterError):  # the public hard limit still applies
        parse_request("AAPL", "3M", paths=50_001, max_paths=10**9)
    assert parse_request("AAPL", "3M", paths=10_000, max_paths=1).paths == 10_000  # floor


def test_config_parses_app_max_paths():
    assert Settings.from_env({}).app_max_paths == 50_000  # default behaviour unchanged
    assert Settings.from_env({"APP_MAX_PATHS": "20000"}).app_max_paths == 20_000
    for bad in ("5000", "60000", "lots"):
        with pytest.raises(ConfigError):
            Settings.from_env({"APP_MAX_PATHS": bad})


def test_config_parses_risk_free_source():
    assert Settings.from_env({}).risk_free_source == "fixed"  # default behaviour unchanged
    assert Settings.from_env({"RISK_FREE_SOURCE": "LIVE"}).risk_free_source == "live"
    with pytest.raises(ConfigError):
        Settings.from_env({"RISK_FREE_SOURCE": "sometimes"})


# -------------------------------------------------------------------------------- report


@pytest.fixture(scope="module")
def app_result(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("report")
    settings = Settings(cache_path=tmp / "r.sqlite3", backoff_base_seconds=0.0)
    clock = FakeClock()
    provider = MultiProvider({"AAPL": make_prices(900, seed=1), "^GSPC": make_prices(900, seed=2)})
    service = MarketDataService(
        provider, SQLiteCache(settings.cache_path, clock=clock), settings,
        limiter=RateLimiter(10_000, 10_000), clock=clock, sleep=lambda s: None,
    )  # fmt: skip
    req = parse_request("AAPL", "3M", "gbm", "shrunk", 10_000, 5, None)
    return run_request(service, settings, req, "2026-11-10", fixed_rate(settings))


NOW = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)


def test_report_is_self_contained_and_has_every_required_section(app_result):
    page = build_html_report(app_result, generated_at=NOW)
    assert page.startswith("<!doctype html>") and "</html>" in page
    assert len(page) > 1_000_000  # Plotly itself is embedded: works offline
    # (the Plotly bundle itself mentions cdn.plot.ly in strings; what matters is no external load)
    assert "<script src=" not in page and "<link " not in page and 'src="http' not in page
    low = page.lower()
    for needle in (
        "not financial advice", "past performance does not guarantee",  # disclaimer
        "how reliable have this model", "can&#x27;t be verified",  # reliability + 6M/1Y note
        "bull / base / bear", "percentiles", "assumptions used", "limitations",
        "fundamentals", "risk-free rate", "fat tails",
        "generated 2026-11-10 12:00 utc", "stock analyzer report: aapl",
    ):  # fmt: skip
        assert needle in low, needle
    assert page.count("plotly-graph-div") >= 3  # price, fan chart, histogram
    assert all(x.lower().split()[0] in low for x in REPORT_LIMITS[:3])


def test_report_without_fundamentals_says_so_and_never_invents_numbers(app_result):
    page = build_html_report(app_result, generated_at=NOW, inline_js=False)
    assert "Fundamental data was not available" in page or "not available" in page.lower()
    assert "0.00x" not in page


def test_small_report_loads_plotly_from_cdn_instead_of_embedding_it(app_result):
    small = build_html_report(app_result, generated_at=NOW, inline_js=False)
    assert len(small) < 600_000 and "cdn.plot.ly" in small


def test_report_escapes_anything_that_could_carry_markup(app_result):
    evil = build_snapshot(
        as_fundamentals(
            {
                "pe_ratio": 10.0,  # a usable value, so the report prints the sector line
                "sector": "<script>alert(1)</script>",
                "industry": "<img src=x onerror=y>",
            },
            warnings=("<b>warn</b>",),
        )
    )
    page = build_html_report(app_result, generated_at=NOW, inline_js=False, fundamentals=evil)
    assert "<script>alert(1)</script>" not in page and "<img src=x" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page and "<b>warn</b>" not in page


def test_report_filename_is_safe(app_result):
    name = report_filename(app_result, NOW)
    assert name == "stock-analyzer-AAPL-20261110.html"
    assert "/" not in name and "\\" not in name and ".." not in name
