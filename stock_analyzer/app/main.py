"""Stock Analyzer dashboard.  Run:  python -m streamlit run stock_analyzer/app/main.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # make `stock_analyzer` importable

import logging  # noqa: E402

import streamlit as st  # noqa: E402

from stock_analyzer.analysis.benchmarks import BENCHMARKS  # noqa: E402
from stock_analyzer.app import glossary as g  # noqa: E402
from stock_analyzer.app import services, views  # noqa: E402
from stock_analyzer.app.layout import page_chrome  # noqa: E402
from stock_analyzer.app.presenters import friendly_error  # noqa: E402
from stock_analyzer.app.runner import AppResult, last_data_date, run_request  # noqa: E402
from stock_analyzer.app.safety import (  # noqa: E402
    AUTO_BENCHMARK,
    AnalysisRequest,
    RequestThrottle,
    parse_request,
)
from stock_analyzer.data.risk_free import with_rate  # noqa: E402
from stock_analyzer.exceptions import StockAnalyzerError  # noqa: E402
from stock_analyzer.models.monte_carlo import (  # noqa: E402
    DRIFT_MODES,
    HORIZONS,
    METHODS,
    MIN_PATHS,
)

log = logging.getLogger("stock_analyzer.app")


@st.cache_data(show_spinner=False, max_entries=32, ttl=3600)
def cached_run(
    _service,
    _settings,
    ticker: str,
    horizon: str,
    method: str,
    drift: str,
    paths: int,
    seed: int | None,
    benchmark: str | None,
    data_date: str,  # in the key: a new price bar means a fresh analysis
    rf_value: float,  # in the key: a different risk-free rate means a fresh analysis
    _rate,
) -> AppResult:
    req = AnalysisRequest(ticker, horizon, method, drift, paths, seed, benchmark)
    return run_request(_service, _settings, req, data_date, _rate)


def _throttle() -> RequestThrottle:
    """Per-session throttle: its state lives in this session's ``st.session_state`` only."""
    stamps = st.session_state.setdefault("request_stamps", [])
    return RequestThrottle(stamps)


page_chrome("Stock Analyzer")
st.title("Stock Analyzer")
st.caption("Describe a stock's past and simulate a range of possible futures. Not a prediction.")

service = services.get_service()
rate = services.get_risk_free()
settings = with_rate(services.get_app_settings(), rate)  # every model reads settings.risk_free_rate

with st.form("request_form"):
    ticker = st.text_input(
        "Ticker",
        value="AAPL",
        max_chars=16,
        key="ticker_input",
        help="For example AAPL, BRK-B, BP.L (London), SAP.DE (Germany), 7203.T (Tokyo).",
    )
    horizon = st.radio(
        "Time horizon", list(HORIZONS), index=2, horizontal=True, key="horizon_input"
    )
    benchmark = st.selectbox(
        "Benchmark (for beta)",
        [AUTO_BENCHMARK, *BENCHMARKS],
        key="benchmark_input",
        format_func=lambda k: (
            "Auto (suggested for this stock's market)"
            if k == AUTO_BENCHMARK
            else BENCHMARKS[k].name
        ),
    )
    with st.expander("Advanced"):
        method = st.selectbox(
            "Simulation method",
            list(METHODS),
            key="method_input",
            help=f"{g.tip('gbm')} {g.tip('bootstrap')}",
        )
        drift = st.selectbox(
            "Expected return assumption", list(DRIFT_MODES), key="drift_input", help=g.tip("drift")
        )
        paths = st.number_input(
            "Number of simulations",
            min_value=MIN_PATHS,
            max_value=settings.app_max_paths,
            value=min(int(settings.mc_paths), settings.app_max_paths),
            step=5000,
            key="paths_input",
        )
        seed = st.number_input(
            "Random seed (optional, makes a run repeatable)",
            min_value=0,
            max_value=2**31 - 1,
            value=None,
            step=1,
            key="seed_input",
        )
    submitted = st.form_submit_button("Analyse", type="primary")

if submitted:
    try:
        req = parse_request(
            ticker,
            horizon,
            method,
            drift,
            int(paths),
            None if seed is None else int(seed),
            benchmark,
            settings.app_max_paths,
        )
        wait = _throttle().try_acquire()
        if wait:
            st.session_state.pop("request", None)
            st.warning(f"You're going a little fast. Please wait about {wait:.0f} seconds.")
        else:
            st.session_state["request"] = req
    except StockAnalyzerError as exc:
        st.session_state.pop("request", None)
        st.error(exc.user_message)

req = st.session_state.get("request")
if req is None:
    st.info(
        "Enter a ticker and press **Analyse**. New here? Read "
        "**How this works** in the sidebar first, then hover over the ⓘ icons for plain-English "
        "explanations of any term."
    )
    st.stop()

try:
    with st.spinner(f"Analysing {req.ticker}... the first run for a stock takes a little longer."):
        data_date = last_data_date(service, req.ticker)
        res = cached_run(
            service,
            settings,
            req.ticker,
            req.horizon,
            req.method,
            req.drift,
            req.paths,
            req.seed,
            req.benchmark,
            data_date,
            rate.value,
            rate,
        )
except Exception as exc:  # noqa: BLE001 - never show a stack trace to a user
    if not isinstance(exc, StockAnalyzerError):
        log.exception("unexpected failure", extra={"ticker": req.ticker})
    st.error(friendly_error(exc))
    st.stop()

tab_summary, tab_price, tab_forecast, tab_risk, tab_fund = st.tabs(
    ["Summary", "Price & indicators", "Possible outcomes", "Risk", "Fundamentals"]
)
with tab_summary:
    views.render_summary(res)
with tab_price:
    views.render_price(res)
with tab_forecast:
    views.render_forecast(res)
with tab_risk:
    views.render_risk(res)
with tab_fund:
    views.render_fundamentals(res)
views.render_export(res)
