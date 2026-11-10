"""Streamlit rendering of one AppResult. Layout only: all numbers and text come from the
existing analysis/model code and ``presenters`` (nothing is re-implemented here)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from stock_analyzer.app import glossary as g
from stock_analyzer.app.charts import PriceChartOptions, fan_chart, price_chart, terminal_histogram
from stock_analyzer.app.presenters import reliability_views, risk_rows, scenario_table
from stock_analyzer.app.runner import AppResult


def render_header(res: AppResult) -> None:
    a = res.forecast.analysis
    cur = a.currency or "currency unknown"
    st.subheader(f"{a.ticker}  ·  {a.last_price:,.2f} {cur}")
    st.caption(
        f"Prices as of {a.as_of}. All prices are in {cur}. "
        + (f"Benchmark: {a.benchmark.name}." if a.benchmark else "")
    )
    for w in a.warnings:
        st.warning(w)


def render_reliability(res: AppResult, key: str) -> None:
    """1M and 3M grades up front; 6M and 1Y get a plain 'can't be verified' note."""
    st.markdown("#### How reliable have this model's ranges been?", help=g.tip("reliability"))
    views = reliability_views(res.forecast.reliability)
    graded = [v for v in views if v.horizon in ("1M", "3M")]
    cols = st.columns(len(graded))
    for col, v in zip(cols, graded, strict=True):
        with col:
            st.metric(f"{v.horizon} horizon", f"{v.icon} {v.title.split(': ', 1)[1]}")
    for v in graded:
        st.markdown(f"**{v.horizon}.** {v.text}")
    for v in views:
        if v.horizon not in ("1M", "3M"):
            st.info(v.text, icon="ℹ️")
    with st.expander("How was this checked?", expanded=False):
        st.write(
            "We replayed this stock's history: at many past dates the model forecast ranges "
            "using only data available then, and we counted how often the real price landed "
            "inside. Past accuracy is not a promise about the future."
        )
        for v in views:
            for line in v.details:
                st.caption(f"{v.horizon}: {line}")


def render_summary(res: AppResult) -> None:
    f = res.forecast
    a = f.analysis
    sim = f.simulations[res.request.horizon]
    render_header(res)
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Volatility (yearly)", f"{a.risk.annualised_volatility:.0%}", help=g.tip("volatility")
    )
    c2.metric("Max drawdown", f"{a.risk.drawdown.max_drawdown:.0%}", help=g.tip("drawdown"))
    c3.metric("Chart picture", a.technical.trend_label, help=g.tip("sma_ema"))
    render_reliability(res, "summary")
    st.markdown(f"#### In plain English ({res.request.horizon} horizon)")
    for line in sim.plain_english(a.currency):
        st.markdown(f"- {line}")
    st.markdown("#### About the stock's recent behaviour")
    for line in a.technical.bullets + a.risk_bullets:
        st.markdown(f"- {line}")


def render_price(res: AppResult) -> None:
    a = res.forecast.analysis
    st.markdown("Chart options")
    c1, c2 = st.columns(2)
    style = c1.radio("Style", ["candlestick", "line"], horizontal=True, key="chart_style")
    months = c2.select_slider(
        "Period shown",
        [3, 6, 12, 24, 60],
        value=24,
        key="chart_months",
        format_func=lambda m: f"{m} months",
    )
    sma = st.multiselect(
        "Simple moving averages",
        [20, 50, 200],
        default=[50, 200],
        key="chart_sma",
        help=g.tip("sma_ema"),
    )
    ema = st.multiselect(
        "Exponential moving averages",
        [20, 50, 200],
        default=[],
        key="chart_ema",
        help=g.tip("sma_ema"),
    )
    d1, d2, d3, d4 = st.columns(4)
    boll = d1.checkbox("Bollinger", key="chart_boll", help=g.tip("bollinger"))
    vol = d2.checkbox("Volume", value=True, key="chart_vol")
    rsi = d3.checkbox("RSI", value=True, key="chart_rsi", help=g.tip("rsi"))
    macd = d4.checkbox("MACD", key="chart_macd", help=g.tip("macd"))
    opts = PriceChartOptions(
        style=style,
        sma=tuple(sma),
        ema=tuple(ema),
        bollinger=boll,
        volume=vol,
        rsi=rsi,
        macd=macd,
        days=int(months * 21),
        levels=tuple(lv.price for lv in a.technical.levels),
    )
    st.plotly_chart(price_chart(res.ohlc, res.indicators, opts, a.currency), width="stretch")
    st.caption(
        "Prices are adjusted for splits and dividends. Dotted horizontal lines mark past "
        "support and resistance. These are descriptions of the past, not forecasts."
    )
    if a.technical.levels:
        st.markdown("**Support and resistance**", help=g.tip("support_resistance"))
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Type": lv.kind.title(),
                        "Price": f"{lv.price:,.2f} {a.currency or ''}".strip(),
                        "From today": f"{lv.distance_pct:+.1%}",
                        "Touches": lv.touches,
                    }
                    for lv in a.technical.levels
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    for b in a.technical.bullets:
        st.markdown(f"- {b}")


def render_forecast(res: AppResult) -> None:
    f = res.forecast
    a = f.analysis
    h = res.request.horizon
    sim, ss = f.simulations[h], f.scenarios[h]
    render_reliability(res, "forecast")
    st.markdown(f"#### Possible price range over {h}")
    st.plotly_chart(fan_chart(sim, a.adj_close, pd.Timestamp(a.as_of), a.currency), width="stretch")
    st.caption(
        "The shaded bands hold 50% and 90% of the simulated paths; thin lines are examples. "
        "This is a spread of possibilities under stated assumptions, not a prediction.",
        help=g.tip("percentile"),
    )
    st.markdown("#### Where the simulated paths ended")
    st.plotly_chart(terminal_histogram(sim, a.currency), width="stretch")
    st.caption("Each bar counts simulated paths that finished at that price.")
    st.markdown("#### Bull / Base / Bear scenarios", help=g.tip("scenario"))
    st.dataframe(scenario_table(ss, a.currency), hide_index=True, width="stretch")
    for note in ss.notes:
        st.caption(note)
    with st.expander("Assumptions and chart context for each scenario"):
        for sc in ss.scenarios:
            st.markdown(f"**{sc.name}**")
            for line in sc.assumptions:
                st.markdown(f"- {line}")
            st.markdown(f"- {sc.indicator_context}")
    st.markdown("#### Percentiles")
    table = sim.summary_table().reset_index()
    table["price"] = table["price"].map(lambda x: f"{x:,.2f} {a.currency or ''}".strip())
    table["return"] = table["return"].map(lambda x: f"{x:+.1%}")
    st.dataframe(
        table.rename(columns={"percentile": "Percentile", "price": "Price", "return": "Change"}),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        f"Method: {sim.method.upper()}, {sim.n_paths:,} simulated paths. "
        f"Sampling error on the probabilities is about ±{sim.monte_carlo_error(0.5):.1%}; "
        "model error is much larger."
    )


def render_risk(res: AppResult) -> None:
    a = res.forecast.analysis
    r = a.risk
    st.caption(f"Based on {r.n_returns} daily returns from {r.window_start} to {r.window_end}.")
    for label, value, key in risk_rows(r, a.benchmark):
        st.metric(label, value, help=g.tip(key))
    st.markdown("#### In plain English")
    for line in a.risk_bullets:
        st.markdown(f"- {line}")
