"""Pure chart builders: data in, Plotly figure out. No Streamlit, no I/O, no hidden state.

Figures have no fixed width, so they shrink to a phone screen; the legend sits under the plot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stock_analyzer.models.monte_carlo import SimulationResult

UP, DOWN, NEUTRAL, BAND = "#2e9e6b", "#d6455d", "#6b7280", "46, 120, 200"
MAX_SAMPLE_PATHS_SHOWN = 25


@dataclass(frozen=True)
class PriceChartOptions:
    style: str = "candlestick"  # "candlestick" | "line"
    sma: tuple[int, ...] = (50, 200)
    ema: tuple[int, ...] = ()
    bollinger: bool = False
    volume: bool = True
    rsi: bool = True
    macd: bool = False
    days: int = 504  # trading days shown (indicators are still computed on the full history)
    levels: tuple[float, ...] = field(default_factory=tuple)  # support/resistance prices


def _layout(fig: go.Figure, height: int, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        title=title,
        margin={"l": 8, "r": 8, "t": 36 if title else 8, "b": 8},
        legend={"orientation": "h", "yanchor": "top", "y": -0.12, "x": 0},
        hovermode="x unified",
    )
    return fig


def price_chart(
    ohlc: pd.DataFrame, ind: pd.DataFrame, options: PriceChartOptions, currency: str | None = None
) -> go.Figure:
    """Price with optional moving averages, Bollinger Bands, volume, RSI and MACD panels."""
    view = ohlc.iloc[-options.days :]
    iv = ind.reindex(view.index)
    panels = ["price"]
    if options.volume:
        panels.append("volume")
    if options.rsi:
        panels.append("rsi")
    if options.macd:
        panels.append("macd")
    heights = {"price": 0.55, "volume": 0.15, "rsi": 0.15, "macd": 0.15}
    weights = [heights[p] for p in panels]
    fig = make_subplots(
        rows=len(panels),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[w / sum(weights) for w in weights],
    )
    row = {name: i + 1 for i, name in enumerate(panels)}

    if options.style == "candlestick":
        fig.add_trace(
            go.Candlestick(
                x=view.index,
                open=view["Open"],
                high=view["High"],
                low=view["Low"],
                close=view["Close"],
                name="Price",
                increasing_line_color=UP,
                decreasing_line_color=DOWN,
            ),
            row=1,
            col=1,
        )
        fig.update_xaxes(rangeslider_visible=False)
    else:
        fig.add_trace(
            go.Scatter(x=view.index, y=view["Close"], name="Price", line={"width": 1.6}),
            row=1,
            col=1,
        )
    for n in options.sma:
        fig.add_trace(
            go.Scatter(x=view.index, y=iv[f"sma_{n}"], name=f"SMA {n}", line={"width": 1.1}),
            row=1,
            col=1,
        )
    for n in options.ema:
        fig.add_trace(
            go.Scatter(
                x=view.index, y=iv[f"ema_{n}"], name=f"EMA {n}", line={"width": 1.1, "dash": "dot"}
            ),
            row=1,
            col=1,
        )
    if options.bollinger:
        for col, label in (("bb_upper", "Bollinger upper"), ("bb_lower", "Bollinger lower")):
            fig.add_trace(
                go.Scatter(
                    x=view.index,
                    y=iv[col],
                    name=label,
                    line={"width": 0.8, "color": NEUTRAL, "dash": "dash"},
                ),
                row=1,
                col=1,
            )
    for price in options.levels:
        fig.add_hline(y=price, line={"width": 0.7, "dash": "dot", "color": NEUTRAL}, row=1, col=1)
    if "volume" in row:
        colours = np.where(view["Close"] >= view["Open"], UP, DOWN)
        fig.add_trace(
            go.Bar(
                x=view.index,
                y=view["Volume"],
                name="Volume",
                marker_color=colours,
                showlegend=False,
            ),
            row=row["volume"],
            col=1,
        )
        fig.update_yaxes(title_text="Volume", row=row["volume"], col=1)
    if "rsi" in row:
        fig.add_trace(
            go.Scatter(x=view.index, y=iv["rsi_14"], name="RSI 14", line={"width": 1.1}),
            row=row["rsi"],
            col=1,
        )
        for level in (30, 70):
            fig.add_hline(
                y=level, line={"width": 0.6, "dash": "dot", "color": NEUTRAL}, row=row["rsi"], col=1
            )
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=row["rsi"], col=1)
    if "macd" in row:
        fig.add_trace(
            go.Bar(
                x=view.index,
                y=iv["macd_hist"],
                name="MACD histogram",
                marker_color=NEUTRAL,
                opacity=0.5,
            ),
            row=row["macd"],
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=view.index, y=iv["macd"], name="MACD", line={"width": 1.1}),
            row=row["macd"],
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=view.index, y=iv["macd_signal"], name="Signal", line={"width": 1.1}),
            row=row["macd"],
            col=1,
        )
        fig.update_yaxes(title_text="MACD", row=row["macd"], col=1)
    fig.update_yaxes(title_text=f"Price ({currency})" if currency else "Price", row=1, col=1)
    return _layout(fig, 380 + 130 * (len(panels) - 1))


def fan_chart(
    sim: SimulationResult,
    history: pd.Series,
    as_of: pd.Timestamp,
    currency: str | None = None,
    history_days: int = 126,
    show_sample_paths: bool = True,
) -> go.Figure:
    """Recent history, then the simulated 5-95% and 25-75% bands, median and sample paths."""
    days = sim.bands.index.to_numpy()
    future = pd.bdate_range(start=as_of, periods=len(days))
    b = sim.bands
    hist = history.iloc[-history_days:]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist.index, y=hist.values, name="Past price", line={"width": 1.6, "color": NEUTRAL}
        )
    )
    if show_sample_paths:
        for i in range(min(MAX_SAMPLE_PATHS_SHOWN, len(sim.sample_paths))):
            fig.add_trace(
                go.Scatter(
                    x=future,
                    y=sim.sample_paths[i],
                    mode="lines",
                    showlegend=i == 0,
                    name="Example simulated paths",
                    hoverinfo="skip",
                    line={"width": 0.6, "color": f"rgba({BAND}, 0.25)"},
                    legendgroup="s",
                )
            )
    for lo, hi, alpha, name in (
        ("p5", "p95", 0.15, "5th-95th percentile"),
        ("p25", "p75", 0.30, "25th-75th percentile"),
    ):
        fig.add_trace(
            go.Scatter(x=future, y=b[hi], line={"width": 0}, showlegend=False, hoverinfo="skip")
        )
        fig.add_trace(
            go.Scatter(
                x=future,
                y=b[lo],
                fill="tonexty",
                name=name,
                line={"width": 0},
                fillcolor=f"rgba({BAND}, {alpha})",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=future, y=b["p50"], name="Median outcome", line={"width": 2, "color": f"rgb({BAND})"}
        )
    )
    fig.update_yaxes(title_text=f"Price ({currency})" if currency else "Price")
    return _layout(fig, 420)


def terminal_histogram(
    sim: SimulationResult, currency: str | None = None, bins: int = 60
) -> go.Figure:
    """Distribution of simulated end prices, with today's price and the 5/50/95 marks."""
    fig = go.Figure(
        go.Histogram(
            x=sim.terminal,
            nbinsx=bins,
            name="Simulated end prices",
            marker_color=f"rgba({BAND}, 0.6)",
            showlegend=False,
        )
    )
    marks = [
        ("Today", sim.s0, NEUTRAL),
        ("5th", sim.percentile(5), DOWN),
        ("Median", sim.percentile(50), f"rgb({BAND})"),
        ("95th", sim.percentile(95), UP),
    ]
    for label, x, colour in marks:
        fig.add_vline(
            x=x,
            line={"width": 1.5, "dash": "dash", "color": colour},
            annotation_text=label,
            annotation_position="top",
        )
    fig.update_xaxes(title_text=f"Simulated price at the end ({currency})" if currency else "")
    fig.update_yaxes(title_text="Number of simulated paths")
    fig.update_layout(bargap=0.02)
    _layout(fig, 360)
    fig.update_layout(margin={"l": 8, "r": 8, "t": 48, "b": 8})  # room for the top labels
    return fig
