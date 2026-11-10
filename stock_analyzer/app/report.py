"""Self-contained HTML report for one analysis. Pure: no Streamlit, no I/O, no network.

One file with its CSS inline and the charts embedded as interactive Plotly figures (the
Plotly library itself is embedded once, about 4-5 MB, so the file works offline; pass
``inline_js=False`` for a small file that loads Plotly from a CDN instead). Everything that
comes from data or users is HTML-escaped. To get a PDF, open the file in a browser and use
Print > Save as PDF (the stylesheet has print rules); we do not ship a PDF engine because the
usual ones need system libraries or a headless browser, which breaks on plain Windows installs.
"""

from __future__ import annotations

import html
from datetime import datetime

import pandas as pd

from stock_analyzer.analysis.fundamentals import FundamentalsSnapshot
from stock_analyzer.app.charts import PriceChartOptions, fan_chart, price_chart, terminal_histogram
from stock_analyzer.app.limitations import DISCLAIMER
from stock_analyzer.app.presenters import reliability_views, risk_rows, scenario_table
from stock_analyzer.app.runner import AppResult
from stock_analyzer.backtest.report import MODEL_VERSION

REPORT_LIMITS = (
    "Simulations are not predictions. They show a spread of outcomes under stated assumptions.",
    "Real returns have fat tails: extreme days happen more often than the bell-curve model "
    "assumes, so the ranges can be too narrow in a crash.",
    "Volatility clusters: calm and turbulent periods come in streaks, which this model only "
    "partly captures.",
    "Regimes change (rates, wars, new regulation, a company's own story). History from a "
    "different regime may mislead.",
    "Thinly traded stocks can move far more than their history suggests, and you may not be "
    "able to trade at the quoted price.",
    "Data comes from a third-party provider and may contain errors or gaps.",
    "The past-accuracy check covers one stock over about ten years. A good grade means the "
    "ranges were about the right width in the past, not that the next price will land in them.",
    "6M and 1Y ranges can't be verified: ten years of history holds too few independent checks.",
)

_CSS = """
:root{--bg:#fff;--fg:#1f2430;--muted:#5d6678;--line:#d9dde5;--card:#f5f7fa;--warn:#fff4d6;
--warnline:#e0b64a;--accent:#2e78c8}
@media (prefers-color-scheme:dark){:root{--bg:#14171d;--fg:#e6e9ef;--muted:#9aa3b5;
--line:#2c323d;--card:#1c2029;--warn:#3a3015;--warnline:#8a6d1f;--accent:#6aa7e8}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,Segoe UI,sans-serif}
main{max-width:920px;margin:0 auto;padding:16px}
h1{font-size:1.7rem;margin:.2em 0}h2{font-size:1.25rem;margin:1.8em 0 .5em;
border-bottom:1px solid var(--line);padding-bottom:.2em}h3{font-size:1.05rem}
.muted{color:var(--muted);font-size:.9rem}
.disclaimer{background:var(--warn);border:1px solid var(--warnline);border-radius:8px;
padding:10px 14px;margin:12px 0}
.grades{display:flex;gap:12px;flex-wrap:wrap}.grade{flex:1 1 200px;background:var(--card);
border:1px solid var(--line);border-radius:8px;padding:10px 14px}
.grade b{font-size:1.4rem}
.note{background:var(--card);border-left:4px solid var(--accent);padding:8px 12px;
margin:8px 0;border-radius:4px}
table{border-collapse:collapse;width:100%;font-size:.92rem;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:6px 9px;text-align:left}th{background:var(--card)}
.chart{margin:8px 0;max-width:100%;overflow-x:auto}
ul{padding-left:1.2em}
@media print{.chart{break-inside:avoid}h2{break-after:avoid}main{max-width:none}}
"""


def _e(x: object) -> str:
    return html.escape(str(x), quote=True)


def _bullets(lines: list[str] | tuple[str, ...]) -> str:
    return "<ul>" + "".join(f"<li>{_e(x)}</li>" for x in lines) + "</ul>" if lines else ""


def _table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, escape=True, border=0)


def _kv_table(rows: list[tuple[str, str]]) -> str:
    body = "".join(f"<tr><th>{_e(k)}</th><td>{_e(v)}</td></tr>" for k, v in rows)
    return f"<table>{body}</table>"


def _fig(fig, include_js: bool | str) -> str:
    inner = fig.to_html(
        full_html=False,
        include_plotlyjs=include_js,
        config={"displaylogo": False, "responsive": True},
    )
    return f'<div class="chart">{inner}</div>'


def build_html_report(
    res: AppResult,
    *,
    generated_at: datetime,
    inline_js: bool = True,
    fundamentals: FundamentalsSnapshot | None = None,
) -> str:
    """The whole report as one HTML string."""
    f = res.forecast
    a = f.analysis
    h = res.request.horizon
    sim, ss = f.simulations[h], f.scenarios[h]
    cur = a.currency
    snap = fundamentals if fundamentals is not None else res.fundamentals
    first_js: bool | str = True if inline_js else "cdn"  # the first chart carries the library

    disclaimer = _e(DISCLAIMER).replace("**", "")  # strip markdown bold from the shared text
    parts: list[str] = [
        f"<h1>Stock Analyzer report: {_e(a.ticker)}</h1>",
        f'<p class="muted">Generated {_e(generated_at.strftime("%Y-%m-%d %H:%M UTC"))}. '
        f"Latest price {a.last_price:,.2f} {_e(cur or 'currency unknown')} as of {_e(a.as_of)}. "
        f"Data source: {_e(res.provider or 'unknown')}. Horizon: {_e(h)}. "
        f"Benchmark: {_e(a.benchmark.name if a.benchmark else 'none')}.</p>",
        f'<div class="disclaimer">{disclaimer}</div>',
    ]
    for w in a.warnings:
        parts.append(f'<div class="note">{_e(w)}</div>')

    # reliability: 1M and 3M graded, 6M and 1Y "can't be verified"
    views = reliability_views(f.reliability)
    graded = [v for v in views if v.horizon in ("1M", "3M")]
    parts.append("<h2>How reliable have this model's ranges been?</h2>")
    parts.append('<div class="grades">')
    for v in graded:
        grade = v.title.split(": ", 1)[1]
        parts.append(
            f'<div class="grade">{_e(v.horizon)} horizon<br><b>{_e(v.icon)} {_e(grade)}</b>'
            f"<br><span class='muted'>{_e(v.text)}</span></div>"
        )
    parts.append("</div>")
    for v in views:
        if v.horizon not in ("1M", "3M"):
            parts.append(f'<div class="note">{_e(v.text)}</div>')
    details = [f"{v.horizon}: {d}" for v in views for d in v.details]
    if details:
        parts.append("<h3>How it was checked</h3>")
        parts.append(
            "<p>We replayed this stock's history: at many past dates the model forecast ranges "
            "using only data available then, and we counted how often the real price landed "
            "inside. Past accuracy is not a promise about the future.</p>"
        )
        parts.append(_bullets(details))

    parts.append(f"<h2>In plain English ({_e(h)} horizon)</h2>")
    parts.append(_bullets(sim.plain_english(cur)))
    parts.append(_bullets(a.technical.bullets + a.risk_bullets))

    parts.append("<h2>Price and indicators</h2>")
    opts = PriceChartOptions(
        sma=(50, 200),
        volume=True,
        rsi=True,
        days=252,
        levels=tuple(x.price for x in a.technical.levels),
    )
    parts.append(_fig(price_chart(res.ohlc, res.indicators, opts, cur), first_js))
    parts.append(
        '<p class="muted">Prices are adjusted for splits and dividends. Dotted lines mark past '
        "support and resistance. These describe the past; they are not forecasts.</p>"
    )

    parts.append(f"<h2>Possible price range over {_e(h)}</h2>")
    parts.append(_fig(fan_chart(sim, a.adj_close, pd.Timestamp(a.as_of), cur), False))
    parts.append(
        '<p class="muted">Shaded bands hold 50% and 90% of the simulated paths. This is a spread '
        "of possibilities under stated assumptions, not a prediction.</p>"
    )
    parts.append(_fig(terminal_histogram(sim, cur), False))
    parts.append(
        '<p class="muted">Each bar counts simulated paths that finished at that price.</p>'
    )

    parts.append("<h2>Bull / Base / Bear scenarios</h2>")
    parts.append(_table(scenario_table(ss, cur)))
    parts.append(_bullets(ss.notes))
    for sc in ss.scenarios:
        parts.append(f"<h3>{_e(sc.name)}</h3>")
        parts.append(_bullets([*sc.assumptions, sc.indicator_context]))

    table = sim.summary_table().reset_index()
    table["price"] = table["price"].map(lambda x: f"{x:,.2f} {cur or ''}".strip())
    table["return"] = table["return"].map(lambda x: f"{x:+.1%}")
    parts.append("<h2>Percentiles</h2>")
    parts.append(
        _table(
            table.rename(columns={"percentile": "Percentile", "price": "Price", "return": "Change"})
        )
    )

    r = a.risk
    parts.append("<h2>Risk</h2>")
    parts.append(
        f'<p class="muted">Based on {r.n_returns} daily returns from {_e(r.window_start)} '
        f"to {_e(r.window_end)}.</p>"
    )
    parts.append(_kv_table([(label, value) for label, value, _ in risk_rows(r, a.benchmark)]))

    parts.append("<h2>Fundamentals</h2>")
    if snap is None or not snap.available:
        parts.append('<div class="note">Fundamental data was not available for this ticker.</div>')
    else:
        if snap.sector or snap.industry:
            where = " / ".join(x for x in (snap.sector, snap.industry) if x)
            parts.append(f'<p class="muted">{_e(where)}</p>')
        parts.append(
            _kv_table(
                [
                    (row.label, row.value + (f"  ({row.note})" if row.note else ""))
                    for row in snap.rows
                ]
            )  # fmt: skip
        )
    if snap is not None:
        parts.append(_bullets(snap.notes))

    cal = sim.calibration
    parts.append("<h2>Assumptions used</h2>")
    parts.append(
        _bullets(
            [
                f"Method: {sim.method.upper()}, {sim.n_paths:,} simulated paths, "
                f"seed {sim.seed if sim.seed is not None else 'not fixed'}.",
                f"Yearly volatility {cal.sigma_annual:.1%}; assumed yearly return "
                f"{cal.mu_annual:.1%} ({cal.drift_mode} drift; historical average "
                f"{cal.mu_hist_annual:.1%} with a standard error of about "
                f"{cal.mu_hist_se:.0%}).",
                f"Calibrated on {cal.n_obs} daily returns from {cal.window_start} to "
                f"{cal.window_end}.",
                (
                    res.risk_free.label
                    if res.risk_free
                    else f"Risk-free rate {cal.risk_free_rate:.2%}."
                ),
                f"Validation model version {MODEL_VERSION}.",
                *cal.warnings,
            ]
        )
    )
    parts.append("<h2>Limitations</h2>")
    parts.append(_bullets(REPORT_LIMITS))
    parts.append(f'<p class="disclaimer">{disclaimer}</p>')

    title = f"Stock Analyzer report: {_e(a.ticker)}"
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{title}</title><style>{_CSS}</style></head><body><main>"
        + "".join(parts)
        + "</main></body></html>"
    )


def report_filename(res: AppResult, generated_at: datetime) -> str:
    """A safe file name: the ticker is validated upstream, but never trust it in a path."""
    safe = "".join(c for c in res.request.ticker if c.isalnum() or c in ".-")[:16] or "report"
    return f"stock-analyzer-{safe}-{generated_at:%Y%m%d}.html"
