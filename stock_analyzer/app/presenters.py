"""Turn analysis results into display-ready rows and text. Pure: no Streamlit, no I/O.

Reliability policy (agreed with the project owner): grades are shown prominently for the
1M and 3M horizons, where 10 years of history holds enough independent checks. For 6M and 1Y
we show a plain "cannot be verified" note instead of a grade, however the numbers fall,
because a handful of overlapping windows cannot tell good calibration from luck.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from stock_analyzer.analysis.benchmarks import Benchmark
from stock_analyzer.analysis.risk import RiskMetrics
from stock_analyzer.backtest.reliability import Reliability
from stock_analyzer.models.scenarios import ScenarioSet

GRADED_HORIZONS = ("1M", "3M")
ALL_HORIZONS = ("1M", "3M", "6M", "1Y")
GRADE_ICON = {"Good": "🟢", "Fair": "🟡", "Poor": "🔴", "Not enough history": "⚪"}


@dataclass(frozen=True)
class ReliabilityView:
    horizon: str
    verifiable: bool
    title: str  # e.g. "1M: Good" or "6M: can't be verified"
    text: str
    details: tuple[str, ...] = ()
    icon: str = "⚪"


def reliability_views(rel: dict[str, Reliability] | None) -> list[ReliabilityView]:
    """One view per horizon, 1M and 3M first. ``rel`` None means the check failed to run."""
    views: list[ReliabilityView] = []
    for h in ALL_HORIZONS:
        r = rel.get(h) if rel else None
        if h in GRADED_HORIZONS:
            if r is None:
                views.append(
                    ReliabilityView(
                        h,
                        False,
                        f"{h}: reliability check unavailable",
                        "We could not run the past-accuracy check for this stock just now.",
                    )
                )
            else:
                views.append(
                    ReliabilityView(
                        h,
                        r.grade != "Not enough history",
                        f"{h}: {r.grade}",
                        r.headline,
                        r.details,
                        GRADE_ICON.get(r.grade, "⚪"),
                    )
                )
        else:
            n = f"only {r.n_independent} independent past checks" if r else "too few past checks"
            views.append(
                ReliabilityView(
                    h,
                    False,
                    f"{h}: can't be verified",
                    f"The {h} ranges can't be verified: with about 10 years of history there are "
                    f"{n}, too few to tell whether the ranges are reliable. Treat them as "
                    "unverified.",
                    r.details if r else (),
                    "⚪",
                )
            )
    return views


def _money(x: float, currency: str | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:,.2f} {currency}" if currency else f"{x:,.2f}"


def scenario_table(ss: ScenarioSet, currency: str | None) -> pd.DataFrame:
    """Bull / Base / Bear table with probability ranges (shown bull first)."""
    rows = []
    for sc in reversed(ss.scenarios):
        rows.append(
            {
                "Scenario": sc.name,
                "Chance": f"{sc.probability:.0%}",
                "If volatility is 20% lower/higher": (
                    f"{sc.probability_low:.0%} to {sc.probability_high:.0%}"
                ),
                "Typical price range": (
                    "n/a"
                    if math.isnan(sc.price_low)
                    else f"{_money(sc.price_low, None)} to {_money(sc.price_high, currency)}"
                ),
                "Middle outcome": _money(sc.target_price, currency),
                "What counts": sc.boundary_text,
            }
        )
    return pd.DataFrame(rows)


def risk_rows(r: RiskMetrics, benchmark: Benchmark | None) -> list[tuple[str, str, str]]:
    """(label, value text, glossary key) for the risk panel."""
    rows = [
        ("Volatility (yearly)", f"{r.annualised_volatility:.1%}", "volatility"),
        ("Sharpe ratio", f"{r.sharpe:.2f}", "sharpe"),
        ("Sortino ratio", f"{r.sortino:.2f}", "sortino"),
        ("Max drawdown", f"{r.drawdown.max_drawdown:.1%}", "drawdown"),
    ]
    b = r.beta
    if b is not None and b.beta == b.beta:
        name = benchmark.name if benchmark else "benchmark"
        rows.append((f"Beta vs {name}", f"{b.beta:.2f} (R² {b.r_squared:.2f})", "beta"))
    for v in r.var:
        rows.append((f"{v.confidence:.0%} 1-day VaR", f"{v.historical:.2%}", "var"))
    if r.var:
        rows.append(
            (
                f"Avg loss on worst {1 - r.var[0].confidence:.0%} of days",
                f"{r.var[0].expected_shortfall:.2%}",
                "es",
            )
        )
    return rows


def friendly_error(exc: BaseException) -> str:
    """Text safe to show a user: our own messages, or a generic one. Never a stack trace."""
    from stock_analyzer.exceptions import StockAnalyzerError

    if isinstance(exc, StockAnalyzerError):
        return exc.user_message
    return "Something unexpected went wrong. Please try again, or try a different ticker."
