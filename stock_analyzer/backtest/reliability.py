"""Plain-English model-reliability grade. ALL grading thresholds live in ``THRESHOLDS``.

Grade meanings (judged on the model's own past forecasts for this stock and horizon):
* Good   - the 50% and 90% ranges contained the real price about as often as claimed.
* Fair   - noticeably off, but usable with the caveat shown.
* Poor   - ranges were far too narrow or too wide; do not rely on them.
* Not enough history - too few independent checks to judge at all.

Coverage is the headline because it is directly checkable: if a model says "90% of outcomes
fall in this range", about 90% of real outcomes should. We grade on the point estimate but
refuse to say "Good" on a small effective sample, since a handful of checks can look
calibrated by luck.
"""

from __future__ import annotations

from dataclasses import dataclass

from stock_analyzer.backtest.report import BacktestReport, ModelMetrics, skill_vs


@dataclass(frozen=True)
class GradeThresholds:
    min_independent: int = 15  # below this: "Not enough history"
    min_independent_for_good: int = 30  # a "Good" needs at least this many independent checks
    good_gap90: float = 0.04  # |coverage90 - 0.90| at most
    good_gap50: float = 0.08  # |coverage50 - 0.50| at most
    fair_gap90: float = 0.08
    fair_gap50: float = 0.12


THRESHOLDS = GradeThresholds()


@dataclass(frozen=True)
class Reliability:
    grade: str  # "Good" | "Fair" | "Poor" | "Not enough history"
    horizon: str
    headline: str
    details: tuple[str, ...]
    coverage90: float | None = None
    coverage50: float | None = None
    n_checks: int = 0
    n_independent: int = 0


def grade_metrics(mm: ModelMetrics, t: GradeThresholds = THRESHOLDS) -> str:
    if mm.n_independent < t.min_independent:
        return "Not enough history"
    g90, g50 = abs(mm.coverage90 - 0.90), abs(mm.coverage50 - 0.50)
    if (
        g90 <= t.good_gap90
        and g50 <= t.good_gap50
        and mm.n_independent >= t.min_independent_for_good
    ):
        return "Good"
    if g90 <= t.fair_gap90 and g50 <= t.fair_gap50:
        return "Fair"
    return "Poor"


def assess(report: BacktestReport, horizon: str, t: GradeThresholds = THRESHOLDS) -> Reliability:
    """Reliability of the report's graded model at one horizon (e.g. ``"3M"``)."""
    label = horizon.strip().upper()
    hr = report.horizons.get(label)
    mm = hr.models.get(report.graded_model) if hr else None
    if hr is None or mm is None:
        return Reliability(
            "Not enough history",
            label,
            f"Not enough price history to check how reliable the {label} ranges have been.",
            ("The model could not be tested on this stock at this horizon.",),
        )
    grade = grade_metrics(mm, t)
    base = (
        f"Over {hr.n_origins} past checks ({hr.n_independent} independent), the model's 90% range "
        f"contained the real price {mm.coverage90:.0%} of the time (it should be 90%) and its "
        f"50% range {mm.coverage50:.0%} (should be 50%)."
    )
    if grade == "Not enough history":
        headline = (
            f"Not enough history: only {hr.n_independent} independent past checks at {label}. "
            + base
        )
    else:
        gap = mm.coverage90 - 0.90
        if abs(gap) <= t.good_gap90:
            read = "The ranges have been about the right width."
        elif gap < 0:
            read = "Treat its ranges as too narrow: real prices left them more often than claimed."
        else:
            read = "Its ranges have been wider than needed: real prices stayed inside more often."
        headline = f"{grade}. {base} {read}"
    details: list[str] = []
    for baseline, text in (("random_walk", "a zero-drift random walk"),
                           ("empirical", "the stock's own past returns")):  # fmt: skip
        if baseline in hr.models:
            s = skill_vs(hr.models[baseline], mm)
            if s == s:
                verdict = "did not beat" if s <= 0.01 else "beat"
                details.append(
                    f"On quantile loss the model {verdict} {text} ({s:+.1%} vs that baseline)."
                )
    details.append(
        "Past checks do not guarantee future accuracy; regimes change and checks that overlap "
        "in time are not independent."
    )
    return Reliability(
        grade, label, headline, tuple(details), mm.coverage90, mm.coverage50,
        hr.n_origins, hr.n_independent,
    )  # fmt: skip
