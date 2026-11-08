"""Bull / base / bear scenarios derived from the simulation.

How the numbers arise (nothing is hand-set):
* Boundaries: the horizon's typical move is sigma * sqrt(T). Bear = ending more than half
  of that below today, bull = more than half of that above, base = in between (never
  narrower than +/-3%). Scaling by volatility keeps the scenarios meaningful for both calm
  and wild stocks and for every horizon.
* Probability = the share of simulated paths ending in each region (so they sum to 100%).
* Price range = the middle half (25th-75th percentile) of the paths inside that region;
  target = their median.
* Sensitivity: the same boundaries are re-tested with volatility 20% lower and 20% higher.
  The resulting spread is shown as a probability range, because the volatility input is
  itself uncertain.

Indicator state is shown next to each scenario as context but is deliberately NOT used to
tilt the probabilities: nothing here shows that chart indicators improve forecasts. Phase 4
(backtesting) is where that would have to be demonstrated before blending it in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from stock_analyzer.analysis.risk import TRADING_DAYS
from stock_analyzer.analysis.technical import TechnicalSnapshot
from stock_analyzer.models.monte_carlo import SimulationResult, describe_horizon

MIN_REGION_PATHS = 20  # below this, a region's price range is not reported


@dataclass(frozen=True)
class Scenario:
    name: str  # "Bear" | "Base" | "Bull"
    probability: float
    probability_low: float  # lowest probability across the volatility sensitivity runs
    probability_high: float
    price_low: float  # middle half of outcomes in this region
    price_high: float
    target_price: float  # median outcome in this region
    boundary_text: str
    assumptions: list[str] = field(default_factory=list)
    indicator_context: str = ""


@dataclass(frozen=True)
class ScenarioSet:
    horizon_days: int
    s0: float
    threshold: float  # half-width of the base scenario as a fraction of today's price
    scenarios: tuple[Scenario, Scenario, Scenario]  # bear, base, bull
    notes: list[str] = field(default_factory=list)

    def by_name(self, name: str) -> Scenario:
        return next(s for s in self.scenarios if s.name == name)


def _masks(terminal: np.ndarray, s0: float, hw: float) -> dict[str, np.ndarray]:
    lo, hi = s0 * (1 - hw), s0 * (1 + hw)
    return {
        "Bear": terminal <= lo,
        "Base": (terminal > lo) & (terminal < hi),
        "Bull": terminal >= hi,
    }


def _context(name: str, tech: TechnicalSnapshot | None) -> str:
    if tech is None or tech.trend_max == 0:
        return "No technical context available."
    label = tech.trend_label
    bullish, bearish = (
        label in {"Bullish", "Leaning bullish"},
        label
        in {
            "Bearish",
            "Leaning bearish",
        },
    )
    if name == "Bull":
        fit = "consistent with" if bullish else "at odds with" if bearish else "neutral towards"
    elif name == "Bear":
        fit = "consistent with" if bearish else "at odds with" if bullish else "neutral towards"
    else:
        fit = "consistent with" if not (bullish or bearish) else "less typical of"
    return (
        f"The chart picture is currently '{label}', {fit} this scenario. Indicators have "
        "limited predictive power, so this does not change the probability."
    )


def build_scenarios(
    base: SimulationResult,
    low_vol: SimulationResult,
    high_vol: SimulationResult,
    technical: TechnicalSnapshot | None = None,
    *,
    currency: str | None = None,
    k: float = 0.5,
    min_threshold: float = 0.03,
) -> ScenarioSet:
    """Build the three scenarios from a base simulation and two volatility-sensitivity runs."""
    s0, h = base.s0, base.horizon_days
    cur = f"{currency} " if currency else ""
    horizon_sigma = base.sigma_used * math.sqrt(h / TRADING_DAYS)
    hw = max(min_threshold, k * horizon_sigma)

    regions = {name: _masks(sim.terminal, s0, hw) for name, sim in
               (("base", base), ("low", low_vol), ("high", high_vol))}  # fmt: skip
    lo_p, hi_p = s0 * (1 - hw), s0 * (1 + hw)
    boundary = {
        "Bear": f"Price ends more than {hw:.1%} below today (below {cur}{lo_p:,.2f}).",
        "Base": f"Price ends within ±{hw:.1%} of today ({cur}{lo_p:,.2f} to {cur}{hi_p:,.2f}).",
        "Bull": f"Price ends more than {hw:.1%} above today (above {cur}{hi_p:,.2f}).",
    }
    c = base.calibration
    common = [
        f"Volatility assumed: {base.sigma_used:.0%} a year; expected return: "
        f"{c.mu_annual:+.1%} a year ({c.drift_mode} drift).",
        f"Time horizon: {describe_horizon(h)}; method: "
        f"{'GBM' if base.method == 'gbm' else 'bootstrapped returns'}, {base.n_paths:,} paths.",
    ]

    scenarios = []
    for name in ("Bear", "Base", "Bull"):
        mask = regions["base"][name]
        probs = [float(np.mean(regions[key][name])) for key in ("base", "low", "high")]
        inside = base.terminal[mask]
        if len(inside) >= MIN_REGION_PATHS:
            p25, med, p75 = np.percentile(inside, [25, 50, 75])
        else:
            p25 = med = p75 = math.nan
        scenarios.append(
            Scenario(
                name=name,
                probability=probs[0],
                probability_low=min(probs),
                probability_high=max(probs),
                price_low=float(p25),
                price_high=float(p75),
                target_price=float(med),
                boundary_text=boundary[name],
                assumptions=[boundary[name], *common],
                indicator_context=_context(name, technical),
            )
        )
    notes = [
        "Probabilities come from the simulation under the stated assumptions; they are not "
        "forecasts. Ranges show how they move if volatility is 20% lower or higher.",
        "Price ranges are the middle half of simulated outcomes within each scenario.",
    ]
    return ScenarioSet(h, s0, hw, tuple(scenarios), notes)  # type: ignore[arg-type]
