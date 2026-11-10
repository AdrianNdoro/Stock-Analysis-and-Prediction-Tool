"""Plain-English definitions used for tooltips and the glossary. No Streamlit imports."""

from __future__ import annotations

GLOSSARY: dict[str, tuple[str, str]] = {
    "volatility": (
        "Volatility",
        "How much the price jumps around, as a yearly percentage. 30% means a typical year "
        "moves the price about 30% up or down from its average path. Higher = bumpier.",
    ),
    "sharpe": (
        "Sharpe ratio",
        "Return earned above a safe rate (like a savings account or T-bill) per unit of "
        "volatility. Higher is better; below 0 means the stock did worse than the safe rate.",
    ),
    "sortino": (
        "Sortino ratio",
        "Like the Sharpe ratio, but only counts downside bumpiness, since upside surprises "
        "are not really 'risk'.",
    ),
    "beta": (
        "Beta",
        "How strongly the stock moved with its market index. 1.0 = moves with the market, "
        "1.5 = moves 50% more, 0.5 = moves half as much, negative = tends to move opposite.",
    ),
    "drawdown": (
        "Maximum drawdown",
        "The biggest fall from a peak to a later low. -40% means an investor who bought at "
        "the top would at one point have been down 40%.",
    ),
    "var": (
        "Value at Risk (VaR)",
        "A 95% one-day VaR of 2.5% means: on 95 out of 100 past days the loss was smaller "
        "than 2.5%, and on the worst 5 it was at least that much.",
    ),
    "es": (
        "Expected Shortfall",
        "The average loss on the very bad days beyond the VaR. It says how bad the bad days "
        "were, not just where they start.",
    ),
    "percentile": (
        "Percentile (5th, 25th, ...)",
        "The 5th percentile is the price that 5 out of 100 simulated outcomes fell below. "
        "The 50th (median) is the middle outcome. The 5th-95th range holds 9 in 10 outcomes.",
    ),
    "rsi": (
        "RSI (14)",
        "A 0-100 gauge of recent buying vs selling pressure. Above 70 is often called "
        "'overbought', below 30 'oversold'. It describes recent moves; it is not a forecast.",
    ),
    "macd": (
        "MACD",
        "Compares a faster and a slower moving average. When the MACD line is above its "
        "signal line, recent momentum is upward; below, downward.",
    ),
    "bollinger": (
        "Bollinger Bands",
        "A band two standard deviations above and below a 20-day average. Prices outside the "
        "band are unusually far from recent levels.",
    ),
    "atr": (
        "ATR (14)",
        "Average True Range: the typical daily price range over 14 days. A rough measure of "
        "how much the price moves in a day.",
    ),
    "sma_ema": (
        "SMA / EMA",
        "Moving averages of the price (simple, or weighting recent days more). They smooth "
        "out day-to-day noise to show the trend.",
    ),
    "support_resistance": (
        "Support / resistance",
        "Price levels where the stock has repeatedly turned around. Support is below today's "
        "price, resistance above. They are patterns in past data, not guarantees.",
    ),
    "reliability": (
        "Model reliability",
        "How often the model's own past ranges contained the price that really happened. A "
        "90% range should contain the real price about 90% of the time.",
    ),
    "scenario": (
        "Bull / Base / Bear",
        "Three groups of simulated outcomes: finishing clearly above, near, or clearly below "
        "today's price. The percentages are the share of simulated paths in each group.",
    ),
    "gbm": (
        "GBM",
        "Geometric Brownian Motion: a simple model where daily returns are random bell-curve "
        "draws. Tends to understate extreme days.",
    ),
    "bootstrap": (
        "Bootstrap",
        "Re-uses the stock's own past daily returns, randomly reshuffled, to build possible "
        "futures. Keeps fat tails but ignores streaks of calm or stormy days.",
    ),
    "drift": (
        "Expected return (drift)",
        "The average yearly return assumed in the simulation. It is the hardest input to "
        "estimate, so the default blends the stock's history with a long-run market assumption.",
    ),
}


def tip(key: str) -> str:
    """Tooltip text for a glossary key (a typo raises KeyError, which a test guards against)."""
    return GLOSSARY[key][1]


def title(key: str) -> str:
    return GLOSSARY[key][0]
