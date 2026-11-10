"""Text for the disclaimer and the "How this works and its limitations" page. No Streamlit."""

from __future__ import annotations

from stock_analyzer.app.glossary import GLOSSARY

DISCLAIMER = (
    "**Educational tool, not financial advice.** Past performance does not guarantee future "
    "results. The simulations show a spread of possible outcomes under stated assumptions; "
    "they are not predictions."
)

HOW_IT_WORKS = """
## What this tool does

You enter a stock or ETF ticker. The tool downloads its daily price history, describes its
recent condition (trend, momentum, risk), and then **simulates thousands of possible price
paths** over the horizon you choose. From those paths it reports a range of outcomes and the
share of paths that ended clearly higher (Bull), near today's price (Base) or clearly lower
(Bear).

**It does not predict the future.** A simulation answers: *"if daily returns kept behaving
like the assumptions below, what spread of outcomes would result?"* Real prices respond to
earnings, news, interest rates and shocks that no price-history model can see.

## The assumptions, and where each one breaks

* **Returns look like random draws (GBM).** The default model treats each day's return as a
  random draw from a bell curve with the stock's recent volatility. *Where it breaks:* real
  returns have **fat tails** (extreme days are far more common than a bell curve says), so
  large moves are understated. The *bootstrap* option re-uses the stock's own past daily
  returns and keeps fat tails.
* **Volatility stays at its recent level.** *Where it breaks:* **volatility clusters**: calm
  spells and stormy spells come in streaks. In a calm period the ranges can be too wide, and
  right after a shock they are usually too narrow. We tested this on synthetic data and it
  shows up as ranges that are wrong in both directions.
* **The recent past is representative.** *Where it breaks:* **regime changes** (a pandemic, a
  rate shock, a company changing its business) make the calibration window misleading.
* **The expected return is uncertain.** A stock's own average return over a few years is very
  noisy, so by default it is blended with a long-run market assumption. Change this under
  Advanced if you want to see the effect.
* **Low liquidity.** Thinly traded stocks have gaps and stale prices; the model does not
  account for the cost or difficulty of trading.
* **Prices are adjusted for splits and dividends**, so results describe total return in the
  stock's own currency. We never mix currencies; benchmarks in another currency are flagged.

## How reliable have these ranges been? (the backtest)

For every stock we replay history: at many past dates we calibrate the model using only data
available then, make a forecast, and check whether the real price landed inside the model's
50% and 90% ranges. A well-behaved 90% range should contain the real price about 90% of the
time.

* **1M and 3M horizons:** with about 10 years of data there are enough independent checks to
  grade the model **Good / Fair / Poor**. The grade only says how the ranges performed *in
  the past, for this stock*.
* **6M and 1Y horizons:** these **can't be verified**. Ten years contain only about 14
  independent 6-month windows and 7 one-year windows, too few to tell skill from luck.
* **Baselines:** the model is also compared with a no-drift random walk and with the
  stock's own past returns. When it does not beat them, we say so.
* **What we found:** in our own tests on 12 large stocks (to November 2026), pooled 90%
  ranges held the real price about 88-91% of the time, close to the target, though
  individual stocks varied (some ran too narrow). Simple fixes such as widening the ranges did
  not improve results on data the fix had not seen, so the model was left unchanged.
* **Limits of the backtest itself:** the tested stocks all survived to today (survivorship
  bias); history is dominated by a few big events (2020, 2022); checks close together in time
  overlap and are not independent; and a good past grade is no promise about the future.

## Data and privacy

* Prices come from a free public data source intended for personal use. It can be delayed,
  occasionally wrong, or unavailable. A public launch would use a licensed provider.
* We do **not** collect personal data, require an account, or run analytics. Tickers you
  look up are cached to speed up later requests; nothing links them to you.
* Requests are rate-limited per session to keep the service responsive for everyone.
"""


def glossary_markdown() -> str:
    lines = ["## Glossary", ""]
    for _key, (name, text) in GLOSSARY.items():
        lines.append(f"**{name}.** {text}")
        lines.append("")
    return "\n".join(lines)
