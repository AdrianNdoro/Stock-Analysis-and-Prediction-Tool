# Stock Analyzer (Phases 1-4: data, analysis, risk, simulation, scenarios, backtesting)

Educational tool. Not financial advice. Past performance does not guarantee future results.
Simulated outcomes are scenarios, never predictions.

## Windows setup (Command Prompt or PowerShell)

```bat
:: from the project folder (the one containing requirements.txt)
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
copy .env.example .env
```

PowerShell activation: `.venv\Scripts\Activate.ps1`. If scripts are blocked, run
`Set-ExecutionPolicy -Scope Process Bypass` once in that window.
If `py -3.11` fails, run `py -0` to see which Python versions are installed.

## Run the tests and the linter

```bat
python -m pytest
ruff check .
```

## Smoke test against the live provider (needs internet)

```bat
python -m stock_analyzer.cli AAPL
python -m stock_analyzer.cli AAPL --fundamentals
python -m stock_analyzer.cli BP.L
```

Run the first command twice: the second run should say `From cache: True`.
The cache file is `data_cache\market_cache.sqlite3`; delete it to start fresh.

## Analysis (Phase 2)

```bat
python -m stock_analyzer.cli AAPL --analyze
python -m stock_analyzer.cli BP.L --analyze                 # auto-picks the FTSE 100 benchmark
python -m stock_analyzer.cli SAP.DE --analyze --benchmark EUROSTOXX50
```

Technical: SMA/EMA 20/50/200, RSI, MACD, Bollinger Bands, ATR, volume trend, support and
resistance, plain-English summary. Risk: annualised volatility, max drawdown, Sharpe, Sortino,
beta (weekly, local-currency returns), historical and parametric VaR plus Expected Shortfall.
Risk settings (`RISK_FREE_RATE`, `RISK_LOOKBACK_YEARS`, `DEFAULT_BENCHMARK`) live in `.env`.
If you created `.env` before Phase 2, add those three lines from `.env.example` (the defaults
work without them).

## Simulation and scenarios (Phase 3)

```bat
python -m stock_analyzer.cli AAPL --simulate --horizon 6M --seed 42
python -m stock_analyzer.cli AAPL --simulate --horizon ALL --method bootstrap
python -m stock_analyzer.cli AAPL --simulate --drift neutral --paths 20000
```

Horizons: 1M, 3M, 6M, 1Y (or ALL). Methods: `gbm` (default) or `bootstrap`. Drift: `shrunk`
(default), `historical`, `neutral`. Paths: 10,000 to 50,000. `--seed` makes a run repeatable.
Results are scenarios under stated assumptions, never predictions. Settings `MC_PATHS`,
`MC_LOOKBACK_YEARS`, `MC_DRIFT_MODE` live in `.env` (defaults work without them).

## Layout

```
stock_analyzer/
  config.py          settings from environment variables
  exceptions.py      errors with friendly user_message text
  logging_setup.py   structured (JSON) logging
  cli.py             smoke-test command
  analysis/
    technical.py     indicators, support/resistance, plain-English summary
    risk.py          volatility, drawdown, Sharpe/Sortino, beta, VaR
    benchmarks.py    benchmark list, suffix-based suggestion, currency helpers
    pipeline.py      MarketData -> AnalysisResult
  models/
    monte_carlo.py   calibration + GBM / bootstrap simulation, percentiles, probabilities
    scenarios.py     bull/base/bear from the simulation, with sensitivity ranges
    forecast.py      analysis -> simulations + scenarios
  data/
    providers/       DataProvider interface + yfinance implementation
    cache.py         shared SQLite cache (WAL, thread/process safe)
    cleaning.py      validation and quality report
    resilience.py    rate limiter + retry with backoff
    service.py       cache -> limit -> retry -> provider -> clean
    validation.py    ticker sanitising
  tests/
```

## Swapping the data provider

Subclass `DataProvider` (`data/providers/base.py`), register it in
`data/providers/__init__.py`, set `DATA_PROVIDER` and `DATA_PROVIDER_API_KEY` in `.env`.

## Known limits (Phase 1)

* yfinance is unofficial and personal-use oriented: fine for development, not for a public launch.
* On free hosts the SQLite file may be wiped on restart; the cache simply rebuilds.
* Fundamentals unit normalisation (dividend yield, debt/equity) comes in Phase 6.
