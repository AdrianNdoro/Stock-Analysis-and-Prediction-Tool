# Stock Analyzer

Describe a stock's past and simulate a range of possible futures, with an honest check of how
reliable those ranges have been. Data, analysis, risk, simulation, scenarios, backtesting, a
dashboard, fundamentals, report export and deployment files.

Educational tool. Not financial advice. Past performance does not guarantee future results.
Simulated outcomes are scenarios, never predictions.

## How to use the tool (for users)

1. Open the dashboard (see **Dashboard** below) and type a ticker: `AAPL`, `BRK-B`, `BP.L`
   (London), `SAP.DE` (Germany), `7203.T` (Tokyo). Pick a time horizon and press **Analyse**.
2. Read **How reliable have this model's ranges been?** first. The 1M and 3M horizons get a
   grade (Good / Fair / Poor / Not enough history) from replaying the stock's own history.
   6M and 1Y say "can't be verified": there is not enough history to check them. Treat
   unverified ranges as rough.
3. The **Possible outcomes** tab shows a fan chart (the shaded bands hold 50% and 90% of
   simulated paths), where the paths ended, and Bull / Base / Bear scenarios with how sensitive
   each probability is to the volatility assumption.
4. **Price & indicators**, **Risk** and **Fundamentals** describe the past. Hover over the
   ⓘ icons for plain-English explanations; the sidebar page **How this works** has a glossary
   and the limitations.
5. **Download a report** (bottom of the page) saves everything as one HTML file you can keep,
   share, or print to PDF.

None of this tells you what a price will do. The ranges are what a simple model says is
plausible if the past volatility continues; real markets have crashes the model underestimates.

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

## Backtesting and model validation (Phase 4)

```bat
python -m stock_analyzer.cli AAPL --backtest                      :: all horizons
python -m stock_analyzer.cli AAPL --backtest --horizon 3M --method bootstrap --window expanding
python -m stock_analyzer.cli BP.L --simulate --horizon 3M --reliability   :: forecast + reliability grade
python -m stock_analyzer.backtest.basket                          :: 12-ticker validation (needs internet)
python -m stock_analyzer.backtest.tuning                          :: remedy experiment with held-out checks
```

A walk-forward backtest steps through history every 21 trading days. At each origin it
calibrates on data up to that day only, forecasts each horizon, and compares with what really
happened. It reports: how often the 50% and 90% ranges contained the real price (with Wilson
confidence intervals sized by the number of *independent* checks), Kupiec and Christoffersen
tests, a PIT histogram, quantile (pinball) loss and interval score versus two baselines
(zero-drift random walk, the stock's own past returns), and direction hit rate. Results are
cached for 7 days in the SQLite cache (key = model version + parameters + last data date).

**Reliability grade** (`backtest/reliability.py`, thresholds in `THRESHOLDS`): *Good* (90% range
within 4 points and 50% range within 8 points of nominal, with at least 30 independent
checks), *Fair* (within 8 / 12 points), *Poor* (worse), *Not enough history* (under 15
independent checks; common for 6M and 1Y, where 10 years holds only 14 and 7 non-overlapping
windows).

**Findings on 12 tickers (live data to Nov 2026, GBM, shrunk drift, 3-year rolling window):**
pooled 90% ranges contained the real price 88-91% of the time and 50% ranges 48-54%, close to
nominal. Volatility inflation and EWMA volatility did not beat plain GBM on proper scores in
held-out tests, so **no model change was made**. The stock-level grades vary (for example
SAP.DE ran too narrow at 3M-1Y). Baselines: the zero-drift random walk reached similar pooled coverage (84-90% at 90%), so
the drift assumption adds little, and loss comparisons are mixed per stock (at 3M the model beat
it for AAPL and lost to it for BP.L). The empirical-quantile baseline was too narrow at long
horizons (pooled 90% coverage 76% at 6M and 57% at 1Y).

**Limits of this validation:** the basket has survivorship bias and was chosen with hindsight;
tickers share market moves so they are not independent experiments; 2020 and 2022 dominate
the sample; checks at the same horizon overlap in time; a constant risk-free rate is used for
all years. A good grade means "the ranges were about the right width in the past", never
"the next price will land in the range".

## Dashboard (Phase 5)

```
python -m streamlit run stock_analyzer/app/main.py
```

Opens at http://localhost:8501. Enter a ticker, pick a horizon, press **Analyse**. Tabs:
Summary (with the reliability panel), Price & indicators, Possible outcomes (fan chart,
histogram, bull/base/bear table), Risk, Fundamentals. The **How this works** page in the sidebar
lists the assumptions, limitations and a glossary; a disclaimer is shown on every page.

- **Reliability:** 1M and 3M show the Phase 4 grade. 6M and 1Y show a plain "can't be
  verified" note, because ten years of history gives too few independent checks.
- **Safety:** tickers go through `normalize_ticker`; horizon, method, drift, paths and seed are
  checked against allow-lists and limits; each session is throttled (8 analyses per minute);
  errors show friendly text, never stack traces.
- **Concurrency:** one `MarketDataService` per process (`st.cache_resource`); analyses are cached
  with `st.cache_data` keyed by the request and the latest price date; sessions share no state.
- **Privacy:** no accounts, no analytics, nothing about users is stored or logged.
- **Tests:** `test_app_pure.py` (helpers, charts) and `test_app_ui.py` (Streamlit `AppTest`
  with a fake provider). The UI tests take about a minute and some memory.

## Fundamentals, report export and risk-free rate (Phase 6)

**Fundamentals** (Fundamentals tab; also `--fundamentals` in the CLI): market cap, P/E (trailing
and expected), P/B, EPS, revenue and earnings growth, debt/equity, dividend yield. Yahoo's units
are converted at the provider to a documented contract (fractions and ratios; see
`data/providers/base.py`) and pinned by tests to a recorded payload
(`tests/fixtures/yahoo_info_sample.json`). Missing data shows "Not available", never zero.
Sector averages are **not** shown: yfinance does not supply them, and we do not invent them.
Gotchas the tests cover: Yahoo reports dividend yield and debt/equity in percent, and
pence-quoted stocks (BP.L) report EPS in another currency.

**Report export:** the **Download a report** button builds one self-contained HTML file
(charts, scenarios, assumptions, reliability note, fundamentals, disclaimer). By default the
Plotly library is embedded (about 5 MB, works offline); tick "Smaller file" for a few hundred KB
that loads Plotly from a CDN. For a PDF open the file in a browser and choose Print > Save as
PDF. No PDF engine is bundled because the usual ones need system libraries that break on plain
Windows installs.

**Risk-free rate:** fixed by default (`RISK_FREE_RATE`). Opt in to the live US 13-week T-bill
yield with `RISK_FREE_SOURCE=live`: cached 12 hours, rounded to the nearest 0.25%, falls back to
the last known value and then to the fixed rate. The UI and report always say which was used. It
is a US rate applied to every currency, so for non-US stocks it is an approximation.

**Operator cap:** `APP_MAX_PATHS` lowers the most simulations a visitor can request.

## Deploying

```bat
docker build -t stock-analyzer .
docker run --rm -p 8501:8501 -v stock-analyzer-data:/data stock-analyzer
```

- [docs/HOSTING.md](docs/HOSTING.md): Streamlit Community Cloud, Render and Hugging Face Spaces,
  with trade-offs.
- [docs/SCALING_AND_LICENSING.md](docs/SCALING_AND_LICENSING.md): known limits, what to change as
  usage grows, why yfinance must be replaced before a public launch, and a provider sketch.
- GitHub Actions (`.github/workflows/ci.yml`) runs ruff and pytest on Python 3.11 and 3.12 for
  every push and pull request, and builds the Docker image.

## Architecture overview

```
 data/providers  ->  data/service  ->  analysis/pipeline  ->  models/forecast  ->  app/runner
 (yfinance, ...)     cache, limit,      technical, risk,       simulation,          one request
  swappable          retry, clean       fundamentals           scenarios            end to end
                                                                  |
                                backtest/ (walk-forward check) ---+--> reliability grade
                                                                  |
                       app/views + app/charts (Streamlit) and app/report (HTML export)
```

Pure functions do the maths; I/O lives in `data/service.py`, the `*_ticker` wrappers, the CLI and
the Streamlit pages. Results are frozen dataclasses. Everything user-visible comes from the same
functions the CLI uses, so the dashboard re-implements no logic.

## Developer guide

```bat
pip install -r requirements-dev.txt
python -m pytest                       :: about 280 tests; the Streamlit UI tests take ~1 minute
ruff check . && ruff format --check .
python -m streamlit run stock_analyzer/app/main.py
```

- Tests use a fake provider and clock (`tests/fakes.py`); no test touches the network. The
  recorded Yahoo payload is a small fixture of public company figures, used only for unit tests.
- Add a data provider: subclass `DataProvider`, follow the units contract in
  `data/providers/base.py`, register it in `data/providers/__init__.py`.
- Changing the model or its defaults changes cached backtests: bump `MODEL_VERSION` in
  `backtest/report.py`. Changing what a cached fundamentals value means: bump the `v2` in the key
  in `data/service.py`.
- Commit messages are conventional (`feat(scope): ...`). Never commit `.env`, caches or `.venv`.

## Layout

```
Dockerfile, .dockerignore, docker/    container image and healthcheck
.github/workflows/ci.yml              lint + tests + image build
docs/                                 HOSTING.md, SCALING_AND_LICENSING.md
stock_analyzer/
  app/               Streamlit dashboard (main.py, pages/, charts, presenters, safety, glossary,
                     report.py for the HTML export)
  config.py          settings from environment variables
  exceptions.py      errors with friendly user_message text
  logging_setup.py   structured (JSON) logging
  cli.py             smoke-test command
  analysis/
    technical.py     indicators, support/resistance, plain-English summary
    risk.py          volatility, drawdown, Sharpe/Sortino, beta, VaR
    benchmarks.py    benchmark list, suffix-based suggestion, currency helpers
    pipeline.py      MarketData -> AnalysisResult
    fundamentals.py  provider values -> labelled snapshot rows (missing is never zero)
  backtest/
    walkforward.py   no-look-ahead engine (rolling/expanding windows, vol remedies)
    distributions.py closed-form GBM, fast bootstrap (backtest-only), baselines
    metrics.py       coverage, Wilson, Kupiec, Christoffersen, PIT, pinball, interval score
    report.py        calibration report + JSON for the cache
    reliability.py   Good/Fair/Poor/Not enough history (thresholds in one place)
    service.py       caching and ticker wrapper
    basket.py        multi-ticker validation script
    tuning.py        held-out remedy experiment
  models/
    monte_carlo.py   calibration + GBM / bootstrap simulation, percentiles, probabilities
    scenarios.py     bull/base/bear from the simulation, with sensitivity ranges
    forecast.py      analysis -> simulations + scenarios
  data/
    providers/       DataProvider interface + yfinance implementation
    cache.py         shared SQLite cache (WAL, thread/process safe)
    cleaning.py      validation and quality report
    resilience.py    rate limiter + retry with backoff
    risk_free.py     fixed or (opt-in) live T-bill rate, cached, with fallbacks
    service.py       cache -> limit -> retry -> provider -> clean
    validation.py    ticker sanitising
  tests/
```

## Swapping the data provider

Subclass `DataProvider` (`data/providers/base.py`), register it in
`data/providers/__init__.py`, set `DATA_PROVIDER` and `DATA_PROVIDER_API_KEY` in `.env`.

## Known limits

* yfinance is unofficial and personal-use oriented: fine for development, not for a public launch
  (see docs/SCALING_AND_LICENSING.md).
* On free hosts the SQLite file may be wiped on restart; the cache simply rebuilds.
* Fundamentals are a snapshot from Yahoo: reported late, sometimes revised, and without sector
  averages. Dividend yield for a stock with an unusual quote unit may still be wrong; the
  provider drops values above 30%.
* The per-session throttle does not stop one person opening many tabs; use edge rate limiting.
