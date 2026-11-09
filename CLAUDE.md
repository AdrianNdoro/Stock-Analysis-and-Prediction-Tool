# PROJECT HANDOFF: Stock Analyzer (continue from Phase 4)

You are taking over a partly built project. Phases 1-3 are finished, tested and working.
Your job is to complete Phases 4-7 **one phase at a time**, stopping for my approval after each.
Read this whole document first, then inspect the repo before changing anything.

---

## 0. FIRST ACTIONS (do these before writing any code)

1. Read the repo: `README.md`, `requirements*.txt`, `.env.example`, `pytest.ini`, `ruff.toml`, and skim every module under `stock_analyzer/`.
2. Run `python -m pytest` and `ruff check .`. Expected: **152 tests pass, ruff clean**. If not, tell me exactly what differs before doing anything else.
3. Run `git status`. If the folder is not a git repo, propose `git init` plus a sensible `.gitignore` check (one already exists) and make an initial commit labelled "Phases 1-3 baseline" once I agree.
4. Reply with: (a) a 10-line summary of your understanding of the codebase, (b) the discrepancies or risks you found, (c) your concrete plan for Phase 4, and (d) at most 5 questions. **Do not start Phase 4 code until I answer.**

---

## 1. ROLE AND WORKING AGREEMENT

Act as a senior quantitative developer and software architect, and as a collaborator, not an order-taker.

- Explain key decisions briefly, flag risks, and **push back** if something I ask is statistically unsound, misleading, or unsafe to ship to other users.
- Build in phases. After each phase **stop and wait for my approval**. For every phase give me: the file tree changes, complete working code, how to run and test it, and what I should check to confirm it works.
- Write unit tests for the maths and for edge cases (bad tickers, short histories, empty data). Explain the statistical assumptions behind each model in 2-3 sentences, including where it breaks (fat tails, volatility clustering, regime changes, low liquidity).
- Keep code simple and readable over clever. Prefer small pure functions; keep I/O at the edges.
- Commit at the end of each phase (conventional message, e.g. `feat(backtest): walk-forward calibration report`) once tests and lint pass. Never commit `.env`, caches, or `.venv`.

### Honesty rules (non-negotiable)
- Never present simulated outcomes as predictions or guarantees.
- If a feature cannot be done reliably or legally for a public tool (e.g. data licensing), say so and propose the best workaround.
- If I ask for something that would produce misleading results (e.g. overfitting a model to look accurate), push back and explain why.

---

## 2. ABOUT ME AND MY ENVIRONMENT

- 21-year-old computer science student, comfortable with Python, interested in data/AI and investing.
- **Develop on Windows.** Every local command you give me must work in **Command Prompt** (and ideally PowerShell). Use `py -3.11 -m venv .venv`, `.venv\Scripts\activate`, `copy`, backslash paths. Deployment can target Linux (Docker or a cloud host).
- Python 3.11+ (code is written for 3.11; `datetime.UTC` is used). Virtual environment via `venv`, not conda.
- Heavy dependencies must have Windows wheels; avoid anything needing a C compiler or system libraries on Windows.

---

## 3. PROJECT GOAL AND SCOPE

A tool that **other people can use**. A user enters a stock ticker and gets:
1. A structured analysis of the stock's past and current condition (technical, fundamental, risk).
2. **Probabilistic possible outcomes**: bull/base/bear scenarios plus a simulated price distribution over a chosen horizon.

The tool must NOT claim to predict the future. It shows scenarios with probabilities, confidence ranges, assumptions, and clearly shows model limits.

**In scope now:** global stocks and ETFs available from standard providers (US-listed like AAPL, NVDA, MSFT; major UK, European and Asian exchanges the provider supports).
**Out of scope for now:** the Nairobi Securities Exchange and any market with patchy free data. Do not build NSE-specific logic or CSV-import workarounds. Keep the data layer extensible so another market can be added later without rewrites.
**Currency:** show prices in the stock's native currency, label it, and never mix currencies in comparisons or benchmarks.

**Multi-user requirements (design from day one):** no user accounts in the MVP (stateless per session, but leave room for accounts/watchlists later); concurrency-safe shared cache; input validation and limits (max paths, allowed horizons, request frequency) with friendly errors; vectorised NumPy and cached expensive work; no secrets in code (`.env` locally, platform secrets in production); no personal data logged or stored (analytics later must be opt-in and anonymous); a permanent visible disclaimer (educational tool, not financial advice, past performance does not guarantee future results) plus a "How this works and its limitations" page; beginner-friendly UX with tooltips for jargon (Sharpe ratio, 5th percentile, etc.).

**Tech stack:** Python 3.11+, pandas, numpy, scipy, statsmodels, scikit-learn, yfinance, plotly, streamlit, sqlite3 or parquet, pytest, ruff, Docker. Suggest alternatives only with a strong reason. (Currently installed: pandas, numpy, yfinance, python-dotenv, pytest, ruff. Add the rest only when a phase needs them, and pin sensibly.)

---

## 4. CURRENT STATE: PHASES 1-3 ARE DONE

### 4.1 Repository layout (repo root = folder containing `requirements.txt`)

```
stock-analyzer/
  .env.example  .gitignore  pytest.ini  ruff.toml  README.md
  requirements.txt  requirements-dev.txt
  stock_analyzer/
    __init__.py  config.py  exceptions.py  logging_setup.py  cli.py
    data/
      providers/  base.py  yfinance_provider.py  __init__.py   (registry: get_provider)
      cache.py  cleaning.py  resilience.py  service.py  types.py  validation.py
    analysis/
      technical.py  risk.py  benchmarks.py  pipeline.py
    models/
      monte_carlo.py  scenarios.py  forecast.py
    tests/   (fakes.py + 15 test modules, 152 tests)
```
Config in `pytest.ini` has `pythonpath = .`; run tests from the repo root with `python -m pytest`. Ruff: line length 100, target py311, rules E,F,I,B,UP (`ruff check .` and `ruff format .`).

### 4.2 Phase 1: data layer (done)
- `DataProvider` ABC (`data/providers/base.py`): `get_history(ticker, start, end=None) -> RawHistory` (DataFrame with `REQUIRED_COLUMNS` = Open, High, Low, Close, Adj Close, Volume; optional Dividends, Stock Splits; plus `currency`, `source`) and `get_fundamentals(ticker) -> dict` containing every key in `FUNDAMENTAL_FIELDS` (None if unknown). Providers map failures to our exceptions (`TickerNotFoundError`, `ProviderRateLimitError`, `ProviderUnavailableError`). `YFinanceProvider` implements it (injectable `ticker_factory` for tests). **Risk to keep flagging:** yfinance is unofficial and personal-use oriented; a public launch needs a licensed API (Twelve Data, Polygon, FMP, Alpha Vantage) behind the same interface, key in `DATA_PROVIDER_API_KEY`.
- `SQLiteCache` (`data/cache.py`): WAL mode, new connection per operation, atomic upsert, `get/set/delete/purge_older_than/is_fresh`, values are opaque strings, injectable clock.
- `resilience.py`: thread-safe token-bucket `RateLimiter`; `call_with_retry` (exponential backoff + jitter, honours `retry_after`, retries only rate-limit/unavailable errors).
- `cleaning.clean_prices(raw, settings) -> (DataFrame, DataQualityReport)`: tz-naive normalised dates (keeps exchange-local trading date), dedupe, sort, drops invalid prices and inconsistent OHLC, reports (never fills) gaps > 7 days, flags (keeps) one-day moves beyond ±40% log return, warns on zero volume, warns under 504 trading days, refuses under 60.
- `MarketDataService` (`data/service.py`): cache -> rate limit -> retry -> provider -> clean -> cache. Single-flight per key (many users, one fetch), negative cache for unknown tickers (60 min), serves stale cache with a visible warning when the provider is down, degrades if the cache is broken or corrupt, warns about unknown currency and sub-unit currencies (GBp/GBX pence, ZAc, ILA), warns on stale last bar. `get_market_data(ticker, allow_index=False) -> MarketData`; `get_fundamentals(ticker) -> Fundamentals` (never raises for provider trouble). `build_default_service(settings)`.
- `normalize_ticker`: strict allow-list regex (letters, digits, `.`, `-`; leading `^` only for benchmark indices), max 16 chars, ASCII only.
- `exceptions.py`: every user-triggerable error has a friendly `user_message`; technical detail stays in `str(exc)`.
- `config.py`: frozen `Settings` loaded from env (`Settings.from_env`, `get_settings()`), validated with `ConfigError`. Env vars: `DATA_PROVIDER, DATA_PROVIDER_API_KEY, CACHE_PATH, CACHE_TTL_PRICES_HOURS, CACHE_TTL_FUNDAMENTALS_HOURS, CACHE_TTL_MISSING_MINUTES, HISTORY_YEARS, RATE_LIMIT_PER_MINUTE, RATE_LIMIT_BURST, MAX_RETRIES, BACKOFF_BASE_SECONDS, MIN_HISTORY_DAYS, RECOMMENDED_HISTORY_DAYS, STALE_BAR_DAYS, RISK_FREE_RATE, RISK_LOOKBACK_YEARS, DEFAULT_BENCHMARK, MC_PATHS, MC_LOOKBACK_YEARS, MC_DRIFT_MODE, LOG_LEVEL, LOG_JSON`.
- Structured logging (`logging_setup.py`): JSON or readable text; log tickers, counts and timings only, never personal data.

### 4.3 Phase 2: technical analysis and risk (done)
- `analysis/technical.py`: `adjusted_ohlc` (scales O/H/L by Adj Close / Close so splits and dividends do not create fake cliffs), `sma, ema` (EMA seeded with first price; differs slightly from SMA-seeded charting packages early on), `rsi` (Wilder, SMA-seeded), `macd`, `bollinger` (population std), `true_range, atr` (Wilder), `compute_indicators` (SMA/EMA 20/50/200, RSI14, MACD, Bollinger, ATR14, volume averages), `find_levels` (swing-point clustering for support/resistance, 1.5% tolerance), `summarize_technicals -> TechnicalSnapshot` (plain-English bullets plus a +/-1 score over up to four trend/momentum signals with labels Bullish / Leaning bullish / Mixed / Leaning bearish / Bearish).
- `analysis/risk.py`: simple daily returns; `annualised_volatility` (x sqrt 252), `cagr`, `max_drawdown -> DrawdownResult`, `sharpe_ratio`, `sortino_ratio` (MAR = daily risk-free), `compute_beta` (weekly returns by default, 3-year lookback, local-currency returns, min 52 obs, R-squared reported), `value_at_risk` (historical, parametric normal, plus Expected Shortfall; sqrt-time scaling for multi-day), `compute_risk -> RiskMetrics`, `summarize_risk` (plain English).
- `analysis/benchmarks.py`: `BENCHMARKS` (SP500 `^GSPC`, FTSE100 `^FTSE`, EUROSTOXX50 `^STOXX50E`, NIKKEI225 `^N225`, HANGSENG `^HSI`, MSCIWORLD `URTH`), `get_benchmark`, `suggest_benchmark(ticker)` by exchange suffix (default S&P 500), `major_currency` (GBp->GBP etc.).
- `analysis/pipeline.py`: pure `analyze(stock, benchmark_data, benchmark, risk_free_rate, lookback_years) -> AnalysisResult` and IO wrapper `analyze_ticker(service, ticker, benchmark=None, settings=None)`. `AnalysisResult` carries `technical, risk, risk_bullets, benchmark, adj_close, warnings`. It warns on currency mismatch (after normalising pence to pounds), on a non-ideal benchmark, and drops beta (with a warning) if the benchmark fetch fails.

### 4.4 Phase 3: Monte Carlo and scenarios (done)
- `models/monte_carlo.py`: `HORIZONS = {1M:21, 3M:63, 6M:126, 1Y:252}`, `MIN_PATHS=10_000`, `MAX_PATHS=50_000` (50k x 252 float64 is ~100 MB), percentiles 5/25/50/75/95. `calibrate(close, risk_free_rate, lookback_years, drift_mode, ...) -> Calibration` estimates volatility from log returns and drift from the mean simple return. **Drift decision:** default `shrunk` = 50/50 blend of the historical mean and a prior (risk-free + 5% equity premium), because the historical mean's standard error (sigma/sqrt(years), often +/-15%/yr) is too large to trust; alternatives `historical` and `neutral` (risk-free). Expected return is capped at +/-50%/yr. Warnings for short windows, extreme drift, and fat tails (excess kurtosis > 3). `simulate(calibration, s0, horizon_days, method='gbm'|'bootstrap', n_paths, seed, sigma_scale, ...) -> SimulationResult`: fully vectorised, **mean-matched** (both methods have the same expected one-day simple return, so they differ in shape not centre), optional seed, bands for a fan chart, a few sample paths, terminal prices. Methods: `prob_above_current`, `prob_gain_more_than(x)`, `prob_loss_more_than(x)`, `expected_range`, `summary_table`, `plain_english`.
- `models/scenarios.py`: bull/base/bear derived from the simulation (base = within +/-0.5 x sigma*sqrt(T) of today, minimum +/-3%). Probability = share of paths in each region; price range = 25th-75th percentile inside the region; target = region median; **sensitivity range** from re-running with volatility x0.8 and x1.2. **Deliberate decision:** indicator state is shown as labelled context ("consistent with / at odds with this scenario") and does NOT tilt probabilities, because nothing yet shows indicators add skill. Phase 4 results decide whether an opt-in tilt is ever justified.
- `models/forecast.py`: pure `forecast(analysis, horizons, method, n_paths, seed, drift_mode, lookback_years) -> ForecastResult` and IO wrapper `forecast_ticker(service, ticker, horizons, ...)`. All runs for one horizon share the seed (common random numbers).
- `cli.py`: `python -m stock_analyzer.cli TICKER [--fundamentals] [--analyze] [--simulate --horizon 1M|3M|6M|1Y|ALL --method gbm|bootstrap --drift shrunk|historical|neutral --paths N --seed N] [--benchmark KEY_OR_TICKER]`.

### 4.5 Conventions already established (keep following them)
- `from __future__ import annotations`; type hints everywhere; docstrings that state assumptions and failure modes; frozen dataclasses for results (use `eq=False` when they hold numpy arrays).
- Pure functions for maths; I/O only in `service.py`, `*_ticker` wrappers, and the CLI.
- Use `Adj Close` (via `adjusted_ohlc`) for returns and indicators; `Close` is the as-traded price. Simple returns for risk metrics, log returns for simulation calibration. 252 trading days per year.
- Never fill missing prices; report them. Never mix currencies silently.
- Tests: deterministic seeds, hand-computed or independently implemented reference values, statistical checks against analytic theory with explicit tolerances, a fake provider/clock in `tests/fakes.py` (`FakeProvider`, `FakeClock`, `make_prices`) and `MultiProvider` in `tests/test_benchmarks_and_pipeline.py`. No network in tests.
- Plain-English text is hedged and states limits; no language that implies prediction.

### 4.6 Known caveats and open items
- Live yfinance behaviour was never exercised by the automated tests (no network when the code was written). Treat the live provider as "works on my machine" until I confirm; the error mapping in `yfinance_provider.py` is heuristic.
- Fundamentals unit normalisation (Yahoo `dividendYield`, `debtToEquity` vary by version) is an explicit TODO for Phase 6.
- Risk-free rate is a configurable constant (default 4%). **Planned:** an opt-in live T-bill fetch, cached, with fallback to the constant.
- Free hosts may have ephemeral disk, so the SQLite cache can reset; acceptable for the MVP.
- Weekly beta uses local-currency returns; currency effects are not included (warned in the output).
- GBM understates fat tails and ignores volatility clustering; bootstrap keeps fat tails but treats days as independent.

---

## 5. REMAINING PHASES (stop for my approval after each)

### Phase 4: Backtesting and model validation (NEXT)
Goal: measure whether the forecasting approach is honest, and tell users how far to trust it.

Build `stock_analyzer/backtest/`:
1. **Walk-forward engine** (no look-ahead, no random shuffles). For each origin date t (stepping e.g. every 21 trading days across the available history), calibrate **only on data up to t**, forecast each horizon, then compare with the realised price at t+h. Support rolling or expanding calibration windows. Include a test proving that changing prices after t cannot change the forecast made at t.
2. **Speed:** a backtest needs hundreds of forecasts. For GBM use closed-form lognormal quantiles (and verify them against the simulator in a test); for bootstrap allow fewer paths via an internal, clearly named option that bypasses `MIN_PATHS` only inside the backtest module. Do not weaken the public limits.
3. **Calibration report:** for 50% (p25-p75) and 90% (p5-p95) intervals, the realised coverage with a binomial confidence interval; probability-integral-transform (PIT) histogram and a uniformity check; pinball (quantile) loss and interval score; direction hit-rate versus a naive baseline. Handle **overlapping windows** honestly: consecutive origins are autocorrelated, so report an effective sample size or use non-overlapping origins for the formal tests. Add Kupiec (unconditional coverage) and, if time allows, Christoffersen (independence) tests.
4. **Baselines to beat:** zero-drift random walk, and an empirical quantile forecast from the stock's own past h-day returns. Report honestly when the model does not beat them.
5. **Model-reliability note** in plain English for non-experts (for example: "Good / Fair / Poor / Not enough history", with the numbers behind it: "the model's 90% range contained the real price 81% of the time over N checks, so treat its ranges as too narrow"). Define the grading thresholds in one place and document them. Wire it into `ForecastResult` (optional field) so the UI can show it.
6. **Findings-driven fixes only if statistically justified:** if coverage is systematically too low (very likely for GBM), propose and test a principled remedy (for example a volatility inflation factor, EWMA/GARCH-style volatility, or defaulting to bootstrap) and report the before/after calibration. Do not tune parameters until the report "looks good": use a held-out period or multiple tickers and say how it generalises.
7. A script/CLI: `python -m stock_analyzer.cli TICKER --backtest [--horizon 3M]` printing the report, plus a multi-ticker validation script over a small basket (e.g. AAPL, MSFT, NVDA, KO, BP.L, SAP.DE) that I can run locally.
8. Cache backtest results (they are expensive) in the existing SQLite cache with a sensible TTL and a key that includes model version and parameters.
9. Tests: synthetic data where the true distribution is known so a *correct* model shows ~nominal coverage and a deliberately mis-specified one (too-low volatility) is detected; short-history handling; no look-ahead; edge cases.

Explain the assumptions and failure modes of the validation itself (regime changes, overlapping windows, survivorship bias in my ticker basket, look-ahead in choosing the basket).

### Phase 5: Streamlit dashboard, input safety, beginner-friendly explanations
- `stock_analyzer/app/` Streamlit app: ticker input, horizon selector (1M/3M/6M/1Y), benchmark selector (default suggested by suffix), an "Advanced" expander (method, drift, paths within limits, seed), interactive Plotly **price chart with indicators** (candlestick or line, SMA/EMA toggles, Bollinger, volume, RSI and MACD sub-panels), a **fan chart** of the Monte Carlo bands with sample paths, a terminal-price histogram, a **scenario table** (probabilities with sensitivity ranges, price ranges, assumptions, indicator context), a **risk metrics panel**, and a **plain-English summary** (reuse the existing `plain_english`/bullet functions; do not re-write the logic in the UI).
- Show the Phase 4 **model reliability note** prominently next to the forecast.
- **Safety and limits:** ticker sanitising (reuse `normalize_ticker`), allowed horizons only, path limits, per-session request throttling, friendly errors from `user_message`, never show stack traces. Concurrency: one `MarketDataService` per process via `st.cache_resource`; `st.cache_data` for expensive analysis keyed by (ticker, horizon, method, drift, paths, seed, data date); no shared mutable state between sessions.
- **Beginner-friendly UX:** a glossary module and tooltips (Sharpe, Sortino, beta, VaR, drawdown, percentiles, RSI, MACD, Bollinger, ATR, support/resistance); short plain-language captions under each chart; mobile-friendly layout where Streamlit allows (single column fallback, no wide fixed-size elements).
- **Permanent visible disclaimer** and a **"How this works and its limitations"** page (assumptions, fat tails, volatility clustering, regime changes, low liquidity, data-provider caveats, what the backtest shows).
- **Privacy:** no personal data logged or stored; no analytics (any later analytics opt-in and anonymous).
- Tests: Streamlit `AppTest` smoke tests (valid ticker, invalid ticker, provider outage, short history), plus unit tests for any chart-building helpers (pure functions returning Plotly figures).

### Phase 6: Deployment, fundamentals, report export (and risk-free rate)
- **Fundamentals snapshot:** P/E, P/B, EPS growth, revenue growth, debt/equity, dividend yield, market cap; compare against sector averages **only if the provider supplies them**; degrade gracefully and say what is missing. Normalise units (Yahoo `dividendYield`, `debtToEquity`) with tests pinned to a recorded sample payload. Never present missing data as zero.
- **Report export:** a self-contained **HTML** report (charts embedded, disclaimer included, assumptions and reliability note included); PDF optional (only if it works on Windows and Linux without heavy system dependencies, otherwise document the "print to PDF" route).
- **Opt-in live T-bill rate** for the risk-free rate (cached, short timeout, fallback to the configured constant, clearly labelled in the UI and report).
- **Docker:** `Dockerfile` (slim Python 3.11+, non-root user, healthcheck, cache directory as a volume), `.dockerignore`, `docker run` instructions that also work from Windows.
- **CI:** GitHub Actions workflow running ruff and pytest on every push and pull request (Python 3.11 and 3.12), with pip caching.
- **Hosting guide:** deployment instructions for at least Streamlit Community Cloud, Render and Hugging Face Spaces, with the trade-offs (cold starts, memory limits, ephemeral disk, secrets handling, custom domain, concurrency). Platform secrets for any API keys; `.env.example` kept current.
- **Scaling and licensing document:** known limits (provider quotas, free-tier CPU/RAM, single-process SQLite) and what to change if usage grows (licensed data feed, Postgres/Redis cache, background workers, rate limiting at the edge). Be explicit that yfinance must be replaced before a public commercial launch, and sketch a `TwelveDataProvider` (or similar) implementing `DataProvider`.
- README: Windows setup, user-facing "how to use this tool", developer guide, architecture overview.

### Phase 7 (optional): ML direction model and sentiment layer
- **ML direction model** (for example gradient boosting) for the probability that price is up over the horizon. Hard conditions: proper **walk-forward validation** (no shuffling, no leakage, features computed only from data available at the time, purged/embargoed splits for overlapping labels), compare against the naive baseline, report **honestly if it does not beat it** (and then do not ship it as a feature). Show it as a separate, clearly labelled panel with its own reliability metrics, never blended silently into the Monte Carlo numbers.
- **Sentiment layer:** pull recent news headlines for the ticker and score sentiment; display it as a **separate input**, never blended into the numbers. Check data licensing and terms for the news source; cache aggressively; handle missing news gracefully.

---

## 6. QUALITY BAR FOR EVERY PHASE
- All tests pass and `ruff check .` is clean before you hand a phase back. Report the test count.
- New maths gets tests against hand-computed values or independent reference implementations, plus statistical tests with explicit tolerances.
- New external-facing text is plain English, hedged, and consistent with the honesty rules.
- Update `README.md` and `.env.example` when behaviour or settings change. Keep Windows instructions accurate.
- Do not silently change the public behaviour or defaults of Phases 1-3. If a fix is needed there, call it out, explain why, and add a regression test.
- At the end of each phase give me: (1) file tree changes, (2) how to run and test, (3) what to check to confirm it works, (4) the statistical assumptions and where they break, (5) risks and open questions, (6) your recommendation for the next phase.

Start with Section 0 now.
