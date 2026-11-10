# Scaling and licensing

This is an engineering note, not legal advice.

## 1. Licensing: replace yfinance before a public launch

yfinance is an unofficial scraper of Yahoo Finance. Yahoo's terms are written for personal use,
and the library can break or be throttled at any time. A public site, and certainly a
commercial one, needs a data feed whose licence allows what you do with it.

What to check in any provider's terms before choosing:

- **Display and redistribution.** Many free tiers allow personal use only. Showing prices,
  charts and derived statistics to the public may need a paid "display" licence.
- **Caching and storage.** Some licences limit how long you may store data (our SQLite cache
  stores it) or forbid building a database from it.
- **Derived data.** Whether forecasts, ranges and reports computed from the data can be shown
  and exported (the HTML report embeds prices).
- **Exchange fees.** Real-time or even delayed data from some exchanges carries separate fees.
  End-of-day data is usually cheaper and is all this tool needs.
- **Attribution and rate limits.** Quotas per minute and per day decide how many visitors you
  can serve (see section 2).

Candidates to evaluate: Twelve Data, Polygon, Financial Modeling Prep, Alpha Vantage, EODHD.
Coverage of non-US exchanges (UK, Europe, Japan) and the price of end-of-day history differ a
lot, so test the tickers you care about. Ask the provider in writing if your use is unclear.

## 2. Known limits today

| Limit | Why | Effect |
|---|---|---|
| Provider quotas | One provider key or scraper IP shared by all visitors | The service rate limiter (default 30 requests per minute) protects the provider but queues visitors; cached tickers cost nothing |
| Free-host CPU and RAM | A 50k-path run is about 100 MB; an uncached analysis (backtest and simulations) takes a few seconds of CPU | A few simultaneous cold analyses are fine; dozens are not. Use `APP_MAX_PATHS` |
| Single-process SQLite | WAL mode allows many readers and one writer; the file is local to one machine | Fine for one container; replicas would not share a cache |
| In-memory analysis cache | `st.cache_data` is per process (32 entries, 1 hour) | Restarts empty it; replicas do not share it |
| Per-session throttle | Keyed to one session, not to a person or IP | One person can open many tabs. Needs edge rate limiting |
| Ephemeral disks | Free hosts reset files on restart | The cache rebuilds; only costs time and provider calls |
| Backtest cost | Walk-forward over about 10 years; bootstrap costs more than GBM | Cached 7 days per ticker and last price date |

## 3. What to change as usage grows

In rough order of when you will need each:

1. **Licensed data feed** (section 4). Required before launch.
2. **Edge rate limiting.** Put a reverse proxy or CDN (Cloudflare, nginx, or the host's
   built-in) in front and limit requests per IP. This is the real answer to abuse.
3. **A shared cache.** Replace `SQLiteCache` with Postgres or Redis behind the same interface
   (`get`, `set`, `delete`, `purge_older_than`, `is_fresh`; values are opaque strings, so it is a
   small adapter). That allows several replicas and survives restarts.
4. **Background workers for the expensive part.** Precompute backtests for popular tickers on a
   schedule and store them under the existing backtest cache key (it already includes model
   version, configuration and last price date), so page loads only read.
5. **More than one replica.** Streamlit sessions use a websocket to one process; use sticky
   sessions at the load balancer. Items 3 and 4 must come first.
6. **Observability without personal data.** Request counts, latency, error rates and cache hit
   rates. Never log who looked at what. Any analytics must be opt-in and anonymous.
7. **Accounts and watchlists.** Out of scope for the MVP; the session state is already kept
   separate from shared state, so accounts can be added later without changing the model code.

## 4. Sketch: a `TwelveDataProvider`

A sketch, **not shipped and not tested**. The endpoint and field names are from memory of the
provider's documentation: verify them, and the licence, before using it. It shows how little of
the app changes: implement `DataProvider`, map failures to our exceptions, register it, set
`DATA_PROVIDER=twelvedata` and `DATA_PROVIDER_API_KEY`.

```python
# stock_analyzer/data/providers/twelvedata_provider.py  (sketch)
from __future__ import annotations

from datetime import date

import pandas as pd
import requests  # add to requirements.txt if you ship this

from stock_analyzer.data.providers.base import FUNDAMENTAL_FIELDS, DataProvider
from stock_analyzer.data.types import RawHistory
from stock_analyzer.exceptions import (
    ProviderRateLimitError,
    ProviderUnavailableError,
    TickerNotFoundError,
)

BASE = "https://api.twelvedata.com"


class TwelveDataProvider(DataProvider):
    name = "twelvedata"

    def __init__(self, api_key: str, session: requests.Session | None = None) -> None:
        self._key, self._http = api_key, session or requests.Session()

    def _get(self, path: str, **params) -> dict:
        try:
            r = self._http.get(f"{BASE}/{path}", params={**params, "apikey": self._key}, timeout=15)
        except requests.RequestException as exc:
            raise ProviderUnavailableError(f"twelvedata: {exc}") from exc
        data = r.json()
        code = data.get("code") if isinstance(data, dict) else None
        if r.status_code == 429 or code == 429:
            raise ProviderRateLimitError("twelvedata rate limit")
        if code in (400, 404):  # unknown symbol
            raise TickerNotFoundError(data.get("message", "unknown symbol"))
        if r.status_code >= 500 or (code and code >= 500):
            raise ProviderUnavailableError(f"twelvedata HTTP {r.status_code}")
        return data

    def _series(
        self, ticker: str, start: date, end: date | None, adjust: str
    ) -> tuple[pd.DataFrame, str | None]:
        data = self._get(
            "time_series",
            symbol=ticker,
            interval="1day",
            adjust=adjust,
            start_date=start.isoformat(),
            end_date=end.isoformat() if end else None,
            outputsize=5000,
            order="ASC",
        )
        df = pd.DataFrame(data["values"]).astype(
            {"open": float, "high": float, "low": float, "close": float, "volume": float}
        )
        df.index = pd.to_datetime(df.pop("datetime"))
        return df, data.get("meta", {}).get("currency")

    def get_history(self, ticker: str, start: date, end: date | None = None) -> RawHistory:
        raw, currency = self._series(ticker, start, end, "none")  # as-traded prices
        adj, _ = self._series(ticker, start, end, "all")  # split+dividend adjusted
        out = pd.DataFrame(
            {
                "Open": raw.open,
                "High": raw.high,
                "Low": raw.low,
                "Close": raw.close,
                "Adj Close": adj.close,
                "Volume": raw.volume,
            }
        )
        return RawHistory(ticker=ticker, prices=out, currency=currency, source=self.name)

    def get_fundamentals(self, ticker: str) -> dict:
        # Fundamentals are usually a higher-priced endpoint. Returning all-None is valid:
        # the app then says "not available" instead of inventing numbers.
        return dict.fromkeys(FUNDAMENTAL_FIELDS)
```

Then register it in `providers/__init__.py` (`if settings.provider == "twelvedata": ...` using
`settings.provider_api_key`). If you supply fundamentals, convert to the units contract in
`providers/base.py` (fractions and ratios, market cap in major currency units) and add a recorded
sample payload test like `tests/fixtures/yahoo_info_sample.json`.
