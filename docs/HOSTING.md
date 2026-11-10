# Hosting guide

How to put the dashboard online. Read **Before you go public** first: the data source matters
more than the host.

> Platform limits and prices below are from memory of each platform's documentation and change
> often. Treat every number as "roughly, check the current docs before relying on it". Anything
> marked **unverified** has not been tried by the project.

## Before you go public

1. **Data licence.** The default provider, yfinance, scrapes Yahoo Finance for personal use. A
   public site that shows its data may breach Yahoo's terms, and it can be blocked or break
   without notice. Fine for a private demo or classmates; before a real public launch, switch to
   a licensed provider (see [SCALING_AND_LICENSING.md](SCALING_AND_LICENSING.md)).
2. **Disclaimer.** It is built into every page. Do not remove it.
3. **Secrets.** Put API keys in the platform's secret/environment settings, never in the repo.
   `.env` is git-ignored and also excluded from the Docker image (`.dockerignore`).
4. **Memory.** One 50,000-path simulation needs about 100 MB. On a 512 MB host set
   `APP_MAX_PATHS=20000` so two visitors cannot exhaust it.
5. **Settings worth setting in production:** `LOG_JSON=true`, `APP_MAX_PATHS`, and optionally
   `RISK_FREE_SOURCE=live` (US 13-week T-bill, cached, falls back to `RISK_FREE_RATE`).

## Run it in Docker locally (Windows, Command Prompt)

Install Docker Desktop and start it, then from the project folder:

```bat
docker build -t stock-analyzer .
docker run --rm -p 8501:8501 -v stock-analyzer-data:/data stock-analyzer
```

Open http://localhost:8501. The named volume `stock-analyzer-data` keeps the SQLite cache
between runs. Pass settings with `-e`, for example `-e APP_MAX_PATHS=20000`, or `--env-file .env`
(Docker's env-file format does not strip quotes, so keep values unquoted).
The container runs as a non-root user and has a healthcheck on Streamlit's
`/_stcore/health`; check it with `docker ps` (look for "healthy").

> **Unverified here:** the image was written and its pieces were tested separately (healthcheck
> script against a live Streamlit, requirements resolution), but it has not been built end to
> end on the development machine because the Docker daemon was not running. The first CI run
> also builds it. If the build fails, send the error and it will be fixed.

## Option A: Streamlit Community Cloud (easiest, free)

1. Push the repo to GitHub.
2. At share.streamlit.io choose **New app**, pick the repo and branch, and set the main file to
   `stock_analyzer/app/main.py`.
3. Under **Advanced settings** choose Python 3.11 or 3.12 and paste secrets/settings as TOML,
   for example `APP_MAX_PATHS = "20000"` (top-level keys are exposed to the app as environment
   variables; check the current docs if a setting is not picked up).
4. Dependencies come from `requirements.txt` in the repo root.

Trade-offs: free and no server to manage; apps go to sleep after a period without visitors and
need a moment to wake (cold start, which re-installs nothing but re-imports everything);
limited RAM (about 1 GB class), so lower `APP_MAX_PATHS`; the disk is ephemeral (the cache
resets on restart, which is acceptable here); no custom domain on the free tier, only a
`*.streamlit.app` address; the code is usually public.

## Option B: Render (Docker web service)

1. New **Web Service**, connect the repo, runtime **Docker** (it finds the `Dockerfile`).
2. Set environment variables in the dashboard. Render injects `PORT`; the image already
   listens on `$PORT`.
3. Set the health check path to `/_stcore/health`.
4. For a persistent cache, attach a disk mounted at `/data` (paid plans only); otherwise the
   cache is rebuilt after each deploy or restart.

Trade-offs: free web services spin down after about 15 minutes idle and take up to a minute to
wake; free RAM is about 512 MB (set `APP_MAX_PATHS=20000`); custom domains with TLS are
supported; secrets live in the dashboard; one instance means one process, which suits this app
(the cache is a local SQLite file, see the scaling document before running several instances).

## Option C: Hugging Face Spaces (Docker Space)

1. Create a Space with SDK **Docker**. Use a separate Space repository (or branch) whose
   `README.md` begins with the YAML block Spaces needs:

   ```yaml
   ---
   title: Stock Analyzer
   sdk: docker
   app_port: 8501
   ---
   ```

2. Copy in the `Dockerfile`, `requirements.txt`, `.streamlit/`, `docker/` and `stock_analyzer/`.
3. Spaces run the container as a non-root user with its own ID, so `/data` may not be writable.
   Add a Space variable `CACHE_PATH=/tmp/market_cache.sqlite3` (the cache is then ephemeral).
4. Add keys under Settings > Variables and secrets.

Trade-offs: generous free CPU/RAM compared with the others; free Spaces sleep after a period of
inactivity; storage is ephemeral; custom domains are not available on the free tier;
**unverified:** exact limits and the user-ID behaviour, check the Spaces Docker documentation.

## Comparison

| | Streamlit Community Cloud | Render (free) | Hugging Face Spaces (free) |
|---|---|---|---|
| Setup effort | Lowest | Low | Low |
| Cold start | Yes (sleeps when idle) | Yes (about 15 min idle) | Yes (sleeps when idle) |
| RAM | ~1 GB class | ~512 MB | Highest of the three |
| Disk | Ephemeral | Ephemeral (disk on paid) | Ephemeral |
| Custom domain | No (free tier) | Yes | Not on free tier |
| Uses the Dockerfile | No | Yes | Yes |
| Secrets | App settings | Dashboard env vars | Space secrets |

## Concurrency, in one paragraph

Each visitor gets their own Streamlit session (own state, own request throttle). All sessions
in one process share one `MarketDataService` (single-flight fetches, token-bucket rate limit,
SQLite cache in WAL mode) and the in-memory analysis cache. Several visitors analysing the same
stock cost one provider call and one analysis. The per-session throttle does not stop one person
opening many tabs; put rate limiting at the edge (see the scaling document) before a real launch.
