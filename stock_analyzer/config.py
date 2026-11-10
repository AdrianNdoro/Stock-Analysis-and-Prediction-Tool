"""Central configuration. All values come from environment variables (``.env`` locally,
platform secrets in production). No secrets live in code."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from stock_analyzer.exceptions import ConfigError


def _num(
    env: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum: float = 0.0,
    maximum: float = float("inf"),
) -> float:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    if value > maximum:
        raise ConfigError(f"{name} must be <= {maximum}, got {value}")
    return value


def _choice(env: Mapping[str, str], name: str, default: str, allowed: tuple[str, ...]) -> str:
    raw = env.get(name, "").strip().lower()
    if not raw:
        return default
    if raw not in allowed:
        raise ConfigError(f"{name} must be one of {allowed}, got {raw!r}")
    return raw


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be true/false, got {raw!r}")


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    provider: str = "yfinance"
    provider_api_key: str | None = None

    cache_path: Path = Path("data_cache/market_cache.sqlite3")
    cache_ttl_prices_hours: float = 24.0
    cache_ttl_fundamentals_hours: float = 24.0
    cache_ttl_missing_minutes: float = 60.0  # remember "ticker not found" to protect the provider

    history_years: int = 10
    rate_limit_per_minute: float = 30.0
    rate_limit_burst: int = 5
    max_retries: int = 3
    backoff_base_seconds: float = 1.0

    min_history_days: int = 60  # hard minimum: below this we refuse to analyse
    recommended_history_days: int = 504  # ~2 trading years: below this we warn
    stale_bar_days: int = 10  # warn if the newest price bar is older than this

    risk_free_rate: float = 0.04  # annual fraction; the fixed value, and the live fallback
    risk_free_source: str = "fixed"  # fixed | live (live = US 13-week T-bill, opt-in)
    risk_lookback_years: float = 3.0  # window for volatility, Sharpe, Sortino, beta and VaR
    default_benchmark: str = "^GSPC"  # S&P 500

    mc_paths: int = 10_000  # simulated paths per run (allowed range 10,000 - 50,000)
    app_max_paths: int = 50_000  # dashboard ceiling; lower it on small hosts (100 MB per 50k run)
    mc_lookback_years: float = 3.0  # history used to calibrate volatility and drift
    mc_drift_mode: str = "shrunk"  # shrunk | historical | neutral

    log_level: str = "INFO"
    log_json: bool = True

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, *, load_dotenv_file: bool = True):
        """Build settings from ``env`` (defaults to ``os.environ`` after loading ``.env``)."""
        if env is None:
            if load_dotenv_file:
                from dotenv import load_dotenv

                load_dotenv(override=False)
            env = os.environ
        d = cls()
        return cls(
            provider=(env.get("DATA_PROVIDER", "").strip() or d.provider).lower(),
            provider_api_key=env.get("DATA_PROVIDER_API_KEY", "").strip() or None,
            cache_path=Path(env.get("CACHE_PATH", "").strip() or d.cache_path),
            cache_ttl_prices_hours=_num(
                env, "CACHE_TTL_PRICES_HOURS", d.cache_ttl_prices_hours, minimum=0.01
            ),
            cache_ttl_fundamentals_hours=_num(
                env, "CACHE_TTL_FUNDAMENTALS_HOURS", d.cache_ttl_fundamentals_hours, minimum=0.01
            ),
            cache_ttl_missing_minutes=_num(
                env, "CACHE_TTL_MISSING_MINUTES", d.cache_ttl_missing_minutes, minimum=0.0
            ),
            history_years=int(_num(env, "HISTORY_YEARS", d.history_years, minimum=1)),
            rate_limit_per_minute=_num(
                env, "RATE_LIMIT_PER_MINUTE", d.rate_limit_per_minute, minimum=1
            ),
            rate_limit_burst=int(_num(env, "RATE_LIMIT_BURST", d.rate_limit_burst, minimum=1)),
            max_retries=int(_num(env, "MAX_RETRIES", d.max_retries, minimum=0)),
            backoff_base_seconds=_num(
                env, "BACKOFF_BASE_SECONDS", d.backoff_base_seconds, minimum=0.0
            ),
            min_history_days=int(_num(env, "MIN_HISTORY_DAYS", d.min_history_days, minimum=10)),
            recommended_history_days=int(
                _num(env, "RECOMMENDED_HISTORY_DAYS", d.recommended_history_days, minimum=10)
            ),
            stale_bar_days=int(_num(env, "STALE_BAR_DAYS", d.stale_bar_days, minimum=1)),
            risk_free_rate=_num(
                env, "RISK_FREE_RATE", d.risk_free_rate, minimum=-0.05, maximum=0.5
            ),
            risk_free_source=_choice(
                env, "RISK_FREE_SOURCE", d.risk_free_source, ("fixed", "live")
            ),
            risk_lookback_years=_num(
                env, "RISK_LOOKBACK_YEARS", d.risk_lookback_years, minimum=1.0, maximum=10.0
            ),
            default_benchmark=(
                env.get("DEFAULT_BENCHMARK", "").strip() or d.default_benchmark
            ).upper(),
            mc_paths=int(_num(env, "MC_PATHS", d.mc_paths, minimum=10_000, maximum=50_000)),
            app_max_paths=int(
                _num(env, "APP_MAX_PATHS", d.app_max_paths, minimum=10_000, maximum=50_000)
            ),
            mc_lookback_years=_num(
                env, "MC_LOOKBACK_YEARS", d.mc_lookback_years, minimum=1.0, maximum=10.0
            ),
            mc_drift_mode=_choice(
                env, "MC_DRIFT_MODE", d.mc_drift_mode, ("shrunk", "historical", "neutral")
            ),
            log_level=(env.get("LOG_LEVEL", "").strip() or d.log_level).upper(),
            log_json=_bool(env, "LOG_JSON", d.log_json),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, loaded once."""
    return Settings.from_env()
