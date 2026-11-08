import pytest

from stock_analyzer.config import Settings
from stock_analyzer.exceptions import ConfigError


def test_defaults():
    s = Settings.from_env({})
    assert s.provider == "yfinance" and s.provider_api_key is None
    assert s.cache_ttl_prices_hours == 24 and s.min_history_days == 60 and s.log_json is True


def test_env_overrides_and_secrets_come_from_env():
    s = Settings.from_env(
        {
            "DATA_PROVIDER": "TwelveData",
            "DATA_PROVIDER_API_KEY": "abc",
            "HISTORY_YEARS": "5",
            "LOG_JSON": "false",
            "CACHE_PATH": "x/y.sqlite3",
        }
    )
    assert s.provider == "twelvedata" and s.provider_api_key == "abc"
    assert s.history_years == 5 and s.log_json is False and str(s.cache_path).endswith("y.sqlite3")


@pytest.mark.parametrize(
    "env",
    [
        {"HISTORY_YEARS": "ten"},
        {"RATE_LIMIT_PER_MINUTE": "0"},
        {"LOG_JSON": "maybe"},
        {"MAX_RETRIES": "-1"},
    ],
)
def test_bad_values_raise_config_error(env):
    with pytest.raises(ConfigError):
        Settings.from_env(env)


def test_risk_settings_defaults_and_bounds():
    s = Settings.from_env({})
    assert (
        s.risk_free_rate == 0.04 and s.risk_lookback_years == 3 and s.default_benchmark == "^GSPC"
    )
    assert Settings.from_env({"RISK_FREE_RATE": "0.0275"}).risk_free_rate == 0.0275
    for bad in ({"RISK_FREE_RATE": "4"}, {"RISK_LOOKBACK_YEARS": "0.5"}):
        with pytest.raises(ConfigError):
            Settings.from_env(bad)


def test_monte_carlo_settings_and_limits():
    s = Settings.from_env({})
    assert (s.mc_paths, s.mc_lookback_years, s.mc_drift_mode) == (10_000, 3.0, "shrunk")
    assert Settings.from_env({"MC_DRIFT_MODE": "Neutral", "MC_PATHS": "20000"}).mc_paths == 20_000
    for bad in ({"MC_PATHS": "100"}, {"MC_PATHS": "999999"}, {"MC_DRIFT_MODE": "magic"}):
        with pytest.raises(ConfigError):
            Settings.from_env(bad)


def test_unknown_provider_is_a_config_error():
    from stock_analyzer.data.providers import get_provider

    with pytest.raises(ConfigError):
        get_provider(Settings(provider="nope"))
