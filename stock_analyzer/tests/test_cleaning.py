import numpy as np
import pandas as pd
import pytest

from stock_analyzer.config import Settings
from stock_analyzer.data.cleaning import clean_prices
from stock_analyzer.exceptions import DataValidationError, InsufficientDataError
from stock_analyzer.tests.fakes import make_prices

S = Settings()


def test_clean_data_passes_through_without_warnings():
    df, rep = clean_prices(make_prices(800), S)
    assert len(df) == 800 and rep.rows_raw == 800 and rep.warnings == []
    assert list(df.columns[:6]) == ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    assert df.index.tz is None and df.index.is_monotonic_increasing


def test_input_is_not_mutated():
    raw = make_prices(300)
    snapshot = raw.copy()
    clean_prices(raw, S)
    pd.testing.assert_frame_equal(raw, snapshot)


def test_duplicates_unsorted_and_timezone_are_handled():
    raw = make_prices(300)
    raw.index = raw.index.tz_localize("America/New_York")
    raw = pd.concat([raw, raw.iloc[[5]]]).sort_index(ascending=False)
    df, rep = clean_prices(raw, S)
    assert rep.dropped["duplicate_dates"] == 1
    assert df.index.tz is None and df.index.is_monotonic_increasing
    assert len(df) == 300


def test_tz_aware_index_keeps_local_trading_date():
    raw = make_prices(300)
    raw.index = raw.index.tz_localize("Asia/Tokyo")  # midnight JST would be the day before in UTC
    df, _ = clean_prices(raw, S)
    assert df.index[-1] == pd.Timestamp("2026-11-03")


def test_invalid_prices_and_inconsistent_ohlc_dropped_and_reported():
    raw = make_prices(300)
    raw.iloc[10, raw.columns.get_loc("Adj Close")] = np.nan
    raw.iloc[20, raw.columns.get_loc("Close")] = -5.0
    raw.iloc[30, raw.columns.get_loc("High")] = raw.iloc[30]["Low"] - 1.0
    df, rep = clean_prices(raw, S)
    assert rep.dropped == {"invalid_price": 2, "inconsistent_ohlc": 1}
    assert len(df) == 297 and any("Removed 3" in w for w in rep.warnings)


def test_gaps_are_reported_not_filled():
    raw = make_prices(400)
    raw = raw.drop(raw.index[100:120])
    df, rep = clean_prices(raw, S)
    assert len(df) == 380 and len(rep.gaps) == 1 and rep.gaps[0]["days"] > 7
    assert any("gap" in w for w in rep.warnings)


def test_outlier_is_flagged_but_kept():
    raw = make_prices(400)
    raw.iloc[200:, raw.columns.get_loc("Adj Close")] *= 0.4  # -60% in one day
    df, rep = clean_prices(raw, S)
    assert len(df) == 400 and len(rep.outlier_dates) == 1
    assert rep.outlier_dates[0] == raw.index[200].strftime("%Y-%m-%d")


def test_short_history_warns_and_tiny_history_errors():
    _, rep = clean_prices(make_prices(300), S)
    assert any("years of history" in w for w in rep.warnings)
    with pytest.raises(InsufficientDataError) as err:
        clean_prices(make_prices(30), S)
    assert "at least 60" in err.value.user_message


def test_zero_volume_warns():
    raw = make_prices(600)
    raw.iloc[:100, raw.columns.get_loc("Volume")] = 0.0
    _, rep = clean_prices(raw, S)
    assert rep.zero_volume_days == 100 and any("thinly" in w for w in rep.warnings)


def test_empty_and_missing_columns():
    with pytest.raises(InsufficientDataError):
        clean_prices(pd.DataFrame(), S)
    with pytest.raises(DataValidationError):
        clean_prices(make_prices(300).drop(columns=["Adj Close"]), S)


def test_optional_columns_are_optional():
    raw = make_prices(600).drop(columns=["Dividends", "Stock Splits"])
    df, _ = clean_prices(raw, S)
    assert "Dividends" not in df.columns
