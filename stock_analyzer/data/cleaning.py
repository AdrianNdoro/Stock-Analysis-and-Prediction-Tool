"""Validate and clean raw OHLCV data.

Principles: never invent prices (gaps are reported, not filled); never silently change what
we keep (everything dropped or suspicious is counted and shown in the quality report);
use ``Adj Close`` for returns because it accounts for splits and dividends.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_analyzer.config import Settings
from stock_analyzer.data.providers.base import OPTIONAL_COLUMNS, REQUIRED_COLUMNS
from stock_analyzer.data.types import DataQualityReport
from stock_analyzer.exceptions import DataValidationError, InsufficientDataError

GAP_DAYS = 7  # calendar days between consecutive bars that we call a "gap"
OUTLIER_ABS_LOG_RETURN = 0.4  # |ln(P_t / P_t-1)| > 0.4  (about -33% / +49% in a day)
ZERO_VOLUME_WARN_SHARE = 0.05
_MAX_LISTED = 5


def _normalise_index(index: pd.Index) -> pd.DatetimeIndex:
    """Tz-naive, midnight-normalised dates. Keeps the exchange-local calendar date."""
    idx = index if isinstance(index, pd.DatetimeIndex) else pd.to_datetime(index, errors="coerce")
    if idx.tz is not None:
        idx = idx.tz_localize(None)  # drop tz but keep local wall time (the trading date)
    return pd.DatetimeIndex(idx.normalize(), name="Date")


def clean_prices(raw: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, DataQualityReport]:
    """Return ``(clean_prices, report)``. Does not modify ``raw``."""
    if raw is None or raw.empty:
        raise InsufficientDataError("provider returned no rows")
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise DataValidationError(f"provider data is missing columns: {missing}")

    rows_raw = len(raw)
    df = raw.copy()
    df.index = _normalise_index(df.index)
    df = df[~df.index.isna()].sort_index()

    dropped: dict[str, int] = {}

    dup = df.index.duplicated(keep="last")
    if dup.any():
        dropped["duplicate_dates"] = int(dup.sum())
        df = df[~dup]

    keep = REQUIRED_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in df.columns]
    df = df[keep].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    bad_price = df[["Close", "Adj Close"]].isna().any(axis=1) | (
        df[["Close", "Adj Close"]] <= 0
    ).any(axis=1)
    if bad_price.any():
        dropped["invalid_price"] = int(bad_price.sum())
        df = df[~bad_price]

    ohlc = df[["Open", "High", "Low"]]
    bad_ohlc = ohlc.isna().any(axis=1) | (df["High"] < df["Low"] - 1e-9) | (ohlc <= 0).any(axis=1)
    if bad_ohlc.any():
        dropped["inconsistent_ohlc"] = int(bad_ohlc.sum())
        df = df[~bad_ohlc]

    df["Volume"] = df["Volume"].fillna(0.0)
    for col in OPTIONAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    n = len(df)
    if n < settings.min_history_days:
        raise InsufficientDataError(
            f"only {n} usable rows",
            user_message=(
                f"Only {n} trading days of price history are available for this ticker. "
                f"We need at least {settings.min_history_days} to give meaningful results."
            ),
        )

    warnings: list[str] = []
    if dropped:
        detail = ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in dropped.items())
        warnings.append(f"Removed {sum(dropped.values())} unusable row(s) ({detail}).")

    days = df.index.to_series().diff().dt.days
    gap_rows = days[days > GAP_DAYS]
    gaps = [
        {
            "start": (idx - pd.Timedelta(days=int(d))).strftime("%Y-%m-%d"),
            "end": idx.strftime("%Y-%m-%d"),
            "days": int(d),
        }
        for idx, d in gap_rows.items()
    ]
    if gaps:
        warnings.append(
            f"{len(gaps)} gap(s) longer than {GAP_DAYS} days in the price history "
            f"(largest: {max(g['days'] for g in gaps)} days). Missing days are not filled."
        )

    log_ret = np.log(df["Adj Close"]).diff()
    out = log_ret[log_ret.abs() > OUTLIER_ABS_LOG_RETURN]
    outlier_dates = [d.strftime("%Y-%m-%d") for d in out.index]
    if outlier_dates:
        shown = ", ".join(outlier_dates[:_MAX_LISTED])
        warnings.append(
            f"{len(outlier_dates)} one-day price move(s) larger than about 33% down / 49% up "
            f"(e.g. {shown}). They are kept, but may reflect real shocks, an unadjusted "
            "corporate action or a data error."
        )

    zero_volume = int((df["Volume"] <= 0).sum())
    if zero_volume / n > ZERO_VOLUME_WARN_SHARE:
        warnings.append(
            f"Trading volume is zero on {zero_volume / n:.0%} of days: this looks thinly "
            "traded, so real prices may jump more than the statistics suggest."
        )

    if n < settings.recommended_history_days:
        warnings.append(
            f"Only about {n / 252:.1f} years of history ({n} trading days). Volatility and "
            "simulation results are less reliable with under ~2 years of data."
        )

    report = DataQualityReport(
        rows_raw=rows_raw,
        rows_clean=n,
        first_date=df.index[0].strftime("%Y-%m-%d"),
        last_date=df.index[-1].strftime("%Y-%m-%d"),
        dropped=dropped,
        gaps=gaps,
        outlier_dates=outlier_dates,
        zero_volume_days=zero_volume,
        warnings=warnings,
    )
    return df.astype("float64"), report
