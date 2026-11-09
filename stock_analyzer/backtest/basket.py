"""Multi-ticker validation:  python -m stock_analyzer.backtest.basket [--method gbm|bootstrap]

Runs the walk-forward backtest over a small basket and prints per-ticker and pooled coverage.

Limits of this check (read before trusting a "good" result):
* Survivorship bias: every ticker here is a company that still exists and is large today.
  Failed companies are missing, which flatters any model on this basket.
* Look-ahead in choosing the basket: the tickers are well known BECAUSE they did well or are
  familiar. The development/held-out split below was fixed before any results were seen.
* Cross-sectional correlation: tickers move with the market, so 12 tickers are far fewer than
  12 independent experiments. Pooled numbers carry no confidence interval for that reason.
* Regimes: 2020 and 2022 dominate the sample; a different decade could grade differently.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

import pandas as pd

from stock_analyzer.analysis.technical import adjusted_ohlc
from stock_analyzer.backtest.reliability import assess
from stock_analyzer.backtest.report import build_report
from stock_analyzer.backtest.walkforward import BacktestConfig, run_walk_forward
from stock_analyzer.config import get_settings
from stock_analyzer.data.service import MarketDataService, build_default_service
from stock_analyzer.exceptions import StockAnalyzerError
from stock_analyzer.logging_setup import setup_logging

# Fixed in advance (before any result was looked at). Mixed US / UK / EU / Asia, large caps + ETF.
DEVELOPMENT_TICKERS = ("AAPL", "MSFT", "KO", "BP.L", "JNJ", "SPY")
HELD_OUT_TICKERS = ("NVDA", "SAP.DE", "JPM", "HSBA.L", "7203.T", "XOM")
BASKET = DEVELOPMENT_TICKERS + HELD_OUT_TICKERS
SPLIT_DATE = date(2023, 1, 1)  # development period: outcomes before it; held-out: origins from it
HORIZONS = ("1M", "3M", "6M", "1Y")


def load_closes(
    service: MarketDataService, tickers: tuple[str, ...] | list[str]
) -> tuple[dict[str, pd.Series], list[str]]:
    """Adjusted closes for each ticker; failures are reported, never silently dropped."""
    closes: dict[str, pd.Series] = {}
    problems: list[str] = []
    for t in tickers:
        try:
            closes[t] = adjusted_ohlc(service.get_market_data(t).prices)["Close"]
        except StockAnalyzerError as exc:
            problems.append(f"{t}: {exc.user_message}")
    return closes, problems


def pooled_coverage(rows: pd.DataFrame, model: str) -> pd.DataFrame:
    """Pooled coverage and pinball loss per horizon across all rows (no confidence interval)."""
    sub = rows[rows["model"] == model]
    out = []
    for label, g in sub.groupby("horizon", sort=False):
        out.append(
            {
                "horizon": label,
                "n": len(g),
                "cov50": float(((g["y"] >= g["q25"]) & (g["y"] <= g["q75"])).mean()),
                "cov90": float(((g["y"] >= g["q05"]) & (g["y"] <= g["q95"])).mean()),
            }
        )
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Walk-forward validation over a basket of tickers.")
    p.add_argument("--tickers", nargs="*", help="override the basket")
    p.add_argument("--method", choices=["gbm", "bootstrap"], default="gbm")
    p.add_argument("--window", choices=["rolling", "expanding"], default="rolling")
    p.add_argument("--vol-inflation", type=float, default=1.0)
    p.add_argument("--vol-model", choices=["sample", "ewma"], default="sample")
    args = p.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_level, settings.log_json)
    tickers = tuple(t.upper() for t in args.tickers) if args.tickers else BASKET
    service = build_default_service(settings)
    closes, problems = load_closes(service, tickers)
    config = BacktestConfig(
        method=args.method,
        drift_mode=settings.mc_drift_mode,
        lookback_years=settings.mc_lookback_years,
        risk_free_rate=settings.risk_free_rate,
        window=args.window,
        horizons=HORIZONS,
        vol_inflation=args.vol_inflation,
        vol_model=args.vol_model,
    )
    print(
        f"Basket validation: model={args.method}, window={args.window}, "
        f"vol_model={args.vol_model}, vol_inflation={args.vol_inflation}"
    )
    all_rows = []
    header = (
        f"{'ticker':<9}" + "".join(f"{h + ' c50/c90 (n)':>22}" for h in HORIZONS) + "  grade@3M"
    )
    print("\n" + header)
    for t, close in closes.items():
        rows = run_walk_forward(close, config)
        rows["ticker"] = t
        all_rows.append(rows)
        report = build_report(t, rows, config, str(close.index[-1].date()))
        cells = []
        for h in HORIZONS:
            hr = report.horizons.get(h)
            mm = hr.models.get(args.method) if hr else None
            cells.append(
                f"{mm.coverage50:>6.0%}/{mm.coverage90:<4.0%} ({hr.n_independent:>3})".rjust(22)
                if mm
                else "n/a".rjust(22)
            )
        print(f"{t:<9}" + "".join(cells) + f"  {assess(report, '3M').grade}")
    if all_rows:
        pooled = pd.concat(all_rows, ignore_index=True)
        print("\nPOOLED over all tickers (target: cov50 = 50%, cov90 = 90%):")
        for name in (args.method, "random_walk", "empirical"):
            tab = pooled_coverage(pooled, name)
            line = "  ".join(f"{r.horizon}: {r.cov50:.0%}/{r.cov90:.0%}" for r in tab.itertuples())
            print(f"  {name:<12} {line}")
        print(
            "\nPooled figures are not independent (tickers share market moves) and the basket "
            "has survivorship bias."
        )
    for pr in problems:
        print(f"WARNING: {pr}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
