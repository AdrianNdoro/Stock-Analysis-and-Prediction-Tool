"""Quick smoke test against the live provider:  python -m stock_analyzer.cli AAPL"""

from __future__ import annotations

import argparse
import sys

from stock_analyzer.analysis.pipeline import AnalysisResult, analyze_ticker
from stock_analyzer.config import get_settings
from stock_analyzer.data.service import build_default_service
from stock_analyzer.exceptions import StockAnalyzerError
from stock_analyzer.logging_setup import setup_logging
from stock_analyzer.models.forecast import ForecastResult, forecast_ticker


def print_analysis(a: AnalysisResult) -> None:
    ccy = a.currency or "?"
    print(f"{a.ticker}  as of {a.as_of}  last price {a.last_price:,.2f} {ccy}")
    print("\nTECHNICAL")
    for b in a.technical.bullets:
        print(f"  - {b}")
    r = a.risk
    print(f"\nRISK  ({r.window_start} to {r.window_end}, {r.n_returns} daily returns)")
    print(f"  Annualised return      {r.annualised_return:+.1%}")
    print(f"  Annualised volatility  {r.annualised_volatility:.1%}")
    ratios = f"{r.sharpe:.2f} / {r.sortino:.2f}"
    print(f"  Sharpe / Sortino       {ratios}   (risk-free {r.risk_free_rate:.1%})")
    d = r.drawdown
    print(f"  Max drawdown           {d.max_drawdown:.1%}  ({d.peak_date} -> {d.trough_date})")
    if r.beta and r.beta.beta == r.beta.beta:
        name = a.benchmark.name if a.benchmark else "benchmark"
        freq = "weekly" if r.beta.frequency == "W" else "daily"
        print(
            f"  Beta vs {name:<14} {r.beta.beta:.2f}  "
            f"(R2 {r.beta.r_squared:.2f}, {r.beta.n_obs} {freq} obs)"
        )
    for v in r.var:
        print(f"  {v.confidence:.0%} 1-day VaR          hist {v.historical:.2%} | "
              f"normal {v.parametric:.2%} | tail avg {v.expected_shortfall:.2%}")  # fmt: skip
    print("\nIN PLAIN ENGLISH")
    for b in a.risk_bullets:
        print(f"  - {b}")
    for w in a.warnings:
        print(f"WARNING: {w}")


def print_forecast(f: ForecastResult) -> None:
    a = f.analysis
    cur = a.currency or "?"
    print(f"{a.ticker}  as of {a.as_of}  last price {a.last_price:,.2f} {cur}")
    for label, sim in f.simulations.items():
        print(f"\n=== {label}: {sim.n_paths:,} paths, {sim.method}, seed {sim.seed} ===")
        print("Price percentiles:")
        for name, row in sim.summary_table().iterrows():
            print(f"  {name:>5}  {row['price']:>10,.2f} {cur}  ({row['return']:+.1%})")
        for line in sim.plain_english(a.currency):
            print(f"  - {line}")
        ss = f.scenarios[label]
        print(f"Scenarios (base = within ±{ss.threshold:.1%} of today):")
        for sc in ss.scenarios:
            rng = f"{sc.price_low:,.2f}-{sc.price_high:,.2f}"
            print(
                f"  {sc.name:<4} {sc.probability:>4.0%} "
                f"(range {sc.probability_low:.0%}-{sc.probability_high:.0%})  "
                f"typical {rng} {cur}, target {sc.target_price:,.2f}"
            )
            print(f"       {sc.indicator_context}")
    for w in a.warnings:
        print(f"WARNING: {w}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and clean prices for a ticker.")
    parser.add_argument("ticker")
    parser.add_argument("--fundamentals", action="store_true", help="also fetch fundamentals")
    parser.add_argument("--analyze", action="store_true", help="technical + risk analysis")
    parser.add_argument("--simulate", action="store_true", help="Monte Carlo + scenarios")
    parser.add_argument("--horizon", default="6M", help="1M, 3M, 6M, 1Y or ALL (default 6M)")
    parser.add_argument("--method", choices=["gbm", "bootstrap"], default="gbm")
    parser.add_argument("--drift", choices=["shrunk", "historical", "neutral"])
    parser.add_argument("--paths", type=int, help="simulations, 10000-50000")
    parser.add_argument("--seed", type=int, help="fixed random seed for reproducible runs")
    parser.add_argument(
        "--benchmark", help="benchmark key or ticker, e.g. SP500, FTSE100, ^N225 (default: auto)"
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_level, settings.log_json)
    try:
        service = build_default_service(settings)
        if args.simulate:
            horizons = ["1M", "3M", "6M", "1Y"] if args.horizon.upper() == "ALL" else [args.horizon]
            result = forecast_ticker(
                service, args.ticker, horizons, benchmark=args.benchmark, method=args.method,
                n_paths=args.paths, seed=args.seed, drift_mode=args.drift, settings=settings,
            )  # fmt: skip
            print_forecast(result)
            return 0
        if args.analyze:
            print_analysis(analyze_ticker(service, args.ticker, args.benchmark, settings))
            return 0
        md = service.get_market_data(args.ticker)
        fund = service.get_fundamentals(args.ticker) if args.fundamentals else None
    except StockAnalyzerError as exc:
        print(f"Error: {exc.user_message}", file=sys.stderr)
        return 1

    q = md.quality
    print(f"Ticker:    {md.ticker}   Source: {md.source}   From cache: {md.from_cache}")
    print(f"Currency:  {md.currency or 'unknown'}")
    print(f"History:   {q.rows_clean} trading days, {q.first_date} -> {q.last_date}")
    print(f"Last bar:  Close {md.prices['Close'].iloc[-1]:.2f}  "
          f"Adj Close {md.prices['Adj Close'].iloc[-1]:.2f}")  # fmt: skip
    for w in md.warnings:
        print(f"WARNING:   {w}")
    if fund:
        print(f"Fundamentals available: {fund.available}")
        for k, v in fund.values.items():
            print(f"  {k:16} {v}")
        for w in fund.warnings:
            print(f"WARNING:   {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
