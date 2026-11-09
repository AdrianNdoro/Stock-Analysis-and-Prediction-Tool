"""Remedy experiment:  python -m stock_analyzer.backtest.tuning

Protocol, fixed BEFORE looking at results (so the report cannot be tuned to look good):
1. Candidates (in the order agreed): the baseline GBM, flat volatility inflation, EWMA
   volatility, then bootstrap.
2. Selection uses ONLY development tickers and ONLY the development period (forecasts whose
   outcomes were realised before SPLIT_DATE). Criterion: proper scores (pinball loss and
   90%-interval score) relative to baseline GBM, averaged over tickers and horizons.
   Coverage is not the criterion: it can be hit by widening everything, which a proper
   score penalises.
3. The selected remedy is then judged on cells it never saw: held-out tickers and/or
   held-out period. Origins straddling the split are dropped from both sides (purged).
4. A remedy is "justified" only if it beats baseline on the development score by a margin,
   and also beats it on the clean held-out cell (held-out tickers x held-out period).
   Otherwise the verdict is: leave the model alone.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from stock_analyzer.backtest.basket import (
    DEVELOPMENT_TICKERS,
    HELD_OUT_TICKERS,
    HORIZONS,
    SPLIT_DATE,
    load_closes,
)
from stock_analyzer.backtest.distributions import QUANTILE_LEVELS
from stock_analyzer.backtest.metrics import interval_score, mean_pinball
from stock_analyzer.backtest.walkforward import (
    QUANTILE_COLUMNS,
    BacktestConfig,
    run_walk_forward,
)
from stock_analyzer.config import get_settings
from stock_analyzer.data.service import build_default_service
from stock_analyzer.logging_setup import setup_logging

MARGIN = 0.02  # a remedy must lower the relative score by at least 2% to count as an improvement

CANDIDATES: dict[str, dict] = {
    "gbm (baseline)": {},
    "gbm x1.10": {"vol_inflation": 1.10},
    "gbm x1.20": {"vol_inflation": 1.20},
    "gbm x1.30": {"vol_inflation": 1.30},
    "gbm EWMA(0.94)": {"vol_model": "ewma", "ewma_weight": 1.0},
    "gbm EWMA 50/50": {"vol_model": "ewma", "ewma_weight": 0.5},
    "bootstrap": {"method": "bootstrap"},
}
BASELINE = "gbm (baseline)"


def _scores(g: pd.DataFrame) -> tuple[float, float]:
    y = g["y"].to_numpy()
    q = g[list(QUANTILE_COLUMNS)].to_numpy()
    return (
        mean_pinball(y, q, QUANTILE_LEVELS),
        interval_score(y, q[:, 0], q[:, 4], 0.90),
    )


def cell(rows: pd.DataFrame, tickers, period: str) -> pd.DataFrame:
    """Rows of one model for a (tickers, period) cell. Origins straddling the split are dropped."""
    split = pd.Timestamp(SPLIT_DATE)
    sub = rows[rows["ticker"].isin(tickers)]
    return sub[sub["outcome"] < split] if period == "dev" else sub[sub["origin"] >= split]


def relative_score(cand: pd.DataFrame, base: pd.DataFrame) -> float:
    """Mean over (ticker, horizon) of the candidate/baseline loss ratio (<1 = better)."""
    ratios = []
    for key, g in cand.groupby(["ticker", "horizon"]):
        b = base[(base["ticker"] == key[0]) & (base["horizon"] == key[1])]
        if len(g) < 10 or len(b) != len(g):
            continue
        pc, ic = _scores(g)
        pb, ib = _scores(b)
        ratios.append(0.5 * (pc / pb + ic / ib))
    return float(np.mean(ratios)) if ratios else float("nan")


def coverage(g: pd.DataFrame) -> tuple[float, float, int]:
    c50 = float(((g["y"] >= g["q25"]) & (g["y"] <= g["q75"])).mean())
    c90 = float(((g["y"] >= g["q05"]) & (g["y"] <= g["q95"])).mean())
    return c50, c90, len(g)


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args(argv)
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_json)
    service = build_default_service(settings)
    tickers = DEVELOPMENT_TICKERS + HELD_OUT_TICKERS
    closes, problems = load_closes(service, tickers)

    base_cfg = BacktestConfig(
        drift_mode=settings.mc_drift_mode,
        lookback_years=settings.mc_lookback_years,
        risk_free_rate=settings.risk_free_rate,
        horizons=HORIZONS,
    )
    results: dict[str, pd.DataFrame] = {}
    for name, change in CANDIDATES.items():
        cfg = base_cfg.with_(**change)
        frames = []
        for t, close in closes.items():
            r = run_walk_forward(close, cfg)
            r = r[r["model"] == cfg.method].assign(ticker=t)
            frames.append(r)
        results[name] = pd.concat(frames, ignore_index=True)
        print(f"ran {name}", flush=True)

    dev_t = [t for t in DEVELOPMENT_TICKERS if t in closes]
    out_t = [t for t in HELD_OUT_TICKERS if t in closes]
    cells = {
        "A dev tickers / dev period (selection)": (dev_t, "dev"),
        "B held-out tickers / dev period": (out_t, "dev"),
        "C dev tickers / held-out period": (dev_t, "held"),
        "D held-out tickers / held-out period (clean)": (out_t, "held"),
    }
    print(f"\nSplit date {SPLIT_DATE}. Relative score = mean(pinball ratio, interval90 ratio) vs")
    print("baseline GBM (lower is better; 1.000 = same). Coverage shown for 1M / 3M / 6M.\n")
    print(f"{'candidate':<16}" + "".join(f"{k[:1]:>8}" for k in cells) + "    D coverage c50/c90")
    selection: dict[str, float] = {}
    for name, rows in results.items():
        rel = []
        for tk, period in cells.values():
            base_c = cell(results[BASELINE], tk, period)
            rel.append(relative_score(cell(rows, tk, period), base_c))
        selection[name] = rel[0]
        d = cell(rows, out_t, "held")
        cov = "  ".join(
            "{:.0%}/{:.0%}".format(*coverage(d[d["horizon"] == h])[:2]) for h in ("1M", "3M", "6M")
        )
        print(f"{name:<16}" + "".join(f"{x:>8.3f}" for x in rel) + f"    {cov}")
    print("\nCells: " + "; ".join(f"{k}" for k in cells))

    best = min(selection, key=selection.get)
    print(f"\nSelected on cell A only: {best} (score {selection[best]:.3f})")
    if best == BASELINE or selection[best] > 1 - MARGIN:
        print(
            "VERDICT: no remedy beats baseline GBM by the required margin on the development set."
        )
    else:
        rel_d = relative_score(
            cell(results[best], out_t, "held"), cell(results[BASELINE], out_t, "held")
        )
        ok = rel_d < 1.0
        print(
            f"Clean held-out cell D: {rel_d:.3f} -> "
            + (
                "VERDICT: improvement holds out of sample."
                if ok
                else "VERDICT: improvement does NOT hold out of sample; do not adopt."
            )
        )
    for pr in problems:
        print(f"WARNING: {pr}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
