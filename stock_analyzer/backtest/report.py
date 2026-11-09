"""Turn walk-forward rows into a calibration report (and serialise it for the cache)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from stock_analyzer.backtest import metrics as m
from stock_analyzer.backtest.distributions import QUANTILE_LEVELS
from stock_analyzer.backtest.walkforward import QUANTILE_COLUMNS, BacktestConfig
from stock_analyzer.models.monte_carlo import parse_horizon

MODEL_VERSION = 1  # bump when the model, scoring or report format changes: it is in the cache key


@dataclass(frozen=True)
class ModelMetrics:
    model: str
    n_origins: int
    n_independent: int  # non-overlapping origins: the honest "effective sample size"
    coverage50: float
    coverage50_lo: float
    coverage50_hi: float
    coverage90: float
    coverage90_lo: float
    coverage90_hi: float
    kupiec50_p: float
    kupiec90_p: float
    christoffersen90_p: float | None
    pit_hist: tuple[int, ...]
    pit_ks_p: float
    pinball: float  # mean over the 5 quantile levels, in log-return units
    interval90: float
    interval50: float
    direction_hit: float
    always_up_hit: float
    up_rate: float
    brier: float  # of P(price up); 0.25 is what a coin-flip forecast scores


@dataclass(frozen=True)
class HorizonReport:
    label: str
    horizon_days: int
    n_origins: int
    n_independent: int
    first_origin: str
    last_origin: str
    models: dict[str, ModelMetrics]


@dataclass(frozen=True)
class BacktestReport:
    ticker: str
    as_of: str
    graded_model: str  # which model the user is looking at: "gbm" or "bootstrap"
    config: dict
    horizons: dict[str, HorizonReport]
    warnings: tuple[str, ...] = field(default_factory=tuple)
    model_version: int = MODEL_VERSION

    # ---- serialisation (JSON-safe dicts; used by the SQLite cache) ----
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> BacktestReport:
        horizons = {
            k: HorizonReport(
                **{
                    **{f: v for f, v in h.items() if f != "models"},
                    "models": {
                        name: ModelMetrics(**{**mm, "pit_hist": tuple(mm["pit_hist"])})
                        for name, mm in h["models"].items()
                    },
                }
            )
            for k, h in d["horizons"].items()
        }
        return cls(
            ticker=d["ticker"],
            as_of=d["as_of"],
            graded_model=d["graded_model"],
            config=d["config"],
            horizons=horizons,
            warnings=tuple(d.get("warnings", ())),
            model_version=d.get("model_version", MODEL_VERSION),
        )


def _metrics_for(model: str, df: pd.DataFrame, h: int, step: int) -> ModelMetrics:
    df = df.sort_values("origin")
    y = df["y"].to_numpy()
    q = df[list(QUANTILE_COLUMNS)].to_numpy()
    pit = df["pit"].to_numpy()
    p_up = df["p_up"].to_numpy()
    n = len(df)
    ind = m.non_overlapping_indices(n, h, step)
    n_ind = len(ind)

    in50 = m.inside_interval(y, q[:, 1], q[:, 3])
    in90 = m.inside_interval(y, q[:, 0], q[:, 4])
    cov50, cov90 = float(in50.mean()), float(in90.mean())
    # Point estimate from all origins; uncertainty from the effective (non-overlapping) size.
    ci50 = m.wilson_interval(cov50 * n_ind, n_ind)
    ci90 = m.wilson_interval(cov90 * n_ind, n_ind)
    miss50, miss90 = ~in50[ind], ~in90[ind]
    k50 = m.kupiec_pof(int(miss50.sum()), n_ind, 0.50)
    k90 = m.kupiec_pof(int(miss90.sum()), n_ind, 0.10)
    chris = m.christoffersen_independence(miss90.astype(int))

    went_up = y > 0
    predicted_up = p_up > 0.5
    return ModelMetrics(
        model=model,
        n_origins=n,
        n_independent=n_ind,
        coverage50=cov50,
        coverage50_lo=ci50[0],
        coverage50_hi=ci50[1],
        coverage90=cov90,
        coverage90_lo=ci90[0],
        coverage90_hi=ci90[1],
        kupiec50_p=k50[1],
        kupiec90_p=k90[1],
        christoffersen90_p=None if chris is None else chris[1],
        pit_hist=m.pit_histogram(pit),
        pit_ks_p=m.pit_uniformity_pvalue(pit[ind]),
        pinball=m.mean_pinball(y, q, QUANTILE_LEVELS),
        interval90=m.interval_score(y, q[:, 0], q[:, 4], 0.90),
        interval50=m.interval_score(y, q[:, 1], q[:, 3], 0.50),
        direction_hit=float(np.mean(predicted_up == went_up)),
        always_up_hit=float(np.mean(went_up)),
        up_rate=float(np.mean(went_up)),
        brier=m.brier_score(p_up, went_up),
    )


def build_report(
    ticker: str,
    rows: pd.DataFrame,
    config: BacktestConfig,
    as_of: str,
    extra_warnings: tuple[str, ...] = (),
) -> BacktestReport:
    """Summarise ``rows`` (from ``run_walk_forward``) per horizon and per model.

    Only origins where every model produced a forecast are compared, so differences between
    the model and its baselines are never caused by different samples.
    """
    horizons: dict[str, HorizonReport] = {}
    warnings = list(extra_warnings)
    for label in (x.upper() for x in config.horizons):
        h = parse_horizon(label)
        sub = rows[rows["horizon"] == label] if len(rows) else rows
        if sub.empty:
            warnings.append(f"Not enough price history to backtest the {label} horizon.")
            continue
        counts = sub.groupby("origin")["model"].nunique()
        complete = counts[counts == counts.max()].index
        sub = sub[sub["origin"].isin(complete)]
        models = {
            name: _metrics_for(name, g, h, config.step_days) for name, g in sub.groupby("model")
        }
        origins = sorted(sub["origin"].unique())
        any_model = next(iter(models.values()))
        horizons[label] = HorizonReport(
            label=label,
            horizon_days=h,
            n_origins=any_model.n_origins,
            n_independent=any_model.n_independent,
            first_origin=str(pd.Timestamp(origins[0]).date()),
            last_origin=str(pd.Timestamp(origins[-1]).date()),
            models=models,
        )
    cfg = {k: (str(v) if v is not None and not isinstance(v, str | int | float | tuple) else v)
           for k, v in asdict(config).items()}  # fmt: skip
    cfg["horizons"] = list(config.horizons)
    return BacktestReport(
        ticker=ticker,
        as_of=as_of,
        graded_model=config.method,
        config=cfg,
        horizons=horizons,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def skill_vs(baseline: ModelMetrics, model: ModelMetrics) -> float:
    """Pinball-loss skill: +10% means 10% lower loss than the baseline; negative = worse."""
    if not baseline.pinball > 0 or math.isnan(baseline.pinball):
        return math.nan
    return 1.0 - model.pinball / baseline.pinball


def format_report(report: BacktestReport) -> list[str]:
    """Plain-text report for the CLI and basket script."""
    out = [
        f"BACKTEST  {report.ticker}  (data to {report.as_of}; model: {report.graded_model}; "
        f"window: {report.config.get('window')}; step {report.config.get('step_days')} days)"
    ]
    for label, hr in report.horizons.items():
        out.append(
            f"\n== {label}: {hr.n_origins} checks ({hr.n_independent} independent), "
            f"origins {hr.first_origin} to {hr.last_origin} =="
        )
        out.append(
            f"{'model':<12}{'cov50%':>8}{'[95% CI]':>14}{'cov90%':>8}{'[95% CI]':>14}"
            f"{'Kup90 p':>9}{'PIT-KS p':>9}{'pinball':>9}{'int90':>8}{'dir hit':>8}"
        )
        for name, mm in hr.models.items():
            # a zero-drift walk has P(up) = 0.5 exactly, so it makes no directional call
            direction = "n/a" if name == "random_walk" else f"{mm.direction_hit:.0%}"
            out.append(
                f"{name:<12}{mm.coverage50:>8.0%}{f'[{mm.coverage50_lo:.0%}-{mm.coverage50_hi:.0%}]':>14}"
                f"{mm.coverage90:>8.0%}{f'[{mm.coverage90_lo:.0%}-{mm.coverage90_hi:.0%}]':>14}"
                f"{mm.kupiec90_p:>9.3f}{mm.pit_ks_p:>9.3f}{mm.pinball:>9.4f}"
                f"{mm.interval90:>8.3f}{direction:>8}"
            )
        mod = hr.models.get(report.graded_model)
        if mod:
            for base in ("random_walk", "empirical"):
                if base in hr.models:
                    s = skill_vs(hr.models[base], mod)
                    verdict = (
                        "better than"
                        if s > 0.01
                        else "worse than"
                        if s < -0.01
                        else "no different from"
                    )
                    out.append(f"  Quantile loss: model is {verdict} {base} ({s:+.1%}).")
            out.append(
                f"  PIT histogram (10 bins, ideal flat): {list(mod.pit_hist)}; "
                f"always-'up' hit rate {mod.always_up_hit:.0%}, "
                f"Brier {mod.brier:.3f} (coin flip 0.250)."
            )
    for w in report.warnings:
        out.append(f"WARNING: {w}")
    return out
