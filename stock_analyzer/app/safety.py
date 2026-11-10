"""Input safety for the UI: whitelist parsing and per-session throttling. Pure, no Streamlit."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from stock_analyzer.analysis.benchmarks import BENCHMARKS
from stock_analyzer.data.validation import normalize_ticker
from stock_analyzer.exceptions import InvalidParameterError, UnknownBenchmarkError
from stock_analyzer.models.monte_carlo import DRIFT_MODES, HORIZONS, MAX_PATHS, METHODS, MIN_PATHS

MAX_SEED = 2**31 - 1
AUTO_BENCHMARK = "auto"


@dataclass(frozen=True)
class AnalysisRequest:
    """A fully validated request. Only this type reaches the analysis code."""

    ticker: str
    horizon: str
    method: str
    drift: str
    paths: int
    seed: int | None
    benchmark: str | None  # None = auto-suggest


def _is_int(x: object) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def parse_request(
    ticker: object,
    horizon: object,
    method: object = "gbm",
    drift: object = "shrunk",
    paths: object = MIN_PATHS,
    seed: object = None,
    benchmark: object = AUTO_BENCHMARK,
    max_paths: int = MAX_PATHS,
) -> AnalysisRequest:
    """Validate everything against allow-lists. Raises StockAnalyzerError with friendly text.

    ``max_paths`` lets an operator lower the ceiling on small hosts (never above MAX_PATHS).
    """
    max_paths = max(MIN_PATHS, min(max_paths, MAX_PATHS))
    symbol = normalize_ticker(ticker)
    if horizon not in HORIZONS:
        raise InvalidParameterError(
            f"horizon {horizon!r}", user_message="Choose a horizon of 1M, 3M, 6M or 1Y."
        )
    if method not in METHODS or drift not in DRIFT_MODES:
        raise InvalidParameterError(f"method/drift {method!r}/{drift!r}")
    if not _is_int(paths) or not MIN_PATHS <= paths <= max_paths:
        raise InvalidParameterError(
            f"paths {paths!r}",
            user_message=f"Number of simulations must be between {MIN_PATHS:,} and {max_paths:,}.",
        )
    if seed is not None and (not _is_int(seed) or not 0 <= seed <= MAX_SEED):
        raise InvalidParameterError(
            f"seed {seed!r}", user_message="The random seed must be a whole number, 0 or more."
        )
    bench = None
    if benchmark not in (None, AUTO_BENCHMARK):
        if benchmark not in BENCHMARKS:
            raise UnknownBenchmarkError(f"benchmark {benchmark!r}")
        bench = str(benchmark)
    return AnalysisRequest(symbol, str(horizon), str(method), str(drift), paths, seed, bench)


class RequestThrottle:
    """Sliding-window limiter for ONE session. State is a plain list so it can live in
    ``st.session_state`` and is never shared between users."""

    def __init__(
        self,
        stamps: list[float] | None = None,
        *,
        max_requests: int = 8,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.stamps = stamps if stamps is not None else []
        self.max_requests = max_requests
        self.window = window_seconds
        self._clock = clock

    def try_acquire(self) -> float:
        """Return 0.0 if allowed (and record it), else the seconds to wait."""
        now = self._clock()
        self.stamps[:] = [t for t in self.stamps if now - t < self.window]
        if len(self.stamps) >= self.max_requests:
            return max(0.1, self.window - (now - self.stamps[0]))
        self.stamps.append(now)
        return 0.0
