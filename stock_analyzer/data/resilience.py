"""Rate limiting and retry-with-backoff to protect against provider throttling."""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable
from typing import TypeVar

from stock_analyzer.exceptions import ProviderRateLimitError, ProviderUnavailableError

log = logging.getLogger(__name__)
T = TypeVar("T")


class RateLimiter:
    """Thread-safe token bucket. ``acquire()`` blocks until a call is allowed.

    ``calls_per_minute`` is the sustained rate; ``burst`` is how many calls may go out
    back-to-back before throttling starts. Waiters reserve their slot under the lock and
    sleep outside it, so many threads queue fairly without busy-waiting.
    """

    def __init__(
        self,
        calls_per_minute: float,
        burst: int = 1,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if calls_per_minute <= 0 or burst < 1:
            raise ValueError("calls_per_minute must be > 0 and burst >= 1")
        self._rate = calls_per_minute / 60.0
        self._capacity = float(burst)
        self._tokens = float(burst)
        self._clock = clock
        self._sleep = sleep
        self._updated = clock()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Wait for a slot. Returns the number of seconds waited."""
        with self._lock:
            now = self._clock()
            self._tokens = min(self._capacity, self._tokens + (now - self._updated) * self._rate)
            self._updated = now
            self._tokens -= 1.0
            wait = 0.0 if self._tokens >= 0 else -self._tokens / self._rate
        if wait > 0:
            self._sleep(wait)
        return wait


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_retries: int,
    base_delay: float,
    max_delay: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
    rng: Callable[[], float] = random.random,
    on_retry: Callable[[int, float, Exception], None] | None = None,
) -> T:
    """Call ``fn``; on throttling/outage errors retry with exponential backoff + jitter.

    Only :class:`ProviderRateLimitError` and :class:`ProviderUnavailableError` are retried.
    Everything else (bad ticker, bad data) is deterministic, so retrying would be pointless.
    A provider-supplied ``retry_after`` is honoured (capped at ``max_delay``).
    """
    attempt = 0
    while True:
        try:
            return fn()
        except (ProviderRateLimitError, ProviderUnavailableError) as exc:
            if attempt >= max_retries:
                raise
            delay = min(max_delay, base_delay * (2**attempt)) * (0.5 + 0.5 * rng())
            retry_after = getattr(exc, "retry_after", None)
            if retry_after:
                delay = max(delay, min(float(retry_after), max_delay))
            if on_retry:
                on_retry(attempt + 1, delay, exc)
            sleep(delay)
            attempt += 1
