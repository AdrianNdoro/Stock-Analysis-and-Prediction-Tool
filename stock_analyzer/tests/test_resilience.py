import pytest

from stock_analyzer.data.resilience import RateLimiter, call_with_retry
from stock_analyzer.exceptions import (
    ProviderRateLimitError,
    ProviderUnavailableError,
    TickerNotFoundError,
)
from stock_analyzer.tests.fakes import FakeClock


def test_rate_limiter_allows_burst_then_throttles():
    clock = FakeClock(0.0)
    waits: list[float] = []

    def sleep(s: float) -> None:
        waits.append(s)
        clock.advance(s)

    rl = RateLimiter(60, burst=2, clock=clock, sleep=sleep)  # 1 call/second sustained
    assert [rl.acquire() for _ in range(4)] == [0.0, 0.0, pytest.approx(1.0), pytest.approx(1.0)]
    clock.advance(10)  # bucket refills (capped at burst)
    assert rl.acquire() == 0.0 and rl.acquire() == 0.0 and rl.acquire() > 0


def test_rate_limiter_rejects_bad_config():
    with pytest.raises(ValueError):
        RateLimiter(0)


def _flaky(failures: list[Exception]):
    state = {"calls": 0}

    def fn():
        state["calls"] += 1
        if failures:
            raise failures.pop(0)
        return "ok"

    return fn, state


def test_retry_succeeds_after_transient_failures_with_exponential_backoff():
    delays: list[float] = []
    fn, state = _flaky([ProviderUnavailableError("x"), ProviderRateLimitError("y")])
    out = call_with_retry(fn, max_retries=3, base_delay=1.0, sleep=delays.append, rng=lambda: 1.0)
    assert out == "ok" and state["calls"] == 3 and delays == [1.0, 2.0]


def test_retry_gives_up_after_max_retries():
    fn, state = _flaky([ProviderUnavailableError("x")] * 10)
    with pytest.raises(ProviderUnavailableError):
        call_with_retry(fn, max_retries=2, base_delay=0.1, sleep=lambda s: None)
    assert state["calls"] == 3


def test_non_transient_errors_are_not_retried():
    fn, state = _flaky([TickerNotFoundError("nope")])
    with pytest.raises(TickerNotFoundError):
        call_with_retry(fn, max_retries=5, base_delay=0.1, sleep=lambda s: None)
    assert state["calls"] == 1


def test_retry_after_is_honoured_and_capped():
    delays: list[float] = []
    fn, _ = _flaky(
        [
            ProviderRateLimitError("x", retry_after=12.0),
            ProviderRateLimitError("y", retry_after=999),
        ]
    )
    call_with_retry(
        fn, max_retries=3, base_delay=0.1, max_delay=30, sleep=delays.append, rng=lambda: 1.0
    )
    assert delays == [12.0, 30.0]
