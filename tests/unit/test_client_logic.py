from jobnlp.harvest.client import RateLimiter, backoff_delay


def test_first_request_has_no_delay() -> None:
    rl = RateLimiter(1.0)
    assert rl.wait("h", now=100.0, sleep=False) == 0.0


def test_second_request_waits_remaining_interval() -> None:
    rl = RateLimiter(1.0)
    rl.wait("h", now=100.0, sleep=False)
    assert rl.wait("h", now=100.25, sleep=False) == 0.75


def test_hosts_are_independent() -> None:
    rl = RateLimiter(1.0)
    rl.wait("a", now=100.0, sleep=False)
    assert rl.wait("b", now=100.0, sleep=False) == 0.0


def test_slow_response_stretches_interval() -> None:
    rl = RateLimiter(1.0)
    rl.observe("h", elapsed_s=4.0)  # excess 2 s, doubled
    assert rl.interval("h") == 5.0


def test_backoff_exponential_and_retry_after() -> None:
    assert [backoff_delay(i) for i in range(3)] == [2.0, 4.0, 8.0]
    assert backoff_delay(0, "17") == 17.0
    assert backoff_delay(20) == 300.0
