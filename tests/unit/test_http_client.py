"""Tests for the HTTP client's politeness, caching and failure handling.

Every upstream interaction is mocked with respx, so the suite is fast and runs
offline. The live-API checks live in ``tests/integration`` behind the
``network`` marker.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import httpx
import pytest
import respx

from prodrome.clients.base import (
    ApiClient,
    ApiError,
    CircuitBreaker,
    HttpCache,
    QuotaExceededError,
    RateLimiter,
    _retry_after_seconds,
)

BASE = "https://api.example.test"


def make_client(tmp_path: Path, **overrides: object) -> ApiClient:
    defaults: dict[str, object] = {
        "base_url": BASE,
        "namespace": "test",
        "cache": HttpCache(tmp_path / "cache"),
        "limiter": RateLimiter(240),
        "max_retries": 3,
        "backoff_base_seconds": 0.001,
        "max_sleep_seconds": 0.01,
    }
    defaults.update(overrides)
    return ApiClient(**defaults)  # type: ignore[arg-type]


class TestCacheKey:
    def test_parameter_order_does_not_matter(self) -> None:
        assert HttpCache.key("n", "u", {"a": 1, "b": 2}) == HttpCache.key(
            "n", "u", {"b": 2, "a": 1}
        )

    def test_api_key_is_excluded(self) -> None:
        """A cache built without a key must be reusable once a key is added."""
        assert HttpCache.key("n", "u", {"a": 1}) == HttpCache.key(
            "n", "u", {"a": 1, "api_key": "s"}
        )

    def test_namespace_separates_sources(self) -> None:
        assert HttpCache.key("openfda", "u", None) != HttpCache.key("dailymed", "u", None)

    def test_different_params_differ(self) -> None:
        assert HttpCache.key("n", "u", {"a": 1}) != HttpCache.key("n", "u", {"a": 2})


class TestCacheStorage:
    def test_round_trip(self, tmp_path: Path) -> None:
        cache = HttpCache(tmp_path)
        cache.put("ns", "abc123", {"hello": "world"})
        assert cache.get("ns", "abc123") == {"hello": "world"}

    def test_miss_returns_none(self, tmp_path: Path) -> None:
        assert HttpCache(tmp_path).get("ns", "nope") is None

    def test_corrupt_entry_is_discarded_not_raised(self, tmp_path: Path) -> None:
        """An interrupted write must not poison every later run."""
        cache = HttpCache(tmp_path)
        path = cache.path_for("ns", "deadbeef")
        path.parent.mkdir(parents=True)
        path.write_text("{not json", encoding="utf-8")
        assert cache.get("ns", "deadbeef") is None
        assert not path.exists()

    def test_disabled_cache_stores_nothing(self, tmp_path: Path) -> None:
        cache = HttpCache(tmp_path, enabled=False)
        cache.put("ns", "k", {"a": 1})
        assert cache.get("ns", "k") is None


class TestRetryAfter:
    def test_delta_seconds(self) -> None:
        response = httpx.Response(429, headers={"Retry-After": "12"})
        assert _retry_after_seconds(response, cap=60) == 12.0

    def test_is_capped(self) -> None:
        """A server may legitimately ask for an hour; a weekly job cannot wait."""
        response = httpx.Response(429, headers={"Retry-After": "3600"})
        assert _retry_after_seconds(response, cap=60) == 60.0

    def test_negative_is_treated_as_malformed(self) -> None:
        """A negative value is not an instruction to sleep negative time.

        Returning None routes the caller to normal exponential backoff instead.
        """
        response = httpx.Response(429, headers={"Retry-After": "-5"})
        assert _retry_after_seconds(response, cap=60) is None

    @pytest.mark.parametrize("value", ["soon", "", "NaN", "12abc", "1e"])
    def test_unparseable_falls_through_to_backoff(self, value: str) -> None:
        response = httpx.Response(429, headers={"Retry-After": value})
        assert _retry_after_seconds(response, cap=60) is None

    def test_absent_header(self) -> None:
        assert _retry_after_seconds(httpx.Response(429), cap=60) is None

    def test_http_date_in_the_future(self) -> None:
        future = dt.datetime.now(tz=dt.UTC) + dt.timedelta(seconds=20)
        stamp = future.strftime("%a, %d %b %Y %H:%M:%S GMT")
        got = _retry_after_seconds(httpx.Response(429, headers={"Retry-After": stamp}), cap=60)
        assert got is not None
        assert 15 <= got <= 25

    def test_http_date_in_the_past_is_malformed_not_zero(self) -> None:
        past = dt.datetime.now(tz=dt.UTC) - dt.timedelta(hours=1)
        stamp = past.strftime("%a, %d %b %Y %H:%M:%S GMT")
        assert (
            _retry_after_seconds(httpx.Response(429, headers={"Retry-After": stamp}), cap=60)
            is None
        )


class TestGetJson:
    @respx.mock
    def test_successful_request_is_cached(self, tmp_path: Path) -> None:
        route = respx.get(f"{BASE}/thing").mock(return_value=httpx.Response(200, json={"ok": True}))
        with make_client(tmp_path) as client:
            assert client.get_json("thing") == {"ok": True}
            assert client.get_json("thing") == {"ok": True}
        assert route.call_count == 1, "second call must be served from cache"

    @respx.mock
    def test_404_is_an_empty_result_not_an_error(self, tmp_path: Path) -> None:
        """openFDA returns 404 for 'no matches', which is a real zero."""
        respx.get(f"{BASE}/thing").mock(return_value=httpx.Response(404, json={"error": "x"}))
        with make_client(tmp_path) as client:
            assert client.get_json("thing") is None
            assert client.stats.empty_results == 1

    @respx.mock
    def test_empty_results_are_cached_too(self, tmp_path: Path) -> None:
        """Otherwise a sparse cohort refetches thousands of known-empty cells."""
        route = respx.get(f"{BASE}/thing").mock(return_value=httpx.Response(404))
        with make_client(tmp_path) as client:
            assert client.get_json("thing") is None
            assert client.get_json("thing") is None
        assert route.call_count == 1

    @respx.mock
    def test_404_raises_when_empty_is_not_allowed(self, tmp_path: Path) -> None:
        respx.get(f"{BASE}/thing").mock(return_value=httpx.Response(404))
        with make_client(tmp_path) as client, pytest.raises(ApiError):
            client.get_json("thing", allow_empty=False)

    @respx.mock
    def test_retries_then_succeeds(self, tmp_path: Path) -> None:
        respx.get(f"{BASE}/thing").mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "0"}),
                httpx.Response(503),
                httpx.Response(200, json={"ok": 1}),
            ]
        )
        with make_client(tmp_path) as client:
            assert client.get_json("thing") == {"ok": 1}
            assert client.stats.retries == 2

    @respx.mock
    def test_retries_are_exhausted_and_reported(self, tmp_path: Path) -> None:
        respx.get(f"{BASE}/thing").mock(return_value=httpx.Response(503))
        with (
            make_client(tmp_path, max_retries=2) as client,
            pytest.raises(ApiError, match="exhausted 3 attempts"),
        ):
            client.get_json("thing")

    @respx.mock
    def test_client_error_is_not_retried(self, tmp_path: Path) -> None:
        """A 400 means we built a bad query; retrying just wastes quota."""
        route = respx.get(f"{BASE}/thing").mock(return_value=httpx.Response(400, text="bad"))
        with make_client(tmp_path) as client, pytest.raises(ApiError, match="HTTP 400"):
            client.get_json("thing")
        assert route.call_count == 1

    @respx.mock
    def test_timeout_is_retried(self, tmp_path: Path) -> None:
        respx.get(f"{BASE}/thing").mock(
            side_effect=[httpx.ConnectTimeout("slow"), httpx.Response(200, json={"ok": 1})]
        )
        with make_client(tmp_path) as client:
            assert client.get_json("thing") == {"ok": 1}

    @respx.mock
    def test_non_json_body_is_an_error(self, tmp_path: Path) -> None:
        respx.get(f"{BASE}/thing").mock(
            return_value=httpx.Response(200, text="<html>maintenance</html>")
        )
        with make_client(tmp_path) as client, pytest.raises(ApiError, match="non-JSON"):
            client.get_json("thing")

    @respx.mock
    def test_budget_is_enforced_before_spending_it(self, tmp_path: Path) -> None:
        respx.get(url__regex=rf"{BASE}/.*").mock(return_value=httpx.Response(200, json={"n": 1}))
        with make_client(tmp_path, request_budget=2) as client:
            client.get_json("a")
            client.get_json("b")
            with pytest.raises(QuotaExceededError, match="request budget of 2"):
                client.get_json("c")

    @respx.mock
    def test_api_key_is_sent_but_not_cached(self, tmp_path: Path) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json={"ok": 1})

        respx.get(url__regex=rf"{BASE}/.*").mock(side_effect=handler)
        with make_client(tmp_path, api_key="s3cret") as client:
            client.get_json("thing", {"limit": 1})
        assert "api_key=s3cret" in str(captured[0].url)
        written = list((tmp_path / "cache" / "test").rglob("*.json"))
        assert written, "response should have been cached"
        assert "s3cret" not in written[0].read_text(encoding="utf-8")

    @respx.mock
    def test_stats_are_accumulated(self, tmp_path: Path) -> None:
        respx.get(url__regex=rf"{BASE}/.*").mock(
            return_value=httpx.Response(200, json={"payload": "x" * 100})
        )
        with make_client(tmp_path) as client:
            client.get_json("a")
            client.get_json("a")
        assert client.stats.requests == 1
        assert client.stats.cache_hits == 1
        assert client.stats.cache_hit_rate == 0.5
        assert client.stats.bytes_received > 100
        assert set(client.stats.as_row()) == {
            "requests",
            "cache_hits",
            "retries",
            "empty_results",
            "total_wait_seconds",
            "bytes_received",
            "cache_hit_rate",
            "failures",
        }


class TestRateLimiter:
    def test_rejects_nonsense_rate(self) -> None:
        with pytest.raises(ValueError, match="per_minute must be >= 1"):
            RateLimiter(0)

    def test_allows_a_burst_up_to_the_limit_without_sleeping(self) -> None:
        limiter = RateLimiter(10, window_seconds=60)
        assert sum(limiter.acquire() for _ in range(10)) == 0.0

    def test_blocks_once_the_window_is_full(self) -> None:
        limiter = RateLimiter(3, window_seconds=0.3)
        for _ in range(3):
            limiter.acquire()
        assert limiter.acquire() > 0.0


class TestFailureTolerance:
    """openFDA's drug/event index returns intermittent 500s while its other
    indexes stay healthy -- observed directly during development. A long run has
    to survive that, but only for data the analysis can proceed without."""

    @respx.mock
    def test_skip_mode_returns_none_and_counts_the_failure(self, tmp_path: Path) -> None:
        respx.get(f"{BASE}/thing").mock(return_value=httpx.Response(500))
        with make_client(tmp_path, max_retries=1) as client:
            assert client.get_json("thing", on_error="skip") is None
            assert client.stats.failures == 1

    @respx.mock
    def test_raise_mode_is_the_default(self, tmp_path: Path) -> None:
        """Essential data must fail loudly: a skipped count becomes a missing cell,
        which is indistinguishable downstream from a real zero."""
        respx.get(f"{BASE}/thing").mock(return_value=httpx.Response(500))
        with make_client(tmp_path, max_retries=1) as client:
            with pytest.raises(ApiError):
                client.get_json("thing")
            assert client.stats.failures == 1

    @respx.mock
    def test_a_skipped_failure_is_not_cached_as_empty(self, tmp_path: Path) -> None:
        """A transient 500 must not be remembered as 'no data'.

        Caching it would make one bad minute permanently poison those cells, and a
        re-run -- the documented remedy -- would replay the poison instead of
        retrying.
        """
        route = respx.get(f"{BASE}/thing").mock(
            side_effect=[
                httpx.Response(500),
                httpx.Response(500),
                httpx.Response(200, json={"ok": 1}),
            ]
        )
        with make_client(tmp_path, max_retries=0) as client:
            assert client.get_json("thing", on_error="skip") is None
            assert client.get_json("thing", on_error="skip") is None
            assert client.get_json("thing") == {"ok": 1}
        assert route.call_count == 3


class TestCircuitBreaker:
    """A high failure rate must change the retry policy, not just be endured.

    openFDA's event index was measured failing roughly 40% of requests. Nine patient
    attempts per request is correct for an occasional failure and pathological at
    that rate -- it turns a ten-minute job into a five-hour one and makes the outage
    worse for everyone else.
    """

    def test_does_not_open_before_the_window_fills(self) -> None:
        breaker = CircuitBreaker(window=10, threshold=0.5)
        for _ in range(9):
            breaker.record(ok=False)
        assert not breaker.is_open, "a few early failures must not trip it"

    def test_opens_once_the_window_is_full_and_failing(self) -> None:
        breaker = CircuitBreaker(window=10, threshold=0.5)
        for _ in range(10):
            breaker.record(ok=False)
        assert breaker.is_open
        assert breaker.failure_rate == 1.0

    def test_closes_again_on_recovery_without_intervention(self) -> None:
        breaker = CircuitBreaker(window=10, threshold=0.5)
        for _ in range(10):
            breaker.record(ok=False)
        for _ in range(10):
            breaker.record(ok=True)
        assert not breaker.is_open
        assert breaker.failure_rate == 0.0

    def test_rejects_a_nonsense_threshold(self) -> None:
        for bad in (0.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="threshold must be in"):
                CircuitBreaker(threshold=bad)

    @respx.mock
    def test_optional_requests_get_a_small_retry_budget(self, tmp_path: Path) -> None:
        """Optional data does not deserve the full retry budget."""
        route = respx.get(f"{BASE}/thing").mock(return_value=httpx.Response(500))
        with make_client(tmp_path, max_retries=8, optional_max_retries=2) as client:
            assert client.get_json("thing", on_error="skip") is None
        assert route.call_count == 3, "2 retries + the initial attempt"

    @respx.mock
    def test_essential_requests_get_the_full_budget_while_healthy(self, tmp_path: Path) -> None:
        route = respx.get(f"{BASE}/thing").mock(return_value=httpx.Response(500))
        with (
            make_client(tmp_path, max_retries=4, optional_max_retries=1) as client,
            pytest.raises(ApiError),
        ):
            client.get_json("thing")
        assert route.call_count == 5

    @respx.mock
    def test_an_open_breaker_cuts_the_essential_budget_too(self, tmp_path: Path) -> None:
        """Otherwise a broadly-degraded upstream makes the run take hours to fail."""
        route = respx.get(f"{BASE}/later").mock(return_value=httpx.Response(500))
        with make_client(tmp_path, max_retries=8, optional_max_retries=1) as client:
            # Prime the breaker open directly: the point under test is the retry
            # budget it produces, not the bookkeeping that opens it.
            client.breaker = CircuitBreaker(window=4, threshold=0.5)
            for _ in range(4):
                client.breaker.record(ok=False)
            assert client.breaker.is_open
            with pytest.raises(ApiError):
                client.get_json("later")
        assert route.call_count == 2, "1 retry + the initial attempt"

    @respx.mock
    def test_the_breaker_records_successes_so_it_can_close(self, tmp_path: Path) -> None:
        respx.get(f"{BASE}/thing").mock(return_value=httpx.Response(200, json={"ok": 1}))
        with make_client(tmp_path) as client:
            client.breaker = CircuitBreaker(window=3, threshold=0.5)
            for _ in range(3):
                client.breaker.record(ok=False)
            assert client.breaker.is_open
            for suffix in ("a", "b", "c"):
                respx.get(f"{BASE}/{suffix}").mock(return_value=httpx.Response(200, json={"ok": 1}))
                client.get_json(suffix)
            assert not client.breaker.is_open
