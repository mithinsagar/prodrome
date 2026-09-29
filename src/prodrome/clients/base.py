"""A polite, reproducible HTTP client.

Three properties matter for this project, and none of them are optional:

*politeness*
    Both sources are free public services funded by taxpayers. The client
    rate-limits below the published ceiling, identifies itself, honours
    ``Retry-After``, and backs off on every server-side failure class.
*reproducibility*
    Responses are cached on disk keyed by the request, so re-running an analysis
    produces byte-identical inputs and reviewing a result does not re-hammer the
    API. The cache key deliberately excludes the API key, so a cache built
    without one is reused by a run that has one.
*auditability*
    Every run accumulates :class:`RequestStats` -- how many calls, how many cache
    hits, how many retries, how much of the daily quota was spent. This is
    written into the run manifest, which is what makes a published number
    traceable to the traffic that produced it.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import math
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

#: Status codes worth retrying. 429 is quota, the 5xx set is upstream trouble.
#: 404 is explicitly *not* here: openFDA returns it for "no matches", which is a
#: legitimate empty answer rather than a failure.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class ApiError(RuntimeError):
    """A request failed in a way retrying will not fix."""

    def __init__(self, message: str, *, status_code: int | None = None, url: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.url = url


class QuotaExceededError(ApiError):
    """The configured request budget for this run is exhausted.

    Raised in preference to letting a job grind on into 429s, so a weekly run
    fails with a legible reason and a partial, still-valid warehouse rather than
    an opaque stall.
    """


@dataclass
class RequestStats:
    """Traffic counters for one client, for the run manifest."""

    requests: int = 0
    cache_hits: int = 0
    retries: int = 0
    empty_results: int = 0
    total_wait_seconds: float = 0.0
    bytes_received: int = 0
    #: Requests that exhausted every retry. Counted rather than merely logged,
    #: because the count is the signal that something systematic is wrong: a run
    #: against a healthy API should end with zero, and a non-zero total on a long
    #: backfill is worth investigating rather than shrugging at.
    failures: int = 0

    @property
    def cache_hit_rate(self) -> float:
        total = self.requests + self.cache_hits
        return self.cache_hits / total if total else 0.0

    def as_row(self) -> dict[str, float | int]:
        return {
            "requests": self.requests,
            "cache_hits": self.cache_hits,
            "retries": self.retries,
            "empty_results": self.empty_results,
            "total_wait_seconds": round(self.total_wait_seconds, 3),
            "bytes_received": self.bytes_received,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "failures": self.failures,
        }


class CircuitBreaker:
    """Stops retrying hard when the upstream is broadly failing.

    A retry policy tuned for the occasional transient fault behaves pathologically
    when failures are systematic: with nine patient attempts per request, a
    ten-minute job becomes a multi-hour one while making the upstream's problem
    worse for everyone else.

    This was built after exactly that happened here -- though the cause turned out
    to be a malformed query of ours rather than an unhealthy API (see
    ``openfda.validate_count_field``). The breaker is kept regardless, because the
    lesson generalises: a client hammering a free public service through a sustained
    failure is badly behaved whoever is at fault, and the cost of the guard is one
    counter.

    It tracks the outcome of the last `window` requests. Above `threshold` failures
    the breaker is "open" and the client cuts its retry budget to a single quick
    attempt -- enough to pick up a recovery on the next call, cheap enough that a
    long tail of failures costs minutes rather than hours. It closes again as soon
    as successes refill the window, so recovery needs no intervention.
    """

    def __init__(self, window: int = 20, threshold: float = 0.5) -> None:
        if not 0 < threshold <= 1:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        self.window = window
        self.threshold = threshold
        self._outcomes: deque[bool] = deque(maxlen=window)
        self._lock = threading.Lock()

    def record(self, *, ok: bool) -> None:
        with self._lock:
            self._outcomes.append(ok)

    @property
    def failure_rate(self) -> float:
        with self._lock:
            if not self._outcomes:
                return 0.0
            return 1.0 - sum(self._outcomes) / len(self._outcomes)

    @property
    def is_open(self) -> bool:
        """True when the upstream looks broadly unhealthy.

        Requires a full window before opening, so a couple of early failures on a
        healthy service do not trip it.
        """
        with self._lock:
            if len(self._outcomes) < self.window:
                return False
        return self.failure_rate >= self.threshold


class RateLimiter:
    """Sliding-window limiter: at most `per_minute` requests in any 60 seconds.

    A sliding window rather than fixed spacing because the openFDA limit is
    expressed per minute, and fixed spacing would leave most of the allowance
    unused during the long stretches when responses are slow anyway.

    Thread-safe, because the label backfill fetches SPL archives concurrently.
    """

    def __init__(self, per_minute: int, *, window_seconds: float = 60.0) -> None:
        if per_minute < 1:
            raise ValueError(f"per_minute must be >= 1, got {per_minute}")
        self.per_minute = per_minute
        self.window_seconds = window_seconds
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Block until a request may be made. Returns seconds actually slept."""
        slept = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                cutoff = now - self.window_seconds
                while self._times and self._times[0] <= cutoff:
                    self._times.popleft()
                if len(self._times) < self.per_minute:
                    self._times.append(now)
                    return slept
                wait = self._times[0] + self.window_seconds - now
            wait = max(wait, 0.01)
            time.sleep(wait)
            slept += wait


class HttpCache:
    """Content-addressed on-disk response cache.

    Layout is ``<root>/<namespace>/<first two hex chars>/<full hash>.json``. The
    two-character fan-out keeps directory listings usable once a full backfill
    has written tens of thousands of entries.
    """

    def __init__(self, root: Path, *, enabled: bool = True) -> None:
        self.root = Path(root)
        self.enabled = enabled

    @staticmethod
    def key(namespace: str, url: str, params: dict[str, Any] | None) -> str:
        """Stable key for a request.

        ``api_key`` is excluded so that turning a key on or off does not
        invalidate an existing cache -- the response body does not depend on it.
        Parameters are sorted so dict ordering cannot change the key.
        """
        material = {
            "namespace": namespace,
            "url": url,
            "params": sorted(
                (k, str(v)) for k, v in (params or {}).items() if k not in {"api_key"}
            ),
        }
        blob = json.dumps(material, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def path_for(self, namespace: str, digest: str) -> Path:
        return self.root / namespace / digest[:2] / f"{digest}.json"

    def get(self, namespace: str, digest: str) -> Any | None:
        if not self.enabled:
            return None
        path = self.path_for(namespace, digest)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A truncated entry from an interrupted write is not worth failing
            # over; drop it and let the caller refetch.
            logger.warning("discarding unreadable cache entry %s", path)
            path.unlink(missing_ok=True)
            return None

    def put(self, namespace: str, digest: str, payload: Any) -> None:
        if not self.enabled:
            return
        path = self.path_for(namespace, digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a sibling then rename, so a crash cannot leave a half-written
        # entry that a later run would read as valid.
        scratch = path.with_suffix(".partial")
        scratch.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        scratch.replace(path)

    def blob_path_for(self, namespace: str, digest: str, suffix: str) -> Path:
        """Location for a non-JSON response body.

        Label archives are multi-megabyte ZIPs, so they are stored as opaque
        blobs rather than base64 inside a JSON envelope -- which would inflate
        them by a third and make the cache directory unreadable to a human
        debugging a bad label.
        """
        return self.root / namespace / digest[:2] / f"{digest}{suffix}"

    def get_bytes(self, namespace: str, digest: str, suffix: str) -> bytes | None:
        if not self.enabled:
            return None
        path = self.blob_path_for(namespace, digest, suffix)
        if not path.is_file():
            return None
        try:
            return path.read_bytes()
        except OSError:
            logger.warning("discarding unreadable cache blob %s", path)
            path.unlink(missing_ok=True)
            return None

    def put_bytes(self, namespace: str, digest: str, suffix: str, payload: bytes) -> None:
        if not self.enabled:
            return
        path = self.blob_path_for(namespace, digest, suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        scratch = path.with_name(path.name + ".partial")
        scratch.write_bytes(payload)
        scratch.replace(path)


def _retry_after_seconds(response: httpx.Response, *, cap: float) -> float | None:
    """Parse ``Retry-After``, returning None when it is absent or unusable.

    The header is attacker- and bug-controlled input. A malformed value, a date
    in the past, or a negative number must not become a sleep duration -- a
    negative value passed to ``time.sleep`` raises, and a huge one would hang a
    scheduled job. Anything unparseable falls through to normal backoff.
    """
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    raw = raw.strip()
    try:
        # The delta-seconds form, which is what openFDA actually sends.
        seconds = float(raw)
    except ValueError:
        # The HTTP-date form. parsedate_to_datetime raises on malformed input and
        # returns a naive datetime when the string carries no zone, so the
        # comparison has to be made in whichever kind we got back.
        try:
            target = parsedate_to_datetime(raw)
        except (TypeError, ValueError):
            logger.warning("ignoring unparseable Retry-After: %r", raw)
            return None
        now = dt.datetime.now(tz=target.tzinfo)
        seconds = (target - now).total_seconds()
    if not math.isfinite(seconds) or seconds < 0:
        # A negative or NaN value is malformed, not an instruction: passing it to
        # time.sleep would raise, and treating it as 0 would defeat the backoff.
        logger.warning("ignoring non-positive Retry-After: %r", raw)
        return None
    return min(seconds, cap)


@dataclass
class ApiClient:
    """A rate-limited, retrying, caching JSON client.

    Not a general-purpose HTTP library: it does exactly what the two upstream
    APIs need, and its narrowness is what lets the retry semantics be precise.
    """

    base_url: str
    namespace: str
    cache: HttpCache
    limiter: RateLimiter
    timeout_seconds: float = 60.0
    max_retries: int = 5
    backoff_base_seconds: float = 1.5
    max_sleep_seconds: float = 60.0
    user_agent: str = "prodrome/0.1"
    api_key: str | None = None
    api_key_param: str = "api_key"
    request_budget: int | None = None
    #: Retry budget for requests whose data is optional. Kept low deliberately:
    #: see CircuitBreaker and the on_error parameter of get_json.
    optional_max_retries: int = 2
    stats: RequestStats = field(default_factory=RequestStats)
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    _client: httpx.Client | None = field(default=None, repr=False)
    # Seeded so backoff jitter is reproducible across runs; never used for secrets.
    _rng: random.Random = field(
        default_factory=lambda: random.Random(0),  # noqa: S311
        repr=False,
    )

    def __post_init__(self) -> None:
        self._client = httpx.Client(
            timeout=self.timeout_seconds,
            headers={"User-Agent": self.user_agent, "Accept": "application/json"},
            follow_redirects=True,
        )

    def __enter__(self) -> ApiClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _sleep(self, seconds: float) -> None:
        seconds = max(0.0, min(seconds, self.max_sleep_seconds))
        if seconds:
            time.sleep(seconds)
            self.stats.total_wait_seconds += seconds

    def _retry_budget(self, on_error: str) -> int:
        """How many retries this request gets.

        Optional data gets a small budget always; essential data gets the full
        budget until the circuit breaker opens, after which it is cut too. Without
        the cut, a broadly-degraded upstream makes the run take hours to fail.
        """
        if on_error == "skip":
            return min(self.optional_max_retries, self.max_retries)
        if self.breaker.is_open:
            logger.debug(
                "circuit breaker open (failure rate %.0f%%); reducing the retry budget",
                self.breaker.failure_rate * 100,
            )
            return min(self.optional_max_retries, self.max_retries)
        return self.max_retries

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff with full jitter.

        Jitter matters even for a single client: a burst of 429s otherwise
        produces a synchronised retry storm against the same second.
        """
        ceiling = min(self.backoff_base_seconds * (2**attempt), self.max_sleep_seconds)
        # Jitter only: this picks a sleep duration, never a secret. A CSPRNG here
        # would be slower for no benefit.
        return self._rng.uniform(0.0, ceiling)

    def get_json(  # noqa: PLR0915
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        allow_empty: bool = True,
        on_error: str = "raise",
    ) -> dict[str, Any] | None:
        """GET a JSON document, with caching, rate limiting and retries.

        Args:
            path: appended to ``base_url``.
            params: query parameters; the API key is added automatically.
            allow_empty: when True, an upstream "no matches" (HTTP 404 on
                openFDA) returns None instead of raising. Callers that treat an
                empty result as a legitimate zero pass True; callers that
                consider it a bug pass False.
            on_error: ``"raise"`` propagates an exhausted-retry failure;
                ``"skip"`` returns None and increments ``stats.failures``. Use
                ``"skip"`` only for data the analysis can proceed without -- a
                skipped *count* silently becomes a missing cell, whereas a skipped
                diagnostic merely leaves a signal unqualified. The distinction is
                the caller's to make, which is why it is not a client-wide setting.

        Returns:
            The decoded body, or None for an allowed empty result or a skipped
            failure.

        Raises:
            QuotaExceededError: request budget for the run is spent.
            ApiError: a non-retryable failure, or retries exhausted with
                ``on_error="raise"``.
        """
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}" if path else self.base_url
        digest = HttpCache.key(self.namespace, url, params)

        cached = self.cache.get(self.namespace, digest)
        if cached is not None:
            self.stats.cache_hits += 1
            return None if cached == {"__empty__": True} else cached

        if self.request_budget is not None and self.stats.requests >= self.request_budget:
            raise QuotaExceededError(
                f"request budget of {self.request_budget} exhausted for namespace "
                f"{self.namespace!r}; rerun to continue from the cache",
                url=url,
            )

        send_params = dict(params or {})
        if self.api_key:
            send_params[self.api_key_param] = self.api_key

        assert self._client is not None, "client used after close()"
        budget = self._retry_budget(on_error)
        last_error: Exception | None = None
        for attempt in range(budget + 1):
            # The limiter blocks internally; record what it cost us so the run
            # manifest shows how much of the wall clock was rate limiting.
            self.stats.total_wait_seconds += self.limiter.acquire()
            self.stats.requests += 1
            try:
                response = self._client.get(url, params=send_params)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                self.stats.retries += 1
                self.breaker.record(ok=False)
                logger.warning(
                    "transport error on %s (attempt %d/%d): %s",
                    url,
                    attempt + 1,
                    budget + 1,
                    exc,
                )
                self._sleep(self._backoff(attempt))
                continue

            self.stats.bytes_received += len(response.content)

            if response.status_code == 404 and allow_empty:
                self.stats.empty_results += 1
                self.breaker.record(ok=True)
                self.cache.put(self.namespace, digest, {"__empty__": True})
                return None

            if response.status_code in RETRYABLE_STATUS:
                self.stats.retries += 1
                self.breaker.record(ok=False)
                wait = _retry_after_seconds(response, cap=self.max_sleep_seconds)
                logger.warning(
                    "HTTP %d on %s (attempt %d/%d), waiting %s",
                    response.status_code,
                    url,
                    attempt + 1,
                    budget + 1,
                    f"{wait:.1f}s (Retry-After)" if wait is not None else "backoff",
                )
                self._sleep(wait if wait is not None else self._backoff(attempt))
                last_error = ApiError(
                    f"HTTP {response.status_code}", status_code=response.status_code, url=url
                )
                continue

            if response.status_code >= 400:
                raise ApiError(
                    f"HTTP {response.status_code} on {url}: {response.text[:300]}",
                    status_code=response.status_code,
                    url=url,
                )

            try:
                payload = response.json()
            except json.JSONDecodeError as exc:
                raise ApiError(f"non-JSON response from {url}: {exc}", url=url) from exc

            self.breaker.record(ok=True)
            self.cache.put(self.namespace, digest, payload)
            return payload  # type: ignore[no-any-return]

        self.stats.failures += 1
        message = f"exhausted {budget + 1} attempts on {url}: {last_error}"
        if on_error == "skip":
            logger.warning("%s -- skipping", message)
            return None
        raise ApiError(message, url=url) from last_error

    def get_bytes(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        suffix: str = ".bin",
        expect_prefix: bytes | None = None,
    ) -> bytes | None:
        """GET an opaque body, with the same caching, limiting and retries.

        Args:
            suffix: file extension for the cache blob, purely so a human can
                browse the cache directory.
            expect_prefix: magic bytes the body must start with. DailyMed answers
                a request for a nonexistent label version with an HTTP 200 and an
                HTML error page, so a status check alone is not enough to tell
                success from failure -- checking for ``PK`` on a ZIP is what
                actually distinguishes them.

        Returns:
            The body, or None when the response did not match `expect_prefix`.
        """
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}" if path else self.base_url
        digest = HttpCache.key(f"{self.namespace}:bytes", url, params)

        cached = self.cache.get_bytes(self.namespace, digest, suffix)
        if cached is not None:
            self.stats.cache_hits += 1
            return None if cached == b"__empty__" else cached

        if self.request_budget is not None and self.stats.requests >= self.request_budget:
            raise QuotaExceededError(
                f"request budget of {self.request_budget} exhausted for namespace "
                f"{self.namespace!r}",
                url=url,
            )

        send_params = dict(params or {})
        if self.api_key:
            send_params[self.api_key_param] = self.api_key

        assert self._client is not None, "client used after close()"
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self.stats.total_wait_seconds += self.limiter.acquire()
            self.stats.requests += 1
            try:
                response = self._client.get(url, params=send_params)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                self.stats.retries += 1
                self._sleep(self._backoff(attempt))
                continue

            self.stats.bytes_received += len(response.content)

            if response.status_code in RETRYABLE_STATUS:
                self.stats.retries += 1
                wait = _retry_after_seconds(response, cap=self.max_sleep_seconds)
                self._sleep(wait if wait is not None else self._backoff(attempt))
                last_error = ApiError(
                    f"HTTP {response.status_code}", status_code=response.status_code, url=url
                )
                continue

            if response.status_code == 404:
                self.stats.empty_results += 1
                self.cache.put_bytes(self.namespace, digest, suffix, b"__empty__")
                return None

            if response.status_code >= 400:
                raise ApiError(
                    f"HTTP {response.status_code} on {url}",
                    status_code=response.status_code,
                    url=url,
                )

            body = response.content
            if expect_prefix is not None and not body.startswith(expect_prefix):
                # A 200 carrying the wrong content type means the resource does
                # not exist despite the status. Cache the negative so a backfill
                # does not re-ask for every missing version on every run.
                logger.debug(
                    "%s returned %d bytes not starting with %r; treating as absent",
                    url,
                    len(body),
                    expect_prefix,
                )
                self.stats.empty_results += 1
                self.cache.put_bytes(self.namespace, digest, suffix, b"__empty__")
                return None

            self.cache.put_bytes(self.namespace, digest, suffix, body)
            return body

        raise ApiError(
            f"exhausted {self.max_retries + 1} attempts on {url}: {last_error}",
            url=url,
        ) from last_error
